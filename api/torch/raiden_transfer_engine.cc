// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/shared_ptr.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "api/torch/kv_cache_manager.h"
#include "core/raw_transfer_core.h"
#include "frameworks/torch/torch_nanobind_utils.h"
#include "kv_cache/kv_cache_manager_base.h"
#include "transport/socket_transport.h"
#include "torch/extension.h"  // IWYU pragma: keep

namespace nb = nanobind;

namespace tpu_raiden::kv_cache {
namespace {

using TensorList = std::vector<at::Tensor>;

[[noreturn]] void ThrowStatus(const std::string& context,
                              const absl::Status& status) {
  throw std::runtime_error(context + ": " + std::string(status.message()));
}

void CheckStatus(const std::string& context, const absl::Status& status) {
  if (!status.ok()) {
    ThrowStatus(context, status);
  }
}

void EmitTimingLog(const std::string& message) {
  LOG(INFO) << message;
  std::cerr << message << std::endl;
}

template <typename T>
T ValueOrThrow(const std::string& context, absl::StatusOr<T> value_or) {
  if (!value_or.ok()) {
    ThrowStatus(context, value_or.status());
  }
  return std::move(value_or).value();
}

absl::Status WriteExact(int fd, const void* buffer, size_t length) {
  const uint8_t* ptr = static_cast<const uint8_t*>(buffer);
  size_t remaining = length;
  while (remaining > 0) {
    ssize_t written = write(fd, ptr, remaining);
    if (written < 0) {
      if (errno == EINTR) continue;
      return absl::InternalError("socket write failed: " +
                                 std::string(std::strerror(errno)));
    }
    if (written == 0) {
      return absl::InternalError("socket closed during write");
    }
    ptr += written;
    remaining -= written;
  }
  return absl::OkStatus();
}

absl::Status ReadExact(int fd, void* buffer, size_t length) {
  uint8_t* ptr = static_cast<uint8_t*>(buffer);
  size_t remaining = length;
  while (remaining > 0) {
    ssize_t bytes_read = read(fd, ptr, remaining);
    if (bytes_read < 0) {
      if (errno == EINTR) continue;
      return absl::InternalError("socket read failed: " +
                                 std::string(std::strerror(errno)));
    }
    if (bytes_read == 0) {
      return absl::InternalError("socket closed during read");
    }
    ptr += bytes_read;
    remaining -= bytes_read;
  }
  return absl::OkStatus();
}

std::pair<std::string, int> SplitEndpoint(const std::string& endpoint) {
  const size_t colon = endpoint.rfind(':');
  if (colon == std::string::npos) {
    throw std::invalid_argument("endpoint must be host:port");
  }
  return {endpoint.substr(0, colon), std::stoi(endpoint.substr(colon + 1))};
}

int ConnectTcp(const std::string& endpoint) {
  auto [host, port] = SplitEndpoint(endpoint);
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    throw std::runtime_error("socket() failed: " +
                             std::string(std::strerror(errno)));
  }
  int opt = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
    close(fd);
    throw std::runtime_error("invalid IPv4 endpoint host: " + host);
  }
  if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    std::string err = std::strerror(errno);
    close(fd);
    throw std::runtime_error("connect(" + endpoint + ") failed: " + err);
  }
  return fd;
}

std::vector<std::vector<at::Tensor>> SingleShardLayers(
    const TensorList& kv_caches) {
  std::vector<std::vector<at::Tensor>> layers;
  layers.reserve(kv_caches.size());
  for (const auto& kv_cache : kv_caches) {
    layers.push_back({kv_cache});
  }
  return layers;
}

class RaidenTransferFuture {
 public:
  RaidenTransferFuture() = default;

  void Add(std::shared_ptr<raiden::PjRtCopyFuture> future) {
    futures_.push_back(std::move(future));
  }

  void Add(raiden::PjRtCopyFuture future) {
    futures_.push_back(
        std::make_shared<raiden::PjRtCopyFuture>(std::move(future)));
  }

  void AddAll(const std::shared_ptr<RaidenTransferFuture>& other) {
    futures_.insert(futures_.end(), other->futures_.begin(),
                    other->futures_.end());
  }

  void Await() {
    for (const auto& future : futures_) {
      if (future) {
        future->Await();
      }
    }
    futures_.clear();
  }

  bool IsReady() const {
    for (const auto& future : futures_) {
      if (future && !future->IsReady()) {
        return false;
      }
    }
    return true;
  }

 private:
  std::vector<std::shared_ptr<raiden::PjRtCopyFuture>> futures_;
};

struct RaidenStageResult {
  std::shared_ptr<RaidenTransferFuture> future;
  std::vector<KVCacheHostSpan> host_spans;
  int64_t total_bytes = 0;
  int64_t copy_segments = 0;
};

class RaidenTransferEngine {
 public:
  RaidenTransferEngine(const TensorList& kv_caches, int64_t tp_rank,
                       int64_t local_control_port, int64_t max_blocks,
                       int64_t num_slots, double timeout_s,
                       bool unsafe_skip_buffer_lock)
      : tp_rank_(tp_rank),
        local_control_port_(static_cast<int>(local_control_port)),
        local_data_port_(static_cast<int>(local_control_port) == 0
                             ? 0
                             : static_cast<int>(local_control_port) + 1),
        max_blocks_(max_blocks),
        num_slots_(num_slots),
        timeout_s_(timeout_s),
        unsafe_skip_buffer_lock_(unsafe_skip_buffer_lock) {
    if (max_blocks_ <= 0) {
      throw std::invalid_argument("max_blocks must be positive");
    }
    if (num_slots <= 0) {
      throw std::invalid_argument("num_slots must be positive");
    }
    RegisterKvCache(kv_caches);
    transport_ = std::make_unique<tpu_raiden::transport::SocketTransport>(
        local_data_port_);
    local_data_port_ = transport_->local_port();
    StartControlServer();
  }

  ~RaidenTransferEngine() { StopControlServer(); }

  std::vector<int64_t> RegisterKvCache(const TensorList& kv_caches) {
    kv_caches_ = kv_caches;
    if (num_slots_ <= 0) {
      throw std::invalid_argument("num_slots must be configured first");
    }
    if (max_blocks_ >
        std::numeric_limits<int>::max() / std::max<int64_t>(num_slots_, 1)) {
      throw std::invalid_argument("host staging block count exceeds int range");
    }
    
    bool has_tpu = false;
    for (const auto& t : kv_caches_) {
      if (t.device().type() == c10::DeviceType::PrivateUse1) {
        has_tpu = true;
        break;
      }
    }
    
    if (has_tpu) {
      const int host_blocks_to_allocate =
          static_cast<int>(num_slots_ * max_blocks_);
      kv_transfer_ = std::make_unique<tpu_raiden::torch::KVCacheManager>(
          SingleShardLayers(kv_caches_), /*block_size=*/1,
          /*local_port=*/std::nullopt, host_blocks_to_allocate,
          /*external_host_ptrs=*/std::nullopt, unsafe_skip_buffer_lock_);
      CheckStatus("configure KVCacheManager host staging slots",
                  kv_transfer_->ConfigureHostStagingSlots(num_slots_,
                                                          max_blocks_));
    }
    InitializeSlotPool(num_slots_);

    std::vector<int64_t> region_ids;
    region_ids.reserve(kv_caches_.size());
    for (size_t i = 0; i < kv_caches_.size(); ++i) {
      region_ids.push_back(static_cast<int64_t>(i));
    }
    return region_ids;
  }

  void RegisterHostBuffers(nb::object /*host_pool*/, int64_t tp_rank) {
    tp_rank_ = tp_rank;
  }

  bool UsesPreparedTpuBuffers() const { return kv_transfer_ != nullptr; }

  nb::tuple StageD2H(int64_t slot_idx, int64_t num_blocks,
                     const std::vector<int64_t>& block_ids) {
    RaidenStageResult result = IssueD2H(slot_idx, num_blocks, block_ids);
    return nb::make_tuple(result.future, kv_caches_,
                          HostSpanMemoryViews(result.host_spans),
                          result.total_bytes);
  }

  void StageD2HSync(int64_t slot_idx, int64_t num_blocks,
                    const std::vector<int64_t>& block_ids) {
    RaidenStageResult result = IssueD2H(slot_idx, num_blocks, block_ids);
    result.future->Await();
  }

  nb::tuple CommitH2D(int64_t slot_idx, int64_t num_blocks,
                      const std::vector<int64_t>& local_block_ids) {
    if (num_blocks != static_cast<int64_t>(local_block_ids.size())) {
      throw std::invalid_argument("num_blocks must match len(local_block_ids)");
    }
    auto t0 = std::chrono::steady_clock::now();
    RaidenStageResult result = IssueH2D(slot_idx, num_blocks, local_block_ids);
    auto t_issued = std::chrono::steady_clock::now();
    result.future->Await();
    auto t_done = std::chrono::steady_clock::now();
    return nb::make_tuple(DurationMs(t0, t_issued),
                          DurationMs(t_issued, t_done), DurationMs(t0, t_done),
                          result.total_bytes);
  }

  nb::object RankLayerViews(int64_t slot_idx, int64_t rank,
                            int64_t num_blocks) {
    if (rank != tp_rank_) {
      throw std::invalid_argument("Raiden internal slots are per-rank only");
    }
    return HostSpanMemoryViews(LayerSpans(slot_idx, num_blocks));
  }

  void UnpackRankLayers(int64_t slot_idx, int64_t rank, int64_t num_blocks,
                        nb::object layer_buffers) {
    if (rank != tp_rank_) {
      throw std::invalid_argument("Raiden internal slots are per-rank only");
    }
    std::vector<KVCacheHostSpan> spans = LayerSpans(slot_idx, num_blocks);
    size_t idx = 0;
    for (nb::handle item : layer_buffers) {
      if (idx >= spans.size()) {
        throw std::invalid_argument("too many layer buffers");
      }
      nb::ndarray<> source = nb::cast<nb::ndarray<>>(item);
      const size_t source_bytes = source.nbytes();
      if (source_bytes != spans[idx].nbytes) {
        throw std::invalid_argument("layer buffer size mismatch");
      }
      std::memcpy(spans[idx].ptr, source.data(), spans[idx].nbytes);
      ++idx;
    }
    if (idx != spans.size()) {
      throw std::invalid_argument("too few layer buffers");
    }
  }

  int64_t SubmitD2H(int64_t slot_idx, int64_t num_blocks,
                    const std::vector<int64_t>& block_ids) {
    PendingOperation op;
    op.future = IssueD2H(slot_idx, num_blocks, block_ids).future;
    return StorePending(std::move(op));
  }

  int64_t SubmitH2D(int64_t slot_idx, int64_t num_blocks,
                    const std::vector<int64_t>& local_block_ids) {
    PendingOperation op;
    op.future = IssueH2D(slot_idx, num_blocks, local_block_ids).future;
    return StorePending(std::move(op));
  }

  int64_t RegisterSend(const std::string& req_id, uint64_t uuid,
                       const std::vector<int64_t>& block_ids) {
    const auto register_start = std::chrono::steady_clock::now();
    if (block_ids.empty()) {
      return 0;
    }

    {
      std::lock_guard<std::mutex> lock(mu_);
      if (pending_acks_.erase(uuid) > 0) {
        done_sending_.insert(req_id);
        return 0;
      }
    }

    auto entry = std::make_shared<SendEntry>();
    entry->req_id = req_id;
    entry->uuid = uuid;
    entry->registered_num_blocks = static_cast<int64_t>(block_ids.size());
    entry->registered_block_ids = block_ids;
    for (int64_t block_id : block_ids) {
      entry->registered_block_set.insert(block_id);
    }
    entry->deadline = DeadlineFromNow();
    entry->register_start = register_start;

    {
      std::lock_guard<std::mutex> lock(mu_);
      send_entries_[uuid] = entry;
    }
    cv_.notify_all();

    std::ostringstream timing;
    timing << "RAIDEN_TIMING event=producer_register"
           << " req_id=" << req_id << " uuid=" << uuid << " rank=" << tp_rank_
           << " blocks=" << block_ids.size() << " enqueue_ms="
           << DurationMs(register_start, std::chrono::steady_clock::now())
           << " failed=0";
    EmitTimingLog(timing.str());
    return static_cast<int64_t>(uuid);
  }

  int64_t SubmitLoad(const std::string& req_id, uint64_t uuid,
                     const std::string& remote_endpoint,
                     const std::vector<int64_t>& remote_block_ids,
                     const std::vector<int64_t>& local_block_ids) {
    auto load_promise = std::make_shared<std::promise<void>>();
    const auto submit_start = std::chrono::steady_clock::now();
    CopyPlan load_plan = BuildLoadCopyPlan(remote_block_ids, local_block_ids);
    {
      std::lock_guard<std::mutex> lock(worker_threads_mu_);
      worker_threads_.emplace_back([this, req_id, uuid, remote_endpoint,
                                    submit_start,
                                    load_plan = std::move(load_plan),
                                    load_promise]() {
        const auto worker_start = std::chrono::steady_clock::now();
        bool failed = false;
        bool report_recv_done = true;
        bool release_only = false;
        int64_t slot_idx = -1;
        int64_t h2h_bytes = 0;
        int64_t h2d_bytes = 0;
        int64_t h2d_segments = 0;
        size_t h2h_layers = 0;
        double slot_ms = 0.0;
        double control_setup_ms = 0.0;
        double descriptor_ms = 0.0;
        double h2h_ms = 0.0;
        double host_reorder_ms = 0.0;
        double h2d_issue_ms = 0.0;
        double h2d_wait_ms = 0.0;
        double h2d_total_ms = 0.0;
        double ack_ms = 0.0;
        try {
          if (load_plan.num_blocks == 0) {
            report_recv_done = false;
            release_only = true;
            const auto ack_start = std::chrono::steady_clock::now();
            AckRemote(remote_endpoint, uuid);
            ack_ms = DurationMs(ack_start, std::chrono::steady_clock::now());
          } else {
            const auto slot_start = std::chrono::steady_clock::now();
            slot_idx = AcquireSlot();
            slot_ms = DurationMs(slot_start, std::chrono::steady_clock::now());

            const auto control_start = std::chrono::steady_clock::now();
            int control_fd = ConnectTcp(remote_endpoint);
            auto control_cleanup =
                std::unique_ptr<int, void (*)(int*)>(&control_fd, [](int* p) {
                  if (p && *p >= 0) close(*p);
                });
            ControlRequestHeader stream_request;
            stream_request.magic = kControlMagic;
            stream_request.op = kOpPullStream;
            stream_request.uuid = uuid;
            stream_request.num_blocks =
                static_cast<uint64_t>(load_plan.num_blocks);
            CheckStatus("control pull stream write",
                        WriteExact(control_fd, &stream_request,
                                   sizeof(stream_request)));
            WriteBlockIds(control_fd, load_plan.producer_remote_block_ids);
            ControlResponseHeader response =
                ReadControlResponseHeader(control_fd);
            control_setup_ms =
                DurationMs(control_start, std::chrono::steady_clock::now());

            std::vector<KVCacheHostSpan> local_spans =
                LayerSpans(slot_idx,
                           static_cast<int64_t>(load_plan.num_blocks));
            if (response.num_layers != local_spans.size()) {
              throw std::runtime_error("remote layer descriptor count mismatch");
            }
            for (size_t i = 0; i < local_spans.size(); ++i) {
              PullLayerDescriptor layer;
              const auto descriptor_start = std::chrono::steady_clock::now();
              CheckStatus("control stream layer descriptor read",
                          ReadExact(control_fd, &layer, sizeof(layer)));
              descriptor_ms += DurationMs(
                  descriptor_start, std::chrono::steady_clock::now());
              if (layer.len != static_cast<uint64_t>(local_spans[i].nbytes)) {
                throw std::runtime_error(
                    "remote layer descriptor size mismatch");
              }
              ++h2h_layers;
              h2h_bytes += static_cast<int64_t>(layer.len);
              peregrine::Request request;
              request.op = peregrine::Op::kRead;
              request.laddr = local_spans[i].ptr;
              request.raddr = reinterpret_cast<uint8_t*>(layer.addr);
              request.len = layer.len;
              const auto h2h_start = std::chrono::steady_clock::now();
              auto handle_or = transport_->Post(
                  EndpointWithPort(remote_endpoint, response.data_port),
                  request);
              if (!handle_or.ok()) {
                ThrowStatus("SocketTransport read failed", handle_or.status());
              }
              auto status_or = transport_->Poll(handle_or.value());
              if (!status_or.ok() ||
                  status_or.value() != peregrine::Status::kSuccess) {
                throw std::runtime_error(
                    "SocketTransport read did not succeed");
              }
              h2h_ms += DurationMs(h2h_start, std::chrono::steady_clock::now());
              if (load_plan.RequiresHostReorder()) {
                const auto reorder_start = std::chrono::steady_clock::now();
                ReorderCompactBlocks(local_spans[i],
                                     load_plan.host_dst_to_src);
                host_reorder_ms +=
                    DurationMs(reorder_start, std::chrono::steady_clock::now());
              }
            }

            const auto h2d_start = std::chrono::steady_clock::now();
            RaidenStageResult h2d = IssueH2D(slot_idx, load_plan.num_blocks,
                                             load_plan.h2d_local_block_ids);
            const auto h2d_issued = std::chrono::steady_clock::now();
            h2d.future->Await();
            const auto h2d_done = std::chrono::steady_clock::now();
            h2d_bytes = h2d.total_bytes;
            h2d_segments = h2d.copy_segments;
            h2d_issue_ms = DurationMs(h2d_start, h2d_issued);
            h2d_wait_ms = DurationMs(h2d_issued, h2d_done);
            h2d_total_ms = DurationMs(h2d_start, h2d_done);

            const auto ack_start = std::chrono::steady_clock::now();
            AckRemote(remote_endpoint, uuid);
            ack_ms = DurationMs(ack_start, std::chrono::steady_clock::now());
          }
        } catch (const std::exception& e) {
          failed = true;
          LOG(ERROR) << "Raiden load failed for req=" << req_id
                     << " uuid=" << uuid << ": " << e.what();
        } catch (...) {
          failed = true;
          LOG(ERROR) << "Raiden load failed for req=" << req_id
                     << " uuid=" << uuid;
        }
        if (slot_idx >= 0) {
          std::lock_guard<std::mutex> lock(mu_);
          ReleaseSlotLocked(slot_idx);
        }
        {
          std::lock_guard<std::mutex> lock(mu_);
          if (report_recv_done) {
            done_recving_.insert(req_id);
            if (failed) failed_recving_.insert(req_id);
          }
        }
        const auto done = std::chrono::steady_clock::now();
        std::ostringstream timing;
        timing << "RAIDEN_TIMING event=consumer_load"
               << " req_id=" << req_id << " uuid=" << uuid
               << " rank=" << tp_rank_ << " endpoint=" << remote_endpoint
               << " blocks=" << load_plan.num_blocks << " layers=" << h2h_layers
               << " bytes=" << h2h_bytes << " h2d_bytes=" << h2d_bytes
               << " h2d_segments=" << h2d_segments
               << " queue_delay_ms=" << DurationMs(submit_start, worker_start)
               << " slot_ms=" << slot_ms
               << " control_setup_ms=" << control_setup_ms
               << " descriptor_ms=" << descriptor_ms << " h2h_ms=" << h2h_ms
               << " host_reorder_ms=" << host_reorder_ms
               << " h2d_issue_ms=" << h2d_issue_ms
               << " h2d_wait_ms=" << h2d_wait_ms
               << " h2d_total_ms=" << h2d_total_ms << " ack_ms=" << ack_ms
               << " total_ms=" << DurationMs(submit_start, done)
               << " release_only=" << (release_only ? 1 : 0)
               << " failed=" << (failed ? 1 : 0);
        EmitTimingLog(timing.str());
        load_promise->set_value();
      });
    }
    PendingOperation op;
    op.load_promise = load_promise;
    return StorePending(std::move(op));
  }

  nb::tuple PollFinished() {
    std::vector<std::string> done_sending;
    std::vector<std::string> done_recving;
    std::vector<std::string> failed_recving;
    {
      std::lock_guard<std::mutex> lock(mu_);
      const auto now = std::chrono::steady_clock::now();
      for (auto it = send_entries_.begin(); it != send_entries_.end();) {
        const auto& entry = it->second;
        if (entry->deadline <= now) {
          done_sending_.insert(entry->req_id);
          ReleaseEntrySlotLocked(entry);
          it = send_entries_.erase(it);
        } else {
          ++it;
        }
      }
      done_sending.assign(done_sending_.begin(), done_sending_.end());
      done_recving.assign(done_recving_.begin(), done_recving_.end());
      failed_recving.assign(failed_recving_.begin(), failed_recving_.end());
      done_sending_.clear();
      done_recving_.clear();
      failed_recving_.clear();
    }
    return nb::make_tuple(done_sending, done_recving, failed_recving);
  }

  std::vector<int64_t> PollTransferOps() {
    std::vector<int64_t> done;
    for (auto it = pending_.begin(); it != pending_.end();) {
      if (!it->second.future || it->second.future->IsReady()) {
        if (it->second.future) {
          it->second.future->Await();
        }
        done.push_back(it->first);
        it = pending_.erase(it);
      } else {
        ++it;
      }
    }
    return done;
  }

  void WaitTransfer(int64_t op_id) {
    auto it = pending_.find(op_id);
    if (it == pending_.end()) {
      throw std::invalid_argument("unknown Raiden transfer op id");
    }
    if (it->second.future) {
      it->second.future->Await();
    }
    if (it->second.load_promise) {
      it->second.load_promise->get_future().wait();
    }
    pending_.erase(it);
  }

  int local_control_port() const { return local_control_port_; }
  int local_data_port() const { return local_data_port_; }

  int64_t CountCopySegmentsForTesting(
      const std::vector<int64_t>& block_ids) const {
    return static_cast<int64_t>(
        Offsets(block_ids, /*source_is_compact=*/false).sizes.size());
  }

  nb::dict SendCopyPlanForTesting(const std::vector<int64_t>& block_ids) const {
    return CopyPlanToDict(BuildProducerCopyPlan(block_ids));
  }

  nb::dict LoadCopyPlanForTesting(
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids) const {
    return CopyPlanToDict(BuildLoadCopyPlan(remote_block_ids, local_block_ids));
  }

 private:
  struct CopySpec {
    std::vector<int64_t> src_offsets;
    std::vector<int64_t> dst_offsets;
    std::vector<int64_t> sizes;
  };

  static KVCacheCopySpec ToKVCacheCopySpec(const CopySpec& spec) {
    return {.src_offsets = spec.src_offsets,
            .dst_offsets = spec.dst_offsets,
            .sizes = spec.sizes};
  }

  struct CopyPlan {
    int64_t num_blocks = 0;
    std::vector<int64_t> requested_remote_block_ids;
    std::vector<int64_t> requested_local_block_ids;
    std::vector<int64_t> producer_remote_block_ids;
    std::vector<int64_t> h2d_local_block_ids;
    std::vector<size_t> host_dst_to_src;
    CopySpec d2h_copy;
    CopySpec h2d_copy;

    bool RequiresHostReorder() const { return !host_dst_to_src.empty(); }
  };

  struct PendingOperation {
    std::shared_ptr<RaidenTransferFuture> future;
    std::shared_ptr<std::promise<void>> load_promise;
  };

  struct SendEntry {
    std::string req_id;
    uint64_t uuid = 0;
    int64_t slot_idx = -1;
    int64_t num_blocks = 0;
    int64_t registered_num_blocks = 0;
    int64_t total_bytes = 0;
    int64_t copy_segments = 0;
    std::vector<int64_t> registered_block_ids;
    std::set<int64_t> registered_block_set;
    std::chrono::steady_clock::time_point register_start;
    std::chrono::steady_clock::time_point d2h_issue_start;
    std::chrono::steady_clock::time_point d2h_issue_done;
    std::chrono::steady_clock::time_point d2h_done;
    bool stage_done = false;
    bool failed = false;
    bool pull_started = false;
    bool slot_released = false;
    std::chrono::steady_clock::time_point deadline;
  };

  void ReleaseEntrySlotLocked(const std::shared_ptr<SendEntry>& entry) {
    if (!entry || entry->slot_idx < 0 || entry->slot_released) return;
    ReleaseSlotLocked(entry->slot_idx);
    entry->slot_released = true;
  }

  struct PullLayerDescriptor {
    uint64_t addr = 0;
    uint64_t len = 0;
  };

  struct alignas(8) ControlRequestHeader {
    uint32_t magic = 0x52414944;  // "RAID"
    uint32_t op = 0;
    uint64_t uuid = 0;
    uint64_t num_blocks = 0;
  };

  struct alignas(8) ControlResponseHeader {
    uint32_t magic = 0x44494152;  // "DIAR"
    int32_t status = 0;
    uint32_t num_layers = 0;
    uint32_t data_port = 0;
    uint64_t message_len = 0;
  };

  static constexpr uint32_t kControlMagic = 0x52414944;
  static constexpr uint32_t kResponseMagic = 0x44494152;
  static constexpr uint32_t kOpAck = 2;
  static constexpr uint32_t kOpPullStream = 3;

  static double DurationMs(std::chrono::steady_clock::time_point start,
                           std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
  }

  static void WriteBlockIds(int fd, const std::vector<int64_t>& block_ids) {
    if (block_ids.empty()) return;
    CheckStatus(
        "control block ids write",
        WriteExact(fd, block_ids.data(), block_ids.size() * sizeof(int64_t)));
  }

  static std::vector<int64_t> ReadBlockIds(int fd, uint64_t num_blocks) {
    if (num_blocks == 0) return {};
    if (num_blocks >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      throw std::invalid_argument("num_blocks is too large");
    }
    std::vector<int64_t> block_ids(static_cast<size_t>(num_blocks));
    CheckStatus(
        "control block ids read",
        ReadExact(fd, block_ids.data(), block_ids.size() * sizeof(int64_t)));
    return block_ids;
  }

  static CopySpec Offsets(const std::vector<int64_t>& block_ids,
                          bool source_is_compact) {
    const int64_t n = static_cast<int64_t>(block_ids.size());
    CopySpec spec;
    spec.src_offsets.reserve(block_ids.size());
    spec.dst_offsets.reserve(block_ids.size());
    spec.sizes.reserve(block_ids.size());
    for (int64_t start = 0; start < n;) {
      int64_t end = start + 1;
      while (end < n && block_ids[end] == block_ids[end - 1] + 1) {
        ++end;
      }
      const int64_t run_size = end - start;
      if (source_is_compact) {
        spec.src_offsets.push_back(start);
        spec.dst_offsets.push_back(block_ids[start]);
      } else {
        spec.src_offsets.push_back(block_ids[start]);
        spec.dst_offsets.push_back(start);
      }
      spec.sizes.push_back(run_size);
      start = end;
    }
    return spec;
  }

  static nb::dict CopySpecToDict(const CopySpec& spec) {
    nb::dict out;
    out["src_offsets"] = spec.src_offsets;
    out["dst_offsets"] = spec.dst_offsets;
    out["sizes"] = spec.sizes;
    return out;
  }

  static nb::dict CopyPlanToDict(const CopyPlan& plan) {
    nb::dict out;
    out["num_blocks"] = plan.num_blocks;
    out["requested_remote_block_ids"] = plan.requested_remote_block_ids;
    out["requested_local_block_ids"] = plan.requested_local_block_ids;
    out["producer_remote_block_ids"] = plan.producer_remote_block_ids;
    out["h2d_local_block_ids"] = plan.h2d_local_block_ids;
    out["host_dst_to_src"] = plan.host_dst_to_src;
    out["requires_host_reorder"] = plan.RequiresHostReorder();
    out["d2h_copy"] = CopySpecToDict(plan.d2h_copy);
    out["h2d_copy"] = CopySpecToDict(plan.h2d_copy);
    return out;
  }

  static std::vector<int64_t> CanonicalSendBlockIds(
      const std::vector<int64_t>& block_ids) {
    std::vector<int64_t> ordered = block_ids;
    std::stable_sort(ordered.begin(), ordered.end());
    return ordered;
  }

  static void ValidateRequestedBlocks(
      const SendEntry& entry, const std::vector<int64_t>& requested_block_ids) {
    if (requested_block_ids.empty()) {
      throw std::invalid_argument(
          "pull stream requested no blocks; use ack-only path");
    }
    std::set<int64_t> seen;
    for (int64_t block_id : requested_block_ids) {
      if (entry.registered_block_set.find(block_id) ==
          entry.registered_block_set.end()) {
        throw std::invalid_argument(
            "pull stream requested block not registered by producer");
      }
      if (!seen.insert(block_id).second) {
        throw std::invalid_argument(
            "pull stream requested duplicate producer block id");
      }
    }
  }

  static CopyPlan BuildProducerCopyPlan(const std::vector<int64_t>& block_ids) {
    CopyPlan plan;
    plan.num_blocks = static_cast<int64_t>(block_ids.size());
    plan.requested_remote_block_ids = block_ids;
    plan.producer_remote_block_ids = CanonicalSendBlockIds(block_ids);
    plan.d2h_copy =
        Offsets(plan.producer_remote_block_ids, /*source_is_compact=*/false);
    return plan;
  }

  static CopyPlan BuildLoadCopyPlan(
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids) {
    if (remote_block_ids.size() != local_block_ids.size()) {
      throw std::invalid_argument(
          "remote_block_ids and local_block_ids must have same length");
    }
    CopyPlan plan;
    plan.num_blocks = static_cast<int64_t>(remote_block_ids.size());
    plan.requested_remote_block_ids = remote_block_ids;
    plan.requested_local_block_ids = local_block_ids;
    if (remote_block_ids.empty()) {
      return plan;
    }
    std::vector<size_t> remote_order(remote_block_ids.size());
    std::vector<size_t> local_order(local_block_ids.size());
    for (size_t i = 0; i < remote_order.size(); ++i) {
      remote_order[i] = i;
      local_order[i] = i;
    }
    std::stable_sort(remote_order.begin(), remote_order.end(),
                     [&](size_t a, size_t b) {
                       return remote_block_ids[a] < remote_block_ids[b];
                     });
    std::stable_sort(local_order.begin(), local_order.end(),
                     [&](size_t a, size_t b) {
                       return local_block_ids[a] < local_block_ids[b];
                     });

    plan.producer_remote_block_ids.reserve(remote_order.size());
    plan.h2d_local_block_ids.reserve(local_order.size());
    plan.host_dst_to_src.reserve(local_order.size());
    std::vector<size_t> source_pos_by_original_idx(remote_order.size());
    for (size_t source_pos = 0; source_pos < remote_order.size();
         ++source_pos) {
      const size_t original_idx = remote_order[source_pos];
      plan.producer_remote_block_ids.push_back(remote_block_ids[original_idx]);
      source_pos_by_original_idx[original_idx] = source_pos;
    }
    bool identity_reorder = true;
    for (size_t dst_pos = 0; dst_pos < local_order.size(); ++dst_pos) {
      const size_t original_idx = local_order[dst_pos];
      const size_t src_pos = source_pos_by_original_idx[original_idx];
      plan.h2d_local_block_ids.push_back(local_block_ids[original_idx]);
      plan.host_dst_to_src.push_back(src_pos);
      identity_reorder = identity_reorder && (src_pos == dst_pos);
    }
    if (identity_reorder) {
      plan.host_dst_to_src.clear();
    }
    plan.h2d_copy =
        Offsets(plan.h2d_local_block_ids, /*source_is_compact=*/true);
    return plan;
  }

  static void ReorderCompactBlocks(const KVCacheHostSpan& compact_blocks,
                                   const std::vector<size_t>& dst_to_src) {
    if (dst_to_src.empty()) return;
    const size_t num_blocks = dst_to_src.size();
    if (num_blocks == 0 || compact_blocks.nbytes % num_blocks != 0) {
      throw std::invalid_argument("host reorder view has invalid block layout");
    }
    const size_t block_bytes =
        static_cast<size_t>(compact_blocks.nbytes) / num_blocks;
    const size_t total_bytes = compact_blocks.nbytes;
    const uint8_t* src = compact_blocks.ptr;
    std::vector<uint8_t> reordered(total_bytes);
    for (size_t dst_idx = 0; dst_idx < num_blocks; ++dst_idx) {
      const size_t src_idx = dst_to_src[dst_idx];
      if (src_idx >= num_blocks) {
        throw std::out_of_range("host reorder source index out of range");
      }
      std::memcpy(reordered.data() + dst_idx * block_bytes,
                  src + src_idx * block_bytes, block_bytes);
    }
    std::memcpy(compact_blocks.ptr, reordered.data(), total_bytes);
  }

  std::vector<KVCacheHostSpan> LayerSpans(int64_t slot_idx,
                                          int64_t num_blocks) {
    if (!kv_transfer_) {
      throw std::runtime_error("KV cache manager is not registered");
    }
    if (num_blocks < 0 || num_blocks > max_blocks_) {
      throw std::out_of_range("num_blocks out of range");
    }
    std::vector<KVCacheHostSpan> spans;
    spans.reserve(kv_transfer_->num_layers());
    for (size_t layer_idx = 0; layer_idx < kv_transfer_->num_layers();
         ++layer_idx) {
      spans.push_back(ValueOrThrow(
          "Failed to get KVCacheManager host staging span",
          kv_transfer_->HostSpan(layer_idx, /*shard_idx=*/0, slot_idx,
                                 num_blocks)));
    }
    return spans;
  }

  static nb::list HostSpanMemoryViews(
      const std::vector<KVCacheHostSpan>& host_spans) {
    nb::list views;
    for (const KVCacheHostSpan& span : host_spans) {
      if (span.nbytes >
          static_cast<size_t>(std::numeric_limits<ssize_t>::max())) {
        throw std::overflow_error("host span is too large for Python view");
      }
      PyObject* mv = PyMemoryView_FromMemory(reinterpret_cast<char*>(span.ptr),
                                             static_cast<ssize_t>(span.nbytes),
                                             PyBUF_WRITE);
      if (mv == nullptr) {
        throw std::runtime_error("Failed to create Python memoryview");
      }
      views.append(nb::steal<nb::object>(mv));
    }
    return views;
  }

  RaidenStageResult IssueD2H(int64_t slot_idx, int64_t num_blocks,
                             const std::vector<int64_t>& block_ids) {
    if (num_blocks != static_cast<int64_t>(block_ids.size())) {
      throw std::invalid_argument("num_blocks must match len(block_ids)");
    }
    CopySpec copy_spec = Offsets(block_ids, /*source_is_compact=*/false);
    KVCacheCopySpec transfer_spec = ToKVCacheCopySpec(copy_spec);
    std::vector<KVCacheHostSpan> host_spans = LayerSpans(slot_idx, num_blocks);
    auto future = std::make_shared<RaidenTransferFuture>();
    int64_t total_bytes = 0;
    for (size_t i = 0; i < kv_transfer_->num_layers(); ++i) {
      total_bytes += static_cast<int64_t>(host_spans[i].nbytes);
      future->Add(ValueOrThrow("Failed to issue D2H transfer",
                               kv_transfer_->D2hToHostSlot(
                                   i, slot_idx, num_blocks, transfer_spec)));
    }
    return {.future = std::move(future),
            .host_spans = std::move(host_spans),
            .total_bytes = total_bytes,
            .copy_segments = static_cast<int64_t>(copy_spec.sizes.size())};
  }

  RaidenStageResult IssueH2D(int64_t slot_idx, int64_t num_blocks,
                             const std::vector<int64_t>& local_block_ids) {
    if (num_blocks != static_cast<int64_t>(local_block_ids.size())) {
      throw std::invalid_argument("num_blocks must match len(local_block_ids)");
    }
    CopySpec copy_spec = Offsets(local_block_ids, /*source_is_compact=*/true);
    KVCacheCopySpec transfer_spec = ToKVCacheCopySpec(copy_spec);
    std::vector<KVCacheHostSpan> host_spans = LayerSpans(slot_idx, num_blocks);
    auto future = std::make_shared<RaidenTransferFuture>();
    int64_t total_bytes = 0;
    for (size_t i = 0; i < kv_transfer_->num_layers(); ++i) {
      total_bytes += static_cast<int64_t>(host_spans[i].nbytes);
      future->Add(ValueOrThrow("Failed to issue H2D transfer",
                               kv_transfer_->H2dFromHostSlot(
                                   i, slot_idx, num_blocks, transfer_spec)));
    }
    return {.future = std::move(future),
            .host_spans = std::move(host_spans),
            .total_bytes = total_bytes,
            .copy_segments = static_cast<int64_t>(copy_spec.sizes.size())};
  }

  int64_t StorePending(PendingOperation op) {
    const int64_t op_id = next_op_id_++;
    pending_[op_id] = std::move(op);
    return op_id;
  }

  std::chrono::steady_clock::time_point DeadlineFromNow() const {
    return std::chrono::steady_clock::now() +
           std::chrono::milliseconds(static_cast<int64_t>(timeout_s_ * 1000.0));
  }

  void InitializeSlotPool(int64_t num_slots) {
    free_slots_.clear();
    for (int64_t slot = 0; slot < num_slots; ++slot) {
      free_slots_.push_back(slot);
    }
  }

  int64_t AcquireSlot() {
    std::lock_guard<std::mutex> lock(mu_);
    return AcquireSlotLocked();
  }

  int64_t AcquireSlotLocked() {
    if (free_slots_.empty()) {
      throw std::runtime_error("Raiden host slot pool exhausted");
    }
    int64_t slot = free_slots_.front();
    free_slots_.pop_front();
    return slot;
  }

  void ReleaseSlotLocked(int64_t slot_idx) {
    if (slot_idx < 0) return;
    free_slots_.push_back(slot_idx);
  }

  std::string EndpointWithPort(const std::string& endpoint, int port) const {
    auto [host, ignored_port] = SplitEndpoint(endpoint);
    (void)ignored_port;
    return host + ":" + std::to_string(port);
  }

  PullLayerDescriptor LayerDescriptor(int64_t slot_idx, int64_t num_blocks,
                                      size_t layer_idx) {
    KVCacheHostSpan span =
        ValueOrThrow("Failed to get KVCacheManager host staging descriptor",
                     kv_transfer_->HostSpan(layer_idx, /*shard_idx=*/0,
                                            slot_idx, num_blocks));
    return {reinterpret_cast<uint64_t>(span.ptr),
            static_cast<uint64_t>(span.nbytes)};
  }

  void StartControlServer() {
    control_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (control_fd_ < 0) {
      throw std::runtime_error("failed to create Raiden control socket");
    }
    int opt = 1;
    setsockopt(control_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(control_fd_, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(local_control_port_);
    if (bind(control_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) <
        0) {
      std::string err = std::strerror(errno);
      close(control_fd_);
      control_fd_ = -1;
      throw std::runtime_error("bind Raiden control socket failed: " + err);
    }
    socklen_t len = sizeof(addr);
    if (getsockname(control_fd_, reinterpret_cast<sockaddr*>(&addr), &len) ==
        0) {
      local_control_port_ = ntohs(addr.sin_port);
    }
    if (listen(control_fd_, 128) < 0) {
      std::string err = std::strerror(errno);
      close(control_fd_);
      control_fd_ = -1;
      throw std::runtime_error("listen Raiden control socket failed: " + err);
    }
    control_thread_ = std::thread([this]() { ControlLoop(); });
  }

  void StopControlServer() {
    stopping_ = true;
    if (control_fd_ >= 0) {
      shutdown(control_fd_, SHUT_RDWR);
      close(control_fd_);
      control_fd_ = -1;
    }
    cv_.notify_all();
    if (control_thread_.joinable()) {
      control_thread_.join();
    }
    {
      std::lock_guard<std::mutex> lock(control_workers_mu_);
      for (auto& thread : control_workers_) {
        if (thread.joinable()) thread.join();
      }
      control_workers_.clear();
    }
    {
      std::lock_guard<std::mutex> lock(worker_threads_mu_);
      for (auto& thread : worker_threads_) {
        if (thread.joinable()) thread.join();
      }
      worker_threads_.clear();
    }
  }

  void ControlLoop() {
    while (!stopping_) {
      pollfd pfd;
      pfd.fd = control_fd_;
      pfd.events = POLLIN;
      int ret = poll(&pfd, 1, 50);
      if (ret <= 0) continue;
      sockaddr_in client_addr;
      socklen_t len = sizeof(client_addr);
      int client_fd =
          accept(control_fd_, reinterpret_cast<sockaddr*>(&client_addr), &len);
      if (client_fd < 0) continue;
      int opt = 1;
      setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
      std::lock_guard<std::mutex> lock(control_workers_mu_);
      control_workers_.emplace_back(
          [this, client_fd]() { HandleControlConnection(client_fd); });
    }
  }

  void HandleControlConnection(int fd) {
    while (!stopping_) {
      ControlRequestHeader request;
      absl::Status s = ReadExact(fd, &request, sizeof(request));
      if (!s.ok()) break;
      if (request.magic != kControlMagic) {
        WriteControlError(fd, "bad control magic");
        break;
      }
      if (request.op == kOpPullStream) {
        HandlePullStreamRequest(fd, request);
      } else if (request.op == kOpAck) {
        AckSend(request.uuid);
        WriteControlOk(fd, 0);
      } else {
        WriteControlError(fd, "unknown control op");
      }
    }
    close(fd);
  }

  void WriteControlOk(int fd, size_t num_layers) {
    ControlResponseHeader response;
    response.magic = kResponseMagic;
    response.status = 0;
    response.num_layers = static_cast<uint32_t>(num_layers);
    response.data_port = static_cast<uint32_t>(local_data_port_);
    response.message_len = 0;
    (void)WriteExact(fd, &response, sizeof(response));
  }

  void WriteControlError(int fd, const std::string& message) {
    ControlResponseHeader response;
    response.magic = kResponseMagic;
    response.status = 1;
    response.num_layers = 0;
    response.data_port = static_cast<uint32_t>(local_data_port_);
    response.message_len = message.size();
    absl::Status s = WriteExact(fd, &response, sizeof(response));
    if (!s.ok()) return;
    if (!message.empty()) {
      (void)WriteExact(fd, message.data(), message.size());
    }
  }

  void HandlePullStreamRequest(int fd, const ControlRequestHeader& request) {
    const auto request_start = std::chrono::steady_clock::now();
    std::shared_ptr<SendEntry> entry;
    std::vector<int64_t> requested_block_ids;
    try {
      if (request.num_blocks > static_cast<uint64_t>(max_blocks_)) {
        throw std::invalid_argument("requested block count exceeds max_blocks");
      }
      requested_block_ids = ReadBlockIds(fd, request.num_blocks);
    } catch (const std::exception& e) {
      WriteControlError(fd, e.what());
      return;
    }

    const auto requested_plan = BuildProducerCopyPlan(requested_block_ids);
    int64_t slot_idx = -1;
    bool failed = false;
    std::string failure_reason;
    size_t num_layers = 0;
    try {
      {
        std::unique_lock<std::mutex> lock(mu_);
        const auto deadline = DeadlineFromNow();
        cv_.wait_until(lock, deadline, [&]() {
          return stopping_ ||
                 send_entries_.find(request.uuid) != send_entries_.end();
        });
        auto it = send_entries_.find(request.uuid);
        if (it == send_entries_.end()) {
          throw std::runtime_error("uuid not registered before timeout");
        }
        entry = it->second;
        if (entry->failed) {
          throw std::runtime_error("producer stage failed");
        }
        if (entry->pull_started) {
          throw std::runtime_error("uuid already has an active pull");
        }
        ValidateRequestedBlocks(*entry,
                                requested_plan.producer_remote_block_ids);
        slot_idx = AcquireSlotLocked();
        entry->slot_idx = slot_idx;
        entry->slot_released = false;
        entry->pull_started = true;
        entry->num_blocks = requested_plan.num_blocks;
        entry->d2h_issue_start = std::chrono::steady_clock::now();
        entry->deadline = DeadlineFromNow();
      }

      RaidenStageResult staged =
          IssueD2H(slot_idx, requested_plan.num_blocks,
                   requested_plan.producer_remote_block_ids);
      {
        std::lock_guard<std::mutex> lock(mu_);
        entry->d2h_issue_done = std::chrono::steady_clock::now();
        entry->total_bytes = staged.total_bytes;
        entry->copy_segments = staged.copy_segments;
      }
      staged.future->Await();
      const auto d2h_done = std::chrono::steady_clock::now();
      {
        std::lock_guard<std::mutex> lock(mu_);
        entry->stage_done = true;
        entry->d2h_done = d2h_done;
        entry->deadline = DeadlineFromNow();
      }
      cv_.notify_all();

      num_layers = staged.host_spans.size();
      WriteControlOk(fd, num_layers);
      for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        PullLayerDescriptor descriptor =
            LayerDescriptor(slot_idx, requested_plan.num_blocks, layer_idx);
        CheckStatus("control stream layer descriptor write",
                    WriteExact(fd, &descriptor, sizeof(descriptor)));
      }
    } catch (const std::exception& e) {
      failed = true;
      failure_reason = e.what();
      LOG(ERROR) << "Raiden pull stream failed for uuid=" << request.uuid
                 << ": " << failure_reason;
      WriteControlError(fd, failure_reason);
    } catch (...) {
      failed = true;
      failure_reason = "unknown error";
      LOG(ERROR) << "Raiden pull stream failed for uuid=" << request.uuid;
      WriteControlError(fd, failure_reason);
    }
    const auto done = std::chrono::steady_clock::now();
    if (entry) {
      {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = send_entries_.find(request.uuid);
        if (it != send_entries_.end()) {
          it->second->stage_done = !failed;
          it->second->failed = failed;
          if (it->second->d2h_issue_done ==
              std::chrono::steady_clock::time_point()) {
            it->second->d2h_issue_done = done;
          }
          if (it->second->d2h_done == std::chrono::steady_clock::time_point()) {
            it->second->d2h_done = done;
          }
        } else if (slot_idx >= 0 && !entry->slot_released) {
          ReleaseSlotLocked(slot_idx);
          entry->slot_released = true;
        }
      }
      cv_.notify_all();
      std::ostringstream stage_timing;
      stage_timing << "RAIDEN_TIMING event=producer_stage"
                   << " req_id=" << entry->req_id << " uuid=" << entry->uuid
                   << " rank=" << tp_rank_
                   << " registered_blocks=" << entry->registered_num_blocks
                   << " blocks=" << entry->num_blocks
                   << " bytes=" << entry->total_bytes
                   << " copy_segments=" << entry->copy_segments
                   << " d2h_issue_ms="
                   << DurationMs(entry->d2h_issue_start, entry->d2h_issue_done)
                   << " d2h_wait_ms="
                   << DurationMs(entry->d2h_issue_done, entry->d2h_done)
                   << " d2h_total_ms="
                   << DurationMs(entry->d2h_issue_start, entry->d2h_done)
                   << " register_to_stage_ms="
                   << DurationMs(entry->register_start, entry->d2h_done)
                   << " failed=" << (failed ? 1 : 0);
      EmitTimingLog(stage_timing.str());
    }
    std::ostringstream timing;
    timing << "RAIDEN_TIMING event=producer_pull_stream"
           << " req_id=" << (entry ? entry->req_id : "")
           << " uuid=" << request.uuid << " rank=" << tp_rank_
           << " registered_blocks="
           << (entry ? entry->registered_num_blocks : 0)
           << " blocks=" << requested_plan.num_blocks
           << " layers=" << num_layers
           << " bytes=" << (entry ? entry->total_bytes : 0)
           << " stream_ms=" << DurationMs(request_start, done)
           << " reason=" << failure_reason << " failed=" << (failed ? 1 : 0);
    EmitTimingLog(timing.str());
  }

  void AckRemote(const std::string& remote_endpoint, uint64_t uuid) {
    int fd = ConnectTcp(remote_endpoint);
    auto cleanup = std::unique_ptr<int, void (*)(int*)>(&fd, [](int* p) {
      if (p && *p >= 0) close(*p);
    });
    ControlRequestHeader request;
    request.magic = kControlMagic;
    request.op = kOpAck;
    request.uuid = uuid;
    request.num_blocks = 0;
    CheckStatus("control ack write", WriteExact(fd, &request, sizeof(request)));
    (void)ReadControlResponseHeader(fd);
  }

  ControlResponseHeader ReadControlResponseHeader(int fd) {
    ControlResponseHeader response;
    CheckStatus("control response read",
                ReadExact(fd, &response, sizeof(response)));
    if (response.magic != kResponseMagic) {
      throw std::runtime_error("bad control response magic");
    }
    if (response.status != 0) {
      std::string message(response.message_len, '\0');
      if (response.message_len > 0) {
        CheckStatus("control error body read",
                    ReadExact(fd, message.data(), message.size()));
      }
      throw std::runtime_error("remote Raiden control error: " + message);
    }
    return response;
  }

  void AckSend(uint64_t uuid) {
    std::shared_ptr<SendEntry> entry;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = send_entries_.find(uuid);
      if (it == send_entries_.end()) {
        pending_acks_.insert(uuid);
        return;
      }
      entry = it->second;
      done_sending_.insert(entry->req_id);
      ReleaseEntrySlotLocked(entry);
      send_entries_.erase(it);
    }
    const auto ack_done = std::chrono::steady_clock::now();
    std::ostringstream timing;
    timing << "RAIDEN_TIMING event=producer_ack"
           << " req_id=" << entry->req_id << " uuid=" << entry->uuid
           << " rank=" << tp_rank_ << " blocks=" << entry->num_blocks
           << " bytes=" << entry->total_bytes
           << " stage_to_ack_ms=" << DurationMs(entry->d2h_done, ack_done)
           << " register_to_ack_ms="
           << DurationMs(entry->register_start, ack_done)
           << " failed=" << (entry->failed ? 1 : 0);
    EmitTimingLog(timing.str());
  }

  TensorList kv_caches_;
  std::unique_ptr<tpu_raiden::torch::KVCacheManager> kv_transfer_;
  int64_t tp_rank_ = 0;
  int local_control_port_ = 0;
  int local_data_port_ = 0;
  int64_t max_blocks_ = 0;
  int64_t num_slots_ = 0;
  double timeout_s_ = 120.0;
  bool unsafe_skip_buffer_lock_ = true;
  int64_t next_op_id_ = 1;
  std::map<int64_t, PendingOperation> pending_;
  std::deque<int64_t> free_slots_;
  std::map<uint64_t, std::shared_ptr<SendEntry>> send_entries_;
  std::set<uint64_t> pending_acks_;
  std::set<std::string> done_sending_;
  std::set<std::string> done_recving_;
  std::set<std::string> failed_recving_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::unique_ptr<tpu_raiden::transport::SocketTransport> transport_;
  int control_fd_ = -1;
  std::atomic<bool> stopping_{false};
  std::thread control_thread_;
  std::mutex control_workers_mu_;
  std::vector<std::thread> control_workers_;
  std::mutex worker_threads_mu_;
  std::vector<std::thread> worker_threads_;
};

void AwaitAll(nb::object futures) {
  if (nb::isinstance<RaidenTransferFuture>(futures)) {
    auto future = nb::cast<std::shared_ptr<RaidenTransferFuture>>(futures);
    future->Await();
    return;
  }
  for (nb::handle item : futures) {
    auto future = nb::cast<std::shared_ptr<RaidenTransferFuture>>(item);
    future->Await();
  }
}

bool IsReady(nb::object futures) {
  if (nb::isinstance<RaidenTransferFuture>(futures)) {
    return nb::cast<std::shared_ptr<RaidenTransferFuture>>(futures)->IsReady();
  }
  for (nb::handle item : futures) {
    if (!nb::cast<std::shared_ptr<RaidenTransferFuture>>(item)->IsReady()) {
      return false;
    }
  }
  return true;
}

}  // namespace

NB_MODULE(_raiden_transfer_engine, m) {
  nb::class_<RaidenTransferFuture>(m, "RaidenTransferFuture")
      .def("Await", &RaidenTransferFuture::Await)
      .def("wait", &RaidenTransferFuture::Await)
      .def("IsReady", &RaidenTransferFuture::IsReady)
      .def("is_ready", &RaidenTransferFuture::IsReady);

  nb::class_<RaidenTransferEngine>(m, "RaidenTransferEngine")
      .def(nb::init<const TensorList&, int64_t, int64_t, int64_t, int64_t,
                    double, bool>(),
           nb::arg("kv_caches"), nb::arg("tp_rank"),
           nb::arg("local_control_port"), nb::arg("max_blocks"),
           nb::arg("num_slots"), nb::arg("timeout_s") = 120.0,
           nb::arg("unsafe_skip_buffer_lock") = true)
      .def_prop_ro("uses_prepared_tpu_buffers",
                   &RaidenTransferEngine::UsesPreparedTpuBuffers)
      .def_prop_ro("local_control_port",
                   &RaidenTransferEngine::local_control_port)
      .def_prop_ro("local_data_port", &RaidenTransferEngine::local_data_port)
      .def("register_kv_cache", &RaidenTransferEngine::RegisterKvCache,
           nb::arg("kv_caches"))
      .def("register_host_buffers", &RaidenTransferEngine::RegisterHostBuffers,
           nb::arg("host_pool"), nb::arg("tp_rank"))
      .def("register_send", &RaidenTransferEngine::RegisterSend,
           nb::arg("req_id"), nb::arg("uuid"), nb::arg("block_ids"))
      .def("submit_load", &RaidenTransferEngine::SubmitLoad, nb::arg("req_id"),
           nb::arg("uuid"), nb::arg("remote_endpoint"),
           nb::arg("remote_block_ids"), nb::arg("local_block_ids"))
      .def("stage_d2h", &RaidenTransferEngine::StageD2H, nb::kw_only(),
           nb::arg("slot_idx"), nb::arg("num_blocks"), nb::arg("block_ids"))
      .def("stage_d2h_sync", &RaidenTransferEngine::StageD2HSync, nb::kw_only(),
           nb::arg("slot_idx"), nb::arg("num_blocks"), nb::arg("block_ids"))
      .def("commit_h2d", &RaidenTransferEngine::CommitH2D, nb::kw_only(),
           nb::arg("slot_idx"), nb::arg("num_blocks"),
           nb::arg("local_block_ids"))
      .def("rank_layer_views", &RaidenTransferEngine::RankLayerViews,
           nb::arg("slot_idx"), nb::arg("rank"), nb::arg("num_blocks"))
      .def("unpack_rank_layers", &RaidenTransferEngine::UnpackRankLayers,
           nb::arg("slot_idx"), nb::arg("rank"), nb::arg("num_blocks"),
           nb::arg("layer_buffers"))
      .def("submit_d2h", &RaidenTransferEngine::SubmitD2H, nb::kw_only(),
           nb::arg("slot_idx"), nb::arg("num_blocks"), nb::arg("block_ids"))
      .def("submit_h2d", &RaidenTransferEngine::SubmitH2D, nb::kw_only(),
           nb::arg("slot_idx"), nb::arg("num_blocks"),
           nb::arg("local_block_ids"))
      .def("poll_finished", &RaidenTransferEngine::PollFinished)
      .def("poll_transfer_ops", &RaidenTransferEngine::PollTransferOps)
      .def("wait_transfer", &RaidenTransferEngine::WaitTransfer,
           nb::arg("op_id"))
      .def("_count_copy_segments_for_testing",
           &RaidenTransferEngine::CountCopySegmentsForTesting,
           nb::arg("block_ids"))
      .def("_send_copy_plan_for_testing",
           &RaidenTransferEngine::SendCopyPlanForTesting, nb::arg("block_ids"))
      .def("_load_copy_plan_for_testing",
           &RaidenTransferEngine::LoadCopyPlanForTesting,
           nb::arg("remote_block_ids"), nb::arg("local_block_ids"));

  m.def("await_all", &AwaitAll, nb::arg("futures"));
  m.def("is_ready", &IsReady, nb::arg("futures"));
}

}  // namespace tpu_raiden::kv_cache
