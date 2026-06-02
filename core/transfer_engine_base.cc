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

// Copyright 2026 Google LLC
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

#include "core/transfer_engine_base.h"

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
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/raw_transfer_core.h"
#include "kv_cache/kv_cache_manager_base.h"

namespace tpu_raiden {
namespace {

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

static void WriteBlockIds(int fd, const std::vector<int64_t>& block_ids) {
  if (block_ids.empty()) return;
  CheckStatus(
      "control block ids write",
      WriteExact(fd, block_ids.data(), block_ids.size() * sizeof(int64_t)));
}

static std::vector<int64_t> ReadBlockIds(int fd, uint64_t num_blocks) {
  if (num_blocks == 0) return {};
  if (num_blocks > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    throw std::invalid_argument("num_blocks is too large");
  }
  std::vector<int64_t> block_ids(static_cast<size_t>(num_blocks));
  CheckStatus(
      "control block ids read",
      ReadExact(fd, block_ids.data(), block_ids.size() * sizeof(int64_t)));
  return block_ids;
}

static CopySpec OffsetsImpl(const std::vector<int64_t>& block_ids,
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

static kv_cache::KVCacheCopySpec ToKVCacheCopySpecImpl(const CopySpec& spec) {
  return {.src_offsets = spec.src_offsets,
          .dst_offsets = spec.dst_offsets,
          .sizes = spec.sizes};
}

static std::vector<int64_t> CanonicalSendBlockIds(
    const std::vector<int64_t>& block_ids) {
  std::vector<int64_t> ordered = block_ids;
  std::stable_sort(ordered.begin(), ordered.end());
  return ordered;
}

static CopyPlan BuildProducerCopyPlan(const std::vector<int64_t>& block_ids) {
  CopyPlan plan;
  plan.num_blocks = static_cast<int64_t>(block_ids.size());
  plan.requested_remote_block_ids = block_ids;
  plan.producer_remote_block_ids = CanonicalSendBlockIds(block_ids);
  plan.d2h_copy =
      OffsetsImpl(plan.producer_remote_block_ids, /*source_is_compact=*/false);
  return plan;
}

static CopyPlan BuildLoadCopyPlan(const std::vector<int64_t>& remote_block_ids,
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
  for (size_t source_pos = 0; source_pos < remote_order.size(); ++source_pos) {
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
      OffsetsImpl(plan.h2d_local_block_ids, /*source_is_compact=*/true);
  return plan;
}

static void ReorderCompactBlocks(
    const kv_cache::KVCacheHostSpan& compact_blocks,
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

static double DurationMs(std::chrono::steady_clock::time_point start,
                         std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

CopySpec TransferEngineBase::Offsets(const std::vector<int64_t>& block_ids,
                                     bool source_is_compact) {
  return OffsetsImpl(block_ids, source_is_compact);
}

kv_cache::KVCacheCopySpec TransferEngineBase::ToKVCacheCopySpec(
    const CopySpec& spec) {
  return ToKVCacheCopySpecImpl(spec);
}

void TransferEngineBase::ValidateRequestedBlocks(
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

TransferEngineBase::TransferEngineBase(
    std::unique_ptr<kv_cache::KVCacheManagerBase> kv_transfer, int64_t tp_rank,
    int64_t local_control_port, int64_t max_blocks, int64_t num_slots,
    double timeout_s, bool unsafe_skip_buffer_lock)
    : kv_transfer_(std::move(kv_transfer)),
      tp_rank_(tp_rank),
      local_control_port_(static_cast<int>(local_control_port)),
      local_data_port_(0),
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
  if (kv_transfer_) {
    ConfigureDataPortFromKvTransfer();
  }
  InitializeSlotPool(num_slots_);
  StartControlServer();
}

TransferEngineBase::~TransferEngineBase() { StopControlServer(); }

int64_t TransferEngineBase::SubmitD2H(int64_t slot_idx, int64_t num_blocks,
                                      const std::vector<int64_t>& block_ids) {
  PendingOperation op;
  op.future = IssueD2H(slot_idx, num_blocks, block_ids).future;
  return StorePending(std::move(op));
}

int64_t TransferEngineBase::SubmitH2D(
    int64_t slot_idx, int64_t num_blocks,
    const std::vector<int64_t>& local_block_ids) {
  PendingOperation op;
  op.future = IssueH2D(slot_idx, num_blocks, local_block_ids).future;
  return StorePending(std::move(op));
}

int64_t TransferEngineBase::RegisterSend(
    const std::string& req_id, uint64_t uuid,
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

int64_t TransferEngineBase::SubmitLoad(
    const std::string& req_id, uint64_t uuid,
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
          CheckStatus(
              "control pull stream write",
              WriteExact(control_fd, &stream_request, sizeof(stream_request)));
          WriteBlockIds(control_fd, load_plan.producer_remote_block_ids);
          ControlResponseHeader response =
              ReadControlResponseHeader(control_fd);
          control_setup_ms =
              DurationMs(control_start, std::chrono::steady_clock::now());

          std::vector<kv_cache::KVCacheHostSpan> local_spans =
              LayerSpans(slot_idx, static_cast<int64_t>(load_plan.num_blocks));
          if (response.num_layers != local_spans.size()) {
            throw std::runtime_error("remote layer descriptor count mismatch");
          }
          PullBlockDescriptor block_descriptor;
          const auto descriptor_start = std::chrono::steady_clock::now();
          CheckStatus("control stream block descriptor read",
                      ReadExact(control_fd, &block_descriptor,
                                sizeof(block_descriptor)));
          descriptor_ms +=
              DurationMs(descriptor_start, std::chrono::steady_clock::now());
          if (block_descriptor.num_blocks !=
              static_cast<uint64_t>(load_plan.num_blocks)) {
            throw std::runtime_error("remote block descriptor size mismatch");
          }

          std::vector<uint8_t*> explicit_dst_ptrs;
          explicit_dst_ptrs.reserve(local_spans.size());
          for (const kv_cache::KVCacheHostSpan& span : local_spans) {
            explicit_dst_ptrs.push_back(span.ptr);
            ++h2h_layers;
            h2h_bytes += static_cast<int64_t>(span.nbytes);
          }

          std::vector<int> remote_staging_block_ids = ContiguousBlockIds(
              block_descriptor.remote_block_base, block_descriptor.num_blocks);
          std::vector<int> local_compact_block_ids =
              ContiguousBlockIds(/*base=*/0, block_descriptor.num_blocks);
          const auto h2h_start = std::chrono::steady_clock::now();
          auto h2h_future_or = kv_transfer_->H2hReadExplicit(
              EndpointWithPort(remote_endpoint, response.data_port),
              remote_staging_block_ids, local_compact_block_ids,
              explicit_dst_ptrs);
          if (!h2h_future_or.ok()) {
            ThrowStatus("BlockTransport pull failed", h2h_future_or.status());
          }
          absl::Status h2h_status = h2h_future_or.value().Await().status();
          if (!h2h_status.ok()) {
            ThrowStatus("BlockTransport pull failed", h2h_status);
          }
          h2h_ms += DurationMs(h2h_start, std::chrono::steady_clock::now());

          if (load_plan.RequiresHostReorder()) {
            const auto reorder_start = std::chrono::steady_clock::now();
            for (const kv_cache::KVCacheHostSpan& span : local_spans) {
              ReorderCompactBlocks(span, load_plan.host_dst_to_src);
            }
            host_reorder_ms +=
                DurationMs(reorder_start, std::chrono::steady_clock::now());
          }

          const auto h2d_start = std::chrono::steady_clock::now();
          StageResult h2d =
              IssueH2D(slot_idx, static_cast<int64_t>(load_plan.num_blocks),
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
          ControlRequestHeader ack_request;
          ack_request.magic = kControlMagic;
          ack_request.op = kOpAck;
          ack_request.uuid = uuid;
          CheckStatus("control ack write", WriteExact(control_fd, &ack_request,
                                                      sizeof(ack_request)));
          ack_ms = DurationMs(ack_start, std::chrono::steady_clock::now());
        }
        load_promise->set_value();
      } catch (const std::exception& e) {
        LOG(ERROR) << "Raiden consumer error during pull pipeline: "
                   << e.what();
        failed = true;
        load_promise->set_exception(std::current_exception());
      } catch (...) {
        LOG(ERROR) << "Raiden consumer unknown error during pull pipeline";
        failed = true;
        load_promise->set_exception(
            std::make_exception_ptr(std::runtime_error("unknown error")));
      }
      if (slot_idx >= 0) {
        std::lock_guard<std::mutex> lock(mu_);
        ReleaseSlotLocked(slot_idx);
      }

      const auto worker_done = std::chrono::steady_clock::now();
      {
        std::lock_guard<std::mutex> lock(mu_);
        if (failed) {
          failed_recving_.insert(req_id);
        } else if (report_recv_done) {
          done_recving_.insert(req_id);
        }
      }

      std::ostringstream timing;
      timing << "RAIDEN_TIMING event=consumer_pipeline"
             << " req_id=" << req_id << " uuid=" << uuid << " rank=" << tp_rank_
             << " release_only=" << (release_only ? 1 : 0)
             << " total_blocks=" << load_plan.num_blocks
             << " total_bytes=" << (h2h_bytes + h2d_bytes)
             << " h2h_bytes=" << h2h_bytes << " h2d_bytes=" << h2d_bytes
             << " h2h_layers=" << h2h_layers << " h2d_segments=" << h2d_segments
             << " enqueue_to_start_ms="
             << DurationMs(submit_start, worker_start)
             << " slot_acquire_ms=" << slot_ms
             << " control_setup_ms=" << control_setup_ms
             << " descriptors_ms=" << descriptor_ms << " h2h_ms=" << h2h_ms
             << " host_reorder_ms=" << host_reorder_ms
             << " h2d_issue_ms=" << h2d_issue_ms
             << " h2d_wait_ms=" << h2d_wait_ms
             << " h2d_total_ms=" << h2d_total_ms << " ack_ms=" << ack_ms
             << " start_to_done_ms=" << DurationMs(worker_start, worker_done)
             << " enqueue_to_done_ms=" << DurationMs(submit_start, worker_done)
             << " failed=" << (failed ? 1 : 0);
      EmitTimingLog(timing.str());
    });
  }
  PendingOperation op;
  op.load_promise = load_promise;
  return StorePending(std::move(op));
}

std::vector<int64_t> TransferEngineBase::PollTransferOps() {
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

void TransferEngineBase::WaitTransfer(int64_t op_id) {
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

std::tuple<std::vector<std::string>, std::vector<std::string>,
           std::vector<std::string>>
TransferEngineBase::PollFinishedRaw() {
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
  return {done_sending, done_recving, failed_recving};
}

StageResult TransferEngineBase::IssueD2H(
    int64_t slot_idx, int64_t num_blocks,
    const std::vector<int64_t>& block_ids) {
  if (num_blocks != static_cast<int64_t>(block_ids.size())) {
    throw std::invalid_argument("num_blocks must match len(block_ids)");
  }
  CopySpec copy_spec = Offsets(block_ids, /*source_is_compact=*/false);
  kv_cache::KVCacheCopySpec transfer_spec = ToKVCacheCopySpec(copy_spec);
  std::vector<kv_cache::KVCacheHostSpan> host_spans =
      LayerSpans(slot_idx, num_blocks);
  auto future = std::make_shared<TransferFuture>();
  int64_t total_bytes = 0;
  for (size_t i = 0; i < kv_transfer_->num_layers(); ++i) {
    total_bytes += static_cast<int64_t>(host_spans[i].nbytes);
    future->Add(ValueOrThrow(
        "Failed to issue D2H transfer",
        kv_transfer_->D2hToHostSlot(i, slot_idx, num_blocks, transfer_spec)));
  }
  return {.future = std::move(future),
          .host_spans = std::move(host_spans),
          .total_bytes = total_bytes,
          .copy_segments = static_cast<int64_t>(copy_spec.sizes.size())};
}

StageResult TransferEngineBase::IssueH2D(
    int64_t slot_idx, int64_t num_blocks,
    const std::vector<int64_t>& local_block_ids) {
  if (num_blocks != static_cast<int64_t>(local_block_ids.size())) {
    throw std::invalid_argument("num_blocks must match len(local_block_ids)");
  }
  CopySpec copy_spec = Offsets(local_block_ids, /*source_is_compact=*/true);
  kv_cache::KVCacheCopySpec transfer_spec = ToKVCacheCopySpec(copy_spec);
  std::vector<kv_cache::KVCacheHostSpan> host_spans =
      LayerSpans(slot_idx, num_blocks);
  auto future = std::make_shared<TransferFuture>();
  int64_t total_bytes = 0;
  for (size_t i = 0; i < kv_transfer_->num_layers(); ++i) {
    total_bytes += static_cast<int64_t>(host_spans[i].nbytes);
    future->Add(ValueOrThrow(
        "Failed to issue H2D transfer",
        kv_transfer_->H2dFromHostSlot(i, slot_idx, num_blocks, transfer_spec)));
  }
  return {.future = std::move(future),
          .host_spans = std::move(host_spans),
          .total_bytes = total_bytes,
          .copy_segments = static_cast<int64_t>(copy_spec.sizes.size())};
}

CommitResult TransferEngineBase::CommitH2DRaw(
    int64_t slot_idx, int64_t num_blocks,
    const std::vector<int64_t>& local_block_ids) {
  if (num_blocks != static_cast<int64_t>(local_block_ids.size())) {
    throw std::invalid_argument("num_blocks must match len(local_block_ids)");
  }
  auto t0 = std::chrono::steady_clock::now();
  StageResult result = IssueH2D(slot_idx, num_blocks, local_block_ids);
  auto t_issued = std::chrono::steady_clock::now();
  result.future->Await();
  auto t_done = std::chrono::steady_clock::now();
  return {.duration_issue_ms = DurationMs(t0, t_issued),
          .duration_wait_ms = DurationMs(t_issued, t_done),
          .duration_total_ms = DurationMs(t0, t_done),
          .total_bytes = result.total_bytes};
}

std::vector<kv_cache::KVCacheHostSpan> TransferEngineBase::LayerSpans(
    int64_t slot_idx, int64_t num_blocks) {
  if (!kv_transfer_) {
    throw std::runtime_error("KV cache manager is not registered");
  }
  if (num_blocks < 0 || num_blocks > max_blocks_) {
    throw std::out_of_range("num_blocks out of range");
  }
  std::vector<kv_cache::KVCacheHostSpan> spans;
  spans.reserve(kv_transfer_->num_layers());
  for (size_t layer_idx = 0; layer_idx < kv_transfer_->num_layers();
       ++layer_idx) {
    spans.push_back(
        ValueOrThrow("Failed to get KVCacheManager host staging span",
                     kv_transfer_->HostSpan(layer_idx, /*shard_idx=*/0,
                                            slot_idx, num_blocks)));
  }
  return spans;
}

void TransferEngineBase::InitializeSlotPool(int64_t num_slots) {
  free_slots_.clear();
  for (int64_t slot = 0; slot < num_slots; ++slot) {
    free_slots_.push_back(slot);
  }
}

int64_t TransferEngineBase::AcquireSlot() {
  std::lock_guard<std::mutex> lock(mu_);
  return AcquireSlotLocked();
}

int64_t TransferEngineBase::AcquireSlotLocked() {
  if (free_slots_.empty()) {
    throw std::runtime_error("Raiden host slot pool exhausted");
  }
  int64_t slot = free_slots_.front();
  free_slots_.pop_front();
  return slot;
}

void TransferEngineBase::ReleaseSlotLocked(int64_t slot_idx) {
  if (slot_idx < 0) return;
  free_slots_.push_back(slot_idx);
}

void TransferEngineBase::ReleaseEntrySlotLocked(
    const std::shared_ptr<SendEntry>& entry) {
  if (!entry || entry->slot_idx < 0 || entry->slot_released) return;
  ReleaseSlotLocked(entry->slot_idx);
  entry->slot_released = true;
}

void TransferEngineBase::StartControlServer() {
  control_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (control_fd_ < 0) {
    throw std::runtime_error("control socket() failed: " +
                             std::string(std::strerror(errno)));
  }
  int opt = 1;
  setsockopt(control_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  setsockopt(control_fd_, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(local_control_port_);

  if (bind(control_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    std::string err = std::strerror(errno);
    close(control_fd_);
    control_fd_ = -1;
    throw std::runtime_error("control bind(" +
                             std::to_string(local_control_port_) +
                             ") failed: " + err);
  }
  socklen_t len = sizeof(addr);
  if (getsockname(control_fd_, reinterpret_cast<sockaddr*>(&addr), &len) < 0) {
    close(control_fd_);
    control_fd_ = -1;
    throw std::runtime_error("getsockname() failed: " +
                             std::string(std::strerror(errno)));
  }
  local_control_port_ = ntohs(addr.sin_port);
  if (listen(control_fd_, 128) < 0) {
    close(control_fd_);
    control_fd_ = -1;
    throw std::runtime_error("listen() failed: " +
                             std::string(std::strerror(errno)));
  }
  stopping_ = false;
  control_thread_ = std::thread([this]() { ControlServerLoop(); });
}

void TransferEngineBase::StopControlServer() {
  stopping_ = true;
  if (control_fd_ >= 0) {
    shutdown(control_fd_, SHUT_RDWR);
    close(control_fd_);
  }
  if (control_thread_.joinable()) {
    control_thread_.join();
  }
  control_fd_ = -1;

  {
    std::lock_guard<std::mutex> lock(control_workers_mu_);
    for (auto& t : control_workers_) {
      if (t.joinable()) t.join();
    }
  }
  {
    std::lock_guard<std::mutex> lock(worker_threads_mu_);
    for (auto& t : worker_threads_) {
      if (t.joinable()) t.join();
    }
  }
}

void TransferEngineBase::ControlServerLoop() {
  while (!stopping_) {
    pollfd pfd;
    pfd.fd = control_fd_;
    pfd.events = POLLIN;
    int r = poll(&pfd, 1, 200);
    if (r < 0) {
      if (errno == EINTR) continue;
      break;
    }
    if (r == 0) continue;
    int client_fd = accept(control_fd_, nullptr, nullptr);
    if (client_fd < 0) {
      if (errno == EINTR) continue;
      break;
    }
    std::lock_guard<std::mutex> lock(control_workers_mu_);
    control_workers_.emplace_back([this, client_fd]() {
      HandleControlConnection(client_fd);
      close(client_fd);
    });
  }
}

void TransferEngineBase::HandleControlConnection(int fd) {
  try {
    ControlRequestHeader req;
    CheckStatus("control request header read",
                ReadExact(fd, &req, sizeof(req)));
    if (req.magic != kControlMagic) {
      throw std::runtime_error("bad control request magic");
    }
    if (req.op == kOpPullStream) {
      ProcessPullStream(fd, req);
    } else {
      throw std::runtime_error("unknown control op code: " +
                               std::to_string(req.op));
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "Raiden producer error in control connection handler: "
               << e.what();
    ControlResponseHeader response;
    response.magic = kResponseMagic;
    response.status = -1;
    std::string message = e.what();
    response.message_len = message.size();
    (void)WriteExact(fd, &response, sizeof(response));
    if (response.message_len > 0) {
      (void)WriteExact(fd, message.data(), message.size());
    }
  }
}

void TransferEngineBase::ProcessPullStream(int fd,
                                           const ControlRequestHeader& req) {
  const auto pull_start = std::chrono::steady_clock::now();
  std::shared_ptr<SendEntry> entry;
  const auto wait_producer_start = std::chrono::steady_clock::now();
  {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [this, &entry, &req]() {
      auto it = send_entries_.find(req.uuid);
      if (it != send_entries_.end()) {
        entry = it->second;
        return true;
      }
      return stopping_.load();
    });
  }
  const auto wait_producer_ms =
      DurationMs(wait_producer_start, std::chrono::steady_clock::now());
  if (stopping_) return;
  if (!entry) {
    throw std::runtime_error(
        "TransferEngineBase is stopping during wait for send entry");
  }

  const auto parse_start = std::chrono::steady_clock::now();
  std::vector<int64_t> requested_block_ids = ReadBlockIds(fd, req.num_blocks);
  ValidateRequestedBlocks(*entry, requested_block_ids);
  const auto parse_ms =
      DurationMs(parse_start, std::chrono::steady_clock::now());

  const auto slot_start = std::chrono::steady_clock::now();
  int64_t slot_idx = AcquireSlot();
  {
    std::lock_guard<std::mutex> lock(mu_);
    entry->slot_idx = slot_idx;
    entry->num_blocks = static_cast<int64_t>(req.num_blocks);
    entry->pull_started = true;
  }
  const auto slot_ms = DurationMs(slot_start, std::chrono::steady_clock::now());

  const auto issue_start = std::chrono::steady_clock::now();
  StageResult d2h = IssueD2H(slot_idx, entry->num_blocks, requested_block_ids);
  entry->total_bytes = d2h.total_bytes;
  entry->copy_segments = d2h.copy_segments;
  const auto issue_ms =
      DurationMs(issue_start, std::chrono::steady_clock::now());
  d2h.future->Await();
  const auto d2h_ms = DurationMs(issue_start, std::chrono::steady_clock::now());
  {
    std::lock_guard<std::mutex> lock(mu_);
    entry->stage_done = true;
    entry->d2h_done = std::chrono::steady_clock::now();
  }

  const auto response_start = std::chrono::steady_clock::now();
  ControlResponseHeader response;
  response.magic = kResponseMagic;
  response.status = 0;
  response.num_layers = static_cast<uint32_t>(d2h.host_spans.size());
  response.data_port = static_cast<uint32_t>(local_data_port_);
  CheckStatus("control stream response header write",
              WriteExact(fd, &response, sizeof(response)));

  PullBlockDescriptor descriptor;
  descriptor.remote_block_base = StagingBlockBase(slot_idx);
  descriptor.num_blocks = static_cast<uint64_t>(entry->num_blocks);
  CheckStatus("control stream block descriptor write",
              WriteExact(fd, &descriptor, sizeof(descriptor)));
  const auto response_ms =
      DurationMs(response_start, std::chrono::steady_clock::now());

  const auto wait_ack_start = std::chrono::steady_clock::now();
  ControlRequestHeader ack_req;
  CheckStatus("control ack read", ReadExact(fd, &ack_req, sizeof(ack_req)));
  if (ack_req.magic != kControlMagic || ack_req.op != kOpAck) {
    throw std::runtime_error("bad control ack magic or op");
  }
  const auto wait_ack_ms =
      DurationMs(wait_ack_start, std::chrono::steady_clock::now());

  AckSend(req.uuid);

  std::ostringstream timing;
  timing << "RAIDEN_TIMING event=producer_pipeline"
         << " req_id=" << entry->req_id << " uuid=" << entry->uuid
         << " rank=" << tp_rank_ << " total_blocks=" << entry->num_blocks
         << " total_bytes=" << entry->total_bytes
         << " total_layers=" << d2h.host_spans.size()
         << " copy_segments=" << entry->copy_segments
         << " wait_producer_ms=" << wait_producer_ms << " parse_ms=" << parse_ms
         << " slot_acquire_ms=" << slot_ms << " d2h_issue_ms=" << issue_ms
         << " d2h_total_ms=" << d2h_ms << " response_ms=" << response_ms
         << " wait_ack_ms=" << wait_ack_ms << " start_to_done_ms="
         << DurationMs(pull_start, std::chrono::steady_clock::now())
         << " failed=0";
  EmitTimingLog(timing.str());
}

std::string TransferEngineBase::EndpointWithPort(const std::string& endpoint,
                                                 int port) const {
  auto [host, ignored_port] = SplitEndpoint(endpoint);
  (void)ignored_port;
  return host + ":" + std::to_string(port);
}

void TransferEngineBase::AckRemote(const std::string& remote_endpoint,
                                   uint64_t uuid) {
  int control_fd = ConnectTcp(remote_endpoint);
  auto control_cleanup =
      std::unique_ptr<int, void (*)(int*)>(&control_fd, [](int* p) {
        if (p && *p >= 0) close(*p);
      });
  ControlRequestHeader stream_request;
  stream_request.magic = kControlMagic;
  stream_request.op = kOpPullStream;
  stream_request.uuid = uuid;
  stream_request.num_blocks = 0;
  CheckStatus("control pull stream write (empty)",
              WriteExact(control_fd, &stream_request, sizeof(stream_request)));
  (void)ReadControlResponseHeader(control_fd);
}

TransferEngineBase::ControlResponseHeader
TransferEngineBase::ReadControlResponseHeader(int fd) {
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

void TransferEngineBase::AckSend(uint64_t uuid) {
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

int64_t TransferEngineBase::StorePending(PendingOperation op) {
  const int64_t op_id = next_op_id_++;
  pending_[op_id] = std::move(op);
  return op_id;
}

std::chrono::steady_clock::time_point TransferEngineBase::DeadlineFromNow()
    const {
  return std::chrono::steady_clock::now() +
         std::chrono::milliseconds(static_cast<int64_t>(timeout_s_ * 1000.0));
}

int64_t TransferEngineBase::CountCopySegmentsForTesting(
    const std::vector<int64_t>& block_ids) const {
  return static_cast<int64_t>(
      Offsets(block_ids, /*source_is_compact=*/false).sizes.size());
}

CopyPlan TransferEngineBase::BuildProducerCopyPlanForTesting(
    const std::vector<int64_t>& block_ids) const {
  return BuildProducerCopyPlan(block_ids);
}

CopyPlan TransferEngineBase::BuildLoadCopyPlanForTesting(
    const std::vector<int64_t>& remote_block_ids,
    const std::vector<int64_t>& local_block_ids) const {
  return BuildLoadCopyPlan(remote_block_ids, local_block_ids);
}

void TransferEngineBase::ConfigureDataPortFromKvTransfer() {
  if (!kv_transfer_) {
    local_data_port_ = 0;
    return;
  }
  std::optional<int> data_port = kv_transfer_->local_port();
  if (!data_port.has_value()) {
    throw std::runtime_error("KVCacheManager BlockTransport is not running");
  }
  local_data_port_ = *data_port;
}

uint64_t TransferEngineBase::StagingBlockBase(int64_t slot_idx) const {
  if (slot_idx < 0) {
    throw std::out_of_range("slot_idx out of range");
  }
  if (slot_idx >
      std::numeric_limits<int64_t>::max() / std::max<int64_t>(max_blocks_, 1)) {
    throw std::out_of_range("staging block base exceeds int64 range");
  }
  const int64_t base = slot_idx * max_blocks_;
  if (base < 0 ||
      base > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    throw std::out_of_range("staging block base exceeds int range");
  }
  return static_cast<uint64_t>(base);
}

std::vector<int> TransferEngineBase::ContiguousBlockIds(uint64_t base,
                                                        uint64_t count) const {
  if (count > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    throw std::out_of_range("block count exceeds int range");
  }
  if (base > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
      count >
          static_cast<uint64_t>(std::numeric_limits<int>::max()) - base + 1) {
    throw std::out_of_range("block id range exceeds int range");
  }
  std::vector<int> ids;
  ids.reserve(static_cast<size_t>(count));
  for (uint64_t i = 0; i < count; ++i) {
    ids.push_back(static_cast<int>(base + i));
  }
  return ids;
}

}  // namespace tpu_raiden
