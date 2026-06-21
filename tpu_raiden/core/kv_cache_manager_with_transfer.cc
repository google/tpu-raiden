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

#include "tpu_raiden/core/kv_cache_manager_with_transfer.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdio.h>
#include <sys/poll.h>
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
#include <exception>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <ratio>
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
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_client.h"
#include "tpu_raiden/core/host_memory_allocator.h"
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/core/tpu_utils.h"
#include "tpu_raiden/kv_cache/kv_cache_manager_base.h"

namespace tpu_raiden {
namespace {

[[noreturn]] void ThrowStatus(absl::string_view context,
                              const absl::Status& status) {
  throw std::runtime_error(absl::StrCat(context, ": ", status.message()));
}

void CheckStatus(absl::string_view context, const absl::Status& status) {
  if (!status.ok()) {
    ThrowStatus(context, status);
  }
}

void EmitTimingLog(absl::string_view message) { LOG(INFO) << message; }

template <typename T>
T ValueOrThrow(absl::string_view context, absl::StatusOr<T> value_or) {
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

std::pair<std::string, int> SplitEndpoint(absl::string_view endpoint) {
  const size_t colon = endpoint.rfind(':');
  if (colon == absl::string_view::npos) {
    throw std::invalid_argument("endpoint must be host:port");
  }
  int port = 0;
  if (!absl::SimpleAtoi(endpoint.substr(colon + 1), &port)) {
    throw std::invalid_argument("invalid port in endpoint");
  }
  return {std::string(endpoint.substr(0, colon)), port};
}

int ConnectTcp(absl::string_view endpoint) {
  auto [host, port] = SplitEndpoint(endpoint);
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    throw std::runtime_error(
        absl::StrCat("socket() failed: ", std::strerror(errno)));
  }
  int opt = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
    close(fd);
    throw std::runtime_error(
        absl::StrCat("invalid IPv4 endpoint host: ", host));
  }
  if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    std::string err = std::strerror(errno);
    close(fd);
    throw std::runtime_error(
        absl::StrCat("connect(", endpoint, ") failed: ", err));
  }
  return fd;
}

static std::string GetPeerIp(int fd) {
  sockaddr_in addr;
  socklen_t len = sizeof(addr);
  if (getpeername(fd, reinterpret_cast<sockaddr*>(&addr), &len) < 0) {
    throw std::runtime_error("getpeername() failed: " +
                             std::string(std::strerror(errno)));
  }
  char ip_buf[INET_ADDRSTRLEN];
  if (inet_ntop(AF_INET, &addr.sin_addr, ip_buf, sizeof(ip_buf)) == nullptr) {
    throw std::runtime_error("inet_ntop() failed: " +
                             std::string(std::strerror(errno)));
  }
  return std::string(ip_buf);
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

static CopySpec BuildH2dCopySpec(const std::vector<int64_t>& src_block_ids,
                                 const std::vector<int64_t>& dst_block_ids) {
  if (src_block_ids.size() != dst_block_ids.size()) {
    throw std::invalid_argument(
        "src and dst block lists must have same length");
  }
  CopySpec spec;
  if (src_block_ids.empty()) {
    return spec;
  }
  const int64_t n = static_cast<int64_t>(src_block_ids.size());
  spec.src_offsets.reserve(n);
  spec.dst_offsets.reserve(n);
  spec.sizes.reserve(n);

  for (int64_t start = 0; start < n;) {
    int64_t end = start + 1;
    while (end < n && src_block_ids[end] == src_block_ids[end - 1] + 1 &&
           dst_block_ids[end] == dst_block_ids[end - 1] + 1) {
      ++end;
    }
    const int64_t run_size = end - start;
    spec.src_offsets.push_back(src_block_ids[start]);
    spec.dst_offsets.push_back(dst_block_ids[start]);
    spec.sizes.push_back(run_size);
    start = end;
  }
  return spec;
}

static CopyPlan BuildLoadCopyPlan(
    const std::vector<int64_t>& remote_block_ids,
    const std::vector<int64_t>& local_block_ids,
    const std::vector<int64_t>& local_host_block_ids) {
  if (remote_block_ids.size() != local_block_ids.size() ||
      local_block_ids.size() != local_host_block_ids.size()) {
    throw std::invalid_argument(
        "remote_block_ids, local_block_ids, and local_host_block_ids must have "
        "same length");
  }
  CopyPlan plan;
  plan.num_blocks = static_cast<int64_t>(remote_block_ids.size());
  plan.requested_remote_block_ids = remote_block_ids;
  plan.requested_local_block_ids = local_block_ids;
  if (remote_block_ids.empty()) {
    return plan;
  }

  // 1. Determine transport order (sorted by remote_block_ids)
  std::vector<size_t> remote_order(remote_block_ids.size());
  for (size_t i = 0; i < remote_order.size(); ++i) {
    remote_order[i] = i;
  }
  std::stable_sort(remote_order.begin(), remote_order.end(),
                   [&](size_t a, size_t b) {
                     return remote_block_ids[a] < remote_block_ids[b];
                   });

  plan.producer_remote_block_ids.reserve(remote_order.size());
  plan.transport_host_block_ids.reserve(remote_order.size());
  for (size_t i = 0; i < remote_order.size(); ++i) {
    const size_t original_idx = remote_order[i];
    plan.producer_remote_block_ids.push_back(remote_block_ids[original_idx]);
    plan.transport_host_block_ids.push_back(local_host_block_ids[original_idx]);
  }

  // 2. Determine H2D copy plan (sorted by local_block_ids for opt)
  std::vector<size_t> local_order(local_block_ids.size());
  for (size_t i = 0; i < local_order.size(); ++i) {
    local_order[i] = i;
  }
  std::stable_sort(local_order.begin(), local_order.end(),
                   [&](size_t a, size_t b) {
                     return local_block_ids[a] < local_block_ids[b];
                   });

  plan.h2d_local_block_ids.reserve(local_order.size());
  plan.h2d_host_block_ids.reserve(local_order.size());
  for (size_t i = 0; i < local_order.size(); ++i) {
    const size_t original_idx = local_order[i];
    plan.h2d_local_block_ids.push_back(local_block_ids[original_idx]);
    plan.h2d_host_block_ids.push_back(local_host_block_ids[original_idx]);
  }

  plan.h2d_copy =
      BuildH2dCopySpec(plan.h2d_host_block_ids, plan.h2d_local_block_ids);
  plan.host_dst_to_src.clear();  // No host reordering needed!
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

CopySpec KVCacheManagerWithTransfer::Offsets(
    const std::vector<int64_t>& block_ids, bool source_is_compact) {
  return OffsetsImpl(block_ids, source_is_compact);
}

kv_cache::KVCacheCopySpec KVCacheManagerWithTransfer::ToKVCacheCopySpec(
    const CopySpec& spec) {
  return ToKVCacheCopySpecImpl(spec);
}

void KVCacheManagerWithTransfer::ValidateRequestedBlocks(
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

KVCacheManagerWithTransfer::KVCacheManagerWithTransfer(
    const std::vector<std::vector<xla::PjRtBuffer*>>& layer_buffers,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    bool unsafe_skip_buffer_lock, int parallelism,
    HostBufferAllocator host_allocator, int64_t node_id,
    int64_t local_control_port, int64_t max_blocks, int64_t num_slots,
    double timeout_s)
    : KVCacheManagerBase(layer_buffers, local_port,
                         host_blocks_to_allocate.has_value()
                             ? *host_blocks_to_allocate
                             : num_slots * max_blocks,
                         unsafe_skip_buffer_lock, parallelism, host_allocator),
      node_id_(node_id),
      local_control_port_(static_cast<int>(local_control_port)),
      local_data_port_(0),
      max_blocks_(max_blocks),
      num_slots_(num_slots),
      timeout_s_(timeout_s),
      unsafe_skip_buffer_lock_(unsafe_skip_buffer_lock) {
  if (local_control_port_ >= 0) {
    if (max_blocks_ <= 0) {
      throw std::invalid_argument("max_blocks must be positive");
    }
    if (num_slots_ <= 0) {
      throw std::invalid_argument("num_slots must be positive");
    }
    auto status = ConfigureHostStagingSlots(num_slots_, max_blocks_);
    if (!status.ok()) {
      throw std::runtime_error("Failed to configure host staging slots: " +
                               std::string(status.message()));
    }
    if (num_layers() > 0) {
      ConfigureDataPortFromKvTransfer();
    }
    InitializeSlotPool(num_slots_);
    StartControlServer();
  }
}

KVCacheManagerWithTransfer::KVCacheManagerWithTransfer(
    size_t num_layers, size_t num_shards, size_t slice_byte_size,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    int parallelism, int64_t node_id, int64_t local_control_port,
    int64_t max_blocks, int64_t num_slots, double timeout_s)
    : KVCacheManagerBase(num_layers, num_shards, slice_byte_size, local_port,
                         host_blocks_to_allocate.has_value()
                             ? *host_blocks_to_allocate
                             : num_slots * max_blocks,
                         parallelism, nullptr),
      node_id_(node_id),
      local_control_port_(static_cast<int>(local_control_port)),
      local_data_port_(0),
      max_blocks_(max_blocks),
      num_slots_(num_slots),
      timeout_s_(timeout_s),
      unsafe_skip_buffer_lock_(false) {
  if (local_control_port_ >= 0) {
    if (max_blocks_ <= 0) {
      throw std::invalid_argument("max_blocks must be positive");
    }
    if (num_slots_ <= 0) {
      throw std::invalid_argument("num_slots must be positive");
    }
    auto status = ConfigureHostStagingSlots(num_slots_, max_blocks_);
    if (!status.ok()) {
      throw std::runtime_error("Failed to configure host staging slots: " +
                               std::string(status.message()));
    }
    if (num_layers > 0) {
      ConfigureDataPortFromKvTransfer();
    }
    InitializeSlotPool(num_slots_);
    StartControlServer();
  }
}

KVCacheManagerWithTransfer::~KVCacheManagerWithTransfer() {
  StopControlServer();
  push_pool_.reset();
  pull_pool_.reset();
  if (num_layers() > 0) {
    SetBlockReadinessCallback(nullptr);
  }
}

int64_t KVCacheManagerWithTransfer::NotifyForRead(
    absl::string_view req_id, uint64_t uuid,
    const std::vector<int64_t>& block_ids) {
  const auto register_start = std::chrono::steady_clock::now();
  if (block_ids.empty()) {
    return 0;
  }

  {
    std::lock_guard<std::mutex> lock(mu_);
    if (pending_acks_.erase(uuid) > 0) {
      done_sending_.insert(std::string(req_id));
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
         << " req_id=" << req_id << " uuid=" << uuid << " node_id=" << node_id_
         << " blocks=" << block_ids.size() << " enqueue_ms="
         << DurationMs(register_start, std::chrono::steady_clock::now())
         << " failed=0";
  EmitTimingLog(timing.str());
  return static_cast<int64_t>(uuid);
}

void KVCacheManagerWithTransfer::StartRead(
    absl::string_view req_id, uint64_t uuid, absl::string_view remote_endpoint,
    const std::vector<int64_t>& remote_block_ids,
    const std::vector<int64_t>& local_block_ids, int parallelism,
    std::optional<std::vector<int64_t>> local_host_block_ids) {
  VLOG(1) << "KVCacheManagerWithTransfer::StartRead (Hybrid Bridge) called. "
             "req_id: "
          << req_id << ", uuid: " << uuid << ", remote: " << remote_endpoint
          << ", Thread: " << std::this_thread::get_id();
  // local_block_ids index the consumer's DEVICE KV cache (the H2D chip dst),
  // not the host staging pool. If the caller didn't supply explicit host
  // staging indices, borrow a staging slot and use its host block range so the
  // inbound push lands at valid host indices (< num_host_blocks). Reusing
  // local_block_ids as host indices overflows the staging buffer (OUT_OF_RANGE)
  // once a device block id exceeds num_host_blocks.
  std::vector<int64_t> host_block_ids;
  int64_t recv_slot = -1;
  if (local_host_block_ids.has_value()) {
    host_block_ids = *local_host_block_ids;
  } else if (!local_block_ids.empty()) {
    std::lock_guard<std::mutex> lock(mu_);
    if (static_cast<int64_t>(local_block_ids.size()) > max_blocks_ ||
        free_slots_.empty()) {
      // Request larger than a slot, or staging pool exhausted: surface as a
      // recv failure (the connector can recompute) rather than throwing.
      failed_recving_.insert(std::string(req_id));
      return;
    }
    recv_slot = AcquireSlotLocked();
    const int64_t base = static_cast<int64_t>(StagingBlockBase(recv_slot));
    host_block_ids.reserve(local_block_ids.size());
    for (size_t k = 0; k < local_block_ids.size(); ++k) {
      host_block_ids.push_back(base + static_cast<int64_t>(k));
    }
  }
  CopyPlan load_plan =
      BuildLoadCopyPlan(remote_block_ids, local_block_ids, host_block_ids);

  {
    std::lock_guard<std::mutex> lock(mu_);
    RecvEntry entry;
    entry.req_id = req_id;
    entry.slot_idx = recv_slot;
    entry.deadline = DeadlineFromNow();
    entry.total_blocks =
        static_cast<int64_t>(load_plan.h2d_host_block_ids.size());
    for (size_t k = 0; k < load_plan.h2d_host_block_ids.size(); ++k) {
      entry.host_to_chip[load_plan.h2d_host_block_ids[k]] =
          load_plan.h2d_local_block_ids[k];
    }
    active_recv_entries_[uuid] = std::move(entry);
  }

  if (load_plan.num_blocks == 0) {
    std::lock_guard<std::mutex> lock(mu_);
    done_recving_.insert(std::string(req_id));
    ReleaseSlotLocked(recv_slot);
    active_recv_entries_.erase(uuid);
    return;
  }

  std::optional<int> target_node = assigned_numa_node();

  push_pool_->Schedule(target_node, [this, req_id = std::string(req_id), uuid,
                                     remote_endpoint =
                                         std::string(remote_endpoint),
                                     load_plan = std::move(load_plan)]() {
    try {
      int control_fd = ConnectTcp(remote_endpoint);
      auto control_cleanup =
          std::unique_ptr<int, void (*)(int*)>(&control_fd, [](int* p) {
            if (p && *p >= 0) close(*p);
          });

      ControlRequestHeader stream_request;
      stream_request.magic = kControlMagic;
      stream_request.op = kOpPullStream;
      stream_request.uuid = uuid;
      stream_request.num_blocks = static_cast<uint64_t>(load_plan.num_blocks);
      stream_request.consumer_data_port =
          static_cast<uint32_t>(local_data_port_);
      CheckStatus(
          "control pull stream write",
          WriteExact(control_fd, &stream_request, sizeof(stream_request)));
      WriteBlockIds(control_fd, load_plan.producer_remote_block_ids);
      WriteBlockIds(control_fd, load_plan.transport_host_block_ids);

      ControlResponseHeader response = ReadControlResponseHeader(control_fd);
      if (response.status != 0) {
        throw std::runtime_error(
            "Remote producer rejected Hybrid Bridge read request");
      }
      VLOG(1) << "StartRead (Hybrid Bridge) successfully registered pull "
                 "request with Producer. req_id: "
              << req_id;
    } catch (const std::exception& e) {
      LOG(ERROR)
          << "Raiden consumer error during Hybrid Bridge StartRead connect: "
          << e.what();
      std::lock_guard<std::mutex> lock(mu_);
      failed_recving_.insert(req_id);
      auto it = active_recv_entries_.find(uuid);
      if (it != active_recv_entries_.end()) {
        ReleaseSlotLocked(it->second.slot_idx);
        active_recv_entries_.erase(it);
      }
    }
  });
}

std::tuple<std::vector<std::string>, std::vector<std::string>,
           std::vector<std::string>>
KVCacheManagerWithTransfer::CompleteReadRaw() {
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
    // Reclaim recv entries whose transfer never completed (e.g. the producer
    // died or never finished pushing). Without this the entry and its host
    // staging slot leak forever, eventually exhausting the slot pool. Surface
    // the timeout as a recv failure so the connector can recompute the blocks.
    for (auto it = active_recv_entries_.begin();
         it != active_recv_entries_.end();) {
      if (it->second.deadline <= now) {
        failed_recving_.insert(it->second.req_id);
        ReleaseSlotLocked(it->second.slot_idx);
        active_recv_entries_.erase(it++);
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

StageResult KVCacheManagerWithTransfer::IssueH2D(
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
  for (const kv_cache::KVCacheHostSpan& span : host_spans) {
    total_bytes += static_cast<int64_t>(span.nbytes);
  }
  auto fut_or = H2d(transfer_spec.src_offsets, transfer_spec.dst_offsets,
                    transfer_spec.sizes, slot_idx);
  if (!fut_or.ok()) {
    throw std::runtime_error("Failed to issue H2D transfer: " +
                             std::string(fut_or.status().message()));
  }
  future->Add(std::move(fut_or.value()));
  return {.future = std::move(future),
          .host_spans = std::move(host_spans),
          .total_bytes = total_bytes,
          .copy_segments = static_cast<int64_t>(copy_spec.sizes.size())};
}

std::vector<kv_cache::KVCacheHostSpan> KVCacheManagerWithTransfer::LayerSpans(
    int64_t slot_idx, int64_t num_blocks) {
  if (num_layers() == 0) {
    throw std::runtime_error("KV cache manager is not registered");
  }
  if (num_blocks < 0 || num_blocks > max_blocks_) {
    throw std::out_of_range("num_blocks out of range");
  }
  std::vector<kv_cache::KVCacheHostSpan> spans;
  spans.reserve(num_layers() * num_shards());
  for (size_t layer_idx = 0; layer_idx < num_layers(); ++layer_idx) {
    for (size_t shard_idx = 0; shard_idx < num_shards(); ++shard_idx) {
      spans.push_back(
          ValueOrThrow("Failed to get KVCacheManager host staging span",
                       HostSpan(layer_idx, shard_idx, slot_idx, num_blocks)));
    }
  }
  return spans;
}

void KVCacheManagerWithTransfer::InitializeSlotPool(int64_t num_slots) {
  free_slots_.clear();
  for (int64_t slot = 0; slot < num_slots; ++slot) {
    free_slots_.push_back(slot);
  }
}

int64_t KVCacheManagerWithTransfer::AcquireSlot() {
  std::lock_guard<std::mutex> lock(mu_);
  return AcquireSlotLocked();
}

int64_t KVCacheManagerWithTransfer::AcquireSlotLocked() {
  if (free_slots_.empty()) {
    throw std::runtime_error("Raiden host slot pool exhausted");
  }
  int64_t slot = free_slots_.front();
  free_slots_.pop_front();
  return slot;
}

void KVCacheManagerWithTransfer::ReleaseSlotLocked(int64_t slot_idx) {
  if (slot_idx < 0) return;
  free_slots_.push_back(slot_idx);
}

void KVCacheManagerWithTransfer::ReleaseEntrySlotLocked(
    const std::shared_ptr<SendEntry>& entry) {
  if (!entry || entry->slot_idx < 0 || entry->slot_released) return;
  RemoveStagingReadinessLocked(entry->slot_idx);
  for (int64_t block_id : entry->registered_block_ids) {
    active_producer_blocks_.erase(block_id);
  }
  ReleaseSlotLocked(entry->slot_idx);
  entry->slot_released = true;
}

std::shared_ptr<KVCacheManagerWithTransfer::StagingReadinessState>
KVCacheManagerWithTransfer::CreateStagingReadiness(int64_t slot_idx,
                                                   int64_t num_blocks) {
  auto state = std::make_shared<StagingReadinessState>();
  state->slot_idx = slot_idx;
  state->num_blocks = num_blocks;
  state->num_layers = num_layers();
  state->num_shards = num_shards();
  state->layers.resize(state->num_layers * state->num_shards);
  {
    std::lock_guard<std::mutex> lock(mu_);
    staging_readiness_[slot_idx] = state;
  }
  return state;
}

void KVCacheManagerWithTransfer::MarkStagingLayerReady(
    const std::shared_ptr<StagingReadinessState>& state, size_t layer_idx,
    size_t shard_idx, absl::Status status) {
  if (!state) return;
  const size_t layer_state_idx = layer_idx * state->num_shards + shard_idx;
  if (layer_state_idx >= state->layers.size()) return;
  {
    std::lock_guard<std::mutex> lock(state->mu);
    state->layers[layer_state_idx].done = true;
    state->layers[layer_state_idx].status = std::move(status);
  }
  state->cv.notify_all();
}

void KVCacheManagerWithTransfer::RemoveStagingReadinessLocked(
    int64_t slot_idx) {
  auto it = staging_readiness_.find(slot_idx);
  if (it == staging_readiness_.end()) return;
  std::shared_ptr<StagingReadinessState> state = it->second;
  staging_readiness_.erase(it);
  {
    std::lock_guard<std::mutex> state_lock(state->mu);
    for (StagingLayerReady& layer : state->layers) {
      if (!layer.done) {
        layer.done = true;
        layer.status = absl::CancelledError("staging slot was released");
      }
    }
  }
  state->cv.notify_all();
}

absl::Status KVCacheManagerWithTransfer::WaitForStagingBlockRead(
    size_t layer_idx, size_t shard_idx, int block_id) {
  if (block_id < 0 || max_blocks_ <= 0) {
    return absl::OkStatus();
  }
  std::shared_ptr<StagingReadinessState> state;
  bool is_legacy = false;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = active_producer_blocks_.find(block_id);
    if (it != active_producer_blocks_.end()) {
      state = it->second;
    } else {
      // Fallback to legacy staging slot lookup
      const int64_t slot_idx = static_cast<int64_t>(block_id) / max_blocks_;
      auto it_legacy = staging_readiness_.find(slot_idx);
      if (it_legacy != staging_readiness_.end()) {
        state = it_legacy->second;
        is_legacy = true;
      }
    }
  }
  if (!state) {
    return absl::OkStatus();
  }
  if (is_legacy) {
    const int64_t local_block_idx =
        static_cast<int64_t>(block_id) % max_blocks_;
    if (local_block_idx < 0 || local_block_idx >= state->num_blocks) {
      return absl::OkStatus();
    }
  }
  if (layer_idx >= state->num_layers || shard_idx >= state->num_shards) {
    return absl::OutOfRangeError(
        "staging readiness layer or shard out of range");
  }
  const size_t layer_state_idx = layer_idx * state->num_shards + shard_idx;
  std::unique_lock<std::mutex> lock(state->mu);
  state->cv.wait(lock, [&]() {
    return state->layers[layer_state_idx].done || stopping_.load();
  });
  if (stopping_.load() && !state->layers[layer_state_idx].done) {
    return absl::CancelledError("Raiden transfer engine is stopping");
  }
  return state->layers[layer_state_idx].status;
}

void KVCacheManagerWithTransfer::StartControlServer() {
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

void KVCacheManagerWithTransfer::StopControlServer() {
  stopping_ = true;
  {
    std::lock_guard<std::mutex> lock(mu_);
    while (!staging_readiness_.empty()) {
      RemoveStagingReadinessLocked(staging_readiness_.begin()->first);
    }
  }
  if (control_fd_ >= 0) {
    shutdown(control_fd_, SHUT_RDWR);
    close(control_fd_);
  }
  if (control_thread_.joinable()) {
    control_thread_.join();
  }
  control_fd_ = -1;
}

void KVCacheManagerWithTransfer::ControlServerLoop() {
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
    std::optional<int> source_node = assigned_numa_node();

    pull_pool_->Schedule(source_node, [this, client_fd]() {
      HandleControlConnection(client_fd);
      close(client_fd);
    });
  }
}

void KVCacheManagerWithTransfer::HandleControlConnection(int fd) {
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

void KVCacheManagerWithTransfer::ProcessPullStream(
    int fd, const ControlRequestHeader& req) {
  std::shared_ptr<SendEntry> entry;
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
  if (stopping_) return;
  if (!entry) {
    throw std::runtime_error(
        "KVCacheManagerWithTransfer is stopping during wait for send entry");
  }

  std::vector<int64_t> src_block_ids = ReadBlockIds(fd, req.num_blocks);
  std::vector<int64_t> dst_block_ids = ReadBlockIds(fd, req.num_blocks);
  ValidateRequestedBlocks(*entry, src_block_ids);

  // Acknowledge acceptance to consumer immediately
  ControlResponseHeader response;
  response.magic = kResponseMagic;
  response.status = 0;
  response.num_layers = static_cast<uint32_t>(num_layers() * num_shards());
  response.data_port = static_cast<uint32_t>(local_data_port_);
  CheckStatus("control stream response header write",
              WriteExact(fd, &response, sizeof(response)));

  std::string peer_ip = GetPeerIp(fd);
  std::string remote_data_endpoint =
      peer_ip + ":" + std::to_string(req.consumer_data_port);

  VLOG(1) << "ProcessPullStream (Hybrid Bridge) successfully acknowledged "
             "consumer. Intercepting and launching StartPushInternal to "
          << remote_data_endpoint;

  // Intercept and Execute Hybrid Push on behalf of remote consumer!
  StartPushInternal(req.uuid, remote_data_endpoint, src_block_ids,
                    dst_block_ids);
}

void KVCacheManagerWithTransfer::StartPushInternal(
    uint64_t uuid, absl::string_view remote_data_endpoint,
    const std::vector<int64_t>& src_block_ids,
    const std::vector<int64_t>& dst_block_ids) {
  // Stage the producer's device KV into a compact host slot
  // ([StagingBlockBase(slot), +n)) and send those slot blocks to the consumer,
  // keeping host offsets within the staging pool.
  int64_t send_slot = -1;
  std::vector<int64_t> host_block_ids;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (static_cast<int64_t>(src_block_ids.size()) > max_blocks_ ||
        free_slots_.empty()) {
      auto it = send_entries_.find(uuid);
      if (it != send_entries_.end()) {
        done_sending_.insert(it->second->req_id);
        ReleaseEntrySlotLocked(it->second);
        send_entries_.erase(it);
      }
      return;
    }
    send_slot = AcquireSlotLocked();
    auto it = send_entries_.find(uuid);
    if (it != send_entries_.end()) it->second->slot_idx = send_slot;
    const int64_t base = static_cast<int64_t>(StagingBlockBase(send_slot));
    host_block_ids.reserve(src_block_ids.size());
    for (size_t k = 0; k < src_block_ids.size(); ++k) {
      host_block_ids.push_back(base + static_cast<int64_t>(k));
    }
  }

  std::vector<int64_t> sizes(src_block_ids.size(), 1);
  auto future_or =
      D2h(src_block_ids, host_block_ids, sizes, /*slot_idx=*/std::nullopt);
  if (!future_or.ok()) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = send_entries_.find(uuid);
    if (it != send_entries_.end()) {
      done_sending_.insert(it->second->req_id);
      ReleaseEntrySlotLocked(it->second);
      send_entries_.erase(it);
    }
    ThrowStatus("Failed to issue D2H in StartPushInternal", future_or.status());
  }

  auto future = std::move(future_or.value());
  future.OnReady(
      [this, uuid, remote_data_endpoint = std::string(remote_data_endpoint),
       host_block_ids,
       dst_block_ids](const absl::StatusOr<raiden::BufferHolders>& s) {
        if (!s.ok()) {
          std::lock_guard<std::mutex> lock(mu_);
          auto it = send_entries_.find(uuid);
          if (it != send_entries_.end()) {
            done_sending_.insert(it->second->req_id);
            ReleaseEntrySlotLocked(it->second);
            send_entries_.erase(it);
          }
          return;
        }
        std::vector<int> src_ints(host_block_ids.begin(), host_block_ids.end());
        std::vector<int> dst_ints(dst_block_ids.begin(), dst_block_ids.end());
        // Push across Data Plane socket!
        auto push_s = H2hWrite(remote_data_endpoint, src_ints, dst_ints, uuid);
        std::lock_guard<std::mutex> lock(mu_);
        auto it = send_entries_.find(uuid);
        if (it != send_entries_.end()) {
          done_sending_.insert(it->second->req_id);
          ReleaseEntrySlotLocked(it->second);
          send_entries_.erase(it);
        }
      });
}

absl::Status KVCacheManagerWithTransfer::OnBlocksReceived(
    const std::vector<int>& block_ids, uint64_t uuid) {
  std::vector<int64_t> host_src_blocks;
  std::vector<int64_t> chip_dst_blocks;
  std::string target_req_id;
  int64_t chunk_size = block_ids.size();
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = active_recv_entries_.find(uuid);
    if (it == active_recv_entries_.end()) {
      return absl::OkStatus();
    }
    target_req_id = it->second.req_id;
    host_src_blocks.reserve(chunk_size);
    chip_dst_blocks.reserve(chunk_size);
    for (int h_id : block_ids) {
      auto map_it = it->second.host_to_chip.find(h_id);
      if (map_it == it->second.host_to_chip.end()) {
        failed_recving_.insert(target_req_id);
        ReleaseSlotLocked(it->second.slot_idx);
        active_recv_entries_.erase(it);
        return absl::FailedPreconditionError(
            "Received unexpected host block ID");
      }
      host_src_blocks.push_back(h_id);
      chip_dst_blocks.push_back(map_it->second);
    }
  }

  VLOG(1) << "OnBlocksReceived (Hybrid Bridge) triggered for UUID " << uuid
          << " with " << chunk_size << " blocks. Launching H2D copy for req_id "
          << target_req_id;

  std::vector<int64_t> copy_sizes(chunk_size, 1);

  auto future_or = H2d(host_src_blocks, chip_dst_blocks, copy_sizes,
                       /*slot_idx=*/std::nullopt);
  if (!future_or.ok()) {
    std::lock_guard<std::mutex> lock(mu_);
    failed_recving_.insert(target_req_id);
    auto it = active_recv_entries_.find(uuid);
    if (it != active_recv_entries_.end()) {
      ReleaseSlotLocked(it->second.slot_idx);
      active_recv_entries_.erase(it);
    }
    return future_or.status();
  }

  auto future = std::move(future_or.value());
  future.OnReady([this, uuid, target_req_id,
                  chunk_size](const absl::StatusOr<raiden::BufferHolders>& s) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = active_recv_entries_.find(uuid);
    if (it == active_recv_entries_.end()) return;
    if (s.ok()) {
      it->second.num_completed_blocks += chunk_size;
      if (it->second.num_completed_blocks == it->second.total_blocks) {
        done_recving_.insert(target_req_id);
        ReleaseSlotLocked(it->second.slot_idx);
        active_recv_entries_.erase(it);
      }
    } else {
      failed_recving_.insert(target_req_id);
      ReleaseSlotLocked(it->second.slot_idx);
      active_recv_entries_.erase(it);
    }
  });
  return absl::OkStatus();
}

std::string KVCacheManagerWithTransfer::EndpointWithPort(
    absl::string_view endpoint, int port) const {
  auto [host, ignored_port] = SplitEndpoint(endpoint);
  (void)ignored_port;
  return absl::StrCat(host, ":", port);
}

void KVCacheManagerWithTransfer::AckRemote(absl::string_view remote_endpoint,
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

KVCacheManagerWithTransfer::ControlResponseHeader
KVCacheManagerWithTransfer::ReadControlResponseHeader(int fd) {
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

void KVCacheManagerWithTransfer::AckSend(uint64_t uuid) {
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
         << " node_id=" << node_id_ << " blocks=" << entry->num_blocks
         << " bytes=" << entry->total_bytes
         << " stage_to_ack_ms=" << DurationMs(entry->d2h_done, ack_done)
         << " register_to_ack_ms="
         << DurationMs(entry->register_start, ack_done)
         << " failed=" << (entry->failed ? 1 : 0);
  EmitTimingLog(timing.str());
}

std::chrono::steady_clock::time_point
KVCacheManagerWithTransfer::DeadlineFromNow() const {
  return std::chrono::steady_clock::now() +
         std::chrono::milliseconds(static_cast<int64_t>(timeout_s_ * 1000.0));
}

void KVCacheManagerWithTransfer::ConfigureDataPortFromKvTransfer() {
  if (num_layers() == 0) {
    local_data_port_ = 0;
    return;
  }
  std::optional<int> data_port = local_port();
  if (!data_port.has_value()) {
    throw std::runtime_error("KVCacheManager BlockTransport is not running");
  }
  local_data_port_ = *data_port;
  SetBlockReadinessCallback(
      [this](size_t layer_idx, size_t shard_idx, int block_id) {
        return WaitForStagingBlockRead(layer_idx, shard_idx, block_id);
      });
}

uint64_t KVCacheManagerWithTransfer::StagingBlockBase(int64_t slot_idx) const {
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

std::vector<int> KVCacheManagerWithTransfer::ContiguousBlockIds(
    uint64_t base, uint64_t count) const {
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

std::optional<int> KVCacheManagerWithTransfer::GetLocalTpuNumaNode(
    xla::PjRtBuffer* buf) const {
  if (buf && buf->device()) {
    int node = GetPjRtDeviceNumaNode(buf->device());
    if (node >= 0) {
      return node;
    }
  }
  return std::nullopt;
}

}  // namespace tpu_raiden
