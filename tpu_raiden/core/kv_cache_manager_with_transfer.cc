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
#include <netdb.h>
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
#include <future>
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
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/errors.h"
#include "tpu_raiden/core/host_memory_allocator.h"
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/core/status_macros.h"
#include "tpu_raiden/core/tpu_utils.h"
#include "tpu_raiden/kv_cache/kv_cache_manager_base.h"

namespace tpu_raiden {
namespace {

constexpr absl::Duration kPendingWorkTimeout = absl::Seconds(30);

[[noreturn]] void ThrowStatus(const std::string& context,
                              const absl::Status& status) {
  throw std::runtime_error(context + ": " + std::string(status.message()));
}

void CheckStatus(const std::string& context, const absl::Status& status) {
  if (!status.ok()) {
    ThrowStatus(context, status);
  }
}

void EmitTimingLog(const std::string& message) { LOG(INFO) << message; }

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
  std::string host;
  int port = 0;
  if (endpoint.empty()) {
    throw std::invalid_argument("endpoint is empty");
  }
  if (endpoint[0] == '[') {
    size_t closing_bracket = endpoint.find(']');
    if (closing_bracket == std::string::npos ||
        closing_bracket + 2 >= endpoint.size() ||
        endpoint[closing_bracket + 1] != ':') {
      throw std::invalid_argument("invalid IPv6 endpoint: " + endpoint);
    }
    host = endpoint.substr(1, closing_bracket - 1);
    port = std::stoi(endpoint.substr(closing_bracket + 2));
  } else {
    size_t colon = endpoint.rfind(':');
    if (colon == std::string::npos) {
      throw std::invalid_argument("endpoint must be host:port");
    }
    host = endpoint.substr(0, colon);
    port = std::stoi(endpoint.substr(colon + 1));
  }
  return {host, port};
}

int ConnectTcp(const std::string& endpoint) {
  auto [host, port] = SplitEndpoint(endpoint);
  struct addrinfo hints;
  struct addrinfo* res = nullptr;
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  std::string port_str = std::to_string(port);
  int err = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res);
  if (err != 0) {
    throw std::runtime_error("Failed to resolve hostname '" + host +
                             "': " + gai_strerror(err));
  }

  int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
  if (fd < 0) {
    freeaddrinfo(res);
    throw std::runtime_error("socket() failed: " +
                             std::string(std::strerror(errno)));
  }
  int opt = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

  if (connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
    std::string err_str = std::strerror(errno);
    close(fd);
    freeaddrinfo(res);
    throw std::runtime_error("connect(" + endpoint + ") failed: " + err_str);
  }
  freeaddrinfo(res);
  return fd;
}

static std::string GetPeerIp(int fd) {
  sockaddr_storage addr;
  socklen_t len = sizeof(addr);
  if (getpeername(fd, reinterpret_cast<sockaddr*>(&addr), &len) < 0) {
    throw std::runtime_error("getpeername() failed: " +
                             std::string(std::strerror(errno)));
  }
  char ip_buf[INET6_ADDRSTRLEN];
  if (addr.ss_family == AF_INET) {
    sockaddr_in* s = reinterpret_cast<sockaddr_in*>(&addr);
    if (inet_ntop(AF_INET, &s->sin_addr, ip_buf, sizeof(ip_buf)) == nullptr) {
      throw std::runtime_error("inet_ntop() failed: " +
                               std::string(std::strerror(errno)));
    }
  } else if (addr.ss_family == AF_INET6) {
    sockaddr_in6* s = reinterpret_cast<sockaddr_in6*>(&addr);
    if (inet_ntop(AF_INET6, &s->sin6_addr, ip_buf, sizeof(ip_buf)) == nullptr) {
      throw std::runtime_error("inet_ntop() failed: " +
                               std::string(std::strerror(errno)));
    }
  } else {
    throw std::runtime_error("unknown socket family");
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

static CopySpec BuildCoalescedCopySpec(
    const std::vector<int64_t>& src_block_ids,
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
      BuildCoalescedCopySpec(plan.h2d_host_block_ids, plan.h2d_local_block_ids);
  plan.host_dst_to_src.clear();  // No host reordering needed!
  return plan;
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
    : KVCacheManagerBase(
          layer_buffers, local_port,
          host_blocks_to_allocate.value_or(num_slots * max_blocks),
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
      throw std::runtime_error(absl::StrCat(
          "Failed to configure host staging slots: ", status.message()));
    }
    if (num_layers() > 0) {
      ConfigureDataPortFromKvTransfer();
    }
    status = InitializeSlotPool(num_slots_);
    if (!status.ok()) {
      throw std::runtime_error(
          absl::StrCat("Failed to initialize slot pool: ", status.message()));
    }
    StartControlServer();
  }
}

KVCacheManagerWithTransfer::KVCacheManagerWithTransfer(
    const std::vector<std::vector<xla::PjRtBuffer*>>& layer_buffers,
    size_t slice_byte_size, const std::vector<int64_t>& dimensions,
    size_t physical_size, std::optional<int> local_port,
    std::optional<int> host_blocks_to_allocate, bool unsafe_skip_buffer_lock,
    int parallelism, HostBufferAllocator host_allocator, int64_t node_id,
    int64_t local_control_port, int64_t max_blocks, int64_t num_slots,
    double timeout_s, std::optional<int> assigned_numa_node)
    : KVCacheManagerBase(
          layer_buffers, local_port,
          host_blocks_to_allocate.value_or(num_slots * max_blocks),
          unsafe_skip_buffer_lock, parallelism, host_allocator),
      node_id_(node_id),
      local_control_port_(static_cast<int>(local_control_port)),
      local_data_port_(0),
      max_blocks_(max_blocks),
      num_slots_(num_slots),
      timeout_s_(timeout_s),
      unsafe_skip_buffer_lock_(unsafe_skip_buffer_lock) {
  if (num_layers() == 0 || num_shards() == 0) {
    return;
  }
  if (local_control_port_ >= 0) {
    if (max_blocks_ <= 0) {
      throw std::invalid_argument("max_blocks must be positive");
    }
    if (num_slots_ <= 0) {
      throw std::invalid_argument("num_slots must be positive");
    }
    auto status = ConfigureHostStagingSlots(num_slots_, max_blocks_);
    if (!status.ok()) {
      throw std::runtime_error(absl::StrCat(
          "Failed to configure host staging slots: ", status.message()));
    }
    if (num_layers() > 0) {
      ConfigureDataPortFromKvTransfer();
    }
    status = InitializeSlotPool(num_slots_);
    if (!status.ok()) {
      throw std::runtime_error(
          absl::StrCat("Failed to initialize slot pool: ", status.message()));
    }
    StartControlServer();
  }
}

KVCacheManagerWithTransfer::KVCacheManagerWithTransfer(
    size_t num_layers, size_t num_shards, size_t slice_byte_size,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    int parallelism, int64_t node_id, int64_t local_control_port,
    int64_t max_blocks, int64_t num_slots, double timeout_s)
    : KVCacheManagerBase(
          num_layers, num_shards, slice_byte_size, local_port,
          host_blocks_to_allocate.value_or(num_slots * max_blocks), parallelism,
          nullptr),
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
      throw std::runtime_error(absl::StrCat(
          "Failed to configure host staging slots: ", status.message()));
    }
    if (num_layers > 0) {
      ConfigureDataPortFromKvTransfer();
    }
    status = InitializeSlotPool(num_slots_);
    if (!status.ok()) {
      throw std::runtime_error(
          absl::StrCat("Failed to initialize slot pool: ", status.message()));
    }
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
  if (host_block_manager_ && !all_slots_.empty()) {
    std::vector<int> blocks_to_unlock;
    blocks_to_unlock.reserve(all_slots_.size() * max_blocks_);
    for (const Slot& slot : all_slots_) {
      for (int block_id : slot.block_ids) {
        blocks_to_unlock.push_back(block_id);
      }
    }
    (void)host_block_manager_->Unlock(blocks_to_unlock);
  }
}

int64_t KVCacheManagerWithTransfer::NotifyForRead(
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
         << " req_id=" << req_id << " uuid=" << uuid << " node_id=" << node_id_
         << " blocks=" << block_ids.size() << " enqueue_ms="
         << DurationMs(register_start, std::chrono::steady_clock::now())
         << " failed=0";
  EmitTimingLog(timing.str());
  return static_cast<int64_t>(uuid);
}

absl::Status KVCacheManagerWithTransfer::RegisterActivePlan(
    uint64_t uuid, const rpc::StartTransferRequest& request, bool is_sender) {
  // 1. Call base class implementation to register the plan in active_plans_
  TF_RETURN_IF_ERROR(kv_cache::KVCacheManagerBase::RegisterActivePlan(
      uuid, request, is_sender));

  // 2. If we are the receiver and the destination memory type is HBM,
  //    populate active_recv_entries_ to enable automatic H2D copy!
  if (!is_sender && request.dst_mem_type() == rpc::MEMORY_TYPE_HBM) {
    std::lock_guard<std::mutex> lock(mu_);
    RecvEntry recv_entry;
    std::string req_id = absl::StrCat("resharded_transfer_", uuid);
    recv_entry.req_id = req_id;

    int64_t total_blocks = 0;
    for (const auto& [src_replica_idx, schedule] :
         request.shard_push_schedules()) {
      std::set<int> unique_blocks_from_this_source;
      for (const auto& push_entry : schedule.entries()) {
        recv_entry.host_to_chip[push_entry.dst_block_id()] =
            push_entry.dst_block_id();
        unique_blocks_from_this_source.insert(push_entry.dst_block_id());
      }
      total_blocks += unique_blocks_from_this_source.size();
    }
    recv_entry.total_blocks = total_blocks;
    recv_entry.num_completed_blocks = 0;
    if (total_blocks > 0) {
      active_recv_entries_[uuid] = std::move(recv_entry);
      LOG(INFO) << "RegisterActivePlan (Receiver): Populated "
                   "active_recv_entries_ for UUID "
                << uuid << " with " << total_blocks
                << " total physical block-pushes (including duplicates across "
                   "sources) for automatic H2D.";
    }
  }
  return absl::OkStatus();
}

std::vector<EndpointDescriptor>
KVCacheManagerWithTransfer::get_local_endpoints() const {
  std::vector<int64_t> all_shards(num_shards_);
  for (size_t i = 0; i < num_shards_; ++i) {
    all_shards[i] = static_cast<int64_t>(i);
  }
  int64_t port =
      local_control_port_ > 0 ? local_control_port_ : local_port().value_or(0);
  return {{absl::StrCat(local_ip(), ":", port), all_shards}};
}

void KVCacheManagerWithTransfer::StartRead(
    const std::string& req_id, uint64_t uuid,
    const std::vector<std::string>& remote_endpoints,
    const std::vector<int64_t>& remote_block_ids,
    const std::vector<int64_t>& local_block_ids, int parallelism,
    std::optional<std::vector<int64_t>> local_host_block_ids) {
  std::string target_ep;
  if (!remote_endpoints.empty()) {
    target_ep = remote_endpoints[0];
  }
  StartRead(req_id, uuid, target_ep, remote_block_ids, local_block_ids,
            parallelism, local_host_block_ids);
}

void KVCacheManagerWithTransfer::StartRead(
    const std::string& req_id, uint64_t uuid,
    const std::vector<EndpointDescriptor>& remote_descriptors,
    const std::vector<int64_t>& remote_block_ids,
    const std::vector<int64_t>& local_block_ids, int parallelism,
    std::optional<std::vector<int64_t>> local_host_block_ids) {
  if (remote_descriptors.empty()) {
    return;
  }
  // TODO(b/agy): Deal with the case where the shards on both sides don't
  // perfectly match. KVCacheManagerWithTransfer is bound to a single NUMA node
  // / single endpoint. Multi-endpoint routing across sockets is orchestrated by
  // the JAX facade.
  if (remote_descriptors.size() != 1) {
    LOG(WARNING) << "KVCacheManagerWithTransfer::StartRead expected 1 remote "
                    "descriptor, got "
                 << remote_descriptors.size()
                 << ". Picking the first endpoint.";
  }
  StartRead(req_id, uuid, remote_descriptors[0].endpoint, remote_block_ids,
            local_block_ids, parallelism, local_host_block_ids);
}

void KVCacheManagerWithTransfer::StartRead(
    const std::string& req_id, uint64_t uuid,
    const std::string& remote_endpoint,
    const std::vector<int64_t>& remote_block_ids,
    const std::vector<int64_t>& local_block_ids, int parallelism,
    std::optional<std::vector<int64_t>> local_host_block_ids) {
  VLOG(1) << "KVCacheManagerWithTransfer::StartRead (Hybrid Bridge) called. "
             "req_id: "
          << req_id << ", uuid: " << uuid << ", remote: " << remote_endpoint
          << ", Thread: " << std::this_thread::get_id();
  // local_block_ids index the consumer's DEVICE KV cache, not the host staging
  // pool; reusing them as host indices overflows the host buffer once a device
  // block id exceeds num_host_blocks. If the caller didn't supply explicit host
  // indices, borrow a staging slot and stage into its reserved host blocks
  // (slot.block_ids -- the real, possibly non-contiguous host blocks).
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
      failed_recving_.insert(req_id);
      return;
    }
    Slot slot = AcquireSlotLocked();
    recv_slot = slot.slot_idx;
    host_block_ids.reserve(local_block_ids.size());
    for (size_t k = 0; k < local_block_ids.size(); ++k) {
      host_block_ids.push_back(slot.block_ids[k]);
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
    entry.chip_block_ids = load_plan.h2d_local_block_ids;
    entry.total_blocks = load_plan.num_blocks;
    entry.num_completed_blocks = 0;
    // Read the H2D source from the actual staged host blocks (coalesced), not a
    // compact 0..n-1 region -- the producer writes into host_block_ids, so the
    // consumer must read back from the same blocks.
    entry.h2d_copy = load_plan.h2d_copy;
    for (size_t i = 0; i < load_plan.transport_host_block_ids.size(); ++i) {
      entry.host_to_chip[load_plan.transport_host_block_ids[i]] =
          load_plan.h2d_local_block_ids[i];
    }
    entry.h2d_dispatch_futures.reserve(load_plan.h2d_local_block_ids.size());
    active_recv_entries_[uuid] = std::move(entry);
  }

  if (load_plan.num_blocks == 0) {
    std::lock_guard<std::mutex> lock(mu_);
    done_recving_.insert(req_id);
    ReleaseSlotLocked(recv_slot);
    active_recv_entries_.erase(uuid);
    return;
  }

  std::optional<int> target_node = assigned_numa_node();

  push_pool_->Schedule(target_node, [this, req_id, uuid, remote_endpoint,
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
      std::string bound_ip = local_ip();
      if (inet_pton(AF_INET6, bound_ip.c_str(), stream_request.consumer_ip) <=
          0) {
        std::memset(stream_request.consumer_ip, 0, 16);
      }
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
  if (num_layers() == 0) {
    throw std::runtime_error("KV cache manager is not registered");
  }
  if (slot_idx < 0 || slot_idx >= num_slots_) {
    throw std::out_of_range("slot_idx out of range");
  }
  if (num_blocks < 0 || num_blocks > max_blocks_) {
    throw std::out_of_range("num_blocks out of range");
  }
  if (num_blocks != static_cast<int64_t>(local_block_ids.size())) {
    throw std::invalid_argument("num_blocks must match len(local_block_ids)");
  }

  // Get the actual host block IDs for the first num_blocks in the slot
  const Slot& slot = all_slots_[slot_idx];
  std::vector<int64_t> host_block_ids;
  host_block_ids.reserve(num_blocks);
  for (int64_t i = 0; i < num_blocks; ++i) {
    host_block_ids.push_back(slot.block_ids[i]);
  }

  // Coalesce contiguous (host, device) block runs
  CopySpec copy_spec = BuildCoalescedCopySpec(host_block_ids, local_block_ids);
  kv_cache::KVCacheCopySpec transfer_spec = ToKVCacheCopySpec(copy_spec);

  // We still calculate host_spans for the result, but we don't use slot_idx
  // in H2d call to avoid slot-based double offsetting in the base class.
  std::vector<kv_cache::KVCacheHostSpan> host_spans =
      LayerSpans(slot_idx, num_blocks);

  auto future = std::make_shared<TransferFuture>();
  int64_t total_bytes = 0;
  for (const kv_cache::KVCacheHostSpan& span : host_spans) {
    total_bytes += static_cast<int64_t>(span.nbytes);
  }

  // Call H2d with slot_idx = std::nullopt to use actual host block IDs
  auto fut_or = H2d(transfer_spec.src_offsets, transfer_spec.dst_offsets,
                    transfer_spec.sizes, /*slot_idx=*/std::nullopt);
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
  const Slot& slot = all_slots_[slot_idx];

  // Coalesce contiguous runs of block IDs in the slot
  struct Run {
    int64_t start_block_id;
    int64_t size;
  };
  std::vector<Run> runs;
  for (int64_t start = 0; start < num_blocks;) {
    int64_t end = start + 1;
    while (end < num_blocks &&
           slot.block_ids[end] == slot.block_ids[end - 1] + 1) {
      ++end;
    }
    runs.push_back({slot.block_ids[start], end - start});
    start = end;
  }

  spans.reserve(num_layers() * num_shards() * runs.size());
  for (size_t layer_idx = 0; layer_idx < num_layers(); ++layer_idx) {
    for (size_t shard_idx = 0; shard_idx < num_shards(); ++shard_idx) {
      const auto& shard_info = layers_[layer_idx].shards[shard_idx];
      for (const auto& run : runs) {
        const size_t byte_offset =
            static_cast<size_t>(run.start_block_id) * slice_byte_size_;
        const size_t nbytes = static_cast<size_t>(run.size) * slice_byte_size_;
        spans.push_back(kv_cache::KVCacheHostSpan{
            .ptr = const_cast<uint8_t*>(shard_info.host_ptr) + byte_offset,
            .nbytes = nbytes,
            .slot_idx = slot_idx,
            .base_major = run.start_block_id,
            .num_major = run.size,
            .layer_idx = layer_idx,
            .shard_idx = shard_idx});
      }
    }
  }
  return spans;
}

absl::Status KVCacheManagerWithTransfer::InitializeSlotPool(int64_t num_slots) {
  if (host_block_manager_->num_free_blocks() < num_slots * max_blocks_) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Insufficient free host blocks to initialize slot pool. Required: ",
        num_slots * max_blocks_,
        ", Available: ", host_block_manager_->num_free_blocks()));
  }
  free_slots_.clear();
  all_slots_.clear();
  all_slots_.reserve(num_slots);
  for (int64_t i = 0; i < num_slots; ++i) {
    ASSIGN_OR_RETURN(std::vector<int> allocated_ids,
                     host_block_manager_->Allocate(max_blocks_,
                                                   /*lock=*/true));
    if (allocated_ids.size() != max_blocks_) {
      return absl::InternalError(absl::StrCat(
          "Slot pool allocation returned incorrect number of blocks: ",
          allocated_ids.size(), ", expected: ", max_blocks_));
    }
    Slot slot{/*slot_idx=*/i, /*block_ids=*/allocated_ids};
    all_slots_.push_back(slot);
    free_slots_.push_back(slot);
  }
  return absl::OkStatus();
}

KVCacheManagerWithTransfer::Slot KVCacheManagerWithTransfer::AcquireSlot() {
  std::lock_guard<std::mutex> lock(mu_);
  return AcquireSlotLocked();
}

KVCacheManagerWithTransfer::Slot
KVCacheManagerWithTransfer::AcquireSlotLocked() {
  if (free_slots_.empty()) {
    throw std::runtime_error("Raiden host slot pool exhausted");
  }
  Slot slot = free_slots_.front();
  free_slots_.pop_front();
  return slot;
}

void KVCacheManagerWithTransfer::ReleaseSlotLocked(int64_t slot_idx) {
  if (slot_idx < 0 || slot_idx >= num_slots_) {
    return;
  }
  free_slots_.push_back(all_slots_[slot_idx]);
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
  control_fd_ = socket(AF_INET6, SOCK_STREAM, 0);
  if (control_fd_ < 0) {
    throw std::runtime_error("control socket() failed: " +
                             std::string(std::strerror(errno)));
  }
  int opt = 1;
  setsockopt(control_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  setsockopt(control_fd_, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

  int ipv6only = 0;
  if (setsockopt(control_fd_, IPPROTO_IPV6, IPV6_V6ONLY, &ipv6only,
                 sizeof(ipv6only)) < 0) {
    LOG(WARNING) << "setsockopt IPV6_V6ONLY=0 failed: " << std::strerror(errno);
  }

  sockaddr_in6 addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin6_family = AF_INET6;
  addr.sin6_addr = in6addr_any;
  addr.sin6_port = htons(local_control_port_);

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
  local_control_port_ = ntohs(addr.sin6_port);
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

  bool ip_is_empty = true;
  for (int i = 0; i < 16; ++i) {
    if (req.consumer_ip[i] != 0) {
      ip_is_empty = false;
      break;
    }
  }

  std::string peer_ip;
  if (!ip_is_empty) {
    char ip_str[INET6_ADDRSTRLEN];
    if (inet_ntop(AF_INET6, req.consumer_ip, ip_str, sizeof(ip_str)) !=
        nullptr) {
      peer_ip = ip_str;
    }
  }
  if (peer_ip.empty()) {
    peer_ip = GetPeerIp(fd);
  }

  std::string remote_data_endpoint;
  if (absl::StrContains(peer_ip, ':')) {
    remote_data_endpoint =
        "[" + peer_ip + "]:" + std::to_string(req.consumer_data_port);
  } else {
    remote_data_endpoint =
        peer_ip + ":" + std::to_string(req.consumer_data_port);
  }

  VLOG(1) << "ProcessPullStream (Hybrid Bridge) successfully acknowledged "
             "consumer. Intercepting and launching StartPushInternal to "
          << remote_data_endpoint;

  // Intercept and Execute Hybrid Push on behalf of remote consumer!
  StartPushInternal(req.uuid, remote_data_endpoint, src_block_ids,
                    dst_block_ids);
}

void KVCacheManagerWithTransfer::StartPushInternal(
    uint64_t uuid, const std::string& remote_data_endpoint,
    const std::vector<int64_t>& src_block_ids,
    const std::vector<int64_t>& dst_block_ids) {
  // Stage the producer's device KV into a host slot (slot.block_ids) and send
  // those host blocks to the consumer, keeping host offsets within the staging
  // pool. Writing D2H straight to host[src_block_id] overflows the host buffer
  // once a device block id exceeds num_host_blocks.
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
    Slot slot = AcquireSlotLocked();
    send_slot = slot.slot_idx;
    auto it = send_entries_.find(uuid);
    if (it != send_entries_.end()) it->second->slot_idx = send_slot;
    host_block_ids.reserve(src_block_ids.size());
    for (size_t k = 0; k < src_block_ids.size(); ++k) {
      host_block_ids.push_back(slot.block_ids[k]);
    }
  }

  // Coalesce contiguous (device,host) block runs into a few large copies. With
  // per-block segments (sizes=1) a contiguous KV range becomes n device copies
  // that flood the command queue with small ops and serialize against prefill
  // GEMMs on the shared TensorCore; coalescing collapses a contiguous range to
  // one copy, matching the pre-Hybrid-Push pull path.
  CopySpec d2h_copy = BuildCoalescedCopySpec(src_block_ids, host_block_ids);
  auto future_or = D2h(d2h_copy.src_offsets, d2h_copy.dst_offsets,
                       d2h_copy.sizes, /*slot_idx=*/std::nullopt);
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
      [this, uuid, remote_data_endpoint, host_block_ids,
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
        if (!push_s.ok()) {
          LOG(ERROR) << "H2hWrite failed: " << push_s.status().ToString();
        }
        std::lock_guard<std::mutex> lock(mu_);
        auto it = send_entries_.find(uuid);
        if (it != send_entries_.end()) {
          done_sending_.insert(it->second->req_id);
          ReleaseEntrySlotLocked(it->second);
          send_entries_.erase(it);
        }
      });
}

absl::Status KVCacheManagerWithTransfer::WaitForPendingWork() {
  LOG(INFO) << "Waiting for pending H2D transfers to complete...";
  const absl::Time start = absl::Now();
  while (true) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (active_recv_entries_.empty()) {
        break;
      }
      const absl::Duration elapsed = absl::Now() - start;
      if (elapsed > kPendingWorkTimeout) {
        return absl::DeadlineExceededError(
            "Timeout waiting for pending H2D transfers");
      }
    }
    absl::SleepFor(absl::Milliseconds(100));
  }
  LOG(INFO) << "All pending H2D transfers completed.";
  return absl::OkStatus();
}

std::string KVCacheManagerWithTransfer::EndpointWithPort(
    const std::string& endpoint, int port) const {
  auto [host, ignored_port] = SplitEndpoint(endpoint);
  (void)ignored_port;
  return host + ":" + std::to_string(port);
}

void KVCacheManagerWithTransfer::AckRemote(const std::string& remote_endpoint,
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

absl::Status KVCacheManagerWithTransfer::OnBlocksReceived(
    const std::vector<int>& block_ids, uint64_t uuid) {
  VLOG(1) << "KVCacheManagerWithTransfer::OnBlocksReceived called. uuid: "
          << uuid << ", received blocks count: " << block_ids.size();

  std::string req_id;
  int64_t recv_slot = -1;
  CopySpec h2d_copy;
  absl::flat_hash_map<int64_t, int64_t> host_to_chip;
  bool found = false;
  std::vector<int> accumulated_host_blocks;

  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = active_recv_entries_.find(uuid);
    if (it != active_recv_entries_.end()) {
      it->second.num_completed_blocks += block_ids.size();
      it->second.accumulated_host_block_ids.insert(
          it->second.accumulated_host_block_ids.end(), block_ids.begin(),
          block_ids.end());

      if (it->second.num_completed_blocks >= it->second.total_blocks) {
        req_id = it->second.req_id;
        h2d_copy = it->second.h2d_copy;
        host_to_chip = it->second.host_to_chip;
        accumulated_host_blocks = it->second.accumulated_host_block_ids;
        recv_slot = it->second.slot_idx;
        found = true;
        active_recv_entries_.erase(it);
      } else {
        VLOG(1) << "OnBlocksReceived: Partial blocks received for uuid " << uuid
                << ", completed: " << it->second.num_completed_blocks << " / "
                << it->second.total_blocks;
        return absl::OkStatus();
      }
    }
  }

  if (!found) {
    // Forward to base class for direct pull operations
    return RaidenManagerBase::OnBlocksReceived(block_ids, uuid);
  }

  if (h2d_copy.src_offsets.empty()) {
    // Rebuild on-the-fly for push transfers registered via RegisterActivePlan
    std::vector<int64_t> host_blocks(accumulated_host_blocks.begin(),
                                     accumulated_host_blocks.end());
    std::vector<int64_t> chip_blocks;
    chip_blocks.reserve(host_blocks.size());
    for (int64_t hb : host_blocks) {
      auto it = host_to_chip.find(hb);
      if (it != host_to_chip.end()) {
        chip_blocks.push_back(it->second);
      } else {
        chip_blocks.push_back(hb);
      }
    }
    h2d_copy = BuildCoalescedCopySpec(host_blocks, chip_blocks);
  }

  VLOG(1) << "OnBlocksReceived: Issuing H2D copy for req_id: " << req_id
          << ", coalesced runs count: " << h2d_copy.src_offsets.size();

  auto future_or = H2d(h2d_copy.src_offsets, h2d_copy.dst_offsets,
                       h2d_copy.sizes, /*slot_idx=*/std::nullopt);
  if (!future_or.ok()) {
    std::lock_guard<std::mutex> lock(mu_);
    failed_recving_.insert(req_id);
    ReleaseSlotLocked(recv_slot);
    return future_or.status();
  }

  auto future = std::move(future_or.value());
  future.OnReady([this, req_id, recv_slot](auto status_or) {
    std::lock_guard<std::mutex> lock(mu_);
    if (status_or.ok()) {
      VLOG(1) << "OnBlocksReceived (H2D copy complete): successfully "
                 "completed receive for req_id: "
              << req_id;
      done_recving_.insert(req_id);
    } else {
      LOG(ERROR) << "OnBlocksReceived (H2D copy failed) for req_id: " << req_id
                 << ", error: " << status_or.status().ToString();
      failed_recving_.insert(req_id);
    }
    // The staging slot held the received KV for the H2D source; release it now
    // that the copy has finished (success or failure).
    ReleaseSlotLocked(recv_slot);
  });

  return absl::OkStatus();
}

}  // namespace tpu_raiden
