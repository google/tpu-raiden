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

#include "tpu_raiden/frameworks/jax/kv_cache_manager.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include <nanobind/nanobind.h>
#include "core/kv_cache_manager_with_transfer.h"
#include "core/raiden_future.h"
#include "core/raw_transfer_core.h"
#include "core/utils.h"
#include "tpu_raiden/frameworks/jax/utils.h"

namespace tpu_raiden {
namespace kv_cache {
namespace jax {

namespace {
UnpackedCache UnpackAndMove(nanobind::list device_arrays) {
  auto layer_buffers = tpu_raiden::jax::UnpackJaxArrays(device_arrays);
  return {std::move(layer_buffers), std::move(device_arrays)};
}
}  // namespace

KVCacheManager::KVCacheManager(nanobind::list device_arrays,
                               std::optional<int> local_port,
                               std::optional<int> host_blocks_to_allocate,
                               bool unsafe_skip_buffer_lock, int parallelism)
    : KVCacheManager(UnpackAndMove(std::move(device_arrays)), local_port,
                     host_blocks_to_allocate, unsafe_skip_buffer_lock,
                     parallelism) {}

KVCacheManager::KVCacheManager(UnpackedCache&& cache,
                               std::optional<int> local_port,
                               std::optional<int> host_blocks_to_allocate,
                               bool unsafe_skip_buffer_lock, int parallelism)
    : device_arrays_(std::move(cache.device_arrays)) {
  num_layers_ = cache.layer_buffers.size();
  num_shards_ = num_layers_ > 0 ? cache.layer_buffers[0].size() : 0;

  slice_byte_size_ = 0;
  if (num_layers_ > 0 && num_shards_ > 0) {
    auto size_or = cache.layer_buffers[0][0]->GetOnDeviceSizeInBytes();
    if (size_or.ok()) {
      slice_byte_size_ = *size_or;
    } else {
      LOG(ERROR)
          << "[Raiden JAX Composite] Failed to get device size for metadata: "
          << size_or.status();
    }
  }

  // Group buffers by NUMA
  std::vector<std::vector<std::vector<xla::PjRtBuffer*>>> numa_buffers(2);
  numa_buffers[0].resize(num_layers_);
  numa_buffers[1].resize(num_layers_);

  shard_mappings_.resize(num_shards_);
  std::vector<size_t> local_shard_counts(2, 0);

  for (size_t s = 0; s < num_shards_; ++s) {
    xla::PjRtBuffer* buf = cache.layer_buffers[0][s];
    int numa = tpu_raiden::GetPjRtDeviceNumaNode(buf->device());
    if (numa < 0 || numa > 1) {
      LOG(WARNING) << "[Raiden JAX Composite] Unknown NUMA node " << numa
                   << " for shard " << s << ". Defaulting to 0.";
      numa = 0;
    }

    size_t local_idx = local_shard_counts[numa]++;
    shard_mappings_[s] = {static_cast<size_t>(numa), local_idx};

    for (size_t l = 0; l < num_layers_; ++l) {
      numa_buffers[numa][l].push_back(cache.layer_buffers[l][s]);
    }
  }

  // Instantiate sub-managers
  sub_managers_.resize(2);
  for (int numa = 0; numa < 2; ++numa) {
    if (local_shard_counts[numa] > 0) {
      std::optional<int> port = local_port;
      if (port.has_value() && port.value() > 0 && numa > 0) {
        port = port.value() + numa;
      }

      auto host_allocator = tpu_raiden::CreateHostMemoryAllocator(
          numa_buffers[numa][0][0]->device()->client());

      sub_managers_[numa] = std::make_shared<KVCacheManagerWithTransfer>(
          numa_buffers[numa], port, host_blocks_to_allocate,
          unsafe_skip_buffer_lock, parallelism, std::move(host_allocator));
      LOG(INFO)
          << "[Raiden JAX Composite] Created sub-manager (sharded) for NUMA "
          << numa << " managing " << local_shard_counts[numa] << " shards.";
    }
  }
}

KVCacheManager::KVCacheManager(nanobind::list kv_caches, int64_t node_id,
                               int64_t local_control_port, int64_t max_blocks,
                               int64_t num_slots, double timeout_s,
                               bool unsafe_skip_buffer_lock)
    : KVCacheManager(UnpackAndMove(std::move(kv_caches)), node_id,
                     local_control_port, max_blocks, num_slots, timeout_s,
                     unsafe_skip_buffer_lock) {}

KVCacheManager::KVCacheManager(UnpackedCache&& cache, int64_t node_id,
                               int64_t local_control_port, int64_t max_blocks,
                               int64_t num_slots, double timeout_s,
                               bool unsafe_skip_buffer_lock)
    : device_arrays_(std::move(cache.device_arrays)) {
  num_layers_ = cache.layer_buffers.size();
  num_shards_ = num_layers_ > 0 ? cache.layer_buffers[0].size() : 0;

  slice_byte_size_ = 0;
  if (num_layers_ > 0 && num_shards_ > 0) {
    auto size_or = cache.layer_buffers[0][0]->GetOnDeviceSizeInBytes();
    if (size_or.ok()) {
      slice_byte_size_ = *size_or;
    } else {
      LOG(ERROR)
          << "[Raiden JAX Composite] Failed to get device size for metadata: "
          << size_or.status();
    }
  }

  // Group buffers by NUMA
  std::vector<std::vector<std::vector<xla::PjRtBuffer*>>> numa_buffers(2);
  numa_buffers[0].resize(num_layers_);
  numa_buffers[1].resize(num_layers_);

  shard_mappings_.resize(num_shards_);
  std::vector<size_t> local_shard_counts(2, 0);

  for (size_t s = 0; s < num_shards_; ++s) {
    xla::PjRtBuffer* buf = cache.layer_buffers[0][s];
    int numa = tpu_raiden::GetPjRtDeviceNumaNode(buf->device());
    if (numa < 0 || numa > 1) {
      LOG(WARNING) << "[Raiden JAX Composite] Unknown NUMA node " << numa
                   << " for shard " << s << ". Defaulting to 0.";
      numa = 0;
    }

    size_t local_idx = local_shard_counts[numa]++;
    shard_mappings_[s] = {static_cast<size_t>(numa), local_idx};

    for (size_t l = 0; l < num_layers_; ++l) {
      numa_buffers[numa][l].push_back(cache.layer_buffers[l][s]);
    }
  }

  // Instantiate sub-managers
  sub_managers_.resize(2);
  for (int numa = 0; numa < 2; ++numa) {
    if (local_shard_counts[numa] > 0) {
      int64_t port = (local_control_port > 0) ? local_control_port + numa
                                              : local_control_port;

      auto host_allocator = tpu_raiden::CreateHostMemoryAllocator(
          numa_buffers[numa][0][0]->device()->client());

      sub_managers_[numa] = std::make_shared<KVCacheManagerWithTransfer>(
          numa_buffers[numa],
          /*local_port=*/std::nullopt,
          /*host_blocks_to_allocate=*/std::nullopt, unsafe_skip_buffer_lock,
          /*parallelism=*/1, std::move(host_allocator), node_id, port,
          max_blocks, num_slots, timeout_s);
      LOG(INFO) << "[Raiden JAX Composite] Created sub-manager (flat) for NUMA "
                << numa << " on port " << port << " managing "
                << local_shard_counts[numa] << " shards.";
    }
  }
}

KVCacheManager::~KVCacheManager() = default;

const uint8_t* KVCacheManager::GetHostPointer(size_t layer_idx,
                                              size_t shard_idx) const {
  const auto& mapping = shard_mappings_[shard_idx];
  if (sub_managers_[mapping.sub_manager_idx] != nullptr) {
    return sub_managers_[mapping.sub_manager_idx]->GetHostPointer(
        layer_idx, mapping.local_shard_idx);
  }
  LOG(FATAL) << "Sub-manager for NUMA " << mapping.sub_manager_idx
             << " is null during GetHostPointer for global shard " << shard_idx;
}

absl::StatusOr<raiden::PjRtCopyFuture> KVCacheManager::StartRead(
    const std::string& req_id, uint64_t uuid,
    const std::string& remote_endpoint,
    const std::vector<int64_t>& remote_block_ids,
    const std::vector<int64_t>& local_block_ids, int parallelism,
    std::optional<std::vector<int64_t>> local_host_block_ids) {
  std::vector<std::string> endpoints;
  size_t pos = 0;
  std::string s = remote_endpoint;
  while ((pos = s.find(",")) != std::string::npos) {
    endpoints.push_back(s.substr(0, pos));
    s.erase(0, pos + 1);
  }
  endpoints.push_back(s);

  std::vector<raiden::PjRtCopyFuture> sub_futures;
  for (size_t i = 0; i < sub_managers_.size(); ++i) {
    if (sub_managers_[i] == nullptr) continue;
    std::string ep = endpoints[0];
    if (i < endpoints.size()) {
      ep = endpoints[i];
    } else {
      LOG(WARNING)
          << "[Raiden JAX Composite] Fewer endpoints than sub-managers. "
          << "Bottlenecking NUMA " << i << " on " << ep;
    }

    std::vector<int64_t> filtered_remote =
        FilterBlockIdsForSubManager(remote_block_ids, i);
    std::vector<int64_t> filtered_local =
        FilterBlockIdsForSubManager(local_block_ids, i);
    std::optional<std::vector<int64_t>> filtered_host = std::nullopt;
    if (local_host_block_ids.has_value()) {
      filtered_host = FilterBlockIdsForSubManager(*local_host_block_ids, i);
    }

    auto res =
        sub_managers_[i]->StartRead(req_id, uuid, ep, filtered_remote,
                                    filtered_local, parallelism, filtered_host);
    if (!res.ok()) {
      return res.status();
    }
    sub_futures.push_back(std::move(res.value()));
  }

  return raiden::JoinPjRtCopyFutures(sub_futures);
}

bool KVCacheManager::NotifyForRead(const std::string& req_id, uint64_t uuid,
                                   const std::vector<int64_t>& block_ids) {
  bool result = false;
  for (size_t i = 0; i < sub_managers_.size(); ++i) {
    if (sub_managers_[i] != nullptr) {
      std::vector<int64_t> filtered = FilterBlockIdsForSubManager(block_ids, i);
      result |= sub_managers_[i]->NotifyForRead(req_id, uuid, filtered);
    }
  }
  return result;
}

std::tuple<std::vector<std::string>, std::vector<std::string>,
           std::vector<std::string>>
KVCacheManager::CompleteReadRaw() {
  std::vector<std::string> done_sending;
  std::vector<std::string> done_recving;
  std::vector<std::string> failed_recving;

  for (auto& mgr : sub_managers_) {
    if (mgr != nullptr) {
      auto [ds, dr, fr] = mgr->CompleteReadRaw();
      done_sending.insert(done_sending.end(), ds.begin(), ds.end());
      done_recving.insert(done_recving.end(), dr.begin(), dr.end());
      failed_recving.insert(failed_recving.end(), fr.begin(), fr.end());
    }
  }
  return {done_sending, done_recving, failed_recving};
}

std::vector<std::string> KVCacheManager::local_ips() const {
  std::vector<std::string> ips;
  for (const auto& mgr : sub_managers_) {
    if (mgr != nullptr) {
      ips.push_back(mgr->local_ip());
    }
  }
  return ips;
}

std::vector<int> KVCacheManager::local_ports() const {
  std::vector<int> ports;
  for (const auto& mgr : sub_managers_) {
    if (mgr != nullptr) {
      ports.push_back(mgr->local_port().value_or(-1));
    }
  }
  return ports;
}

std::vector<int> KVCacheManager::local_control_ports() const {
  std::vector<int> ports;
  for (const auto& mgr : sub_managers_) {
    if (mgr != nullptr) {
      ports.push_back(mgr->local_control_port());
    }
  }
  return ports;
}

absl::StatusOr<tpu_raiden::RaidenFuture> KVCacheManager::H2d(
    const std::vector<int64_t>& src_offsets,
    const std::vector<int64_t>& dst_offsets,
    const std::vector<int64_t>& copy_sizes) {
  std::vector<raiden::PjRtCopyFuture> sub_futures;
  for (auto& mgr : sub_managers_) {
    if (mgr != nullptr) {
      auto res = mgr->H2d(src_offsets, dst_offsets, copy_sizes);
      if (!res.ok()) return res.status();
      sub_futures.push_back(std::move(res.value()));
    }
  }
  auto joined = raiden::JoinPjRtCopyFutures(sub_futures);
  return tpu_raiden::RaidenFuture{joined};
}

absl::StatusOr<tpu_raiden::RaidenFuture> KVCacheManager::D2h(
    const std::vector<int64_t>& src_offsets,
    const std::vector<int64_t>& dst_offsets,
    const std::vector<int64_t>& copy_sizes) {
  std::vector<raiden::PjRtCopyFuture> sub_futures;
  for (auto& mgr : sub_managers_) {
    if (mgr != nullptr) {
      auto res = mgr->D2h(src_offsets, dst_offsets, copy_sizes);
      if (!res.ok()) return res.status();
      sub_futures.push_back(std::move(res.value()));
    }
  }
  auto joined = raiden::JoinPjRtCopyFutures(sub_futures);
  return tpu_raiden::RaidenFuture{joined};
}

absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
KVCacheManager::D2hAutoAllocate(const std::vector<int64_t>& src_offsets,
                                const std::vector<int64_t>& copy_sizes) {
  std::vector<raiden::PjRtCopyFuture> sub_futures;
  std::vector<int> allocated_ids;
  bool first = true;
  for (auto& mgr : sub_managers_) {
    if (mgr != nullptr) {
      auto res = mgr->D2hAutoAllocate(src_offsets, copy_sizes);
      if (!res.ok()) return res.status();
      if (first) {
        allocated_ids = res.value().first;
        first = false;
      } else {
        if (allocated_ids != res.value().first) {
          return absl::InternalError(
              "[Raiden JAX Composite] NUMA sub-managers allocated different "
              "block IDs in D2hAutoAllocate.");
        }
      }
      sub_futures.push_back(std::move(res.value().second));
    }
  }
  auto joined = raiden::JoinPjRtCopyFutures(sub_futures);
  return std::make_pair(allocated_ids, joined);
}

absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
KVCacheManager::H2hWrite(std::string peer,
                         const std::vector<int>& src_block_ids) {
  if (auto* mgr = GetSingleActiveSubManager()) {
    return mgr->H2hWrite(peer, src_block_ids);
  }
  return absl::UnimplementedError(
      "H2hWrite is not supported with multiple active NUMA sub-managers.");
}

absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
KVCacheManager::H2hRead(std::string peer,
                        const std::vector<int>& src_block_ids) {
  if (auto* mgr = GetSingleActiveSubManager()) {
    return mgr->H2hRead(peer, src_block_ids);
  }
  return absl::UnimplementedError(
      "H2hRead is not supported with multiple active NUMA sub-managers.");
}

int64_t KVCacheManager::local_control_port() const {
  for (const auto& mgr : sub_managers_) {
    if (mgr != nullptr) {
      return mgr->local_control_port();
    }
  }
  return -1;
}

void KVCacheManager::Close() { sub_managers_.clear(); }

KVCacheManagerWithTransfer* KVCacheManager::GetSingleActiveSubManager() const {
  KVCacheManagerWithTransfer* active = nullptr;
  for (const auto& mgr : sub_managers_) {
    if (mgr != nullptr) {
      if (active != nullptr) {
        return nullptr;  // More than one active manager
      }
      active = mgr.get();
    }
  }
  return active;
}

std::vector<int64_t> KVCacheManager::FilterBlockIdsForSubManager(
    const std::vector<int64_t>& block_ids, size_t sub_manager_idx) const {
  if (block_ids.empty() || num_shards_ == 0) {
    return block_ids;
  }
  // If not sharded, pass as-is.
  if (block_ids.size() % num_shards_ != 0) {
    return block_ids;
  }

  size_t B = block_ids.size() / num_shards_;
  size_t local_S = sub_managers_[sub_manager_idx]->num_shards();
  std::vector<int64_t> filtered(B * local_S);

  std::vector<size_t> local_to_global(local_S);
  for (size_t s = 0; s < num_shards_; ++s) {
    if (shard_mappings_[s].sub_manager_idx == sub_manager_idx) {
      local_to_global[shard_mappings_[s].local_shard_idx] = s;
    }
  }

  for (size_t i = 0; i < B; ++i) {
    for (size_t local_s = 0; local_s < local_S; ++local_s) {
      size_t global_s = local_to_global[local_s];
      filtered[i * local_S + local_s] = block_ids[i * num_shards_ + global_s];
    }
  }
  return filtered;
}

}  // namespace jax
}  // namespace kv_cache
}  // namespace tpu_raiden
