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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_FRAMEWORKS_JAX_KV_CACHE_MANAGER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_FRAMEWORKS_JAX_KV_CACHE_MANAGER_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#ifndef WITHOUT_PYTHON
#include <nanobind/nanobind.h>
#endif
#include "absl/status/statusor.h"
#include "tpu_raiden/core/kv_cache_manager_with_transfer.h"
#include "tpu_raiden/core/tpu_utils.h"

namespace xla {
class PjRtBuffer;
}  // namespace xla

namespace tpu_raiden {
namespace kv_cache {
class KVCacheListener;
}  // namespace kv_cache

namespace kv_cache {
namespace jax {

struct UnpackedCache {
  std::vector<std::vector<xla::PjRtBuffer*>> layer_buffers;
#ifndef WITHOUT_PYTHON
  nanobind::list device_arrays;
#endif
};

class KVCacheManager {
 public:
  KVCacheManager(const KVCacheManager&) = delete;
  KVCacheManager& operator=(const KVCacheManager&) = delete;
  KVCacheManager(KVCacheManager&&);
  KVCacheManager& operator=(KVCacheManager&&);

#ifndef WITHOUT_PYTHON
  // JAX sharded constructor E2E (cache-only by default)
  KVCacheManager(nanobind::list device_arrays,
                 std::optional<int> local_port = std::nullopt,
                 std::optional<int> host_blocks_to_allocate = std::nullopt,
                 bool unsafe_skip_buffer_lock = false, int parallelism = 1);

  // New transfer-enabled constructor (flat list of arrays, single shard per
  // layer)
  KVCacheManager(nanobind::list kv_caches, int64_t node_id,
                 int64_t local_control_port, int64_t max_blocks,
                 int64_t num_slots, double timeout_s,
                 bool unsafe_skip_buffer_lock, int parallelism);
#endif

  // FFI metadata constructor (cache-only by default)
  KVCacheManager(size_t num_layers, size_t num_shards, size_t slice_byte_size,
                 std::optional<int> local_port,
                 std::optional<int> host_blocks_to_allocate,
                 int parallelism = 1);

  // Test-only constructor for sub-manager mock injection
  explicit KVCacheManager(
      std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> sub_managers);

  ~KVCacheManager();

#ifndef WITHOUT_PYTHON
  nanobind::list kv_caches() const {
    return device_arrays_.value_or(nanobind::list());
  }
#endif

  // Forwarding methods delegating to sub_managers_
  size_t num_layers() const;
  size_t num_shards() const;
  size_t slice_byte_size() const;

  std::optional<int> local_port() const;
  int local_control_port() const;
  int64_t node_id() const;
  std::vector<std::string> listener_addresses() const;

  uint8_t* GetHostPointer(size_t layer_idx, size_t shard_idx);
  const uint8_t* GetHostPointer(size_t layer_idx, size_t shard_idx) const;
  size_t GetHostSize(size_t layer_idx, size_t shard_idx);

  int64_t NotifyForRead(const std::string& req_id, uint64_t uuid,
                        const std::vector<int64_t>& block_ids);

  void StartRead(
      const std::string& req_id, uint64_t uuid,
      const std::string& remote_endpoint,
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids, int parallelism = 1,
      std::optional<std::vector<int64_t>> local_host_block_ids = std::nullopt);

  std::vector<EndpointDescriptor> get_local_endpoints() const;

  void SetSubmanagerShardsForTesting(
      const std::vector<std::vector<int64_t>>& assignment) {
    submanager_to_global_shards_ = assignment;
  }

  void StartRead(
      const std::string& req_id, uint64_t uuid,
      const std::vector<EndpointDescriptor>& remote_descriptors,
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids, int parallelism = 1,
      std::optional<std::vector<int64_t>> local_host_block_ids = std::nullopt);

  std::tuple<std::vector<std::string>, std::vector<std::string>,
             std::vector<std::string>>
  CompleteReadRaw();

  absl::StatusOr<raiden::PjRtCopyFuture> H2d(
      const std::vector<int64_t>& src_offsets_major_dim = {},
      const std::vector<int64_t>& dst_offsets_major_dim = {},
      const std::vector<int64_t>& copy_sizes_major_dim = {},
      std::optional<int64_t> slot_idx = std::nullopt,
      std::optional<size_t> layer_idx = std::nullopt,
      std::optional<size_t> shard_idx = std::nullopt);

  absl::StatusOr<raiden::PjRtCopyFuture> D2h(
      const std::vector<int64_t>& src_offsets_major_dim = {},
      const std::vector<int64_t>& dst_offsets_major_dim = {},
      const std::vector<int64_t>& copy_sizes_major_dim = {},
      std::optional<int64_t> slot_idx = std::nullopt,
      std::optional<size_t> layer_idx = std::nullopt,
      std::optional<size_t> shard_idx = std::nullopt);

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
  D2hAutoAllocate(const std::vector<int64_t>& src_offsets_major_dim = {},
                  const std::vector<int64_t>& copy_sizes_major_dim = {});

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hWrite(
      std::string peer, const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids = {}, uint64_t uuid = 0);

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hWrite(
      const std::vector<EndpointDescriptor>& remote_descriptors,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids = {}, uint64_t uuid = 0);

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hRead(
      std::string peer, const std::vector<int>& src_block_ids);

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hRead(
      const std::vector<EndpointDescriptor>& remote_descriptors,
      const std::vector<int>& src_block_ids);

 private:
#ifndef WITHOUT_PYTHON
  // Private constructor for sharded (cache-only)
  KVCacheManager(UnpackedCache&& cache, std::optional<int> local_port,
                 std::optional<int> host_blocks_to_allocate,
                 bool unsafe_skip_buffer_lock, int parallelism);

  // Private constructor for flat (transfer-enabled)
  KVCacheManager(UnpackedCache&& cache, int64_t node_id,
                 int64_t local_control_port, int64_t max_blocks,
                 int64_t num_slots, double timeout_s,
                 bool unsafe_skip_buffer_lock, int parallelism);

  std::optional<nanobind::list> device_arrays_;
#endif

  void InitSubManagers(
      const std::vector<std::vector<xla::PjRtBuffer*>>& layer_buffers,
      std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
      bool unsafe_skip_buffer_lock, int parallelism, int64_t node_id,
      int64_t local_control_port, int64_t max_blocks, int64_t num_slots,
      double timeout_s);

  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> sub_managers_;
  std::vector<std::pair<int, int>> global_shard_to_submanager_;
  std::vector<std::vector<int64_t>> submanager_to_global_shards_;
  size_t total_num_shards_ = 0;

  std::map<std::string, int> done_sending_counts_;
  std::map<std::string, int> done_recving_counts_;
  std::set<std::string> failed_recving_set_;
  // We maintain a KVCacheListener for each sub-manager. Since sub-managers are
  // partitioned by NUMA node to optimize memory bandwidth, we can have multiple
  // listeners per host (one for each NUMA node/sub-manager).
  std::vector<std::unique_ptr<tpu_raiden::kv_cache::KVCacheListener>>
      listeners_;
};

}  // namespace jax
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_FRAMEWORKS_JAX_KV_CACHE_MANAGER_H_
