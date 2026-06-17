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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_FRAMEWORKS_JAX_KV_CACHE_MANAGER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_FRAMEWORKS_JAX_KV_CACHE_MANAGER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#ifndef WITHOUT_PYTHON
#include <nanobind/nanobind.h>
#endif
#include "absl/status/statusor.h"
#include "core/kv_cache_manager_with_transfer.h"
#include "core/raiden_future.h"

namespace xla {
class PjRtBuffer;
}  // namespace xla

namespace tpu_raiden {
namespace kv_cache {
namespace jax {

struct UnpackedCache {
  std::vector<std::vector<xla::PjRtBuffer*>> layer_buffers;
#ifndef WITHOUT_PYTHON
  nanobind::list device_arrays;
#endif
};

class __attribute__((visibility("default"))) KVCacheManager {
 public:
#ifndef WITHOUT_PYTHON
  // JAX sharded constructor E2E (cache-only by default)
  KVCacheManager(
      nanobind::list device_arrays,
      std::optional<int> local_port = std::nullopt,
      std::optional<int> host_blocks_to_allocate = std::nullopt,
      bool unsafe_skip_buffer_lock = false, int parallelism = 1);

  // Transfer-enabled constructor (flat list of arrays, single shard per layer)
  KVCacheManager(nanobind::list kv_caches, int64_t node_id,
                 int64_t local_control_port, int64_t max_blocks,
                 int64_t num_slots, double timeout_s,
                 bool unsafe_skip_buffer_lock);
#endif

  ~KVCacheManager();

  // Metadata getters for Python
  size_t num_layers() const { return num_layers_; }
  size_t num_shards() const { return num_shards_; }
  size_t slice_byte_size() const { return slice_byte_size_; }
  int64_t local_control_port() const;
  void Close();

  const uint8_t* GetHostPointer(size_t layer_idx, size_t shard_idx) const;

#ifndef WITHOUT_PYTHON
  nanobind::list kv_caches() const {
    return device_arrays_.value_or(nanobind::list());
  }

  // Python/FFI delegation methods
  absl::StatusOr<raiden::PjRtCopyFuture> StartRead(
      const std::string& req_id, uint64_t uuid,
      const std::string& remote_endpoint,
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids, int parallelism,
      std::optional<std::vector<int64_t>> local_host_block_ids = std::nullopt);

  bool NotifyForRead(const std::string& req_id, uint64_t uuid,
                     const std::vector<int64_t>& block_ids);

  std::tuple<std::vector<std::string>, std::vector<std::string>,
             std::vector<std::string>>
  CompleteReadRaw();

  // Resolved endpoints for Python
  std::vector<std::string> local_ips() const;
  std::vector<int> local_ports() const;
  std::vector<int> local_control_ports() const;

  // Python H2d/D2h
  absl::StatusOr<tpu_raiden::RaidenFuture> H2d(
      const std::vector<int64_t>& src_offsets = {},
      const std::vector<int64_t>& dst_offsets = {},
      const std::vector<int64_t>& copy_sizes = {});

  absl::StatusOr<tpu_raiden::RaidenFuture> D2h(
      const std::vector<int64_t>& src_offsets = {},
      const std::vector<int64_t>& dst_offsets = {},
      const std::vector<int64_t>& copy_sizes = {});

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
  D2hAutoAllocate(const std::vector<int64_t>& src_offsets = {},
                  const std::vector<int64_t>& copy_sizes = {});

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hWrite(
      std::string peer, const std::vector<int>& src_block_ids);

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hRead(
      std::string peer, const std::vector<int>& src_block_ids);
#endif

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
                 bool unsafe_skip_buffer_lock);

  std::optional<nanobind::list> device_arrays_;
  std::vector<std::shared_ptr<KVCacheManagerWithTransfer>> sub_managers_;

  KVCacheManagerWithTransfer* GetSingleActiveSubManager() const;

  struct ShardMapping {
    size_t sub_manager_idx;
    size_t local_shard_idx;
  };
  std::vector<ShardMapping> shard_mappings_;
  std::vector<int64_t> FilterBlockIdsForSubManager(
      const std::vector<int64_t>& block_ids, size_t sub_manager_idx) const;
#endif

  size_t num_layers_ = 0;
  size_t num_shards_ = 0;
  size_t slice_byte_size_ = 0;
};

}  // namespace jax
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_FRAMEWORKS_JAX_KV_CACHE_MANAGER_H_
