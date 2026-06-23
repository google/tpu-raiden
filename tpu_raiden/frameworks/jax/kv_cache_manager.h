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
#include <optional>
#include <vector>

#ifndef WITHOUT_PYTHON
#include <nanobind/nanobind.h>
#endif
#include "tpu_raiden/core/kv_cache_manager_with_transfer.h"

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

class KVCacheManager : public KVCacheManagerWithTransfer {
 public:
#ifndef WITHOUT_PYTHON
  // JAX sharded constructor E2E (cache-only by default)
  KVCacheManager(
      nanobind::list device_arrays,
      std::optional<int> local_port = std::nullopt,
      std::optional<int> host_blocks_to_allocate = std::nullopt,
      bool unsafe_skip_buffer_lock = false, int parallelism = 1);

  // New transfer-enabled constructor (flat list of arrays, single shard per
  // layer)
  KVCacheManager(nanobind::list kv_caches, int64_t node_id,
                 int64_t local_control_port, int64_t max_blocks,
                 int64_t num_slots, double timeout_s,
                 bool unsafe_skip_buffer_lock);
#endif

  // FFI metadata constructor (cache-only by default)
  KVCacheManager(size_t num_layers, size_t num_shards, size_t slice_byte_size,
                 std::optional<int> local_port,
                 std::optional<int> host_blocks_to_allocate,
                 int parallelism = 1);

  ~KVCacheManager() override;

#ifndef WITHOUT_PYTHON
  nanobind::list kv_caches() const {
    return device_arrays_.value_or(nanobind::list());
  }
#endif

 private:
#ifndef WITHOUT_PYTHON
  // Private constructor for sharded (cache-only)
  KVCacheManager(UnpackedCache&& cache,
                 std::optional<int> local_port,
                 std::optional<int> host_blocks_to_allocate,
                 bool unsafe_skip_buffer_lock, int parallelism);

  // Private constructor for flat (transfer-enabled)
  KVCacheManager(UnpackedCache&& cache, int64_t node_id,
                 int64_t local_control_port, int64_t max_blocks,
                 int64_t num_slots, double timeout_s,
                 bool unsafe_skip_buffer_lock);

  std::optional<nanobind::list> device_arrays_;
#endif
};

}  // namespace jax
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_FRAMEWORKS_JAX_KV_CACHE_MANAGER_H_
