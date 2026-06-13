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

#ifndef THIRD_PARTY_TPU_RAIDEN_FRAMEWORKS_JAX_KV_CACHE_MANAGER_H_
#define THIRD_PARTY_TPU_RAIDEN_FRAMEWORKS_JAX_KV_CACHE_MANAGER_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#ifndef WITHOUT_PYTHON
#include <nanobind/nanobind.h>
#else
namespace nanobind {
struct list {
  list() = default;
  ~list() = default;
  list(const list&) = default;
  list& operator=(const list&) = default;
};
}  // namespace nanobind
#endif
#include "core/kv_cache_manager_with_transfer.h"

namespace xla {
class PjRtBuffer;
}  // namespace xla

namespace tpu_raiden {
namespace kv_cache {
namespace jax {

struct UnpackedCache {
  std::vector<std::vector<xla::PjRtBuffer*>> layer_buffers;
  nanobind::list device_arrays;
};

class KVCacheManager : public KVCacheManagerWithTransfer {
 public:
  // JAX sharded constructor E2E (cache-only by default)
  KVCacheManager(
      nanobind::list device_arrays, int block_size = 1,
      std::optional<int> local_port = std::nullopt,
      std::optional<int> host_blocks_to_allocate = std::nullopt,
      std::optional<std::vector<uintptr_t>> external_host_ptrs = std::nullopt,
      bool unsafe_skip_buffer_lock = false, int parallelism = 1);

  // New transfer-enabled constructor (flat list of arrays, single shard per
  // layer)
  KVCacheManager(nanobind::list kv_caches, int64_t tp_rank,
                 int64_t local_control_port, int64_t max_blocks,
                 int64_t num_slots, double timeout_s,
                 bool unsafe_skip_buffer_lock);

  // FFI metadata constructor (cache-only by default)
  KVCacheManager(size_t num_layers, size_t num_shards, size_t slice_byte_size,
                 int block_size, std::optional<int> local_port,
                 std::optional<int> host_blocks_to_allocate,
                 int parallelism = 1);

  ~KVCacheManager() override;

  nanobind::list kv_caches() const {
    return device_arrays_.value_or(nanobind::list());
  }

 private:
  // Private constructor for sharded (cache-only)
  KVCacheManager(UnpackedCache&& cache, int block_size,
                 std::optional<int> local_port,
                 std::optional<int> host_blocks_to_allocate,
                 std::optional<std::vector<uintptr_t>> external_host_ptrs,
                 bool unsafe_skip_buffer_lock, int parallelism);

  // Private constructor for flat (transfer-enabled)
  KVCacheManager(UnpackedCache&& cache, int64_t tp_rank,
                 int64_t local_control_port, int64_t max_blocks,
                 int64_t num_slots, double timeout_s,
                 bool unsafe_skip_buffer_lock);

  std::optional<nanobind::list> device_arrays_;
};

}  // namespace jax
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_FRAMEWORKS_JAX_KV_CACHE_MANAGER_H_
