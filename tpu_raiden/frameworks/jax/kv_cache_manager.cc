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

#include <cstdint>
#include <utility>
#include <vector>

#include "core/utils.h"
#include "tpu_raiden/frameworks/jax/utils.h"

namespace nb = nanobind;

namespace tpu_raiden {
namespace kv_cache {
namespace jax {

namespace {
UnpackedCache UnpackAndMove(nanobind::list device_arrays) {
  auto layer_buffers = tpu_raiden::jax::UnpackJaxArrays(device_arrays);
  return {std::move(layer_buffers), std::move(device_arrays)};
}
}  // namespace

KVCacheManager::KVCacheManager(
    nb::list device_arrays, std::optional<int> local_port,
    std::optional<int> host_blocks_to_allocate,
    bool unsafe_skip_buffer_lock, int parallelism)
    // NOTE: To achieve zero-copy initialization while remaining robust against
    // unspecified C++ function/constructor argument evaluation order (Clang
    // typically evaluates left-to-right, GCC evaluates right-to-left), we
    // enforce sequencing through a helper function `UnpackAndMove`. The
    // function call boundary acts as a strict sequencing barrier, guaranteeing
    // that `UnpackJaxArrays` is fully evaluated before the Python list handle
    // is moved into ownership, preventing use-after-move segfaults.
    : KVCacheManager(UnpackAndMove(std::move(device_arrays)),
                     local_port, host_blocks_to_allocate,
                     unsafe_skip_buffer_lock, parallelism) {}

KVCacheManager::KVCacheManager(
    UnpackedCache&& cache, std::optional<int> local_port,
    std::optional<int> host_blocks_to_allocate,
    bool unsafe_skip_buffer_lock, int parallelism)
    : KVCacheManagerWithTransfer(
          cache.layer_buffers, local_port, host_blocks_to_allocate,
          unsafe_skip_buffer_lock, parallelism,
          tpu_raiden::CreateHostMemoryAllocator(
              cache.layer_buffers.empty() || cache.layer_buffers[0].empty()
                  ? nullptr
                  : cache.layer_buffers[0][0]->device()->client()),
          /*node_id=*/0,
          /*local_control_port=*/-1,
          /*max_blocks=*/0,
          /*num_slots=*/0,
          /*timeout_s=*/120.0),
      device_arrays_(std::move(cache.device_arrays)) {}

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
    : KVCacheManagerWithTransfer(
          cache.layer_buffers,
          /*local_port=*/std::nullopt,
          /*host_blocks_to_allocate=*/std::nullopt,
          unsafe_skip_buffer_lock,
          /*parallelism=*/1,
          tpu_raiden::CreateHostMemoryAllocator(
              cache.layer_buffers.empty() || cache.layer_buffers[0].empty()
                  ? nullptr
                  : cache.layer_buffers[0][0]->device()->client()),
          node_id, local_control_port, max_blocks, num_slots, timeout_s),
      device_arrays_(std::move(cache.device_arrays)) {}

KVCacheManager::KVCacheManager(size_t num_layers, size_t num_shards,
                               size_t slice_byte_size,
                               std::optional<int> local_port,
                               std::optional<int> host_blocks_to_allocate,
                               int parallelism)
    : KVCacheManagerWithTransfer(num_layers, num_shards, slice_byte_size,
                                 local_port,
                                 host_blocks_to_allocate, parallelism) {}

KVCacheManager::~KVCacheManager() = default;

}  // namespace jax
}  // namespace kv_cache
}  // namespace tpu_raiden
