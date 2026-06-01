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

#include "frameworks/jax/kv_cache_manager.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "core/utils.h"
#include "frameworks/jax/utils.h"

namespace nb = nanobind;

namespace tpu_raiden {
namespace kv_cache {
namespace jax {

KVCacheManager::KVCacheManager(
    nb::list device_arrays, int block_size, std::optional<int> local_port,
    std::optional<int> host_blocks_to_allocate,
    std::optional<std::vector<uintptr_t>> external_host_ptrs,
    bool unsafe_skip_buffer_lock, int parallelism)
    : KVCacheManager(tpu_raiden::jax::UnpackJaxArrays(device_arrays),
                     block_size, local_port, host_blocks_to_allocate,
                     external_host_ptrs, unsafe_skip_buffer_lock, parallelism,
                     std::move(device_arrays)) {}

KVCacheManager::KVCacheManager(
    std::vector<std::vector<xla::PjRtBuffer*>> layer_buffers, int block_size,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    std::optional<std::vector<uintptr_t>> external_host_ptrs,
    bool unsafe_skip_buffer_lock, int parallelism, nanobind::list device_arrays)
    : KVCacheManagerBase(layer_buffers, block_size, local_port,
                         host_blocks_to_allocate,
                         tpu_raiden::CastExternalPointers(external_host_ptrs),
                         unsafe_skip_buffer_lock, parallelism,
                         tpu_raiden::CreateHostMemoryAllocator(
                             layer_buffers.empty() || layer_buffers[0].empty()
                                 ? nullptr
                                 : layer_buffers[0][0]->device()->client())),
      device_arrays_(std::move(device_arrays)) {}

KVCacheManager::~KVCacheManager() = default;

}  // namespace jax
}  // namespace kv_cache
}  // namespace tpu_raiden
