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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "xla/pjrt/pjrt_client.h"
#include "core/kv_cache_manager_with_transfer.h"
#include "tpu_raiden/frameworks/jax/kv_cache_manager.h"

namespace tpu_raiden {
namespace kv_cache {
namespace jax {

// Stub-implement the Python nb::list constructors so we don't require FFI utils
// or nanobind linking
KVCacheManager::KVCacheManager(
    nanobind::list device_arrays, std::optional<int> local_port,
    std::optional<int> host_blocks_to_allocate,
    std::optional<std::vector<uintptr_t>> external_host_ptrs,
    bool unsafe_skip_buffer_lock, int parallelism)
    : KVCacheManagerWithTransfer({}, local_port,
                                 host_blocks_to_allocate, std::nullopt,
                                 unsafe_skip_buffer_lock, parallelism,
                                 /*host_allocator=*/nullptr, /*tp_rank=*/0,
                                 /*local_control_port=*/-1, /*max_blocks=*/0,
                                 /*num_slots=*/0, /*timeout_s=*/120.0) {
  throw std::runtime_error(
      "Python KVCacheManager constructor is unsupported in pure C++ FFI unit "
      "tests.");
}

KVCacheManager::KVCacheManager(
    UnpackedCache&& cache, std::optional<int> local_port,
    std::optional<int> host_blocks_to_allocate,
    std::optional<std::vector<uintptr_t>> external_host_ptrs,
    bool unsafe_skip_buffer_lock, int parallelism)
    : KVCacheManagerWithTransfer(cache.layer_buffers, local_port,
                                 host_blocks_to_allocate, std::nullopt,
                                 unsafe_skip_buffer_lock, parallelism,
                                 /*host_allocator=*/nullptr, /*tp_rank=*/0,
                                 /*local_control_port=*/-1, /*max_blocks=*/0,
                                 /*num_slots=*/0, /*timeout_s=*/120.0) {
  throw std::runtime_error(
      "Python KVCacheManager constructor is unsupported in pure C++ FFI unit "
      "tests.");
}

KVCacheManager::KVCacheManager(nanobind::list kv_caches, int64_t tp_rank,
                               int64_t local_control_port, int64_t max_blocks,
                               int64_t num_slots, double timeout_s,
                               bool unsafe_skip_buffer_lock)
    : KVCacheManagerWithTransfer(
          {}, std::nullopt, std::nullopt, std::nullopt,
          unsafe_skip_buffer_lock, /*parallelism=*/1,
          /*host_allocator=*/nullptr, tp_rank, local_control_port, max_blocks,
          num_slots, timeout_s) {
  throw std::runtime_error(
      "Python KVCacheManager constructor is unsupported in pure C++ FFI unit "
      "tests.");
}

KVCacheManager::KVCacheManager(UnpackedCache&& cache, int64_t tp_rank,
                               int64_t local_control_port, int64_t max_blocks,
                               int64_t num_slots, double timeout_s,
                               bool unsafe_skip_buffer_lock)
    : KVCacheManagerWithTransfer(
          cache.layer_buffers, std::nullopt, std::nullopt,
          std::nullopt, unsafe_skip_buffer_lock,
          /*parallelism=*/1, /*host_allocator=*/nullptr, tp_rank,
          local_control_port, max_blocks, num_slots, timeout_s) {
  throw std::runtime_error(
      "Python KVCacheManager constructor is unsupported in pure C++ FFI unit "
      "tests.");
}

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
