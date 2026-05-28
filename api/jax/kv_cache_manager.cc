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

#include "api/jax/kv_cache_manager.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "api/jax/jax_utils.h"

namespace nb = nanobind;

namespace tpu_raiden {
namespace kv_cache {
namespace jax {

namespace {

// Helper to extract flat matrix of raw PJRT pointers from list of JAX Arrays
// E2E
std::vector<std::vector<xla::PjRtBuffer*>> UnpackPjrtBuffers(
    nb::list device_arrays) {
  size_t num_layers = nb::len(device_arrays);
  if (num_layers == 0) return {};

  size_t num_shards =
      nb::len(nb::cast<nb::list>(device_arrays[0].attr("addressable_shards")));
  std::vector<std::vector<xla::PjRtBuffer*>> layer_buffers;
  layer_buffers.reserve(num_layers);

  for (size_t l = 0; l < num_layers; ++l) {
    nb::object dst = device_arrays[l];
    xla::ifrt::Array* dst_ifrt_array =
        ::jax::GetIfrtArrayFromPyObject(dst.ptr());
    auto* dst_compat_arr = ::jax::CastToPjRtCompatibleArray(dst_ifrt_array);
    if (dst_compat_arr == nullptr) {
      throw std::runtime_error("Not a PjRt compatible array");
    }

    auto dst_buffers = dst_compat_arr->pjrt_buffers();
    if (dst_buffers.size() != num_shards) {
      throw std::runtime_error(
          "Number of shards mismatch across layers during unpack");
    }

    std::vector<xla::PjRtBuffer*> shard_buffers;
    shard_buffers.reserve(num_shards);
    for (size_t i = 0; i < num_shards; ++i) {
      shard_buffers.push_back(dst_buffers[i].get());
    }
    layer_buffers.push_back(std::move(shard_buffers));
  }
  return layer_buffers;
}

// Helper to cast external host addresses cleanly E2E
std::optional<std::vector<const uint8_t*>> CastExternalPointers(
    const std::optional<std::vector<uintptr_t>>& external_host_ptrs) {
  if (!external_host_ptrs.has_value()) return std::nullopt;
  std::vector<const uint8_t*> cast_ptrs;
  cast_ptrs.reserve(external_host_ptrs->size());
  for (uintptr_t addr : *external_host_ptrs) {
    cast_ptrs.push_back(reinterpret_cast<const uint8_t*>(addr));
  }
  return cast_ptrs;
}

}  // namespace

KVCacheManager::KVCacheManager(
    nb::list device_arrays, int block_size, std::optional<int> local_port,
    std::optional<int> host_blocks_to_allocate,
    std::optional<std::vector<uintptr_t>> external_host_ptrs,
    bool unsafe_skip_buffer_lock, int parallelism)
    : KVCacheManagerBase(UnpackPjrtBuffers(device_arrays), block_size,
                         local_port, host_blocks_to_allocate,
                         CastExternalPointers(external_host_ptrs),
                         unsafe_skip_buffer_lock, parallelism),
      device_arrays_(std::move(device_arrays)) {}

KVCacheManager::~KVCacheManager() = default;

}  // namespace jax
}  // namespace kv_cache
}  // namespace tpu_raiden
