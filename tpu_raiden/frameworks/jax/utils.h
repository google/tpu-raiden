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

/* Copyright 2026 The TPU Raiden Authors. All Rights Reserved.
==============================================================================*/

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_FRAMEWORKS_JAX_UTILS_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_FRAMEWORKS_JAX_UTILS_H_

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include "xla/pjrt/pjrt_client.h"
#include "tpu_raiden/frameworks/jax/jax_utils.h"

namespace tpu_raiden {
namespace jax {

inline std::vector<std::vector<raiden::RaidenBufferHandle>> UnpackJaxArrays(
    const nanobind::list& jax_arrays, bool unsafe_skip_buffer_lock = false) {
  size_t num_layers = nanobind::len(jax_arrays);
  if (num_layers == 0) return {};

  size_t num_shards = nanobind::len(
      nanobind::cast<nanobind::list>(jax_arrays[0].attr("addressable_shards")));
  std::vector<std::vector<raiden::RaidenBufferHandle>> layer_buffers;

  layer_buffers.reserve(num_layers);

  for (size_t l = 0; l < num_layers; ++l) {
    nanobind::object dst = nanobind::cast<nanobind::object>(jax_arrays[l]);
    std::vector<raiden::RaidenBufferHandle> shard_buffers =
        ::jax::ExtractPjRtBuffersFromPyArray(dst, unsafe_skip_buffer_lock);

    if (shard_buffers.size() != num_shards) {
      throw std::runtime_error(
          "Number of shards mismatch across layers during unpack");
    }

    try {
      nanobind::object utils_mod = nanobind::module_::import_(
          "tpu_raiden.api.jax.utils");
      nanobind::object get_permutation_fn =
          utils_mod.attr("get_shard_sorting_permutation");
      nanobind::list permutation =
          nanobind::cast<nanobind::list>(get_permutation_fn(dst));

      std::vector<raiden::RaidenBufferHandle> sorted_shard_buffers;
      sorted_shard_buffers.reserve(num_shards);
      for (size_t i = 0; i < num_shards; ++i) {
        size_t src_idx = nanobind::cast<size_t>(permutation[i]);
        if (src_idx >= shard_buffers.size()) {
          throw std::runtime_error("Permutation index out of bounds");
        }
        sorted_shard_buffers.push_back(std::move(shard_buffers[src_idx]));
      }
      shard_buffers = std::move(sorted_shard_buffers);
    } catch (const std::exception& e) {
      throw std::runtime_error(std::string("Failed to sort shards: ") +
                               e.what());
    }

    layer_buffers.push_back(std::move(shard_buffers));
  }
  return layer_buffers;
}

}  // namespace jax
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_FRAMEWORKS_JAX_UTILS_H_
