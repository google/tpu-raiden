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
#include "xla/python/ifrt/array.h"
#include "tpu_raiden/frameworks/jax/jax_utils.h"

namespace tpu_raiden {
namespace jax {

inline std::vector<std::vector<xla::PjRtBuffer*>> UnpackJaxArrays(
    const nanobind::list& jax_arrays) {
  size_t num_layers = nanobind::len(jax_arrays);
  if (num_layers == 0) return {};

  size_t num_shards = nanobind::len(
      nanobind::cast<nanobind::list>(jax_arrays[0].attr("addressable_shards")));
  std::vector<std::vector<xla::PjRtBuffer*>> layer_buffers;
  layer_buffers.reserve(num_layers);

  for (size_t l = 0; l < num_layers; ++l) {
    nanobind::handle dst = jax_arrays[l];
    xla::ifrt::Array* dst_ifrt_array =
        ::jax::GetIfrtArrayFromPyObject(dst.ptr());
    if (dst_ifrt_array == nullptr) {
      throw std::runtime_error("Failed to extract JAX IFRT Array pointer");
    }
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

}  // namespace jax
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_FRAMEWORKS_JAX_UTILS_H_
