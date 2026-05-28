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

#include "api/jax/weight_synchronizer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include <nanobind/nanobind.h>
#include "api/jax/jax_utils.h"

namespace nb = nanobind;

namespace tpu_raiden {
namespace jax {

namespace {

// Unpacks JAX python array lists into flat PJRT buffers matrix E2E!
std::vector<std::vector<xla::PjRtBuffer*>> UnpackJaxWeights(
    const nb::list& jax_arrays) {
  size_t num_layers = nb::len(jax_arrays);
  if (num_layers == 0) return {};

  size_t num_shards =
      nb::len(nb::cast<nb::list>(jax_arrays[0].attr("addressable_shards")));
  std::vector<std::vector<xla::PjRtBuffer*>> layer_buffers;
  layer_buffers.reserve(num_layers);

  for (size_t l = 0; l < num_layers; ++l) {
    nb::object dst = jax_arrays[l];
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

}  // namespace

WeightSynchronizer::WeightSynchronizer(const nb::list& jax_arrays,
                                       std::optional<int> local_port,
                                       int parallelism,
                                       bool unsafe_skip_buffer_lock)
    : weight_sync::WeightSynchronizerBase(
          UnpackJaxWeights(jax_arrays), local_port,
          /*external_host_ptrs=*/std::nullopt, unsafe_skip_buffer_lock,
          parallelism) {}

WeightSynchronizer::~WeightSynchronizer() = default;

}  // namespace jax
}  // namespace tpu_raiden
