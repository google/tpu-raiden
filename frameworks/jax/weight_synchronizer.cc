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

#include "frameworks/jax/weight_synchronizer.h"

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
#include "frameworks/jax/jax_utils.h"
#include "frameworks/jax/utils.h"

namespace nb = nanobind;

namespace tpu_raiden {
namespace jax {

WeightSynchronizer::WeightSynchronizer(const nb::list& jax_arrays,
                                       std::optional<int> local_port,
                                       int parallelism,
                                       bool unsafe_skip_buffer_lock)
    : weight_sync::WeightSynchronizerBase(
          tpu_raiden::jax::UnpackJaxArrays(jax_arrays), local_port,
          /*external_host_ptrs=*/std::nullopt, unsafe_skip_buffer_lock,
          parallelism) {}

WeightSynchronizer::~WeightSynchronizer() = default;

}  // namespace jax
}  // namespace tpu_raiden
