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

#include "tpu_raiden/frameworks/torch/weight_synchronizer.h"

#include <optional>
#include <string>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "tpu_raiden/frameworks/torch/torch_utils.h"
#include "weight_sync/weight_synchronizer_base.h"

namespace tpu_raiden {
namespace torch {

WeightSynchronizer::WeightSynchronizer(
    const std::vector<std::vector<at::Tensor>>& device_tensors,
    std::optional<int> local_port, int parallelism,
    std::optional<std::string> local_ip)
    : weight_sync::WeightSynchronizerBase(
          UnpackTorchTensors(device_tensors), local_port,
          /*external_host_ptrs=*/std::nullopt,
          /*unsafe_skip_buffer_lock=*/true, parallelism,
          /*control_port=*/std::nullopt) {}

WeightSynchronizer::~WeightSynchronizer() = default;

}  // namespace torch
}  // namespace tpu_raiden
