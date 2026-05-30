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

#include "frameworks/torch/torch_utils.h"

#include <stdexcept>
#include <utility>
#include <vector>

#include "frameworks/torch/torch_tpu_utils.h"

namespace tpu_raiden {
namespace torch {

std::vector<std::vector<xla::PjRtBuffer*>> UnpackTorchTensors(
    const std::vector<std::vector<at::Tensor>>& device_tensors) {
  size_t num_layers = device_tensors.size();
  if (num_layers == 0) return {};

  size_t num_shards = device_tensors[0].size();
  std::vector<std::vector<xla::PjRtBuffer*>> layer_buffers;
  layer_buffers.reserve(num_layers);

  for (size_t l = 0; l < num_layers; ++l) {
    if (device_tensors[l].size() != num_shards) {
      throw std::invalid_argument(
          "Symmetrical shards count mismatch across layers during PyTorch "
          "unpack");
    }
    std::vector<xla::PjRtBuffer*> shard_buffers;
    shard_buffers.reserve(num_shards);

    for (size_t sh = 0; sh < num_shards; ++sh) {
      const at::Tensor& tensor = device_tensors[l][sh];
      shard_buffers.push_back(UnpackTorchTensor(tensor));
    }
    layer_buffers.push_back(std::move(shard_buffers));
  }
  return layer_buffers;
}

}  // namespace torch
}  // namespace tpu_raiden
