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

#include "tpu_raiden/frameworks/torch/torch_utils.h"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "third_party/py/torch/aten/src/ATen/core/TensorBody.h"
#include "xla/pjrt/pjrt_client.h"
#include "tpu_raiden/frameworks/torch/torch_tpu_utils.h"

namespace tpu_raiden {
namespace torch {

UnpackedTensors UnpackTorchTensors(
    const std::vector<std::vector<at::Tensor>>& device_tensors) {
  UnpackedTensors out;
  size_t num_layers = device_tensors.size();
  if (num_layers == 0) return out;

  size_t num_shards = device_tensors[0].size();
  out.buffers.reserve(num_layers);

  for (size_t l = 0; l < num_layers; ++l) {
    if (device_tensors[l].size() != num_shards) {
      throw std::invalid_argument(
          "Symmetrical shards count mismatch across layers during PyTorch "
          "unpack");
    }
    std::vector<xla::PjRtBuffer*> shard_buffers;
    shard_buffers.reserve(num_shards);
    for (size_t sh = 0; sh < num_shards; ++sh) {
      UnpackedTensor unpacked = UnpackTorchTensor(device_tensors[l][sh]);
      shard_buffers.push_back(unpacked.buffer);
      // Retain every owning ref so view-materialized buffers survive for the
      // lifetime of `out.refs`. (The test mock returns no ref -- nullopt --
      // since it hands back pre-registered buffers with no materialization.)
      if (unpacked.ref) {
        out.refs.push_back(std::move(*unpacked.ref));
      }
    }
    out.buffers.push_back(std::move(shard_buffers));
  }
  return out;
}

}  // namespace torch
}  // namespace tpu_raiden
