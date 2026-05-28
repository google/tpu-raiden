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

#include "api/torch/weight_synchronizer.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "xla/pjrt/pjrt_client.h"
#include "weight_sync/weight_synchronizer_base.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/materialize.h"

namespace tpu_raiden {
namespace torch {

namespace {

// Unpacks PyTorch sharded weight tensors matrix into raw PjRtBuffer pointers
// E2E!
std::vector<std::vector<xla::PjRtBuffer*>> UnpackTorchWeights(
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
          "weights unpack");
    }
    std::vector<xla::PjRtBuffer*> shard_buffers;
    shard_buffers.reserve(num_shards);

    for (size_t sh = 0; sh < num_shards; ++sh) {
      const at::Tensor& tensor = device_tensors[l][sh];
      if (tensor.device().type() != at::DeviceType::PrivateUse1) {
        throw std::invalid_argument(
            "Tensor must reside on TPU device private use space");
      }
      if (!tensor.is_contiguous()) {
        throw std::invalid_argument("Tensor must be contiguous");
      }

      // Materialize the underlying PyTorch TPU DeviceBufferRef E2E
      auto status_or_ref = torch_tpu::GetMaterialized(
          tensor, torch_tpu::MaterializationReason::kCpuTransfer);
      if (!status_or_ref.ok()) {
        throw std::runtime_error("Failed to materialize TPU weights tensor: " +
                                 std::string(status_or_ref.status().message()));
      }
      torch_tpu::DeviceBufferRef buffer_ref = std::move(status_or_ref.value());

      // Extract raw PjRtBuffer device pointer
      auto status_or_buf = buffer_ref.AwaitBuffer();
      if (!status_or_buf.ok()) {
        throw std::runtime_error(
            "Failed to fetch PjRtBuffer from TPU weights reference: " +
            std::string(status_or_buf.status().message()));
      }
      shard_buffers.push_back(status_or_buf.value());
    }
    layer_buffers.push_back(std::move(shard_buffers));
  }
  return layer_buffers;
}

}  // namespace

WeightSynchronizer::WeightSynchronizer(
    const std::vector<std::vector<at::Tensor>>& device_tensors,
    std::optional<int> local_port, int parallelism)
    : weight_sync::WeightSynchronizerBase(
          UnpackTorchWeights(device_tensors), local_port,
          /*external_host_ptrs=*/std::nullopt,
          /*unsafe_skip_buffer_lock=*/true, parallelism) {}

WeightSynchronizer::~WeightSynchronizer() = default;

}  // namespace torch
}  // namespace tpu_raiden
