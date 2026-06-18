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

#include "tpu_raiden/frameworks/torch/torch_tpu_utils.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


#include "torch_tpu/eager/tensor_to_buffer.h"

namespace tpu_raiden {
namespace torch {

UnpackedTensor UnpackTorchTensor(const at::Tensor& tensor) {
  if (tensor.device().type() != at::DeviceType::PrivateUse1) {
    throw std::invalid_argument(
        "Tensor must reside on TPU device private use space");
  }
  if (!tensor.is_contiguous()) {
    throw std::invalid_argument("Tensor must be contiguous");
  }

  auto status_or_ref = torch_tpu::MaterializeAndReturn(
      tensor, torch_tpu::MaterializationReason::kCpuTransfer);
  if (!status_or_ref.ok()) {
    throw std::runtime_error("Failed to materialize TPU tensor: " +
                             std::string(status_or_ref.status().message()));
  }
  torch_tpu::DeviceBufferRef buffer_ref = std::move(status_or_ref.value());

  auto status_or_buf = buffer_ref.AwaitBuffer();
  if (!status_or_buf.ok()) {
    throw std::runtime_error("Failed to fetch PjRtBuffer from TPU reference: " +
                             std::string(status_or_buf.status().message()));
  }
  // Return the buffer AND the owning ref. For a view tensor whose layout
  // differs from its base, the materialized buffer above is a fresh allocation
  // owned only by `buffer_ref`; dropping the ref here would free it and leave a
  // dangling pointer (the cause of the D2h/H2d segfault when the manager later
  // dispatches a copy). Move the ref out so the caller can keep it alive.
  return UnpackedTensor{status_or_buf.value(), std::move(buffer_ref)};
}

}  // namespace torch
}  // namespace tpu_raiden
