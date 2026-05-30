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

#include "frameworks/torch/torch_tpu_utils.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace tpu_raiden {
namespace torch {

void ValidateTpuTensor(const at::Tensor& tensor, const char* role) {
  if (tensor.device().type() != at::DeviceType::PrivateUse1) {
    throw std::invalid_argument(std::string(role) +
                                " must reside on TPU device private use space");
  }
  if (!tensor.is_contiguous()) {
    throw std::invalid_argument(std::string(role) + " must be contiguous");
  }
}

torch_tpu::DeviceBufferRef GetMaterializedBufferRef(const at::Tensor& tensor) {
  auto status_or_ref = torch_tpu::MaterializeAndReturn(
      tensor, torch_tpu::MaterializationReason::kCpuTransfer);
  if (!status_or_ref.ok()) {
    throw std::runtime_error("Failed to materialize TPU tensor: " +
                             std::string(status_or_ref.status().message()));
  }
  return std::move(status_or_ref.value());
}

xla::PjRtBuffer* GetPjRtBuffer(const torch_tpu::DeviceBufferRef& buffer_ref) {
  auto status_or_buf = buffer_ref.AwaitBuffer();
  if (!status_or_buf.ok()) {
    throw std::runtime_error("Failed to fetch PjRtBuffer from TPU reference: " +
                             std::string(status_or_buf.status().message()));
  }
  return status_or_buf.value();
}

xla::PjRtBuffer* UnpackTorchTensor(const at::Tensor& tensor) {
  ValidateTpuTensor(tensor, "Tensor");
  torch_tpu::DeviceBufferRef buffer_ref = GetMaterializedBufferRef(tensor);
  return GetPjRtBuffer(buffer_ref);
}

}  // namespace torch
}  // namespace tpu_raiden
