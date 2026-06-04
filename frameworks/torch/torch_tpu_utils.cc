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

#include "torch_tpu/eager/device_buffer.h"

namespace tpu_raiden {
namespace torch {

xla::PjRtBuffer* UnpackTorchTensor(const at::Tensor& tensor) {
  if (tensor.device().type() != at::DeviceType::PrivateUse1) {
    throw std::invalid_argument(
        "Tensor must reside on TPU device private use space");
  }
  if (!tensor.is_contiguous()) {
    throw std::invalid_argument("Tensor must be contiguous");
  }
  if (tensor.storage_offset() != 0) {
    throw std::invalid_argument(
        "Tensor must be a contiguous base tensor with zero storage offset");
  }

  const torch_tpu::DeviceBufferRef* base_buffer_ref =
      static_cast<const torch_tpu::DeviceBufferRef*>(
          tensor.storage().data_ptr().get_context());
  if (base_buffer_ref == nullptr) {
    throw std::runtime_error(
        "Tensor storage does not contain a TorchTPU DeviceBufferRef");
  }

  auto status_or_buf = base_buffer_ref->GetOrMaterializeBuffer();
  if (!status_or_buf.ok()) {
    throw std::runtime_error("Failed to fetch PjRtBuffer from TPU reference: " +
                             std::string(status_or_buf.status().message()));
  }
  return status_or_buf.value();
}

}  // namespace torch
}  // namespace tpu_raiden
