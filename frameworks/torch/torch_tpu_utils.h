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

#ifndef THIRD_PARTY_TPU_RAIDEN_FRAMEWORKS_TORCH_TORCH_TPU_UTILS_TORCH_TPU_UTILS_H_
#define THIRD_PARTY_TPU_RAIDEN_FRAMEWORKS_TORCH_TORCH_TPU_UTILS_TORCH_TPU_UTILS_H_

#include <vector>

#include "ATen/core/TensorBody.h"
#include "xla/pjrt/pjrt_client.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/materialize.h"
#include "torch_tpu/eager/tensor_to_buffer.h"

namespace tpu_raiden {
namespace torch {

// Validates that the tensor is a TPU tensor and is contiguous.
// Throws std::invalid_argument if validation fails.
void ValidateTpuTensor(const at::Tensor& tensor, const char* role);

// Materializes the TPU tensor and returns a DeviceBufferRef.
// Throws std::runtime_error if materialization fails.
torch_tpu::DeviceBufferRef GetMaterializedBufferRef(const at::Tensor& tensor);

// Extracts the raw PjRtBuffer pointer from the DeviceBufferRef.
// Throws std::runtime_error if extraction fails.
xla::PjRtBuffer* GetPjRtBuffer(const torch_tpu::DeviceBufferRef& buffer_ref);

// Unpacks a single PyTorch tensor into a raw PjRtBuffer pointer.
// Throws exceptions if validation or materialization fails.
xla::PjRtBuffer* UnpackTorchTensor(const at::Tensor& tensor);

}  // namespace torch
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_FRAMEWORKS_TORCH_TORCH_TPU_UTILS_TORCH_TPU_UTILS_H_
