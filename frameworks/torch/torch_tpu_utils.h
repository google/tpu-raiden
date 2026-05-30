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
#include "c10/core/Allocator.h"
#include "xla/pjrt/pjrt_client.h"

namespace tpu_raiden {
namespace torch {

// Unpacks a single PyTorch tensor into a raw PjRtBuffer pointer.
// Throws exceptions if validation or materialization fails.
xla::PjRtBuffer* UnpackTorchTensor(const at::Tensor& tensor);

// Allocates a TPU pinned host buffer of the given size.
c10::DataPtr AllocateTpuPinnedHostBuffer(size_t size_bytes);

}  // namespace torch
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_FRAMEWORKS_TORCH_TORCH_TPU_UTILS_TORCH_TPU_UTILS_H_
