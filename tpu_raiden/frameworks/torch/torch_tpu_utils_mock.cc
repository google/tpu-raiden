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

#include "tpu_raiden/frameworks/torch/torch_tpu_utils_mock.h"

#include <optional>
#include <stdexcept>
#include <unordered_map>

#include "ATen/core/TensorBody.h"
#include "xla/pjrt/pjrt_client.h"
#include "tpu_raiden/frameworks/torch/torch_tpu_utils.h"

namespace tpu_raiden {
namespace torch {

namespace {
std::unordered_map<void*, xla::PjRtBuffer*>& GetMockMap() {
  static auto* m = new std::unordered_map<void*, xla::PjRtBuffer*>();
  return *m;
}
}  // namespace

void RegisterMockTensor(const at::Tensor& tensor, xla::PjRtBuffer* buffer) {
  GetMockMap()[tensor.unsafeGetTensorImpl()] = buffer;
}

UnpackedTensor UnpackTorchTensor(const at::Tensor& tensor) {
  auto it = GetMockMap().find(tensor.unsafeGetTensorImpl());
  if (it == GetMockMap().end()) {
    throw std::runtime_error(
        "Mock tensor not registered. Call RegisterMockTensor first.");
  }
  // Mock has no real materialization, hence no DeviceBufferRef to hand back.
  return UnpackedTensor{it->second, std::nullopt};
}

}  // namespace torch
}  // namespace tpu_raiden
