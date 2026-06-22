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

#include "third_party/py/torch/aten/src/ATen/core/TensorBody.h"
#include "third_party/py/torch/torch/headeronly/core/DeviceType.h"

#include "torch_tpu/eager/structured_log_buffer.h"
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
  if (tensor.storage_offset() != 0) {
    throw std::invalid_argument(
        "Tensor must start at storage offset 0: raw transfer block-addresses "
        "the base storage buffer, so an offset (sliced) view would read/write "
        "the wrong blocks.");
  }

  // Raw transfer DMAs physical bytes directly into/out of the buffer, so it
  // must operate on the tensor's actual STORAGE buffer -- the one the model
  // reads and writes -- NOT a materialized copy. MaterializeAndReturn returns
  // the storage buffer only for a base tensor; for a layout-reinterpreting view
  // (e.g. vLLM's `empty().set_(kv_cache.untyped_storage()).view(5D)`, where the
  // model's on-device layout is tiled so the 5D reshape's layout differs from
  // the base) it returns a SEPARATE materialized buffer. DMAing into that copy
  // silently drops H2d (the model never sees the reload) and makes D2h read a
  // one-off snapshot. GetBaseBuffer always returns the buffer backing the
  // tensor's storage, so the DMA lands in the live cache regardless of view.
  auto status_or_ref = torch_tpu::GetBaseBuffer(tensor);
  if (!status_or_ref.ok()) {
    throw std::runtime_error("Failed to resolve base device buffer: " +
                             std::string(status_or_ref.status().message()));
  }
  torch_tpu::DeviceBufferRef base_ref = std::move(status_or_ref.value());

  // The base buffer is the storage and may carry different logical dimensions
  // than the tensor's view. Raw transfer slices it by its major (block)
  // dimension, so the tensor must map onto the base 1:1: same leading block
  // count and same total bytes (a full, layout-preserving reinterpret of the
  // block-major storage). A partial / differently-blocked view would address
  // the wrong bytes -- reject it loudly rather than corrupt silently.
  if (base_ref.dimensions().empty() ||
      base_ref.dimensions()[0] != tensor.size(0)) {
    throw std::invalid_argument(
        "Tensor's leading (block) dimension does not match its base storage "
        "buffer; raw transfer requires a layout-preserving full view of the "
        "block-major storage.");
  }
  if (static_cast<int64_t>(base_ref.size_bytes()) !=
      static_cast<int64_t>(tensor.nbytes())) {
    throw std::invalid_argument(
        "Tensor does not span its entire base storage buffer; raw transfer "
        "requires a full reinterpret of the storage, not a partial view.");
  }

  auto status_or_buf = base_ref.AwaitBuffer();
  if (!status_or_buf.ok()) {
    throw std::runtime_error("Failed to fetch PjRtBuffer from TPU reference: " +
                             std::string(status_or_buf.status().message()));
  }
  // Return the buffer AND the owning base ref. The ref pins the storage buffer
  // backing the tensor for as long as the caller keeps it (manager lifetime /
  // transfer future), so the raw PjRtBuffer* stays valid.
  return UnpackedTensor{status_or_buf.value(), std::move(base_ref)};
}

}  // namespace torch
}  // namespace tpu_raiden
