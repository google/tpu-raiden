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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_TPU_PJRT_MANAGER_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_TPU_PJRT_MANAGER_H_

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape.h"

namespace tpu_raiden {

// TpuPjrtManager manages a standalone TPU PjRtClient lifecycle and
// provides utility methods for buffer creation and memory management.
class TpuPjrtManager {
 public:
  // Returns a thread-safe global instance of the manager. Guaranteed to be
  // initialized.
  static absl::StatusOr<TpuPjrtManager*> GetDefault();

  ~TpuPjrtManager() = default;

  TpuPjrtManager(const TpuPjrtManager&) = delete;
  TpuPjrtManager& operator=(const TpuPjrtManager&) = delete;

  // Returns the underlying PjRtClient. Guaranteed to be non-null.
  xla::PjRtClient* client() const { return client_.get(); }

  // Returns the default TPU device (device 0). Guaranteed to be non-null.
  xla::PjRtDevice* GetDefaultDevice() const { return default_device_; }

  // Allocates an uninitialized buffer on the default device.
  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> AllocateBuffer(
      const xla::Shape& shape);

  // Creates an on-device buffer initialized with host data.
  // The host data is copied synchronously.
  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> BufferFromHost(
      const void* data, xla::PrimitiveType type,
      absl::Span<const int64_t> dims);

  // Creates an on-device buffer initialized with host data on a specific
  // device.
  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> BufferFromHost(
      const void* data, xla::PrimitiveType type, absl::Span<const int64_t> dims,
      xla::PjRtDevice* device);

  // Creates an on-device buffer from an XLA Literal.
  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> BufferFromLiteral(
      const xla::LiteralSlice& literal);

 private:
  TpuPjrtManager() = default;

  // Initialize the TPU PJRT client.
  absl::Status Initialize();

  std::unique_ptr<xla::PjRtClient> client_;
  xla::PjRtDevice* default_device_ = nullptr;  // Not owned
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_TPU_PJRT_MANAGER_H_
