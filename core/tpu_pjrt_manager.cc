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

#include "core/tpu_pjrt_manager.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_tpu/xla_tpu_pjrt_client.h"
#include "xla/shape.h"

namespace tpu_raiden {

namespace {
absl::Mutex g_manager_mutex;
TpuPjrtManager* g_default_manager ABSL_GUARDED_BY(g_manager_mutex) = nullptr;
}  // namespace

absl::StatusOr<TpuPjrtManager*> TpuPjrtManager::GetDefault() {
  absl::MutexLock lock(g_manager_mutex);
  if (g_default_manager == nullptr) {
    auto manager = std::unique_ptr<TpuPjrtManager>(new TpuPjrtManager());
    absl::Status status = manager->Initialize();
    if (!status.ok()) {
      return status;
    }
    g_default_manager = manager.release();
  }
  return g_default_manager;
}

absl::Status TpuPjrtManager::Initialize() {
  if (client_ != nullptr) {
    return absl::OkStatus();
  }

  // Call the OpenXLA TPU client factory
  auto client_or = xla::GetXlaPjrtTpuClient();
  if (!client_or.ok()) {
    return client_or.status();
  }
  client_ = std::move(client_or).value();

  if (client_->addressable_device_count() == 0) {
    client_.reset();
    return absl::InternalError(
        "TPU PjRtClient initialized but no addressable devices found.");
  }

  default_device_ = client_->addressable_devices()[0];
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> TpuPjrtManager::AllocateBuffer(
    const xla::Shape& shape) {
  auto memory_space_or = default_device_->default_memory_space();
  if (!memory_space_or.ok()) {
    return memory_space_or.status();
  }
  return client_->CreateUninitializedBuffer(shape, memory_space_or.value());
}

absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> TpuPjrtManager::BufferFromHost(
    const void* data, xla::PrimitiveType type, absl::Span<const int64_t> dims) {
  return BufferFromHost(data, type, dims, default_device_);
}

absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> TpuPjrtManager::BufferFromHost(
    const void* data, xla::PrimitiveType type, absl::Span<const int64_t> dims,
    xla::PjRtDevice* device) {
  if (device == nullptr) {
    device = default_device_;
  }
  auto memory_space_or = device->default_memory_space();
  if (!memory_space_or.ok()) {
    return memory_space_or.status();
  }

  std::vector<int64_t> dims_vec(dims.begin(), dims.end());
  return client_->BufferFromHostBuffer(
      data, type, dims_vec, /*byte_strides=*/std::nullopt,
      xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
      /*on_done_with_host_buffer=*/nullptr, memory_space_or.value(),
      /*device_layout=*/nullptr);
}

absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>>
TpuPjrtManager::BufferFromLiteral(const xla::LiteralSlice& literal) {
  auto memory_space_or = default_device_->default_memory_space();
  if (!memory_space_or.ok()) {
    return memory_space_or.status();
  }
  return client_->BufferFromHostLiteral(literal, memory_space_or.value());
}

}  // namespace tpu_raiden
