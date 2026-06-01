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

#include "core/host_memory_allocator.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/pjrt_client.h"

namespace tpu_raiden {

absl::StatusOr<std::unique_ptr<HostMemoryAllocator>>
HostMemoryAllocator::Create(xla::PjRtClient* pjrt_client) {
  if (pjrt_client != nullptr) {
    return XlaHostMemoryAllocator::Create(pjrt_client);
  }
  return std::unique_ptr<HostMemoryAllocator>(
      std::make_unique<MallocHostMemoryAllocator>());
}

absl::StatusOr<std::unique_ptr<XlaHostMemoryAllocator>>
XlaHostMemoryAllocator::Create(xla::PjRtClient* absl_nonnull pjrt_client) {
  xla::HostMemoryAllocator* host_allocator =
      pjrt_client->GetHostMemoryAllocator();
  if (host_allocator == nullptr) {
    return absl::FailedPreconditionError("XLA HostMemoryAllocator is null.");
  }
  return std::unique_ptr<XlaHostMemoryAllocator>(
      new XlaHostMemoryAllocator(host_allocator));
}

XlaHostMemoryAllocator::XlaHostMemoryAllocator(
    xla::HostMemoryAllocator* host_allocator)
    : host_allocator_(host_allocator) {}

absl::StatusOr<HostBufferAllocation> XlaHostMemoryAllocator::Allocate(
    size_t size_bytes) {
  // Allocate high-performance DMA pinned memory directly from PJRT Client
  xla::HostMemoryAllocator::OwnedPtr data =
      host_allocator_->Allocate(size_bytes);
  if (data == nullptr && size_bytes > 0) {
    return absl::ResourceExhaustedError(absl::StrCat(
        "Failed to allocate PJRT pinned host buffer of size: ", size_bytes));
  }

  HostBufferAllocation alloc;
  alloc.ptr = static_cast<uint8_t*>(data.get());
  alloc.size = size_bytes;

  // Package the unique_ptr inside the shared_ptr owner to keep the allocation
  // alive
  auto shared_data =
      std::make_shared<xla::HostMemoryAllocator::OwnedPtr>(std::move(data));
  alloc.owner = std::shared_ptr<void>(shared_data, shared_data->get());
  return alloc;
}

absl::StatusOr<HostBufferAllocation> MallocHostMemoryAllocator::Allocate(
    size_t size_bytes) {
  void* ptr = nullptr;
  if (size_bytes > 0) {
    int err = posix_memalign(&ptr, 64, size_bytes);
    if (err != 0) {
      return absl::InternalError(
          absl::StrCat("Failed to allocate unpinned fallback buffer via "
                       "posix_memalign of size: ",
                       size_bytes));
    }
    std::memset(ptr, 0, size_bytes);
  }
  HostBufferAllocation alloc;
  alloc.ptr = static_cast<uint8_t*>(ptr);
  alloc.size = size_bytes;
  alloc.owner = std::shared_ptr<void>(ptr, [](void* p) { std::free(p); });
  return alloc;
}

}  // namespace tpu_raiden
