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

#include <sys/mman.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/pjrt/pjrt_client.h"
#include "core/tpu_utils.h"

namespace tpu_raiden {

absl::StatusOr<std::unique_ptr<HostMemoryAllocator>>
HostMemoryAllocator::Create(xla::PjRtClient* pjrt_client) {
  if (pjrt_client != nullptr && pjrt_client->platform_name() != "cpu") {
    return XlaHostMemoryAllocator::Create(pjrt_client);
  }
  return std::unique_ptr<HostMemoryAllocator>(
      std::make_unique<MallocHostMemoryAllocator>());
}

absl::StatusOr<std::unique_ptr<XlaHostMemoryAllocator>>
XlaHostMemoryAllocator::Create(xla::PjRtClient* absl_nonnull pjrt_client) {
  return std::unique_ptr<XlaHostMemoryAllocator>(
      new XlaHostMemoryAllocator(pjrt_client));
}

XlaHostMemoryAllocator::XlaHostMemoryAllocator(xla::PjRtClient* client)
    : client_(client) {}

absl::StatusOr<HostBufferAllocation> XlaHostMemoryAllocator::AllocateDmaMapped(
    size_t size_bytes) {
  return Allocate(size_bytes);
}

absl::StatusOr<HostBufferAllocation>
XlaHostMemoryAllocator::AllocateDmaMappedForDevice(
    size_t size_bytes, const xla::PjRtDevice* device) {
  int numa_node = GetPjRtDeviceNumaNode(device);
  VLOG(1) << "[ALLOCATOR] Device: "
          << (device ? device->DebugString() : "nullptr")
          << ", resolved NUMA node: " << numa_node;
  if (numa_node >= 0) {
    // Bind this thread's allocations to the TPU's local NUMA node
    SetThreadMempolicy(2, numa_node);  // MPOL_BIND

    auto alloc_or = AllocateDmaMapped(size_bytes);
    if (alloc_or.ok()) {
      // Touch one byte per page (4KB) using a volatile pointer to force
      // physical page allocation (first-touch) on the bound NUMA node.
      // The volatile cast prevents the compiler from optimizing these writes
      // away.
      volatile uint8_t* p =
          static_cast<volatile uint8_t*>(alloc_or.value().ptr);
      for (size_t i = 0; i < size_bytes; i += 4096) {
        p[i] = 0;
      }
    }

    // Restore default policy
    SetThreadMempolicy(0);  // MPOL_DEFAULT

    return alloc_or;
  }
  return AllocateDmaMapped(size_bytes);
}

absl::StatusOr<HostBufferAllocation> XlaHostMemoryAllocator::Allocate(
    size_t size_bytes) {
  if (size_bytes == 0) {
    HostBufferAllocation alloc;
    alloc.ptr = nullptr;
    alloc.size = 0;
    return alloc;
  }

  // DMA mapping (DmaMap) requires the buffer size to be a multiple of
  // and at least 4096 bytes (page size). We round up the allocation size
  // to satisfy this requirement.
  size_t aligned_size = (size_bytes + 4095) & ~4095;

  // Allocate fresh virtual memory directly from the OS using mmap
  // to bypass any allocator pooling (both glibc and PJRT) and guarantee
  // that physical pages are allocated fresh when touched.
  void* ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) {
    return absl::ResourceExhaustedError(
        absl::StrCat("mmap failed for size: ", aligned_size));
  }

  // Register the memory with the device DMA engine
  auto status = client_->DmaMap(ptr, aligned_size);
  if (!status.ok()) {
    munmap(ptr, aligned_size);
    return absl::InternalError(
        absl::StrCat("DmaMap failed: ", status.message()));
  }

  HostBufferAllocation alloc;
  alloc.ptr = static_cast<uint8_t*>(ptr);
  alloc.size = size_bytes;  // Keep the requested size for the user

  // Package the munmap and DmaUnmap in the deleter of the shared_ptr owner
  xla::PjRtClient* client = client_;
  alloc.owner = std::shared_ptr<void>(ptr, [client, aligned_size](void* p) {
    (void)client->DmaUnmap(p);
    munmap(p, aligned_size);
  });

  return alloc;
}

absl::StatusOr<HostBufferAllocation> MallocHostMemoryAllocator::Allocate(
    size_t size_bytes) {
  void* ptr = nullptr;
  if (size_bytes > 0) {
    int err = posix_memalign(&ptr, 64, size_bytes);
    if (err != 0) {
      return absl::ResourceExhaustedError(
          absl::StrCat("posix_memalign failed with error: ", err));
    }
  }

  HostBufferAllocation alloc;
  alloc.ptr = static_cast<uint8_t*>(ptr);
  alloc.size = size_bytes;
  alloc.owner = std::shared_ptr<void>(ptr, std::free);
  return alloc;
}

}  // namespace tpu_raiden
