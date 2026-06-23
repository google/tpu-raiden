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

#include "tpu_raiden/core/host_memory_allocator.h"

#include <sys/mman.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/pjrt_client.h"
#include "tpu_raiden/core/tpu_utils.h"

namespace tpu_raiden {
namespace {
thread_local const xla::PjRtDevice* g_current_device = nullptr;
}  // namespace

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
  g_current_device = device;
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
      volatile uint8_t* p =
          static_cast<volatile uint8_t*>(alloc_or.value().ptr);
      for (size_t i = 0; i < size_bytes; i += 4096) {
        p[i] = 0;
      }
    }

    // Restore default policy
    SetThreadMempolicy(0);  // MPOL_DEFAULT

    g_current_device = nullptr;
    return alloc_or;
  }
  auto alloc_or = AllocateDmaMapped(size_bytes);
  g_current_device = nullptr;
  return alloc_or;
}

absl::StatusOr<HostBufferAllocation> XlaHostMemoryAllocator::Allocate(
    size_t size_bytes) {
  if (size_bytes == 0) {
    HostBufferAllocation alloc;
    alloc.ptr = nullptr;
    alloc.size = 0;
    return alloc;
  }

  bool is_tpuv7 = false;
  if (client_ != nullptr) {
    absl::string_view version = client_->platform_version();
    if (absl::StrContains(version, "7") || absl::StrContains(version, "v7")) {
      is_tpuv7 = true;
    }
    if (!client_->devices().empty() && client_->devices()[0] != nullptr) {
      absl::string_view kind = client_->devices()[0]->device_kind();
      if (absl::StrContains(kind, "7") || absl::StrContains(kind, "v7")) {
        is_tpuv7 = true;
      }
    }
  }

  const bool is_multi_process =
      (std::getenv("PJRT_LOCAL_PROCESS_RANK") != nullptr ||
       std::getenv("RANK") != nullptr || std::getenv("LOCAL_RANK") != nullptr);

  const bool skip_dma_map = (is_tpuv7 && is_multi_process);

  // HYBRID PATH: When skipping DmaMap on multi-process TPUv7 pods, delegate
  // directly to libtpu's internal host memory allocator pool. This yields
  // pre-mapped VFIO staged DMA memory (~175 GB/s H2D / ~128 GB/s D2H) rather
  // than raw unpinned OS memory (~106 GB/s), saving ~70 GB/s on MPMD workloads.
  if (skip_dma_map && client_ != nullptr &&
      client_->GetHostMemoryAllocator() != nullptr) {
    xla::HostMemoryAllocator::AllocateOptions alloc_opts;
    if (g_current_device != nullptr) {
      alloc_opts.numa_node = GetPjRtDeviceNumaNode(g_current_device);
      alloc_opts.local_device_id = g_current_device->local_device_id();
    }
    xla::HostMemoryAllocator* host_alloc = client_->GetHostMemoryAllocator();
    xla::HostMemoryAllocator::OwnedPtr data =
        host_alloc->Allocate(size_bytes, alloc_opts);
    if (data != nullptr) {
      HostBufferAllocation alloc;
      alloc.ptr = data.get();
      alloc.size = size_bytes;
      alloc.owner = std::shared_ptr<void>(
          data.get(), [d = std::move(data)](void*) mutable { d.reset(); });
      return alloc;
    }
  }

  size_t aligned_size = (size_bytes + 4095) & ~4095;

  void* ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) {
    return absl::ResourceExhaustedError(
        absl::StrCat("mmap failed for size: ", aligned_size));
  }

  bool dma_mapped = false;

  if (!skip_dma_map) {
    auto status = client_->DmaMap(ptr, aligned_size);
    if (status.ok()) {
      dma_mapped = true;
    } else {
      munmap(ptr, aligned_size);
      return absl::InternalError(
          absl::StrCat("DmaMap failed: ", status.message()));
    }
  }

  HostBufferAllocation alloc;
  alloc.ptr = static_cast<uint8_t*>(ptr);
  alloc.size = size_bytes;

  xla::PjRtClient* client = client_;
  alloc.owner =
      std::shared_ptr<void>(ptr, [client, aligned_size, dma_mapped](void* p) {
        if (dma_mapped) {
          (void)client->DmaUnmap(p);
        }
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
