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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_HOST_MEMORY_ALLOCATOR_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_HOST_MEMORY_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_client.h"

namespace tpu_raiden {

struct HostBufferAllocation {
  uint8_t* ptr = nullptr;
  size_t size = 0;
  std::shared_ptr<void> owner;
};

// Returns a HostBufferAllocation of at least the requested size for a given
// device. If device is nullptr, it allocates on the default/current NUMA node.
// On failure, returns a non-OK status.
using HostBufferAllocator = std::function<absl::StatusOr<HostBufferAllocation>(
    size_t, const xla::PjRtDevice*)>;

// High-performance host memory allocator that allocates DMA-capable pinned
// memory using PJRT APIs or standard fallback allocations.
class HostMemoryAllocator {
 public:
  virtual ~HostMemoryAllocator() = default;

  static absl::StatusOr<std::unique_ptr<HostMemoryAllocator>> Create(
      xla::PjRtClient* pjrt_client);

  virtual absl::StatusOr<HostBufferAllocation> Allocate(size_t size_bytes) = 0;

  // Allocates host memory registered with the device DMA engine
  // (PjRtClient::DmaMap), so raw C-API D2H/H2D into it run as a true async DMA
  // instead of a synchronous staged copy (the producer-D2H bottleneck). The
  // default implementation falls back to Allocate (e.g. CPU / no DMA support).
  virtual absl::StatusOr<HostBufferAllocation> AllocateDmaMapped(
      size_t size_bytes) {
    return Allocate(size_bytes);
  }

  // Device-aware version of AllocateDmaMapped. Attempts to allocate host
  // memory on the NUMA node local to the given device.
  virtual absl::StatusOr<HostBufferAllocation> AllocateDmaMappedForDevice(
      size_t size_bytes, const xla::PjRtDevice* device) {
    return AllocateDmaMapped(size_bytes);
  }
};

class XlaHostMemoryAllocator : public HostMemoryAllocator {
 public:
  static absl::StatusOr<std::unique_ptr<XlaHostMemoryAllocator>> Create(
      xla::PjRtClient* absl_nonnull pjrt_client);

  absl::StatusOr<HostBufferAllocation> Allocate(size_t size_bytes) override;

  absl::StatusOr<HostBufferAllocation> AllocateDmaMapped(
      size_t size_bytes) override;

  absl::StatusOr<HostBufferAllocation> AllocateDmaMappedForDevice(
      size_t size_bytes, const xla::PjRtDevice* device) override;

 private:
  explicit XlaHostMemoryAllocator(xla::PjRtClient* client);

  xla::PjRtClient* client_ = nullptr;
};

class MallocHostMemoryAllocator : public HostMemoryAllocator {
 public:
  absl::StatusOr<HostBufferAllocation> Allocate(size_t size_bytes) override;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_HOST_MEMORY_ALLOCATOR_H_
