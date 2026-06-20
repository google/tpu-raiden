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

/* Copyright 2026 The TPU Raiden Authors. All Rights Reserved.
==============================================================================*/

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_UTILS_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape.h"
#include "tpu_raiden/core/host_memory_allocator.h"

namespace tpu_raiden {

inline std::optional<std::vector<const uint8_t*>> CastExternalPointers(
    const std::optional<std::vector<uintptr_t>>& external_host_ptrs) {
  if (!external_host_ptrs.has_value()) return std::nullopt;
  std::vector<const uint8_t*> cast_ptrs;
  cast_ptrs.reserve(external_host_ptrs->size());
  for (uintptr_t addr : *external_host_ptrs) {
    cast_ptrs.push_back(reinterpret_cast<const uint8_t*>(addr));
  }
  return cast_ptrs;
}

inline HostBufferAllocator CreateHostMemoryAllocator(xla::PjRtClient* client) {
  auto allocator_or = HostMemoryAllocator::Create(client);
  if (!allocator_or.ok()) {
    absl::Status status = allocator_or.status();
    return [status](size_t size_bytes, const xla::PjRtDevice* device)
               -> absl::StatusOr<HostBufferAllocation> { return status; };
  }
  std::shared_ptr<HostMemoryAllocator> allocator =
      std::move(allocator_or).value();
  return [allocator](size_t size_bytes, const xla::PjRtDevice* device)
             -> absl::StatusOr<HostBufferAllocation> {
    return allocator->AllocateDmaMappedForDevice(size_bytes, device);
  };
}

struct RawCopyChunk {
  int64_t src_offset;
  int64_t dst_offset;
  int64_t size_bytes;
};

inline absl::Status ValidatePartialSpec(
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  bool present = !src_offsets_major_dim.empty() ||
                 !dst_offsets_major_dim.empty() ||
                 !copy_sizes_major_dim.empty();
  if (present &&
      (src_offsets_major_dim.size() != dst_offsets_major_dim.size() ||
       src_offsets_major_dim.size() != copy_sizes_major_dim.size())) {
    return absl::InvalidArgumentError(
        "src_offsets_major_dim, dst_offsets_major_dim, and "
        "copy_sizes_major_dim must have the same length");
  }
  for (size_t i = 0; i < src_offsets_major_dim.size(); ++i) {
    if (src_offsets_major_dim[i] < 0 || dst_offsets_major_dim[i] < 0 ||
        copy_sizes_major_dim[i] < 0) {
      return absl::InvalidArgumentError(
          "raw copy offsets and sizes must be non-negative");
    }
  }
  return absl::OkStatus();
}

inline bool IsPartialCopy(const xla::Shape& shape,
                          const std::vector<int64_t>& src_offsets_major_dim,
                          const std::vector<int64_t>& dst_offsets_major_dim,
                          const std::vector<int64_t>& copy_sizes_major_dim) {
  if (src_offsets_major_dim.empty()) return false;
  if (shape.dimensions().empty()) return true;
  const int64_t full_major_dim = shape.dimensions(0);
  for (size_t i = 0; i < src_offsets_major_dim.size(); ++i) {
    if (src_offsets_major_dim[i] != 0 || dst_offsets_major_dim[i] != 0 ||
        copy_sizes_major_dim[i] != full_major_dim) {
      return true;
    }
  }
  return false;
}

inline absl::Status ValidatePartialAlignment(const xla::Shape& shape,
                                             int64_t slice_byte_size) {
  if (shape.dimensions().size() < 3) {
    return absl::InvalidArgumentError(
        "Only rank >= 3 TPU tensors support partial raw copies");
  }
  if (slice_byte_size % 4096 != 0) {
    return absl::InvalidArgumentError(
        "Partial raw copies require a major-dimension slice size aligned to "
        "4096 bytes");
  }
  return absl::OkStatus();
}

inline absl::StatusOr<std::vector<RawCopyChunk>> ComputeAndValidateChunks(
    int64_t slice_byte_size, int64_t physical_size, int64_t max_cpu_size,
    bool is_partial, const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim, bool is_d2h) {
  std::vector<RawCopyChunk> chunks;
  if (!is_partial) {
    if (is_d2h) {
      if (max_cpu_size < physical_size) {
        return absl::InvalidArgumentError(
            "Destination CPU tensor is too small");
      }
    } else {
      if (max_cpu_size < physical_size) {
        return absl::InvalidArgumentError("Source CPU tensor is too small");
      }
    }
    chunks.push_back({0, 0, physical_size});
  } else {
    chunks.reserve(src_offsets_major_dim.size());
    for (size_t i = 0; i < src_offsets_major_dim.size(); ++i) {
      const int64_t src_offset = src_offsets_major_dim[i] * slice_byte_size;
      const int64_t dst_offset = dst_offsets_major_dim[i] * slice_byte_size;
      const int64_t size_to_copy = copy_sizes_major_dim[i] * slice_byte_size;
      if (is_d2h) {
        if (src_offset + size_to_copy > physical_size) {
          return absl::InvalidArgumentError(
              "Copy range exceeds source TPU buffer");
        }
        if (dst_offset + size_to_copy > max_cpu_size) {
          return absl::InvalidArgumentError(
              "Copy range exceeds destination CPU tensor");
        }
      } else {
        if (src_offset + size_to_copy > max_cpu_size) {
          return absl::InvalidArgumentError(
              "Copy range exceeds source CPU tensor");
        }
        if (dst_offset + size_to_copy > physical_size) {
          return absl::InvalidArgumentError(
              "Copy range exceeds destination TPU buffer");
        }
      }
      chunks.push_back({src_offset, dst_offset, size_to_copy});
    }
  }
  return chunks;
}

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_UTILS_H_
