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
#include <vector>

#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_client.h"
#include "core/host_memory_allocator.h"

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

inline HostBufferAllocator CreatePinnedHostAllocator(xla::PjRtClient* client) {
  auto allocator = std::make_shared<PinnedHostAllocator>(client);
  return
      [allocator](size_t size_bytes) -> absl::StatusOr<HostBufferAllocation> {
        return allocator->Allocate(size_bytes);
      };
}

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_UTILS_H_
