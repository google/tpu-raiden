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

#include "frameworks/torch/kv_cache_manager.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_client.h"
#include "core/host_memory_allocator.h"
#include "frameworks/torch/torch_tpu_utils.h"
#include "frameworks/torch/torch_utils.h"
#include "kv_cache/kv_cache_manager_base.h"

namespace tpu_raiden {
namespace torch {

namespace {

std::optional<std::vector<const uint8_t*>> CastExternalPointers(
    const std::optional<std::vector<uintptr_t>>& external_host_ptrs) {
  if (!external_host_ptrs.has_value()) return std::nullopt;
  std::vector<const uint8_t*> cast_ptrs;
  cast_ptrs.reserve(external_host_ptrs->size());
  for (uintptr_t addr : *external_host_ptrs) {
    cast_ptrs.push_back(reinterpret_cast<const uint8_t*>(addr));
  }
  return cast_ptrs;
}

kv_cache::HostBufferAllocator TorchPinnedHostAllocator(
    xla::PjRtClient* client) {
  auto allocator = std::make_shared<PinnedHostAllocator>(client);
  return
      [allocator](size_t size_bytes) -> absl::StatusOr<HostBufferAllocation> {
        return allocator->Allocate(size_bytes);
      };
}

}  // namespace

KVCacheManager::KVCacheManager(
    const std::vector<std::vector<at::Tensor>>& device_tensors, int block_size,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    std::optional<std::vector<uintptr_t>> external_host_ptrs,
    bool unsafe_skip_buffer_lock, int parallelism)
    : kv_cache::KVCacheManagerBase(
          UnpackTorchTensors(device_tensors), block_size, local_port,
          host_blocks_to_allocate, CastExternalPointers(external_host_ptrs),
          unsafe_skip_buffer_lock, parallelism,
          TorchPinnedHostAllocator(device_tensors.empty() ||
                                           device_tensors[0].empty()
                                       ? nullptr
                                       : UnpackTorchTensor(device_tensors[0][0])
                                             ->device()
                                             ->client())) {}

KVCacheManager::~KVCacheManager() = default;

}  // namespace torch
}  // namespace tpu_raiden
