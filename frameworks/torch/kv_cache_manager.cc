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
#include "core/utils.h"
#include "frameworks/torch/torch_tpu_utils.h"
#include "frameworks/torch/torch_utils.h"
#include "kv_cache/kv_cache_manager_base.h"

namespace tpu_raiden {
namespace torch {

KVCacheManager::KVCacheManager(
    const std::vector<std::vector<at::Tensor>>& device_tensors, int block_size,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    std::optional<std::vector<uintptr_t>> external_host_ptrs,
    bool unsafe_skip_buffer_lock, int parallelism)
    : kv_cache::KVCacheManagerBase(
          UnpackTorchTensors(device_tensors), block_size, local_port,
          host_blocks_to_allocate,
          tpu_raiden::CastExternalPointers(external_host_ptrs),
          unsafe_skip_buffer_lock, parallelism,
          tpu_raiden::CreatePinnedHostAllocator(
              device_tensors.empty() || device_tensors[0].empty()
                  ? nullptr
                  : UnpackTorchTensor(device_tensors[0][0])
                        ->device()
                        ->client())) {}

KVCacheManager::~KVCacheManager() = default;

}  // namespace torch
}  // namespace tpu_raiden
