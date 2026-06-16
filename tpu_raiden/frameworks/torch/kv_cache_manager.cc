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

#include "tpu_raiden/frameworks/torch/kv_cache_manager.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "xla/pjrt/pjrt_client.h"
#include "core/kv_cache_manager_with_transfer.h"
#include "core/utils.h"
#include "tpu_raiden/frameworks/torch/torch_tpu_utils.h"
#include "tpu_raiden/frameworks/torch/torch_utils.h"

namespace tpu_raiden {
namespace torch {
namespace {

using TensorList = std::vector<at::Tensor>;

std::vector<std::vector<at::Tensor>> SingleShardLayers(
    const TensorList& kv_caches) {
  std::vector<std::vector<at::Tensor>> layers;
  layers.reserve(kv_caches.size());
  for (const auto& kv_cache : kv_caches) {
    layers.push_back({kv_cache});
  }
  return layers;
}

}  // namespace

KVCacheManager::KVCacheManager(
    const std::vector<std::vector<at::Tensor>>& device_tensors,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    bool unsafe_skip_buffer_lock, int parallelism)
    : KVCacheManagerWithTransfer(
          UnpackTorchTensors(device_tensors), local_port,
          host_blocks_to_allocate,
          unsafe_skip_buffer_lock, parallelism,
          tpu_raiden::CreateHostMemoryAllocator(
              device_tensors.empty() || device_tensors[0].empty()
                  ? nullptr
                  : UnpackTorchTensor(device_tensors[0][0])
                        ->device()
                        ->client()),
          /*node_id=*/0,
          /*local_control_port=*/-1,
          /*max_blocks=*/0,
          /*num_slots=*/0,
          /*timeout_s=*/120.0) {}

KVCacheManager::KVCacheManager(const std::vector<at::Tensor>& kv_caches,
                               int64_t node_id, int64_t local_control_port,
                               int64_t max_blocks, int64_t num_slots,
                               double timeout_s, bool unsafe_skip_buffer_lock,
                               std::optional<std::string> local_ip)
    : KVCacheManagerWithTransfer(
          UnpackTorchTensors(SingleShardLayers(kv_caches)),
          /*local_port=*/std::nullopt,
          /*host_blocks_to_allocate=*/std::nullopt, unsafe_skip_buffer_lock,
          /*parallelism=*/1,
          tpu_raiden::CreateHostMemoryAllocator(
              kv_caches.empty()
                  ? nullptr
                  : UnpackTorchTensor(kv_caches[0])->device()->client()),
          node_id, local_control_port, max_blocks, num_slots, timeout_s),
      kv_caches_(kv_caches) {}

KVCacheManager::~KVCacheManager() = default;

}  // namespace torch
}  // namespace tpu_raiden
