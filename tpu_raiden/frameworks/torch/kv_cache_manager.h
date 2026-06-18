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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_API_TORCH_KV_CACHE_MANAGER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_API_TORCH_KV_CACHE_MANAGER_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "tpu_raiden/core/kv_cache_manager_with_transfer.h"

namespace tpu_raiden {
namespace torch {

class KVCacheManager : public KVCacheManagerWithTransfer {
 public:
  // PyTorch sharded constructor E2E (cache-only by default)
  KVCacheManager(
      const std::vector<std::vector<at::Tensor>>& device_tensors,
      std::optional<int> local_port = std::nullopt,
      std::optional<int> host_blocks_to_allocate = std::nullopt,
      bool unsafe_skip_buffer_lock = false, int parallelism = 1);

  // New transfer-enabled constructor (flat list of tensors, single shard per
  // layer)
  KVCacheManager(const std::vector<at::Tensor>& kv_caches, int64_t node_id,
                 int64_t local_control_port, int64_t max_blocks,
                 int64_t num_slots, double timeout_s,
                 bool unsafe_skip_buffer_lock);

  ~KVCacheManager() override;

  // Transfer-specific methods merged from TransferEngine
  std::vector<int64_t> RegisterKvCache(
      const std::vector<at::Tensor>& kv_caches);

  void RegisterHostBuffers(int64_t node_id);

  const std::vector<at::Tensor>& kv_caches() const { return kv_caches_; }

 private:
  std::vector<at::Tensor> kv_caches_;
};

}  // namespace torch
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_API_TORCH_KV_CACHE_MANAGER_H_
