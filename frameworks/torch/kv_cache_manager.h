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

#ifndef THIRD_PARTY_TPU_RAIDEN_API_TORCH_KV_CACHE_MANAGER_H_
#define THIRD_PARTY_TPU_RAIDEN_API_TORCH_KV_CACHE_MANAGER_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_client.h"
#include "kv_cache/kv_cache_manager_base.h"

namespace tpu_raiden {
namespace torch {

class KVCacheManager : public kv_cache::KVCacheManagerBase {
 public:
  // PyTorch sharded constructor E2E
  KVCacheManager(
      const std::vector<std::vector<at::Tensor>>& device_tensors,
      int block_size = 1, std::optional<int> local_port = std::nullopt,
      std::optional<int> host_blocks_to_allocate = std::nullopt,
      std::optional<std::vector<uintptr_t>> external_host_ptrs = std::nullopt,
      bool unsafe_skip_buffer_lock = false, int parallelism = 1);

  ~KVCacheManager() override;
};

}  // namespace torch
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_API_TORCH_KV_CACHE_MANAGER_H_
