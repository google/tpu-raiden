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

#ifndef THIRD_PARTY_TPU_RAIDEN_FRAMEWORKS_TORCH_RAIDEN_TRANSFER_ENGINE_H_
#define THIRD_PARTY_TPU_RAIDEN_FRAMEWORKS_TORCH_RAIDEN_TRANSFER_ENGINE_H_

#include <memory>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "core/transfer_engine_base.h"
#include "kv_cache/kv_cache_manager_base.h"

namespace tpu_raiden::torch {

class RaidenTransferEngine : public TransferEngineBase {
 public:
  RaidenTransferEngine(const std::vector<at::Tensor>& kv_caches,
                       int64_t tp_rank, int64_t local_control_port,
                       int64_t max_blocks, int64_t num_slots, double timeout_s,
                       bool unsafe_skip_buffer_lock);

  ~RaidenTransferEngine() override = default;

  std::vector<int64_t> RegisterKvCache(
      const std::vector<at::Tensor>& kv_caches);

  void RegisterHostBuffers(int64_t tp_rank);

  const std::vector<at::Tensor>& kv_caches() const { return kv_caches_; }

 private:
  static std::unique_ptr<kv_cache::KVCacheManagerBase> CreateKvCacheManager(
      const std::vector<at::Tensor>& kv_caches, int64_t num_slots,
      int64_t max_blocks, bool unsafe_skip_buffer_lock);

  std::vector<at::Tensor> kv_caches_;
};

}  // namespace tpu_raiden::torch

#endif  // THIRD_PARTY_TPU_RAIDEN_FRAMEWORKS_TORCH_RAIDEN_TRANSFER_ENGINE_H_
