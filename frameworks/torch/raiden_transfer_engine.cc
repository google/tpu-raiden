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

#include "frameworks/torch/raiden_transfer_engine.h"

#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "core/transfer_engine_base.h"
#include "frameworks/torch/kv_cache_manager.h"
#include "kv_cache/kv_cache_manager_base.h"

namespace tpu_raiden::torch {
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

RaidenTransferEngine::RaidenTransferEngine(const TensorList& kv_caches,
                                           int64_t tp_rank,
                                           int64_t local_control_port,
                                           int64_t max_blocks,
                                           int64_t num_slots, double timeout_s,
                                           bool unsafe_skip_buffer_lock)
    : TransferEngineBase(CreateKvCacheManager(kv_caches, num_slots, max_blocks,
                                              unsafe_skip_buffer_lock),
                         tp_rank, local_control_port, max_blocks, num_slots,
                         timeout_s, unsafe_skip_buffer_lock),
      kv_caches_(kv_caches) {}

std::vector<int64_t> RaidenTransferEngine::RegisterKvCache(
    const TensorList& kv_caches) {
  kv_caches_ = kv_caches;
  if (num_slots_ <= 0) {
    throw std::invalid_argument("num_slots must be configured first");
  }
  if (max_blocks_ >
      std::numeric_limits<int>::max() / std::max<int64_t>(num_slots_, 1)) {
    throw std::invalid_argument("host staging block count exceeds int range");
  }

  bool has_tpu = false;
  for (const auto& t : kv_caches_) {
#ifdef RAIDEN_TORCH_USE_MOCKS
    if (t.device().type() == c10::DeviceType::PrivateUse1 ||
        t.device().type() == c10::DeviceType::CPU) {
#else
    if (t.device().type() == c10::DeviceType::PrivateUse1) {
#endif
      has_tpu = true;
      break;
    }
  }

  if (has_tpu) {
    const int host_blocks_to_allocate =
        static_cast<int>(num_slots_ * max_blocks_);
    kv_transfer_ = std::make_unique<tpu_raiden::torch::KVCacheManager>(
        SingleShardLayers(kv_caches_), /*block_size=*/1,
        /*local_port=*/std::nullopt, host_blocks_to_allocate,
        /*external_host_ptrs=*/std::nullopt, unsafe_skip_buffer_lock_);
    auto status =
        kv_transfer_->ConfigureHostStagingSlots(num_slots_, max_blocks_);
    if (!status.ok()) {
      throw std::runtime_error("Failed to configure KVCacheManager: " +
                               std::string(status.message()));
    }
    ConfigureDataPortFromKvTransfer();
  } else {
    kv_transfer_ = nullptr;
    local_data_port_ = 0;
  }
  InitializeSlotPool(num_slots_);

  std::vector<int64_t> region_ids;
  region_ids.reserve(kv_caches_.size());
  for (size_t i = 0; i < kv_caches_.size(); ++i) {
    region_ids.push_back(static_cast<int64_t>(i));
  }
  return region_ids;
}

void RaidenTransferEngine::RegisterHostBuffers(int64_t tp_rank) {
  tp_rank_ = tp_rank;
}

std::unique_ptr<kv_cache::KVCacheManagerBase>
RaidenTransferEngine::CreateKvCacheManager(const TensorList& kv_caches,
                                           int64_t num_slots,
                                           int64_t max_blocks,
                                           bool unsafe_skip_buffer_lock) {
  bool has_tpu = false;
  for (const auto& t : kv_caches) {
#ifdef RAIDEN_TORCH_USE_MOCKS
    if (t.device().type() == c10::DeviceType::PrivateUse1 ||
        t.device().type() == c10::DeviceType::CPU) {
#else
    if (t.device().type() == c10::DeviceType::PrivateUse1) {
#endif
      has_tpu = true;
      break;
    }
  }
  if (has_tpu) {
    const int host_blocks_to_allocate =
        static_cast<int>(num_slots * max_blocks);
    auto kv_transfer = std::make_unique<tpu_raiden::torch::KVCacheManager>(
        SingleShardLayers(kv_caches), /*block_size=*/1,
        /*local_port=*/std::nullopt, host_blocks_to_allocate,
        /*external_host_ptrs=*/std::nullopt, unsafe_skip_buffer_lock);
    auto status = kv_transfer->ConfigureHostStagingSlots(num_slots, max_blocks);
    if (!status.ok()) {
      throw std::runtime_error("Failed to configure KVCacheManager: " +
                               std::string(status.message()));
    }
    return kv_transfer;
  }
  return nullptr;
}
}  // namespace tpu_raiden::torch
