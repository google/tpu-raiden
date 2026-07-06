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
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "xla/pjrt/pjrt_client.h"
#include "tpu_raiden/core/kv_cache_manager_with_transfer.h"
#include "tpu_raiden/core/utils.h"
#include "tpu_raiden/frameworks/torch/torch_utils.h"
#include "tpu_raiden/kv_cache/kv_cache_listener.h"
#include "tpu_raiden/kv_cache/kv_cache_manager_base.h"

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

std::string FormatAddressWithPort(absl::string_view ip, int port) {
  if (absl::StrContains(ip, ':')) {
    return absl::StrCat("[", ip, "]:", port);
  }
  return absl::StrCat(ip, ":", port);
}

}  // namespace

KVCacheManager::UnpackedLayers KVCacheManager::UnpackLayers(
    const std::vector<std::vector<at::Tensor>>& device_tensors) {
  // Retain the owning DeviceBufferRefs: for view tensors the materialized
  // buffers are fresh allocations owned solely by these refs, so they must
  // outlive every D2h/H2d the manager dispatches.
  UnpackedTensors u = UnpackTorchTensors(device_tensors);
  UnpackedLayers unpacked;
  unpacked.buffers = std::move(u.buffers);
  unpacked.refs = std::move(u.refs);
  if (!unpacked.buffers.empty() && !unpacked.buffers[0].empty()) {
    unpacked.client = unpacked.buffers[0][0]->device()->client();
  }
  return unpacked;
}

KVCacheManager::KVCacheManager(
    const std::vector<std::vector<at::Tensor>>& device_tensors,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    bool unsafe_skip_buffer_lock, int parallelism)
    : KVCacheManager(UnpackLayers(device_tensors), local_port,
                     host_blocks_to_allocate, unsafe_skip_buffer_lock,
                     parallelism, /*node_id=*/0, /*local_control_port=*/-1,
                     /*max_blocks=*/0, /*num_slots=*/0, /*timeout_s=*/120.0,
                     /*kv_caches=*/{}) {}

KVCacheManager::KVCacheManager(
    UnpackedLayers unpacked, std::optional<int> local_port,
    std::optional<int> host_blocks_to_allocate, bool unsafe_skip_buffer_lock,
    int parallelism, int64_t node_id, int64_t local_control_port,
    int64_t max_blocks, int64_t num_slots, double timeout_s,
    std::vector<at::Tensor> kv_caches, const kv_cache::BufferSpec& buffer_spec)
    : KVCacheManagerWithTransfer(
          std::move(unpacked.buffers), local_port, host_blocks_to_allocate,
          unsafe_skip_buffer_lock, parallelism,
          tpu_raiden::CreateHostMemoryAllocator(
              unpacked.client, max_blocks,
              (unpacked.buffers.empty() || unpacked.buffers[0].empty() ||
               unpacked.buffers[0][0] == nullptr)
                  ? 0
                  : unpacked.buffers[0][0]->GetOnDeviceSizeInBytes().value_or(
                        0)),
          node_id, local_control_port, max_blocks, num_slots, timeout_s,
          /*metrics_collector=*/nullptr, buffer_spec),
      kv_caches_(std::move(kv_caches)),
      // Move the keep-alive refs in AFTER the base ctor has acquired the
      // buffers; they pin the materialized device buffers for our lifetime.
      buffer_refs_(std::move(unpacked.refs)) {}

KVCacheManager::KVCacheManager(const std::vector<at::Tensor>& kv_caches,
                               int64_t node_id, int64_t local_control_port,
                               int64_t max_blocks, int64_t num_slots,
                               double timeout_s, bool unsafe_skip_buffer_lock,
                               int parallelism,
                               std::optional<int> listener_port,
                               const kv_cache::BufferSpec& buffer_spec)
    : KVCacheManager(UnpackLayers(SingleShardLayers(kv_caches)),
                     /*local_port=*/std::nullopt,
                     /*host_blocks_to_allocate=*/std::nullopt,
                     unsafe_skip_buffer_lock, parallelism, node_id,
                     local_control_port, max_blocks, num_slots, timeout_s,
                     kv_caches, buffer_spec) {
  if (listener_port) {
    listener_ = std::make_unique<tpu_raiden::kv_cache::KVCacheListener>(
        this, *listener_port);
  }
}

KVCacheManager::~KVCacheManager() = default;

std::optional<int> KVCacheManager::listener_port() const {
  if (listener_) {
    return listener_->listener_port();
  }
  return std::nullopt;
}

bool KVCacheManager::is_listener_active() const {
  if (listener_) {
    return listener_->is_active();
  }
  return false;
}

std::string KVCacheManager::transfer_address() const {
  auto port = local_port();
  if (!port.has_value()) return "";
  return FormatAddressWithPort(local_ip(), *port);
}

std::string KVCacheManager::listener_address() const {
  auto port = listener_port();
  if (!port.has_value()) return "";
  return FormatAddressWithPort(local_ip(), *port);
}

}  // namespace torch
}  // namespace tpu_raiden
