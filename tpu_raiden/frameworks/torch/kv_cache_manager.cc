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
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_client.h"
#include "tpu_raiden/core/kv_cache_manager_with_transfer.h"
#include "tpu_raiden/core/utils.h"
#include "tpu_raiden/frameworks/torch/torch_utils.h"
#include "tpu_raiden/kv_cache/kv_cache_listener.h"

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
  unpacked.logical_dimensions = std::move(u.logical_dimensions);
  unpacked.logical_slice_byte_size = u.logical_slice_byte_size;
  unpacked.logical_physical_size = u.logical_physical_size;
  unpacked.has_logical_metadata = u.has_logical_metadata;
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

KVCacheManager::KVCacheManager(UnpackedLayers unpacked,
                               std::optional<int> local_port,
                               std::optional<int> host_blocks_to_allocate,
                               bool unsafe_skip_buffer_lock, int parallelism,
                               int64_t node_id, int64_t local_control_port,
                               int64_t max_blocks, int64_t num_slots,
                               double timeout_s,
                               std::vector<at::Tensor> kv_caches)
    : KVCacheManagerWithTransfer(
          unpacked.buffers,
          unpacked.has_logical_metadata ? unpacked.logical_slice_byte_size : 0,
          unpacked.has_logical_metadata ? unpacked.logical_dimensions
                                        : std::vector<int64_t>{},
          unpacked.has_logical_metadata ? unpacked.logical_physical_size : 0,
          local_port, host_blocks_to_allocate, unsafe_skip_buffer_lock,
          parallelism,
          tpu_raiden::CreateHostMemoryAllocator(
              unpacked.client, max_blocks,
              (unpacked.buffers.empty() || unpacked.buffers[0].empty() ||
               unpacked.buffers[0][0] == nullptr)
                  ? 0
                  : unpacked.buffers[0][0]->GetOnDeviceSizeInBytes().value_or(
                        0)),
          node_id, local_control_port, max_blocks, num_slots, timeout_s),
      kv_caches_(std::move(kv_caches)),
      // Move the keep-alive refs in AFTER the base ctor has acquired the
      // buffers; they pin the materialized device buffers for our lifetime.
      buffer_refs_(std::move(unpacked.refs)) {}

KVCacheManager::KVCacheManager(const std::vector<at::Tensor>& kv_caches,
                               int64_t node_id, int64_t local_control_port,
                               int64_t max_blocks, int64_t num_slots,
                               double timeout_s, bool unsafe_skip_buffer_lock,
                               int parallelism,
                               std::optional<int> listener_port)
    : KVCacheManager(UnpackLayers(SingleShardLayers(kv_caches)),
                     /*local_port=*/std::nullopt,
                     /*host_blocks_to_allocate=*/std::nullopt,
                     unsafe_skip_buffer_lock, parallelism, node_id,
                     local_control_port, max_blocks, num_slots, timeout_s,
                     kv_caches) {
  if (listener_port) {
    listener_ = std::make_unique<tpu_raiden::kv_cache::KVCacheListener>(
        this, *listener_port);
  }
}

KVCacheManager::KVCacheManager(size_t num_layers, size_t num_shards,
                               size_t slice_byte_size, int64_t node_id,
                               std::optional<int> local_port,
                               std::optional<int> host_blocks_to_allocate,
                               int parallelism)
    : KVCacheManagerWithTransfer(
          num_layers, num_shards, slice_byte_size, local_port,
          host_blocks_to_allocate, parallelism, node_id,
          /*local_control_port=*/-1,
          /*max_blocks=*/host_blocks_to_allocate.value_or(0),
          /*num_slots=*/0, /*timeout_s=*/120.0),
      kv_caches_({}) {}

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

absl::Status KVCacheManager::PushRegisteredPlan(
    uint64_t uuid, const std::string& peer,
    const std::vector<int>& src_block_ids,
    const std::vector<int>& dst_block_ids, int layer_idx, int parallelism) {
  if (peer.empty()) {
    return absl::InvalidArgumentError("peer must not be empty");
  }
  if (src_block_ids.empty()) {
    return absl::InvalidArgumentError("src_block_ids must not be empty");
  }
  if (src_block_ids.size() != dst_block_ids.size()) {
    return absl::InvalidArgumentError(
        "src_block_ids and dst_block_ids must have the same length");
  }
  InitTransportServer();
  // Copy the transport pointer and release server_init_mu_ before the blocking
  // Push: holding the lock across it serializes concurrent pushes from the
  // same manager (one per destination peer).
  tpu_raiden::transport::BlockTransport* transport = nullptr;
  {
    absl::MutexLock lock(server_init_mu_);
    transport = server_.get();
  }
  if (!transport) {
    return absl::FailedPreconditionError("Transport server is not running");
  }
  auto status_or = transport->Push(
      peer, src_block_ids, dst_block_ids, parallelism,
      tpu_raiden::transport::MajorOrder::kLayerMajor, uuid, layer_idx);
  if (!status_or.ok()) {
    return status_or.status();
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> KVCacheManager::ReadBlockBytes(size_t layer_idx,
                                                           int block_id,
                                                           size_t shard_idx) {
  if (block_id < 0) {
    return absl::InvalidArgumentError("block_id must be non-negative");
  }
  const size_t block_bytes = bytes_per_block();
  const size_t host_size = GetHostSize(layer_idx, shard_idx);
  const uint8_t* base = GetHostPointer(layer_idx, shard_idx);
  if (base == nullptr) {
    return absl::OutOfRangeError("layer or shard index out of range");
  }
  const size_t block = static_cast<size_t>(block_id);
  if (block_bytes == 0 || block > host_size / block_bytes ||
      block * block_bytes + block_bytes > host_size) {
    return absl::OutOfRangeError("block range exceeds host buffer");
  }
  const char* ptr = reinterpret_cast<const char*>(base + block * block_bytes);
  return std::string(ptr, block_bytes);
}

absl::Status KVCacheManager::WriteBlockBytes(size_t layer_idx, int block_id,
                                             absl::string_view payload,
                                             size_t shard_idx) {
  if (block_id < 0) {
    return absl::InvalidArgumentError("block_id must be non-negative");
  }
  const size_t block_bytes = bytes_per_block();
  if (payload.size() != block_bytes) {
    return absl::InvalidArgumentError(
        absl::StrCat("payload size must equal block size: got ", payload.size(),
                     ", expected ", block_bytes));
  }
  const size_t host_size = GetHostSize(layer_idx, shard_idx);
  uint8_t* base = GetHostPointer(layer_idx, shard_idx);
  if (base == nullptr) {
    return absl::OutOfRangeError("layer or shard index out of range");
  }
  const size_t block = static_cast<size_t>(block_id);
  if (block_bytes == 0 || block > host_size / block_bytes ||
      block * block_bytes + block_bytes > host_size) {
    return absl::OutOfRangeError("block range exceeds host buffer");
  }
  std::memcpy(base + block * block_bytes, payload.data(), block_bytes);
  return absl::OkStatus();
}

}  // namespace torch
}  // namespace tpu_raiden
