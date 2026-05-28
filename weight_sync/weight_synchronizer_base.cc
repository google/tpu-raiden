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

#include "weight_sync/weight_synchronizer_base.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

ABSL_FLAG(size_t, raiden_weight_sync_host_buffer_scratchpad_size, 256 * 1024,
          "Amount of scratchpad to allocate to host buffers for resharding "
          "pulls.");

namespace tpu_raiden {
namespace weight_sync {

WeightSynchronizerBase::WeightSynchronizerBase(
    const std::vector<std::vector<xla::PjRtBuffer*>>& layer_buffers,
    std::optional<int> local_port,
    std::optional<std::vector<const uint8_t*>> external_host_ptrs,
    bool unsafe_skip_buffer_lock, int parallelism)
    : tpu_raiden::RaidenManagerBase(
          layer_buffers.size(),
          layer_buffers.empty() ? 0 : layer_buffers[0].size(),
          layer_buffers.empty()
              ? 0
              : layer_buffers[0][0]->GetOnDeviceSizeInBytes().value(),
          /*block_size=*/1, local_port, parallelism) {
  if (num_layers_ == 0 || num_shards_ == 0) {
    return;
  }

  xla::PjRtBuffer* first_buffer = layer_buffers[0][0];
  physical_size_ = first_buffer->GetOnDeviceSizeInBytes().value();
  extension_ = raiden::GetRawBufferExtension(first_buffer, &c_api_);

  // Symmetrically register single, unified memory blocks representing the
  // entire weights buffer!
  shard_factor_ = 1;
  major_dim_size_ = 1;

  size_t shard_idx = 0;
  layers_.reserve(num_layers_);
  buffer_holds_.reserve(num_layers_);

  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    const auto& dst_buffers = layer_buffers[layer_idx];
    if (dst_buffers.size() != num_shards_) {
      throw std::runtime_error(
          "Number of shards mismatch across layers during weight sync init");
    }

    LayerInfoBase layer_info;
    layer_info.shards.reserve(num_shards_);
    std::vector<raiden::BufferHoldAndAlias> hold_info;
    hold_info.reserve(num_shards_);

    for (size_t i = 0; i < num_shards_; ++i) {
      xla::PjRtBuffer* dst_buffer = dst_buffers[i];
      ShardBufferInfoBase shard_info;

      shard_info.device_size = dst_buffer->GetOnDeviceSizeInBytes().value();
      if (shard_info.device_size < physical_size_) {
        throw std::runtime_error(
            "Device buffer shard size smaller than physical weights size");
      }

      // Since weight sync acts on the entire buffer, the host size maps exactly
      // to the physical size!
      size_t alloc_size =
          physical_size_ +
          absl::GetFlag(FLAGS_raiden_weight_sync_host_buffer_scratchpad_size);
      if (external_host_ptrs.has_value()) {
        if (shard_idx < external_host_ptrs->size()) {
          shard_info.host_ptr = (*external_host_ptrs)[shard_idx];
        } else {
          throw std::invalid_argument("External host pointers size mismatch");
        }
        shard_info.host_size = alloc_size;
        shard_idx++;
      } else {
        void* ptr = nullptr;
        if (alloc_size > 0) {
          if (posix_memalign(&ptr, 64, alloc_size) != 0) {
            throw std::runtime_error("Failed to allocate host weights buffer");
          }
          std::memset(ptr, 0, alloc_size);
        }
        shard_info.owned_host_buffer =
            std::unique_ptr<uint8_t[], void (*)(void*)>(
                static_cast<uint8_t*>(ptr), [](void* p) { free(p); });
        shard_info.host_ptr = shard_info.owned_host_buffer.get();
        shard_info.host_size = alloc_size;
      }

      auto status_or_hold = raiden::BufferHoldAndAlias::Acquire(
          dst_buffer, c_api_, extension_, unsafe_skip_buffer_lock);
      if (!status_or_hold.ok()) {
        throw std::runtime_error(
            std::string("Failed to acquire weights PJRT hold: ") +
            std::string(status_or_hold.status().message()));
      }
      hold_info.push_back(std::move(status_or_hold.value()));
      layer_info.shards.push_back(std::move(shard_info));
    }
    layers_.push_back(std::move(layer_info));
    buffer_holds_.push_back(std::move(hold_info));
  }
}

WeightSynchronizerBase::WeightSynchronizerBase(
    size_t num_layers, size_t num_shards, size_t slice_byte_size,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    int parallelism)
    : tpu_raiden::RaidenManagerBase(num_layers, num_shards, slice_byte_size,
                                    /*block_size=*/1, local_port, parallelism) {
  physical_size_ = slice_byte_size_;
  shard_factor_ = 1;
  major_dim_size_ = 1;

  layers_.reserve(num_layers_);
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    LayerInfoBase layer_info;
    layer_info.shards.reserve(num_shards_);

    for (size_t i = 0; i < num_shards_; ++i) {
      ShardBufferInfoBase shard_info;

      size_t alloc_size =
          physical_size_ +
          absl::GetFlag(FLAGS_raiden_weight_sync_host_buffer_scratchpad_size);
      std::cerr << "[C++ WS] CPU-only constructor: physical_size_="
                << physical_size_ << ", alloc_size=" << alloc_size << std::endl;
      void* ptr = nullptr;
      if (alloc_size > 0) {
        if (posix_memalign(&ptr, 64, alloc_size) != 0) {
          throw std::runtime_error("Failed to allocate host weights buffer");
        }
        std::memset(ptr, 0, alloc_size);
      }
      shard_info.owned_host_buffer =
          std::unique_ptr<uint8_t[], void (*)(void*)>(
              static_cast<uint8_t*>(ptr), [](void* p) { free(p); });
      shard_info.host_ptr = shard_info.owned_host_buffer.get();
      shard_info.host_size = alloc_size;

      layer_info.shards.push_back(std::move(shard_info));
    }
    layers_.push_back(std::move(layer_info));
  }
}

WeightSynchronizerBase::~WeightSynchronizerBase() = default;

absl::StatusOr<raiden::PjRtCopyFuture> WeightSynchronizerBase::H2d() {
  if (buffer_holds_.empty()) {
    return raiden::PjRtCopyFuture({});
  }
  raiden::PjRtCopyFuture acc({});
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    const auto& layer_info = layers_[layer_idx];
    const auto& layer_holds = buffer_holds_[layer_idx];
    for (size_t i = 0; i < num_shards_; ++i) {
      const auto& shard_info = layer_info.shards[i];
      const auto& shard_hold = layer_holds[i];

      std::vector<xla::Future<>> shard_futures;
      xla::Future<> future = shard_hold.CopyRawHostToDevice(shard_info.host_ptr,
                                                            0, physical_size_);
      shard_futures.push_back(std::move(future));
      acc.Append(std::move(shard_futures), shard_hold);
    }
  }
  return acc;
}

absl::StatusOr<raiden::PjRtCopyFuture> WeightSynchronizerBase::H2dChunk(
    size_t shard_idx, size_t host_offset_bytes, size_t device_offset_bytes,
    size_t size_bytes) {
  if (buffer_holds_.empty()) {
    return raiden::PjRtCopyFuture({});
  }
  if (shard_idx >= num_shards_) {
    return absl::InvalidArgumentError("Invalid shard index");
  }
  const auto& layer_info = layers_[0];
  const auto& layer_holds = buffer_holds_[0];
  const auto& shard_info = layer_info.shards[shard_idx];
  const auto& shard_hold = layer_holds[shard_idx];

  const uint8_t* src_ptr = shard_info.host_ptr + host_offset_bytes;

  std::vector<xla::Future<>> shard_futures;
  xla::Future<> future = shard_hold.CopyRawHostToDevice(
      const_cast<uint8_t*>(src_ptr), device_offset_bytes, size_bytes
  );
  shard_futures.push_back(std::move(future));

  raiden::PjRtCopyFuture acc({});
  acc.Append(std::move(shard_futures), shard_hold);
  return acc;
}

absl::StatusOr<raiden::PjRtCopyFuture> WeightSynchronizerBase::D2h() {
  if (buffer_holds_.empty()) {
    return raiden::PjRtCopyFuture({});
  }
  raiden::PjRtCopyFuture acc({});
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    const auto& layer_info = layers_[layer_idx];
    const auto& layer_holds = buffer_holds_[layer_idx];
    for (size_t i = 0; i < num_shards_; ++i) {
      const auto& shard_info = layer_info.shards[i];
      const auto& shard_hold = layer_holds[i];
      uint8_t* dst_host_ptr = const_cast<uint8_t*>(shard_info.host_ptr);

      std::vector<xla::Future<>> shard_futures;
      xla::Future<> future =
          shard_hold.CopyRawDeviceToHost(dst_host_ptr, 0, physical_size_);
      shard_futures.push_back(std::move(future));
      acc.Append(std::move(shard_futures), shard_hold);
    }
  }
  return acc;
}

absl::Status WeightSynchronizerBase::PushWeights(
    const std::vector<std::string>& peers) {
  if (peers.empty()) {
    return absl::InvalidArgumentError(
        "Peer list cannot be empty for trainer weights sync");
  }

  // 1. Automatically copy latest weights from Device TPU HBM onto Host CPU!
  TF_ASSIGN_OR_RETURN(raiden::PjRtCopyFuture d2h_future, D2h());
  d2h_future.Await();

  // 2. Run high-speed parallel sockets H2H write to push host weights to peers!
  std::vector<int> weights_block_id = {0};
  for (const std::string& peer : peers) {
    TF_RETURN_IF_ERROR(
        H2hWriteDirect(peer, weights_block_id, /*entity_id=*/0).status());
  }
  return absl::OkStatus();
}

absl::Status WeightSynchronizerBase::PullWeights(const std::string& source) {
  if (source.empty()) {
    return absl::InvalidArgumentError(
        "Source peer address cannot be empty for weight pull");
  }

  // 1. Run high-speed parallel sockets H2H read to pull weights from source!
  std::vector<int> weights_block_id = {0};
  TF_RETURN_IF_ERROR(
      H2hReadDirect(source, weights_block_id, /*entity_id=*/0).status());

  // 2. Automatically copy the received staging weights onto local Device TPU
  // HBM!
  TF_ASSIGN_OR_RETURN(raiden::PjRtCopyFuture h2d_future, H2d());
  h2d_future.Await();

  return absl::OkStatus();
}

void WeightSynchronizerBase::SetExternalHostBuffer(
    const std::vector<raiden::BufferHoldAndAlias>& buffer_holds) {
  size_t idx = 0;
  for (size_t l = 0; l < num_layers_; ++l) {
    for (size_t sh = 0; sh < num_shards_; ++sh) {
      if (idx < buffer_holds.size()) {
        auto u_ptr_or = buffer_holds[idx].buffer->client()->UnsafeBufferPointer(
            buffer_holds[idx].buffer);
        if (u_ptr_or.ok()) {
          layers_[l].shards[sh].host_ptr =
              reinterpret_cast<uint8_t*>(u_ptr_or.value());
          layers_[l].shards[sh].host_size =
              buffer_holds[idx].buffer->GetOnDeviceSizeInBytes().value();
        }
        idx++;
      }
    }
  }
}

absl::Status WeightSynchronizerBase::OnDataReceived() {
  // Automatically copy all received staging weights from Host onto Device TPU
  // HBM!
  TF_ASSIGN_OR_RETURN(raiden::PjRtCopyFuture h2d_future, H2d());
  h2d_future.Await();
  return absl::OkStatus();
}

}  // namespace weight_sync
}  // namespace tpu_raiden
