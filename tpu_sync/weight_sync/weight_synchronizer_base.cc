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

#include "tpu_sync/weight_sync/weight_synchronizer_base.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/core/raiden_manager_base.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "tpu_sync/transport/buffer_push_task.h"
#include "tpu_sync/weight_sync/tiling_utils.h"
#include "tpu_sync/weight_sync/weight_synchronizer_listener.h"

ABSL_FLAG(size_t, raiden_weight_sync_host_buffer_scratchpad_size, 256 * 1024,
          "Amount of scratchpad to allocate to host buffers for resharding "
          "pulls.");

namespace tpu_raiden {
namespace weight_sync {

WeightSynchronizerBase::WeightSynchronizerBase(
    const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers,
    std::optional<int> local_port,
    std::optional<std::vector<const uint8_t*>> external_host_ptrs,
    bool unsafe_skip_buffer_lock, int parallelism,
    std::optional<int> listener_port, std::optional<std::string> bind_ip,
    std::vector<std::string> layer_names, bool auto_h2d)
    : tpu_raiden::RaidenManagerBase(
          layer_buffers.size(),
          layer_buffers.empty() ? 0 : layer_buffers[0].size(),
          layer_buffers.empty() ? 0
                                : layer_buffers[0][0].GetOnDeviceSizeInBytes(),
          local_port, parallelism, bind_ip),
      auto_h2d_(auto_h2d) {
  if (layer_names.empty()) {
    layer_names_.reserve(num_layers_);
    for (size_t i = 0; i < num_layers_; ++i) {
      layer_names_.push_back(absl::StrCat("weights_", i));
    }
  } else {
    layer_names_ = std::move(layer_names);
  }
  DetectAndAssignNumaNode(layer_buffers);

  if (num_layers_ == 0 || num_shards_ == 0) {
    return;
  }

  const auto& first_handle = layer_buffers[0][0];
  physical_size_ = first_handle.GetOnDeviceSizeInBytes();

  if (!first_handle.is_common_buffer && first_handle.c_hold) {
    c_api_ = first_handle.c_hold->c_api;
    extension_ = first_handle.c_hold->extension;
  }

  shard_factor_ = 1;
  major_dim_size_ = 1;

  xla::PjRtClient* client = nullptr;
  if (first_handle.buffer) {
    client = const_cast<xla::PjRtClient*>(first_handle.buffer->client());
  } else if (first_handle.device) {
    client = const_cast<xla::PjRtClient*>(first_handle.device->client());
  }

  std::unique_ptr<HostMemoryAllocator> host_allocator;
  if (client) {
    auto alloc = HostMemoryAllocator::Create(client);
    if (alloc.ok()) {
      host_allocator = *std::move(alloc);
    }
  }

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
      const auto& dst_buffer = dst_buffers[i];
      ShardBufferInfoBase shard_info;

      shard_info.device_size = dst_buffer.GetOnDeviceSizeInBytes();

      size_t alloc_size =
          shard_info.device_size +
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
        if (alloc_size > 0) {
          if (host_allocator && dst_buffer.device) {
            auto alloc = host_allocator->AllocateDmaMappedForDevice(
                alloc_size, dst_buffer.device);
            if (alloc.ok()) {
              shard_info.host_ptr = (*alloc).ptr;
              shard_info.host_size = alloc_size;
              shard_info.host_owner = (*alloc).owner;
            }
          }
          if (shard_info.host_ptr == nullptr) {
            void* ptr = nullptr;
            if (posix_memalign(&ptr, 64, alloc_size) != 0) {
              throw std::runtime_error(
                  "Failed to allocate host weights buffer");
            }
            std::memset(ptr, 0, alloc_size);
            shard_info.owned_host_buffer =
                std::unique_ptr<uint8_t[], void (*)(void*)>(
                    static_cast<uint8_t*>(ptr), [](void* p) { free(p); });
            shard_info.host_ptr = shard_info.owned_host_buffer.get();
            shard_info.host_size = alloc_size;
          }
        }
      }

      hold_info.push_back(dst_buffer);
      layer_info.shards.push_back(std::move(shard_info));
    }
    layers_.push_back(std::move(layer_info));
    buffer_holds_.push_back(std::move(hold_info));
  }

  if (listener_port) {
    listener_ =
        std::make_unique<WeightSynchronizerListener>(this, *listener_port);
  }
}

WeightSynchronizerBase::WeightSynchronizerBase(
    size_t num_layers, size_t num_shards, size_t slice_byte_size,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    int parallelism, std::optional<int> listener_port,
    std::optional<std::string> bind_ip, std::vector<std::string> layer_names,
    bool auto_h2d)
    : WeightSynchronizerBase(num_layers, num_shards,
                             std::vector<size_t>(num_layers, slice_byte_size),
                             local_port, host_blocks_to_allocate, parallelism,
                             listener_port, bind_ip, std::move(layer_names),
                             auto_h2d) {}

WeightSynchronizerBase::WeightSynchronizerBase(
    size_t num_layers, size_t num_shards, std::vector<size_t> slice_byte_sizes,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    int parallelism, std::optional<int> listener_port,
    std::optional<std::string> bind_ip, std::vector<std::string> layer_names,
    bool auto_h2d)
    : tpu_raiden::RaidenManagerBase(
          num_layers, num_shards,
          slice_byte_sizes.empty() ? 0 : slice_byte_sizes[0], local_port,
          parallelism, bind_ip),
      auto_h2d_(auto_h2d) {
  if (layer_names.empty()) {
    layer_names_.reserve(num_layers_);
    for (size_t i = 0; i < num_layers_; ++i) {
      layer_names_.push_back(absl::StrCat("weights_", i));
    }
  } else {
    layer_names_ = std::move(layer_names);
  }
  physical_size_ = slice_byte_size_;
  shard_factor_ = 1;
  major_dim_size_ = 1;

  layers_.reserve(num_layers_);
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    size_t current_slice_byte_size = (layer_idx < slice_byte_sizes.size())
                                         ? slice_byte_sizes[layer_idx]
                                         : slice_byte_size_;
    LayerInfoBase layer_info;
    layer_info.shards.reserve(num_shards_);

    for (size_t i = 0; i < num_shards_; ++i) {
      ShardBufferInfoBase shard_info;
      shard_info.device_size = current_slice_byte_size;

      size_t alloc_size =
          current_slice_byte_size +
          absl::GetFlag(FLAGS_raiden_weight_sync_host_buffer_scratchpad_size);
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

  if (listener_port) {
    listener_ =
        std::make_unique<WeightSynchronizerListener>(this, *listener_port);
  }
}

std::optional<int> WeightSynchronizerBase::listener_port() const {
  if (listener_) {
    return listener_->listener_port();
  }
  return std::nullopt;
}

bool WeightSynchronizerBase::is_listener_active() const {
  if (listener_) {
    return listener_->is_active();
  }
  return false;
}

WeightSynchronizerBase::~WeightSynchronizerBase() = default;

absl::StatusOr<raiden::PjRtCopyFuture> WeightSynchronizerBase::H2d(
    uint64_t uuid) {
  if (buffer_holds_.empty()) {
    return raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{});
  }
  std::vector<bool> active_skip;
  {
    absl::MutexLock lock(skip_tiling_mu_);
    if (uuid != 0) {
      auto it = uuid_to_skip_tiling_.find(uuid);
      if (it != uuid_to_skip_tiling_.end()) {
        active_skip = it->second;
      }
    }
    if (active_skip.empty()) {
      if (!latest_skip_tiling_.empty()) {
        active_skip = latest_skip_tiling_;
      } else {
        active_skip = std::vector<bool>(num_layers_, false);
      }
    }
  }

  std::vector<xla::Future<raiden::BufferHolder>> shard_futures_to_join;
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    const auto& layer_info = layers_[layer_idx];
    const auto& layer_holds = buffer_holds_[layer_idx];
    size_t layer_size = block_bytes(layer_idx);
    for (size_t i = 0; i < num_shards_; ++i) {
      const auto& shard_info = layer_info.shards[i];
      const auto& shard_hold = layer_holds[i];

      const xla::Layout* xla_layout = nullptr;
      if (shard_hold.shape.has_layout()) {
        xla_layout = &shard_hold.shape.layout();
      }
      bool is_tiled = xla_layout && !xla_layout->tiles().empty();
      if (layer_idx < active_skip.size() && active_skip[layer_idx]) {
        is_tiled = false;
      }

      std::vector<xla::Future<>> shard_futures;
      if (is_tiled) {
        auto temp_buffer = std::make_shared<std::vector<uint8_t>>(layer_size);
        auto status = tpu_raiden::weight_sync::TileBuffer(
            shard_info.host_ptr, temp_buffer->data(), shard_hold.shape,
            *xla_layout);
        if (!status.ok()) {
          return status;
        }

        xla::Future<> future =
            shard_hold.CopyRawHostToDevice(temp_buffer->data(), 0, layer_size);
        xla::Future<> mapped_future = future.Map([temp_buffer]() {});
        shard_futures.push_back(std::move(mapped_future));
      } else {
        xla::Future<> future =
            shard_hold.CopyRawHostToDevice(shard_info.host_ptr, 0, layer_size);
        shard_futures.push_back(std::move(future));
      }
      shard_futures_to_join.push_back(raiden::CreateBufferFuture(
          std::move(shard_futures), shard_hold.c_hold, shard_hold.common_hold));
    }
  }
  return raiden::PjRtCopyFuture::FromFuture(
      xla::JoinFutures(absl::MakeSpan(shard_futures_to_join)));
}

absl::StatusOr<raiden::PjRtCopyFuture> WeightSynchronizerBase::D2h(
    uint64_t uuid) {
  if (buffer_holds_.empty()) {
    return raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{});
  }
  std::vector<bool> active_skip;
  {
    absl::MutexLock lock(skip_tiling_mu_);
    if (uuid != 0) {
      auto it = uuid_to_skip_tiling_.find(uuid);
      if (it != uuid_to_skip_tiling_.end()) {
        active_skip = it->second;
      }
    }
    if (active_skip.empty()) {
      if (!latest_skip_tiling_.empty()) {
        active_skip = latest_skip_tiling_;
      } else {
        active_skip = std::vector<bool>(num_layers_, false);
      }
    }
  }

  std::vector<xla::Future<raiden::BufferHolder>> shard_futures_to_join;
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    const auto& layer_info = layers_[layer_idx];
    const auto& layer_holds = buffer_holds_[layer_idx];
    size_t layer_size = block_bytes(layer_idx);
    for (size_t i = 0; i < num_shards_; ++i) {
      const auto& shard_info = layer_info.shards[i];
      const auto& shard_hold = layer_holds[i];
      uint8_t* dst_host_ptr = const_cast<uint8_t*>(shard_info.host_ptr);

      const xla::Layout* xla_layout = nullptr;
      if (shard_hold.shape.has_layout()) {
        xla_layout = &shard_hold.shape.layout();
      }
      bool is_tiled = xla_layout && !xla_layout->tiles().empty();
      if (layer_idx < active_skip.size() && active_skip[layer_idx]) {
        is_tiled = false;
      }

      std::vector<xla::Future<>> shard_futures;
      if (is_tiled) {
        auto temp_buffer = std::make_shared<std::vector<uint8_t>>(layer_size);
        uint8_t* temp_buffer_ptr = temp_buffer->data();

        xla::Future<> copy_future =
            shard_hold.CopyRawDeviceToHost(temp_buffer_ptr, 0, layer_size);

        xla::Future<> detile_future = copy_future.Map(
            [temp_buffer, dst_host_ptr, shape = shard_hold.shape,
             layout = *xla_layout]() -> absl::Status {
              return tpu_raiden::weight_sync::DetileBuffer(
                  temp_buffer->data(), dst_host_ptr, shape, layout);
            });

        shard_futures.push_back(std::move(detile_future));
      } else {
        xla::Future<> future =
            shard_hold.CopyRawDeviceToHost(dst_host_ptr, 0, layer_size);
        shard_futures.push_back(std::move(future));
      }
      shard_futures_to_join.push_back(raiden::CreateBufferFuture(
          std::move(shard_futures), shard_hold.c_hold, shard_hold.common_hold));
    }
  }
  return raiden::PjRtCopyFuture::FromFuture(
      xla::JoinFutures(absl::MakeSpan(shard_futures_to_join)));
}

absl::Status WeightSynchronizerBase::PushWeights(
    const std::vector<std::string>& peers) {
  if (peers.empty()) {
    return absl::InvalidArgumentError(
        "Peer list cannot be empty for trainer weights sync");
  }

  TF_ASSIGN_OR_RETURN(raiden::PjRtCopyFuture d2h_future, D2h());
  TF_RETURN_IF_ERROR(d2h_future.Await());

  std::vector<int> weights_block_id = {0};
  for (const std::string& peer : peers) {
    TF_RETURN_IF_ERROR(H2hWriteDirect(peer, weights_block_id).status());
  }
  return absl::OkStatus();
}

absl::Status WeightSynchronizerBase::PushWeightsResharded(
    const tpu_raiden::rpc::StartTransferRequest& request) {
  StoreSkipTiling(request.uuid(), request);
  int fallback_layer_idx = -1;
  bool checked_fallback = false;
  auto get_fallback_layer_idx = [&]() -> absl::StatusOr<int> {
    if (checked_fallback) return fallback_layer_idx;
    checked_fallback = true;
    if (request.src_units().empty()) {
      return absl::InvalidArgumentError("src_units list cannot be empty");
    }
    std::string data_name = request.src_units(0).data_name();
    for (size_t l = 0; l < layer_names_.size(); ++l) {
      if (layer_names_[l] == data_name) {
        fallback_layer_idx = static_cast<int>(l);
        break;
      }
    }
    if (fallback_layer_idx == -1) {
      return absl::NotFoundError(absl::StrCat(
          "Layer name not found in WeightSynchronizer: ", data_name));
    }
    return fallback_layer_idx;
  };

  std::vector<bool> skip_tiling(num_layers_, false);
  for (const auto& [layer_idx, skip_val] : request.skip_tiling()) {
    if (layer_idx >= 0 && static_cast<size_t>(layer_idx) < num_layers_) {
      skip_tiling[layer_idx] = skip_val;
    }
  }

  if (!request.skip_d2h()) {
    VLOG(1) << "PushWeightsResharded: Executing D2H copy.";
    TF_ASSIGN_OR_RETURN(raiden::PjRtCopyFuture d2h_future, D2h(request.uuid()));
    TF_RETURN_IF_ERROR(d2h_future.Await());
  } else {
    VLOG(1) << "PushWeightsResharded: Skipping D2H copy.";
  }

  std::vector<transport::BufferPushTask> tasks;
  const auto& schedules = request.shard_push_schedules();
  for (size_t i = 0; i < num_shards_; ++i) {
    auto it = schedules.find(static_cast<int32_t>(i));
    if (it == schedules.end()) {
      continue;
    }
    const auto& schedule = it->second;

    for (const auto& entry : schedule.entries()) {
      int layer_idx_to_use = -1;
      if (entry.has_layer_idx()) {
        layer_idx_to_use = entry.layer_idx();
      } else {
        TF_ASSIGN_OR_RETURN(layer_idx_to_use, get_fallback_layer_idx());
      }
      if (layer_idx_to_use < 0 ||
          static_cast<size_t>(layer_idx_to_use) >= num_layers_) {
        return absl::InvalidArgumentError(
            absl::StrCat("Layer index out of bounds: ", layer_idx_to_use));
      }
      const uint8_t* base_host_ptr = GetHostPointer(layer_idx_to_use, i);
      if (base_host_ptr == nullptr) {
        return absl::InternalError(
            "Host pointer is null during resharded push");
      }
      size_t shard_host_size = GetHostSize(layer_idx_to_use, i);

      const std::string& dst_peer = entry.dst_peer();
      size_t dst_shard_idx = entry.dst_shard_idx();
      size_t count = entry.count() > 0 ? entry.count() : 1;
      size_t src_stride = entry.src_stride_bytes();
      size_t dst_stride = entry.dst_stride_bytes();
      size_t dst_offset = entry.dst_offset_bytes();
      size_t src_offset = entry.src_offset_bytes();
      size_t size = entry.size_bytes();

      for (size_t c = 0; c < count; ++c) {
        size_t curr_src_offset = src_offset + c * src_stride;
        size_t curr_dst_offset = dst_offset + c * dst_stride;

        if (curr_src_offset + size > shard_host_size) {
          return absl::InvalidArgumentError("Push range out of bounds");
        }

        const uint8_t* data_ptr = base_host_ptr + curr_src_offset;
        tasks.push_back({
            .peer = dst_peer,
            .buffer_id = static_cast<size_t>(layer_idx_to_use),
            .dst_shard_idx = dst_shard_idx,
            .dst_offset_bytes = curr_dst_offset,
            .data_ptr = data_ptr,
            .size_bytes = size,
        });
      }
    }
  }

  if (!tasks.empty()) {
    TF_RETURN_IF_ERROR(PushWeightsChunks(tasks, parallelism_, request.uuid()));
  }
  return absl::OkStatus();
}

absl::Status WeightSynchronizerBase::BindWeights(
    const std::vector<std::vector<raiden::RaidenBufferHandle>>& layer_buffers) {
  if (layer_buffers.size() != num_layers_) {
    return absl::InvalidArgumentError("Number of layers mismatch");
  }
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    if (layer_buffers[layer_idx].size() != num_shards_) {
      return absl::InvalidArgumentError("Number of shards mismatch");
    }
    for (size_t shard_idx = 0; shard_idx < num_shards_; ++shard_idx) {
      const auto& new_buffer = layer_buffers[layer_idx][shard_idx];
      size_t new_size = new_buffer.GetOnDeviceSizeInBytes();
      if (new_size != layers_[layer_idx].shards[shard_idx].device_size) {
        return absl::InvalidArgumentError("Buffer size mismatch");
      }
    }
  }

  if (!layer_buffers.empty() && !layer_buffers[0].empty()) {
    const auto& first_handle = layer_buffers[0][0];
    physical_size_ = first_handle.GetOnDeviceSizeInBytes();
    if (!first_handle.is_common_buffer && first_handle.c_hold) {
      c_api_ = first_handle.c_hold->c_api;
      extension_ = first_handle.c_hold->extension;
    }
  }

  buffer_holds_.clear();
  buffer_holds_.reserve(num_layers_);
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    std::vector<raiden::BufferHoldAndAlias> hold_info;
    hold_info.reserve(num_shards_);
    for (size_t shard_idx = 0; shard_idx < num_shards_; ++shard_idx) {
      hold_info.push_back(layer_buffers[layer_idx][shard_idx]);
    }
    buffer_holds_.push_back(std::move(hold_info));
  }
  return absl::OkStatus();
}

absl::Status WeightSynchronizerBase::OnDataReceived() {
  if (!auto_h2d_) {
    return absl::OkStatus();
  }
  TF_ASSIGN_OR_RETURN(raiden::PjRtCopyFuture h2d_future, H2d());
  return h2d_future.Await();
}

void WeightSynchronizerBase::StoreSkipTiling(
    uint64_t uuid, const tpu_raiden::rpc::StartTransferRequest& request) {
  std::vector<bool> skip(num_layers_, false);
  for (const auto& [layer_idx, skip_val] : request.skip_tiling()) {
    if (layer_idx >= 0 && static_cast<size_t>(layer_idx) < num_layers_) {
      skip[layer_idx] = skip_val;
    }
  }
  absl::MutexLock lock(skip_tiling_mu_);
  latest_skip_tiling_ = skip;
  uuid_to_skip_tiling_[uuid] = std::move(skip);
}

absl::Status WeightSynchronizerBase::OnBlocksReceived(
    const std::vector<int>& block_ids, uint64_t uuid) {
  if (!auto_h2d_) {
    return absl::OkStatus();
  }
  TF_ASSIGN_OR_RETURN(raiden::PjRtCopyFuture h2d_future, H2d(uuid));
  return h2d_future.Await();
}

void WeightSynchronizerBase::ForgetPushProgress(uint64_t uuid) {
  RaidenManagerBase::ForgetPushProgress(uuid);
  absl::MutexLock lock(skip_tiling_mu_);
  uuid_to_skip_tiling_.erase(uuid);
}

}  // namespace weight_sync
}  // namespace tpu_raiden
