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

#include "tpu_raiden/core/raiden_manager_base.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/semaphore.h"
#include "tpu_raiden/core/host_memory_allocator.h"
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/core/raw_transfer_impl.h"
#include "tpu_raiden/core/status_macros.h"
#include "tpu_raiden/core/tpu_utils.h"
#include "tpu_raiden/transport/block_transport.h"

namespace tpu_raiden {

xla::Future<> ReturnFuture(const absl::Status& status) {
  return xla::Future<>(status);
}

void RaidenManagerBase::DetectAndAssignNumaNode(
    const std::vector<std::vector<xla::PjRtBuffer*>>& layer_buffers) {
  std::vector<int> unique_numa_nodes;
  for (const auto& layer : layer_buffers) {
    for (xla::PjRtBuffer* buf : layer) {
      if (buf && buf->device()) {
        int node = GetPjRtDeviceNumaNode(buf->device());
        if (node >= 0) {
          bool found = false;
          for (int n : unique_numa_nodes) {
            if (n == node) {
              found = true;
              break;
            }
          }
          if (!found) unique_numa_nodes.push_back(node);
        }
      }
    }
  }
  if (!unique_numa_nodes.empty()) {
    assigned_numa_node_ = unique_numa_nodes[0];
    if (unique_numa_nodes.size() > 1) {
      LOG(WARNING) << "Incoming PJRT buffers are associated with more than one "
                      "NUMA node ("
                   << unique_numa_nodes[0] << " vs " << unique_numa_nodes[1]
                   << "). Picking the first detected NUMA node: "
                   << unique_numa_nodes[0];
    }
  }
}

RaidenManagerBase::RaidenManagerBase(size_t num_layers, size_t num_shards,
                                     size_t slice_byte_size,
                                     std::optional<int> local_port,
                                     int parallelism)
    : num_layers_(num_layers),
      num_shards_(num_shards),
      slice_byte_size_(slice_byte_size),
      parallelism_(parallelism) {
  shard_factor_ = 1;

  int port = local_port.value_or(0);
  server_ = std::make_unique<tpu_raiden::transport::BlockTransport>(this, port);
}

RaidenManagerBase::~RaidenManagerBase() {
  if (server_) {
    server_.reset();
  }
}

std::optional<int> RaidenManagerBase::local_port() const {
  if (server_) return server_->local_port();
  return std::nullopt;
}

uint8_t* RaidenManagerBase::GetHostPointer(size_t layer_idx, size_t shard_idx) {
  if (layer_idx >= layers_.size() ||
      shard_idx >= layers_[layer_idx].shards.size()) {
    return nullptr;
  }
  return const_cast<uint8_t*>(layers_[layer_idx].shards[shard_idx].host_ptr);
}

size_t RaidenManagerBase::GetHostSize(size_t layer_idx, size_t shard_idx) {
  if (layer_idx >= layers_.size() ||
      shard_idx >= layers_[layer_idx].shards.size()) {
    return 0;
  }
  return layers_[layer_idx].shards[shard_idx].host_size;
}

const uint8_t* RaidenManagerBase::GetHostPointer(size_t layer_idx,
                                                 size_t shard_idx) const {
  if (layer_idx >= layers_.size() ||
      shard_idx >= layers_[layer_idx].shards.size()) {
    return nullptr;
  }
  return layers_[layer_idx].shards[shard_idx].host_ptr;
}

void RaidenManagerBase::SetExternalHostPointers(
    const std::vector<const uint8_t*>& host_ptrs,
    const std::vector<size_t>& host_sizes) {
  size_t idx = 0;
  for (size_t l = 0; l < layers_.size(); ++l) {
    for (size_t sh = 0; sh < layers_[l].shards.size(); ++sh) {
      if (idx < host_ptrs.size() && idx < host_sizes.size()) {
        layers_[l].shards[sh].host_ptr = host_ptrs[idx];
        layers_[l].shards[sh].host_size = host_sizes[idx];
        idx++;
      }
    }
  }
}

absl::StatusOr<std::vector<int>> RaidenManagerBase::H2hWriteDirect(
    const std::string& peer, const std::vector<int>& src_block_ids,
    const std::vector<int>& dst_block_ids, uint64_t uuid) {
  if (!server_) {
    return absl::FailedPreconditionError("Transport server is not running");
  }
  return server_->Push(peer, src_block_ids, dst_block_ids, parallelism_,
                       tpu_raiden::transport::MajorOrder::kLayerMajor, uuid);
}

absl::StatusOr<std::vector<int>> RaidenManagerBase::H2hReadDirect(
    const std::string& peer, const std::vector<int>& src_block_ids) {
  if (!server_) {
    return absl::FailedPreconditionError("Transport server is not running");
  }
  return server_->Pull(peer, src_block_ids, {}, {}, parallelism_);
}

absl::Status RaidenManagerBase::PullWeightsChunk(
    const std::string& source, size_t src_shard_idx, size_t src_offset_bytes,
    size_t dst_shard_idx, size_t dst_offset_bytes, size_t size_bytes) {
  if (!server_) {
    return absl::FailedPreconditionError("Transport server is not running");
  }
  return server_->PullBuffer(source, /*buffer_id=*/0, src_shard_idx,
                             src_offset_bytes, dst_shard_idx, dst_offset_bytes,
                             size_bytes);
}

absl::Status RaidenManagerBase::PushWeightsChunk(const std::string& peer,
                                                 size_t dst_shard_idx,
                                                 size_t dst_offset_bytes,
                                                 const uint8_t* data_ptr,
                                                 size_t size_bytes) {
  if (!server_) {
    return absl::FailedPreconditionError("Transport server is not running");
  }
  return server_->PushBuffer(peer, /*buffer_id=*/0, dst_shard_idx,
                             dst_offset_bytes, data_ptr, size_bytes);
}

size_t RaidenManagerBase::bytes_per_block() const { return slice_byte_size_; }

}  // namespace tpu_raiden
