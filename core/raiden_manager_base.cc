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

#include "core/raiden_manager_base.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace tpu_raiden {

RaidenManagerBase::RaidenManagerBase(size_t num_layers, size_t num_shards,
                                     size_t slice_byte_size, int block_size,
                                     std::optional<int> local_port,
                                     int parallelism)
    : num_layers_(num_layers),
      num_shards_(num_shards),
      slice_byte_size_(slice_byte_size),
      block_size_(block_size),
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
  if (layer_idx >= num_layers_ || shard_idx >= num_shards_) {
    return nullptr;
  }
  return const_cast<uint8_t*>(layers_[layer_idx].shards[shard_idx].host_ptr);
}

size_t RaidenManagerBase::GetHostSize(size_t layer_idx, size_t shard_idx) {
  if (layer_idx >= num_layers_ || shard_idx >= num_shards_) {
    return 0;
  }
  return layers_[layer_idx].shards[shard_idx].host_size;
}

const uint8_t* RaidenManagerBase::GetHostPointer(size_t layer_idx,
                                                 size_t shard_idx) const {
  if (layer_idx >= num_layers_ || shard_idx >= num_shards_) {
    return nullptr;
  }
  return layers_[layer_idx].shards[shard_idx].host_ptr;
}

void RaidenManagerBase::SetExternalHostPointers(
    const std::vector<const uint8_t*>& host_ptrs,
    const std::vector<size_t>& host_sizes) {
  size_t idx = 0;
  for (size_t l = 0; l < num_layers_; ++l) {
    for (size_t sh = 0; sh < num_shards_; ++sh) {
      if (idx < host_ptrs.size()) {
        layers_[l].shards[sh].host_ptr = host_ptrs[idx];
        layers_[l].shards[sh].host_size = host_sizes[idx];
        idx++;
      }
    }
  }
}

absl::StatusOr<std::vector<int>> RaidenManagerBase::H2hWriteDirect(
    const std::string& peer, const std::vector<int>& src_block_ids,
    int64_t entity_id) {
  if (!server_) {
    return absl::FailedPreconditionError("Transport server is not running");
  }
  return server_->Push(peer, src_block_ids, parallelism_);
}

absl::StatusOr<std::vector<int>> RaidenManagerBase::H2hReadDirect(
    const std::string& peer, const std::vector<int>& src_block_ids,
    int64_t entity_id) {
  if (!server_) {
    return absl::FailedPreconditionError("Transport server is not running");
  }
  return server_->Pull(peer, src_block_ids, parallelism_);
}

absl::Status RaidenManagerBase::PullWeightsChunk(
    const std::string& source, size_t src_shard_idx, size_t src_offset_bytes,
    size_t dst_shard_idx, size_t dst_offset_bytes, size_t size_bytes) {
  if (!server_) {
    return absl::FailedPreconditionError("Transport server is not running");
  }
  return server_->PullWeightsChunk(source, src_shard_idx, src_offset_bytes,
                                   dst_shard_idx, dst_offset_bytes, size_bytes);
}

}  // namespace tpu_raiden
