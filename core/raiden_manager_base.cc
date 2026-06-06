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
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "core/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/semaphore.h"
#include "core/host_memory_allocator.h"
#include "core/raw_transfer_core.h"
#include "core/raw_transfer_impl.h"
#include "transport/block_transport.h"

namespace tpu_raiden {

xla::Future<> ReturnFuture(const absl::Status& status) {
  return xla::Future<>(status);
}

RaidenManagerBase::RaidenManagerBase(size_t num_layers, size_t num_shards,
                                     size_t slice_byte_size, int block_size,
                                     std::optional<int> local_port,
                                     int parallelism, size_t max_staging_blocks)
    : num_layers_(num_layers),
      num_shards_(num_shards),
      slice_byte_size_(slice_byte_size),
      block_size_(block_size),
      parallelism_(parallelism) {
  shard_factor_ = 1;

  int port = local_port.value_or(0);
  server_ = std::make_unique<tpu_raiden::transport::BlockTransport>(this, port);

  semaphore_ = std::make_unique<xla::Semaphore>(max_staging_blocks);
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

void RaidenManagerBase::SetBlockReadinessCallback(
    BlockReadinessCallback callback) {
  absl::MutexLock l(&block_readiness_mu_);
  block_readiness_callback_ = std::move(callback);
}

absl::Status RaidenManagerBase::WaitForBlockRead(size_t layer_idx,
                                                 size_t shard_idx,
                                                 int block_id) {
  BlockReadinessCallback callback;
  {
    absl::MutexLock l(&block_readiness_mu_);
    callback = block_readiness_callback_;
  }
  if (!callback) {
    return absl::OkStatus();
  }
  return callback(layer_idx, shard_idx, block_id);
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
  return server_->Pull(peer, src_block_ids, {}, {}, parallelism_);
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

size_t RaidenManagerBase::bytes_per_block() const {
  return block_size_ * slice_byte_size_;
}

xla::Future<> RaidenManagerBase::RemoteD2DBlockWrite(const BlockMetadata& src,
                                                     const BlockMetadata& dst,
                                                     size_t size_bytes) {
  auto [promise, future] = xla::MakePromise<void>();
  if (!semaphore_) {
    promise.Set(absl::FailedPreconditionError(
        "D2D transfer helper is not initialized (missing semaphore)"));
    return future;
  }
  if (!src.pjrt_client) {
    promise.Set(absl::FailedPreconditionError(
        "D2D transfer helper is missing pjrt_client in src metadata"));
    return future;
  }

  auto shared_promise =
      std::make_shared<xla::Promise<void>>(std::move(promise));

  std::thread([this, src, dst, size_bytes, shared_promise]() mutable {
    // Acquire staging block budget.
    semaphore_->Acquire(1);

    // Perform transfer
    xla::Future<> d2d_future = DoD2DTransfer(src, dst, size_bytes);

    d2d_future.OnReady([this, shared_promise](const absl::Status& status) {
      // Release staging block budget.
      semaphore_->Release(1);
      shared_promise->Set(status);
    });
  }).detach();

  return future;
}

xla::Future<> RaidenManagerBase::DoD2DTransfer(const BlockMetadata& src,
                                               const BlockMetadata& dst,
                                               size_t size_bytes) {
  auto [promise, future] = xla::MakePromise<void>();
  ASSIGN_OR_RETURN(std::unique_ptr<HostMemoryAllocator> allocator,
                        HostMemoryAllocator::Create(src.pjrt_client),
                        _.LogError().With(ReturnFuture));

  // 1. Allocate local staging memory
  ASSIGN_OR_RETURN(auto host_allocation, allocator->Allocate(size_bytes),
                        _.LogError().With(ReturnFuture));

  // 2. Copy data from device to host staging memory (D2H)
  xla::PjRtBuffer* src_buffer = static_cast<xla::PjRtBuffer*>(src.data_ptr);

  std::vector<xla::PjRtBuffer*> src_buffers = {src_buffer};
  std::vector<uint8_t*> dst_ptrs = {host_allocation.ptr};
  std::vector<size_t> dst_sizes = {host_allocation.size};

  ASSIGN_OR_RETURN(
      raiden::PjRtCopyFuture d2h_future,
      raiden::transfer_d2h_core(src_buffers, dst_ptrs, dst_sizes,
                                /*src_offsets_major_dim=*/{},
                                /*dst_offsets_major_dim=*/{},
                                /*copy_sizes_major_dim=*/{}),
      _.LogError().With(ReturnFuture));

  // Wrap promise in shared_ptr because OnReady lambda needs to be copyable
  auto shared_promise =
      std::make_shared<xla::Promise<void>>(std::move(promise));

  // 3. Trigger H2H inside d2h_future ready callback
  d2h_future.OnReady(
      [this, dst, host_allocation, shared_promise](
          const absl::StatusOr<raiden::BufferHolders>& status_or) mutable {
        if (!status_or.ok()) {
          shared_promise->Set(absl::InternalError(
              absl::StrCat("D2H copy failed: ", status_or.status().message())));
          return;
        }

        // 4. Start H2H Push in a separate thread to avoid blocking PJRT
        // callback.
        std::thread([this, dst, host_allocation, shared_promise]() mutable {
          absl::Status s = server_->WriteBlockDirect(dst.address, dst.block_id,
                                                     host_allocation.ptr,
                                                     host_allocation.size);
          shared_promise->Set(s);
        }).detach();
      });

  return future;
}
absl::Status RaidenManagerBase::OnSingleBlockReceived(int block_id,
                                                      size_t size_bytes) {
  RecvCallback cb;
  {
    absl::MutexLock l(recv_mu_);
    auto it = recv_callbacks_.find(block_id);
    if (it != recv_callbacks_.end()) {
      cb = std::move(it->second);
      recv_callbacks_.erase(it);
    }
  }
  if (cb) {
    return cb(block_id, size_bytes);
  }
  return absl::OkStatus();
}

xla::Future<> RaidenManagerBase::RemoteD2DBlockReceive(
    int block_id, raiden::BufferHoldAndAlias hold, size_t size_bytes) {
  if (size_bytes != bytes_per_block()) {
    return xla::Future<>(absl::InvalidArgumentError(absl::StrCat(
        "size_bytes (", size_bytes, ") must match bytes_per_block (",
        bytes_per_block(), ")")));
  }
  if (!hold.buffer) {
    return xla::Future<>(absl::FailedPreconditionError(
        "Destination PjRtBuffer in hold is null"));
  }

  auto [promise, future] = xla::MakePromise<void>();
  auto shared_promise =
      std::make_shared<xla::Promise<void>>(std::move(promise));

  auto callback = [this, hold, size_bytes, shared_promise](
                      int block_id_received,
                      size_t size_bytes_received) -> absl::Status {
    if (size_bytes_received != size_bytes) {
      auto status = absl::InvalidArgumentError("Received size mismatch");
      shared_promise->Set(status);
      return status;
    }

    uint8_t* host_ptr = GetBlockHostPointer(0, 0, block_id_received);
    if (host_ptr == nullptr) {
      auto status = absl::FailedPreconditionError("Host pointer is null");
      shared_promise->Set(status);
      return status;
    }

    int64_t device_offset_bytes =
        static_cast<int64_t>(block_id_received) * bytes_per_block();

    xla::Future<> copy_future =
        hold.CopyRawHostToDevice(host_ptr, device_offset_bytes, size_bytes);

    copy_future.OnReady([hold, shared_promise](const absl::Status& status) {
      shared_promise->Set(status);
    });

    return absl::OkStatus();
  };

  {
    absl::MutexLock l(recv_mu_);
    recv_callbacks_[block_id] = std::move(callback);
  }

  return future;
}

}  // namespace tpu_raiden
