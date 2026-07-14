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

#include "tpu_raiden/core/controller/raiden_controller.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "grpcpp/channel.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "tpu_raiden/core/controller/worker_service_client.h"
#include "tpu_raiden/kv_cache/logical_block_manager.h"
#include "tpu_raiden/proto/worker_service.pb.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace controller {

RaidenController::RaidenController(const rpc::RaidenIdProto& unit,
                                   const std::string& worker_address,
                                   int num_blocks, int num_shards,
                                   int64_t shard_size_bytes)
    : RaidenController(unit,
                       grpc::CreateChannel(worker_address,
                                           grpc::InsecureChannelCredentials()),
                       num_blocks, num_shards, shard_size_bytes) {}

RaidenController::RaidenController(
    const rpc::RaidenIdProto& unit,
    std::shared_ptr<grpc::Channel> worker_channel, int num_blocks,
    int num_shards, int64_t shard_size_bytes)
    : RaidenController(unit,
                       std::make_unique<WorkerServiceClient>(worker_channel),
                       num_blocks, num_shards, shard_size_bytes) {}

RaidenController::RaidenController(const rpc::RaidenIdProto& unit,
                                   std::unique_ptr<WorkerServiceClient> client,
                                   int num_blocks, int num_shards,
                                   int64_t shard_size_bytes)
    : unit_(unit),
      client_(std::move(client)),
      num_shards_(num_shards),
      shard_size_bytes_(shard_size_bytes),
      block_manager_(
          std::make_unique<kv_cache::LogicalBlockManager>(num_blocks)) {
  proto::CreateBuffersRequest request;
  *request.mutable_unit() = unit_;
  all_sharded_buffers_.reserve(num_blocks);

  for (int block_id = 0; block_id < num_blocks; ++block_id) {
    auto* spec = request.add_buffers();
    spec->set_num_shards(num_shards_);
    spec->set_size_bytes(shard_size_bytes_);
  }

  auto resp_or = client_->CreateBuffers(request);
  if (!resp_or.ok()) {
    throw std::runtime_error(absl::StrCat(
        "Failed to create buffers on worker in RaidenController constructor: ",
        resp_or.status().message()));
  }
  if (!resp_or->success()) {
    throw std::runtime_error(
        absl::StrCat("WorkerService CreateBuffers failed in RaidenController "
                     "constructor: ",
                     resp_or->message()));
  }
  if (resp_or->buffers_size() != num_blocks) {
    throw std::runtime_error(
        absl::StrCat("WorkerService returned unexpected number of buffers: ",
                     resp_or->buffers_size(), " vs expected ", num_blocks));
  }

  for (int block_id = 0; block_id < num_blocks; ++block_id) {
    const auto& sharded_buf = resp_or->buffers(block_id);
    for (const auto& handle_proto : sharded_buf.buffer_handles()) {
      handle_to_block_id_[handle_proto.handle()] = block_id;
    }
    all_sharded_buffers_.push_back(sharded_buf);
  }
}

RaidenController::~RaidenController() {
  if (!client_ || all_sharded_buffers_.empty()) return;

  proto::DeleteBuffersRequest request;
  *request.mutable_unit() = unit_;
  for (const auto& sharded_buf : all_sharded_buffers_) {
    *request.add_sharded_buffers() = sharded_buf;
  }

  auto resp_or = client_->DeleteBuffers(request);
  if (!resp_or.ok()) {
    LOG(ERROR) << "Failed to delete buffers on worker in ~RaidenController: "
               << resp_or.status().message();
  } else if (!resp_or->success()) {
    LOG(ERROR) << "WorkerService DeleteBuffers failed in ~RaidenController: "
               << resp_or->message();
  }
}

absl::StatusOr<std::vector<proto::BufferProto>> RaidenController::Allocate(
    int num_blocks) {
  if (num_blocks <= 0) {
    return absl::InvalidArgumentError("num_blocks must be positive");
  }

  std::vector<int> block_ids;
  {
    absl::MutexLock lock(mutex_);
    ASSIGN_OR_RETURN(block_ids,
                     block_manager_->Allocate(num_blocks, /*lock=*/true));
  }

  std::vector<proto::BufferProto> sharded_buffers;
  sharded_buffers.reserve(block_ids.size());
  for (int block_id : block_ids) {
    if (block_id < 0 || block_id >= all_sharded_buffers_.size()) {
      return absl::InternalError(
          absl::StrCat("Allocated block_id out of range: ", block_id,
                       ", total buffers: ", all_sharded_buffers_.size()));
    }
    sharded_buffers.push_back(all_sharded_buffers_[block_id]);
  }

  return sharded_buffers;
}

absl::Status RaidenController::Deallocate(
    absl::Span<const proto::BufferProto> sharded_buffers) {
  if (sharded_buffers.empty()) {
    return absl::OkStatus();
  }

  std::vector<int> block_ids_to_unlock;
  block_ids_to_unlock.reserve(sharded_buffers.size());

  for (const auto& sharded_buf : sharded_buffers) {
    if (sharded_buf.buffer_handles().empty()) {
      return absl::InvalidArgumentError("ShardedBuffer has no buffer_handles");
    }
    uint64_t handle = sharded_buf.buffer_handles(0).handle();
    auto it = handle_to_block_id_.find(handle);
    if (it == handle_to_block_id_.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "ShardedBuffer handle not recognized by this RaidenController: ",
          handle));
    }
    block_ids_to_unlock.push_back(it->second);
  }

  absl::MutexLock lock(mutex_);
  RETURN_IF_ERROR(block_manager_->Unlock(block_ids_to_unlock));
  return absl::OkStatus();
}

absl::StatusOr<proto::TransferBuffersResponse>
RaidenController::TransferBuffers(rpc::MemoryType src_mem_type,
                                  rpc::MemoryType dst_mem_type,
                                  absl::Span<const int64_t> src_offsets,
                                  absl::Span<const int64_t> dst_offsets,
                                  absl::Span<const int64_t> copy_sizes,
                                  absl::string_view peer) {
  if (!client_) {
    return absl::FailedPreconditionError(
        "WorkerServiceClient is not initialized");
  }
  if (src_offsets.empty() || src_offsets.size() != dst_offsets.size()) {
    return absl::InvalidArgumentError(
        "Source and destination offsets must have the same non-zero length");
  }
  if (!copy_sizes.empty() && copy_sizes.size() != src_offsets.size()) {
    return absl::InvalidArgumentError(
        "copy_sizes, if provided, must match the length of src_offsets");
  }
  proto::TransferBuffersRequest request;
  auto* transfer = request.mutable_transfer();
  transfer->set_src_mem_type(src_mem_type);
  transfer->set_dst_mem_type(dst_mem_type);
  transfer->mutable_src_offsets()->Add(src_offsets.begin(), src_offsets.end());
  transfer->mutable_dst_offsets()->Add(dst_offsets.begin(), dst_offsets.end());
  transfer->mutable_copy_sizes()->Add(copy_sizes.begin(), copy_sizes.end());
  if (!peer.empty()) {
    transfer->set_peer(std::string(peer));
  }
  return client_->TransferBuffers(request);
}

}  // namespace controller
}  // namespace tpu_raiden
