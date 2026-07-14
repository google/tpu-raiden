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

#include "tpu_raiden/core/controller/worker_service_impl.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tpu_raiden/core/host_memory_allocator.h"
#include "tpu_raiden/core/kv_manager_holder.h"
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/proto/worker_service.pb.h"

namespace tpu_raiden {
namespace controller {
namespace {

bool IsD2H(const proto::TransferBufferSpec& transfer) {
  return transfer.src_mem_type() == rpc::MEMORY_TYPE_HBM &&
         (transfer.dst_mem_type() == rpc::MEMORY_TYPE_DRAM ||
          transfer.dst_mem_type() == rpc::MEMORY_TYPE_UNSPECIFIED);
}

bool IsH2D(const proto::TransferBufferSpec& transfer) {
  return (transfer.src_mem_type() == rpc::MEMORY_TYPE_DRAM ||
          transfer.src_mem_type() == rpc::MEMORY_TYPE_UNSPECIFIED) &&
         transfer.dst_mem_type() == rpc::MEMORY_TYPE_HBM;
}

bool IsH2H(const proto::TransferBufferSpec& transfer) {
  return transfer.src_mem_type() == rpc::MEMORY_TYPE_DRAM &&
         transfer.dst_mem_type() == rpc::MEMORY_TYPE_DRAM;
}

}  // namespace

WorkerServiceImpl::WorkerServiceImpl(
    std::shared_ptr<HostMemoryAllocator> allocator,
    KVManagerHolder transfer_manager)
    : allocator_(allocator ? std::move(allocator)
                           : std::make_shared<MallocHostMemoryAllocator>()),
      transfer_manager_(std::move(transfer_manager)) {}

grpc::Status WorkerServiceImpl::CreateBuffers(
    grpc::ServerContext* context, const proto::CreateBuffersRequest* request,
    proto::CreateBuffersResponse* response) {
  absl::MutexLock lock(mutex_);
  for (const auto& spec : request->buffers()) {
    if (spec.num_shards() <= 0 || spec.size_bytes() <= 0) {
      response->set_success(false);
      response->set_message("num_shards and size_bytes must be positive");
      return grpc::Status::OK;
    }
    proto::BufferProto sharded_buf;
    for (int32_t s = 0; s < spec.num_shards(); ++s) {
      auto alloc_or = allocator_->Allocate(spec.size_bytes());
      if (!alloc_or.ok()) {
        response->set_success(false);
        response->set_message(absl::StrCat("Failed to allocate memory: ",
                                           alloc_or.status().message()));
        return grpc::Status::OK;
      }
      BufferHandle handle = next_buffer_handle_++;
      buffers_[handle] = std::move(*alloc_or);
      sharded_buf.add_buffer_handles()->set_handle(handle);
    }
    *response->add_buffers() = std::move(sharded_buf);
  }
  response->set_success(true);
  response->set_message("Buffers created successfully");
  return grpc::Status::OK;
}

grpc::Status WorkerServiceImpl::DeleteBuffers(
    grpc::ServerContext* context, const proto::DeleteBuffersRequest* request,
    proto::DeleteBuffersResponse* response) {
  absl::MutexLock lock(mutex_);
  for (const auto& sharded_buffer : request->sharded_buffers()) {
    if (sharded_buffer.buffer_handles().empty()) {
      response->set_success(false);
      response->set_message("BufferProto has no buffer_handles");
      return grpc::Status::OK;
    }
    for (const auto& handle_proto : sharded_buffer.buffer_handles()) {
      BufferHandle handle = handle_proto.handle();
      auto it = buffers_.find(handle);
      if (it == buffers_.end()) {
        response->set_success(false);
        response->set_message(
            absl::StrCat("Buffer not found for deletion: handle ", handle));
        return grpc::Status::OK;
      }
      buffers_.erase(it);
    }
  }
  response->set_success(true);
  response->set_message("Buffers deleted successfully");
  return grpc::Status::OK;
}

grpc::Status WorkerServiceImpl::TransferBuffers(
    grpc::ServerContext* context, const proto::TransferBuffersRequest* request,
    proto::TransferBuffersResponse* response) {
  absl::MutexLock lock(mutex_);
  const auto& transfer = request->transfer();
  bool is_d2h = IsD2H(transfer);
  bool is_h2d = IsH2D(transfer);
  bool is_h2h = IsH2H(transfer);
  if (!is_d2h && !is_h2d && !is_h2h) {
    response->set_success(false);
    response->set_message(
        "Only D2H, H2D, and H2H (DRAM to DRAM) transfers are "
        "currently supported");
    return grpc::Status::OK;
  }
  if (is_h2h && transfer.peer().empty()) {
    response->set_success(false);
    response->set_message("Peer address must be provided for H2H transfers");
    return grpc::Status::OK;
  }
  if (is_h2h && transfer.copy_sizes_size() > 0) {
    for (int64_t size : transfer.copy_sizes()) {
      if (size != 1) {
        response->set_success(false);
        response->set_message(
            "H2H transfers only support copy size of 1 per segment");
        return grpc::Status::OK;
      }
    }
  }
  if (transfer.src_offsets_size() == 0 ||
      transfer.src_offsets_size() != transfer.dst_offsets_size()) {
    response->set_success(false);
    response->set_message(
        "Source and destination offsets must have the same non-zero length");
    return grpc::Status::OK;
  }
  if (transfer.copy_sizes_size() > 0 &&
      transfer.copy_sizes_size() != transfer.src_offsets_size()) {
    response->set_success(false);
    response->set_message(
        "copy_sizes, if provided, must match the length of src_offsets");
    return grpc::Status::OK;
  }

  if (!transfer_manager_) {
    response->set_success(false);
    response->set_message(
        "Transfer manager is not configured on WorkerService");
    return grpc::Status::OK;
  }

  std::vector<int64_t> src_offsets(transfer.src_offsets().begin(),
                                   transfer.src_offsets().end());
  std::vector<int64_t> dst_offsets(transfer.dst_offsets().begin(),
                                   transfer.dst_offsets().end());
  std::vector<int64_t> copy_sizes;
  if (transfer.copy_sizes_size() > 0) {
    copy_sizes.assign(transfer.copy_sizes().begin(),
                      transfer.copy_sizes().end());
  } else {
    copy_sizes.assign(src_offsets.size(), 1);
  }

  absl::StatusOr<raiden::PjRtCopyFuture> future_or;
  if (is_d2h) {
    future_or = transfer_manager_.D2h(src_offsets, dst_offsets, copy_sizes);
  } else if (is_h2d) {
    future_or = transfer_manager_.H2d(src_offsets, dst_offsets, copy_sizes);
  } else {
    future_or =
        transfer_manager_.H2hWrite(transfer.peer(), src_offsets, dst_offsets);
  }
  if (!future_or.ok()) {
    response->set_success(false);
    response->set_message(absl::StrCat(
        is_d2h ? "D2H" : (is_h2d ? "H2D" : "H2H"),
        " transfer dispatch failed: ", future_or.status().message()));
    return grpc::Status::OK;
  }
  absl::Status status = future_or->Await();
  if (!status.ok()) {
    response->set_success(false);
    response->set_message(
        absl::StrCat("Transfer execution failed: ", status.message()));
    return grpc::Status::OK;
  }
  response->set_success(true);
  response->set_message("Buffers transferred successfully");
  return grpc::Status::OK;
}

absl::StatusOr<HostBufferAllocation> WorkerServiceImpl::GetBuffer(
    BufferHandle handle) const {
  absl::MutexLock lock(mutex_);
  auto it = buffers_.find(handle);
  if (it == buffers_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Buffer not found: handle ", handle));
  }
  return it->second;
}

size_t WorkerServiceImpl::GetBufferCount() const {
  absl::MutexLock lock(mutex_);
  return buffers_.size();
}

}  // namespace controller
}  // namespace tpu_raiden
