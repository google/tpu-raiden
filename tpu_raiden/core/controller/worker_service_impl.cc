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
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "third_party/grpc/include/grpcpp/server_context.h"
#include "third_party/grpc/include/grpcpp/support/status.h"
#include "tpu_raiden/core/host_memory_allocator.h"
#include "tpu_raiden/proto/worker_service.pb.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace controller {

WorkerServiceImpl::WorkerServiceImpl(
    std::shared_ptr<HostMemoryAllocator> allocator)
    : allocator_(allocator ? std::move(allocator)
                           : std::make_shared<MallocHostMemoryAllocator>()) {}

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
