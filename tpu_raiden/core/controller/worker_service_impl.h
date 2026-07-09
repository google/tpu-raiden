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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_WORKER_SERVICE_IMPL_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_WORKER_SERVICE_IMPL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "third_party/grpc/include/grpcpp/server_context.h"
#include "third_party/grpc/include/grpcpp/support/status.h"
#include "tpu_raiden/core/host_memory_allocator.h"
#include "tpu_raiden/proto/worker_service.grpc.pb.h"
#include "tpu_raiden/proto/worker_service.pb.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace controller {

using BufferHandle = uint64_t;

// Implementation of the WorkerService gRPC service running on transfer workers.
// Manages allocation and deallocation of sharded host memory buffers.
class WorkerServiceImpl final : public proto::WorkerService::Service {
 public:
  // Constructs a WorkerServiceImpl with the given host memory allocator. If
  // allocator is nullptr, defaults to MallocHostMemoryAllocator.
  explicit WorkerServiceImpl(
      std::shared_ptr<HostMemoryAllocator> allocator = nullptr);

  grpc::Status CreateBuffers(grpc::ServerContext* context,
                             const proto::CreateBuffersRequest* request,
                             proto::CreateBuffersResponse* response) override;

  grpc::Status DeleteBuffers(grpc::ServerContext* context,
                             const proto::DeleteBuffersRequest* request,
                             proto::DeleteBuffersResponse* response) override;

  // Retrieves an allocated buffer shard for inspection or transfer operations.
  // Returns NotFoundError if the buffer handle is invalid.
  absl::StatusOr<HostBufferAllocation> GetBuffer(BufferHandle handle) const;

  // Overload accepting BufferHandleProto for convenience.
  absl::StatusOr<HostBufferAllocation> GetBuffer(
      const proto::BufferHandleProto& handle_proto) const {
    return GetBuffer(BufferHandle(handle_proto.handle()));
  }

  // Returns the total number of currently allocated buffer shards across all
  // buffers.
  size_t GetBufferCount() const;

 private:
  std::shared_ptr<HostMemoryAllocator> allocator_;
  mutable absl::Mutex mutex_;
  uint64_t next_buffer_handle_ ABSL_GUARDED_BY(mutex_) = 1;
  absl::flat_hash_map<BufferHandle, HostBufferAllocation> buffers_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace controller
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_WORKER_SERVICE_IMPL_H_
