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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_CONTROLLER_WORKER_SERVICE_SERVER_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_CONTROLLER_WORKER_SERVICE_SERVER_H_

#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/server.h"
#include "tpu_raiden/core/controller/worker_service_impl.h"
#include "tpu_raiden/core/host_memory_allocator.h"
#include "tpu_raiden/core/kv_manager_holder.h"

namespace tpu_raiden {
namespace controller {

// A singleton gRPC server wrapper that hosts WorkerServiceImpl on an ephemeral
// or specified port. Reused across wrapper instances to avoid port conflicts
// and rebinding overhead.
class WorkerServiceServer {
 public:
  static WorkerServiceServer& GetInstance();

  // Starts the gRPC server hosting WorkerServiceImpl on the specified port.
  // If the server is already started, updates the transfer manager (if
  // provided) and returns OkStatus().
  absl::Status StartServer(std::shared_ptr<HostMemoryAllocator> host_allocator,
                           int port) {
    return StartServer(std::move(host_allocator), KVManagerHolder(), port);
  }

  absl::Status StartServer(std::shared_ptr<HostMemoryAllocator> host_allocator,
                           KVManagerHolder transfer_manager, int port);

  void SetTransferManager(KVManagerHolder transfer_manager);

  // Returns the port the gRPC server is listening on. Returns 0 if the server
  // is not running or failed to start.
  int GetGrpcPort() const;

 private:
  WorkerServiceServer() = default;
  ~WorkerServiceServer() = default;
  WorkerServiceServer(const WorkerServiceServer&) = delete;
  WorkerServiceServer& operator=(const WorkerServiceServer&) = delete;

  mutable absl::Mutex mutex_;
  std::unique_ptr<WorkerServiceImpl> worker_service_ ABSL_GUARDED_BY(mutex_);
  std::unique_ptr<grpc::Server> grpc_server_ ABSL_GUARDED_BY(mutex_);
  int grpc_port_ ABSL_GUARDED_BY(mutex_) = 0;
  bool started_ ABSL_GUARDED_BY(mutex_) = false;
};

}  // namespace controller
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_CONTROLLER_WORKER_SERVICE_SERVER_H_
