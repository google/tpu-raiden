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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_CONTROLLER_SERVER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_CONTROLLER_SERVER_H_

#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/server.h"
#include "tpu_raiden/core/controller/controller_service.h"

namespace tpu_raiden {
namespace core {
namespace controller {

// A singleton gRPC server wrapper that hosts RaidenControllerServiceImpl on an
// ephemeral or specified port. Reused across wrapper instances to avoid port
// conflicts and rebinding overhead.
class ControllerServer {
 public:
  static ControllerServer& GetInstance();

  // Starts the gRPC server hosting RaidenControllerServiceImpl on the specified
  // port (pass 0 for an ephemeral port).
  // If the server is already started, updates the worker registry (if
  // provided) and returns OkStatus().
  absl::Status StartServer(int port = 0) {
    return StartServer(/*worker_registry=*/nullptr, port);
  }

  absl::Status StartServer(std::shared_ptr<WorkerRegistry> worker_registry,
                           int port = 0);

  void SetWorkerRegistry(std::shared_ptr<WorkerRegistry> worker_registry);

  std::shared_ptr<WorkerRegistry> GetWorkerRegistry() const;

  // Returns the port the gRPC server is listening on. Returns 0 if the server
  // is not running or failed to start.
  int GetGrpcPort() const;



 private:
  // Returns a pointer to the hosted RaidenControllerServiceImpl instance, or
  // nullptr if the server has not been started.
  RaidenControllerServiceImpl* GetControllerService() const;

  ControllerServer() = default;
  ~ControllerServer() = default;
  ControllerServer(const ControllerServer&) = delete;
  ControllerServer& operator=(const ControllerServer&) = delete;

  mutable absl::Mutex mutex_;
  std::unique_ptr<RaidenControllerServiceImpl> controller_service_
      ABSL_GUARDED_BY(mutex_);
  std::unique_ptr<grpc::Server> grpc_server_ ABSL_GUARDED_BY(mutex_);
  int grpc_port_ ABSL_GUARDED_BY(mutex_) = 0;
  bool started_ ABSL_GUARDED_BY(mutex_) = false;
};

using RaidenControllerServer = ControllerServer;

}  // namespace controller
}  // namespace core
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_CONTROLLER_SERVER_H_
