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

#include "tpu_raiden/core/controller/controller_server.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server_builder.h"
#include "tpu_raiden/core/controller/controller_service.h"

namespace tpu_raiden {
namespace core {
namespace controller {

ControllerServer& ControllerServer::GetInstance() {
  static ControllerServer* instance = new ControllerServer();
  return *instance;
}

absl::Status ControllerServer::StartServer(
    std::shared_ptr<WorkerRegistry> worker_registry, int port) {
  if (port < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid gRPC port: ", port));
  }

  absl::MutexLock lock(mutex_);
  if (started_) {
    if (port != 0 && port != grpc_port_) {
      return absl::FailedPreconditionError(
          absl::StrCat("Server already started on port ", grpc_port_,
                       ", cannot start on requested port ", port));
    }
    if (worker_registry && controller_service_) {
      controller_service_->SetWorkerRegistry(std::move(worker_registry));
    }
    return absl::OkStatus();
  }

  controller_service_ =
      std::make_unique<RaidenControllerServiceImpl>(std::move(worker_registry));

  std::string server_address = absl::StrCat("[::]:", port);
  grpc::ServerBuilder builder;
  int selected_port = 0;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials(),
                           &selected_port);
  builder.RegisterService(controller_service_.get());
  grpc_server_ = builder.BuildAndStart();

  if (!grpc_server_ || selected_port == 0) {
    grpc_server_.reset();
    controller_service_.reset();
    return absl::InternalError(
        absl::StrCat("Failed to start gRPC server on port: ", port, " (",
                     server_address, ")"));
  }

  grpc_port_ = selected_port;
  started_ = true;
  return absl::OkStatus();
}

void ControllerServer::SetWorkerRegistry(
    std::shared_ptr<WorkerRegistry> worker_registry) {
  absl::MutexLock lock(mutex_);
  if (controller_service_) {
    controller_service_->SetWorkerRegistry(std::move(worker_registry));
  }
}

std::shared_ptr<WorkerRegistry> ControllerServer::GetWorkerRegistry() const {
  absl::MutexLock lock(mutex_);
  return controller_service_ ? controller_service_->worker_registry() : nullptr;
}

int ControllerServer::GetGrpcPort() const {
  absl::MutexLock lock(mutex_);
  return grpc_port_;
}



RaidenControllerServiceImpl* ControllerServer::GetControllerService() const {
  absl::MutexLock lock(&mutex_);
  return controller_service_.get();
}

}  // namespace controller
}  // namespace core
}  // namespace tpu_raiden
