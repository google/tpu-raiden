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

#include "tpu_raiden/core/kv_cache_manager_with_worker_service.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "tpu_raiden/core/controller/worker_service_impl.h"
#include "tpu_raiden/core/host_memory_allocator.h"
#include "tpu_raiden/core/kv_cache_manager_with_transfer.h"

namespace tpu_raiden {
namespace {

class WorkerServiceServer {
 public:
  static WorkerServiceServer& GetInstance() {
    static WorkerServiceServer* instance = new WorkerServiceServer();
    return *instance;
  }

  absl::Status StartServer(std::shared_ptr<HostMemoryAllocator> host_allocator,
                           int port) {
    if (port < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid gRPC port: ", port));
    }

    absl::MutexLock lock(mutex_);
    if (started_) {
      return absl::OkStatus();
    }

    worker_service_ = std::make_unique<controller::WorkerServiceImpl>(
        std::move(host_allocator));

    std::string server_address = absl::StrCat("[::]:", port);
    grpc::ServerBuilder builder;
    int selected_port = 0;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials(),
                             &selected_port);
    builder.RegisterService(worker_service_.get());
    grpc_server_ = builder.BuildAndStart();

    if (!grpc_server_ || selected_port == 0) {
      grpc_server_.reset();
      worker_service_.reset();
      return absl::InternalError(
          absl::StrCat("Failed to start gRPC server on port: ", port, " (",
                       server_address, ")"));
    }

    grpc_port_ = selected_port;
    started_ = true;
    return absl::OkStatus();
  }

  int GetGrpcPort() const {
    absl::MutexLock lock(mutex_);
    return grpc_port_;
  }

 private:
  WorkerServiceServer() = default;

  mutable absl::Mutex mutex_;
  std::unique_ptr<controller::WorkerServiceImpl> worker_service_
      ABSL_GUARDED_BY(mutex_);
  std::unique_ptr<grpc::Server> grpc_server_ ABSL_GUARDED_BY(mutex_);
  int grpc_port_ ABSL_GUARDED_BY(mutex_) = 0;
  bool started_ ABSL_GUARDED_BY(mutex_) = false;
};

}  // namespace

KVCacheManagerWithWorkerService::KVCacheManagerWithWorkerService(
    std::unique_ptr<KVCacheManagerWithTransfer> transfer_manager,
    std::shared_ptr<HostMemoryAllocator> host_allocator, int grpc_port)
    : transfer_manager_(std::move(transfer_manager)) {
  absl::Status status = WorkerServiceServer::GetInstance().StartServer(
      std::move(host_allocator), grpc_port);
  if (!status.ok()) {
    throw std::runtime_error(absl::StrCat(
        "Failed to start gRPC server in KVCacheManagerWithWorkerService: ",
        status.message()));
  }
}

KVCacheManagerWithWorkerService::~KVCacheManagerWithWorkerService() = default;

int KVCacheManagerWithWorkerService::GetGrpcPort() const {
  return WorkerServiceServer::GetInstance().GetGrpcPort();
}

}  // namespace tpu_raiden
