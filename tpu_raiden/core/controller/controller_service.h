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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_CONTROLLER_SERVICE_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_CONTROLLER_SERVICE_H_

#include <grpcpp/grpcpp.h>

#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tpu_raiden/proto/controller_service.grpc.pb.h"
#include "tpu_raiden/proto/controller_service.pb.h"

namespace tpu_raiden {
namespace core {
namespace controller {

struct WorkerRegistration {
  std::string worker_id;
  std::string raiden_worker_endpoint;
  std::string raiden_transfer_endpoint;
};

class RaidenControllerServiceImpl final
    : public ::tpu_raiden::tpu_raiden::proto::RaidenControllerService::Service {
 public:
  RaidenControllerServiceImpl() = default;
  ~RaidenControllerServiceImpl() override = default;

  // Disallow copy and assign
  RaidenControllerServiceImpl(const RaidenControllerServiceImpl&) = delete;
  RaidenControllerServiceImpl& operator=(const RaidenControllerServiceImpl&) =
      delete;

  grpc::Status RegisterWorker(
      grpc::ServerContext* context,
      const ::tpu_raiden::tpu_raiden::proto::RegisterWorkerRequest* request,
      ::tpu_raiden::tpu_raiden::proto::RegisterWorkerResponse* response)
      override;

  // Retrieves all registered workers.
  std::vector<WorkerRegistration> GetRegisteredWorkers() const;

  // Retrieves a specific worker registration by ID.
  absl::StatusOr<WorkerRegistration> GetWorker(
      absl::string_view worker_id) const;

 private:
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<std::string, WorkerRegistration> workers_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace controller
}  // namespace core
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_CONTROLLER_SERVICE_H_
