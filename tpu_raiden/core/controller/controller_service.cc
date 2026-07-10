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

#include "tpu_raiden/core/controller/controller_service.h"

#include <grpcpp/grpcpp.h>

#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tpu_raiden/proto/controller_service.pb.h"

namespace tpu_raiden {
namespace core {
namespace controller {

grpc::Status RaidenControllerServiceImpl::RegisterWorker(
    grpc::ServerContext* context,
    const ::tpu_raiden::tpu_raiden::proto::RegisterWorkerRequest* request,
    ::tpu_raiden::tpu_raiden::proto::RegisterWorkerResponse* response) {
  absl::MutexLock lock(mutex_);

  if (request->worker_id().empty()) {
    response->set_success(false);
    response->set_error_message("worker_id cannot be empty");
    return grpc::Status::OK;
  }
  if (request->raiden_worker_endpoint().empty() &&
      request->raiden_transfer_endpoint().empty()) {
    response->set_success(false);
    response->set_error_message(
        "at least one of raiden_worker_endpoint or raiden_transfer_endpoint"
        " must be provided");
    return grpc::Status::OK;
  }

  WorkerRegistration reg = {
      .worker_id = request->worker_id(),
      .raiden_worker_endpoint = request->raiden_worker_endpoint(),
      .raiden_transfer_endpoint = request->raiden_transfer_endpoint(),
  };

  workers_[reg.worker_id] = std::move(reg);

  response->set_success(true);
  return grpc::Status::OK;
}

std::vector<WorkerRegistration>
RaidenControllerServiceImpl::GetRegisteredWorkers() const {
  absl::MutexLock lock(mutex_);
  std::vector<WorkerRegistration> result;
  result.reserve(workers_.size());
  for (const auto& [_, reg] : workers_) {
    result.push_back(reg);
  }
  return result;
}

absl::StatusOr<WorkerRegistration> RaidenControllerServiceImpl::GetWorker(
    absl::string_view worker_id) const {
  absl::MutexLock lock(mutex_);
  auto it = workers_.find(worker_id);
  if (it == workers_.end()) {
    return absl::NotFoundError(absl::StrCat("Worker not found: ", worker_id));
  }
  return it->second;
}

}  // namespace controller
}  // namespace core
}  // namespace tpu_raiden
