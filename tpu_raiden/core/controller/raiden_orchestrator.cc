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

#include "tpu_raiden/core/controller/raiden_orchestrator.h"

#include <string>

#include "grpcpp/grpcpp.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/support/status.h"
#include "tpu_raiden/proto/orchestrator_service.pb.h"

namespace tpu_raiden {

::grpc::Status RaidenOrchestrator::RegisterController(
    ::grpc::ServerContext* context,
    const proto::RegisterControllerRequest* request,
    proto::RegisterControllerResponse* response) {
  if (!request->has_raiden_id() || request->address().empty()) {
    return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT,
                          "Missing raiden_id or address.");
  }

  std::string id_key = absl::StrCat(request->raiden_id().job_name(), "/",
                                    request->raiden_id().job_replica_id(), "/",
                                    request->raiden_id().data_name());

  absl::MutexLock lock(mutex_);
  registry_[id_key] = request->address();

  LOG(INFO) << "Registered RaidenId with address: " << request->address();
  response->set_success(true);
  response->set_message("Successfully registered controller.");

  return ::grpc::Status::OK;
}

::grpc::Status RaidenOrchestrator::ResolveController(
    ::grpc::ServerContext* context,
    const proto::ResolveControllerRequest* request,
    proto::ResolveControllerResponse* response) {
  if (!request->has_raiden_id()) {
    return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT,
                          "Missing raiden_id.");
  }

  std::string id_key = absl::StrCat(request->raiden_id().job_name(), "/",
                                    request->raiden_id().job_replica_id(), "/",
                                    request->raiden_id().data_name());

  absl::MutexLock lock(mutex_);
  auto it = registry_.find(id_key);
  if (it != registry_.end()) {
    response->set_success(true);
    response->set_message("Controller resolved.");
    response->set_address(it->second);
    return ::grpc::Status::OK;
  }

  response->set_success(false);
  response->set_message("Controller not found.");
  return ::grpc::Status(::grpc::StatusCode::NOT_FOUND,
                        "Controller not registered.");
}

}  // namespace tpu_raiden
