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

#include "tpu_raiden/core/controller/controller_client.h"

#include <grpcpp/grpcpp.h>

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/status.h"
#include "tpu_raiden/proto/controller_service.grpc.pb.h"
#include "tpu_raiden/proto/controller_service.pb.h"

namespace tpu_raiden {
namespace core {
namespace controller {

RaidenControllerClient::RaidenControllerClient(
    std::shared_ptr<grpc::Channel> channel)
    : stub_(::tpu_raiden::tpu_raiden::proto::RaidenControllerService::NewStub(
          channel)) {}

RaidenControllerClient::RaidenControllerClient(absl::string_view endpoint)
    : stub_(::tpu_raiden::tpu_raiden::proto::RaidenControllerService::NewStub(
          grpc::CreateChannel(std::string(endpoint),
                              grpc::InsecureChannelCredentials()))) {}

absl::Status RaidenControllerClient::RegisterWorker(
    absl::string_view worker_id, absl::string_view raiden_worker_endpoint,
    absl::string_view raiden_transfer_endpoint) {
  ::tpu_raiden::tpu_raiden::proto::RegisterWorkerRequest request;
  request.set_worker_id(std::string(worker_id));
  request.set_raiden_worker_endpoint(std::string(raiden_worker_endpoint));
  request.set_raiden_transfer_endpoint(std::string(raiden_transfer_endpoint));

  ::tpu_raiden::tpu_raiden::proto::RegisterWorkerResponse response;
  grpc::ClientContext context;
  grpc::Status status = stub_->RegisterWorker(&context, request, &response);

  if (!status.ok()) {
    return absl::InternalError(status.error_message());
  }
  if (!response.success()) {
    return absl::FailedPreconditionError(response.error_message());
  }
  return absl::OkStatus();
}

}  // namespace controller
}  // namespace core
}  // namespace tpu_raiden
