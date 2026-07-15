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

#include "tpu_raiden/core/controller/orchestrator_service_client.h"

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/status.h"
#include "tpu_raiden/proto/orchestrator_service.grpc.pb.h"
#include "tpu_raiden/proto/orchestrator_service.pb.h"

namespace tpu_raiden {
namespace controller {

OrchestratorServiceClient::OrchestratorServiceClient(
    std::shared_ptr<grpc::Channel> channel)
    : stub_(proto::OrchestratorService::NewStub(channel)) {}

std::unique_ptr<OrchestratorServiceClient> OrchestratorServiceClient::Create(
    absl::string_view orchestrator_address) {
  auto channel = grpc::CreateChannel(std::string(orchestrator_address),
                                     grpc::InsecureChannelCredentials());
  return std::make_unique<OrchestratorServiceClient>(channel);
}

absl::StatusOr<proto::RegisterControllerResponse>
OrchestratorServiceClient::RegisterController(
    const proto::RegisterControllerRequest& request) {
  proto::RegisterControllerResponse response;
  grpc::ClientContext context;
  grpc::Status status = stub_->RegisterController(&context, request, &response);

  if (!status.ok()) {
    return absl::Status(static_cast<absl::StatusCode>(status.error_code()),
                        status.error_message());
  }
  return response;
}

absl::StatusOr<proto::ResolveControllerResponse>
OrchestratorServiceClient::ResolveController(
    const proto::ResolveControllerRequest& request) {
  proto::ResolveControllerResponse response;
  grpc::ClientContext context;
  grpc::Status status = stub_->ResolveController(&context, request, &response);

  if (!status.ok()) {
    return absl::Status(static_cast<absl::StatusCode>(status.error_code()),
                        status.error_message());
  }
  return response;
}

absl::Status OrchestratorServiceClient::RegisterController(
    const rpc::RaidenIdProto& unit, absl::string_view controller_address) {
  proto::RegisterControllerRequest request;
  *request.mutable_raiden_id() = unit;
  request.set_address(std::string(controller_address));
  auto resp_or = RegisterController(request);
  if (!resp_or.ok()) return resp_or.status();
  if (!resp_or->success()) {
    return absl::FailedPreconditionError(resp_or->message());
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> OrchestratorServiceClient::ResolveController(
    const rpc::RaidenIdProto& peer_id) {
  proto::ResolveControllerRequest request;
  *request.mutable_raiden_id() = peer_id;
  auto resp_or = ResolveController(request);
  if (!resp_or.ok()) return resp_or.status();
  if (!resp_or->success()) {
    return absl::NotFoundError(resp_or->message());
  }
  return resp_or->address();
}

}  // namespace controller
}  // namespace tpu_raiden
