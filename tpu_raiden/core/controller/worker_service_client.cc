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

#include "tpu_raiden/core/controller/worker_service_client.h"

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "third_party/grpc/include/grpcpp/client_context.h"
#include "third_party/grpc/include/grpcpp/support/status.h"
#include "tpu_raiden/proto/worker_service.grpc.pb.h"
#include "tpu_raiden/proto/worker_service.pb.h"

namespace tpu_raiden {
namespace controller {

WorkerServiceClient::WorkerServiceClient(std::shared_ptr<grpc::Channel> channel)
    : stub_(proto::WorkerService::NewStub(channel)) {}

absl::StatusOr<proto::CreateBuffersResponse> WorkerServiceClient::CreateBuffers(
    const proto::CreateBuffersRequest& request) {
  proto::CreateBuffersResponse response;
  grpc::ClientContext context;

  grpc::Status status = stub_->CreateBuffers(&context, request, &response);
  if (!status.ok()) {
    return absl::InternalError(
        absl::StrCat("CreateBuffers RPC failed: ", status.error_message()));
  }
  return response;
}

absl::StatusOr<proto::DeleteBuffersResponse> WorkerServiceClient::DeleteBuffers(
    const proto::DeleteBuffersRequest& request) {
  proto::DeleteBuffersResponse response;
  grpc::ClientContext context;

  grpc::Status status = stub_->DeleteBuffers(&context, request, &response);
  if (!status.ok()) {
    return absl::InternalError(
        absl::StrCat("DeleteBuffers RPC failed: ", status.error_message()));
  }
  return response;
}

}  // namespace controller
}  // namespace tpu_raiden
