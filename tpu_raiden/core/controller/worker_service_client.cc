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
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "grpcpp/client_context.h"
#include "grpcpp/support/status.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_raiden/proto/worker_service.grpc.pb.h"
#include "tpu_raiden/proto/worker_service.pb.h"

namespace tpu_raiden {
namespace controller {

WorkerServiceClient::WorkerServiceClient(std::shared_ptr<grpc::Channel> channel)
    : stub_(proto::WorkerService::NewStub(channel)) {}

tsl::Future<proto::CreateBuffersResponse> WorkerServiceClient::CreateBuffers(
    const proto::CreateBuffersRequest& request) {
  auto [promise, future] = tsl::MakePromise<proto::CreateBuffersResponse>();
  auto context = std::make_shared<grpc::ClientContext>();
  auto response = std::make_shared<proto::CreateBuffersResponse>();

  stub_->async()->CreateBuffers(
      context.get(), &request, response.get(),
      [context, response,
       promise = std::move(promise).ToShared()](grpc::Status status) {
        if (!status.ok()) {
          promise->Set(absl::InternalError(absl::StrCat(
              "CreateBuffers RPC failed: ", status.error_message())));
        } else {
          promise->Set(std::move(*response));
        }
      });
  return future;
}

tsl::Future<proto::DeleteBuffersResponse> WorkerServiceClient::DeleteBuffers(
    const proto::DeleteBuffersRequest& request) {
  auto [promise, future] = tsl::MakePromise<proto::DeleteBuffersResponse>();
  auto context = std::make_shared<grpc::ClientContext>();
  auto response = std::make_shared<proto::DeleteBuffersResponse>();

  stub_->async()->DeleteBuffers(
      context.get(), &request, response.get(),
      [context, response,
       promise = std::move(promise).ToShared()](grpc::Status status) {
        if (!status.ok()) {
          promise->Set(absl::InternalError(absl::StrCat(
              "DeleteBuffers RPC failed: ", status.error_message())));
        } else {
          promise->Set(std::move(*response));
        }
      });
  return future;
}

tsl::Future<> WorkerServiceClient::TransferBuffers(
    const proto::TransferBuffersRequest& request) {
  auto [promise, future] = tsl::MakePromise<>();
  auto context = std::make_shared<grpc::ClientContext>();
  auto response = std::make_shared<proto::TransferBuffersResponse>();

  stub_->async()->TransferBuffers(
      context.get(), &request, response.get(),
      [context, response,
       promise = std::move(promise).ToShared()](grpc::Status status) {
        if (!status.ok()) {
          promise->Set(absl::InternalError(absl::StrCat(
              "TransferBuffers RPC failed: ", status.error_message())));
        } else if (!response->success()) {
          promise->Set(absl::InternalError(response->message()));
        } else {
          promise->Set(absl::OkStatus());
        }
      });
  return future;
}

}  // namespace controller
}  // namespace tpu_raiden
