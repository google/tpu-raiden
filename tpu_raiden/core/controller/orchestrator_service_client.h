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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_ORCHESTRATOR_SERVICE_CLIENT_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_ORCHESTRATOR_SERVICE_CLIENT_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "grpcpp/channel.h"
#include "tpu_raiden/proto/orchestrator_service.grpc.pb.h"
#include "tpu_raiden/proto/orchestrator_service.pb.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace controller {

// Client for interacting with the OrchestratorService gRPC endpoint to
// discover peers.
class OrchestratorServiceClient {
 public:
  explicit OrchestratorServiceClient(std::shared_ptr<grpc::Channel> channel);

  // Helper method to connect to the orchestrator using a string address.
  static std::unique_ptr<OrchestratorServiceClient> Create(
      absl::string_view orchestrator_address);

  // Registers the controller's address with the orchestrator.
  absl::StatusOr<proto::RegisterControllerResponse> RegisterController(
      const proto::RegisterControllerRequest& request);

  absl::Status RegisterController(const rpc::RaidenIdProto& unit,
                                  absl::string_view controller_address);

  // Resolves the address of a controller by RaidenId.
  absl::StatusOr<proto::ResolveControllerResponse> ResolveController(
      const proto::ResolveControllerRequest& request);

  absl::StatusOr<std::string> ResolveController(
      const rpc::RaidenIdProto& peer_id);

 private:
  std::unique_ptr<proto::OrchestratorService::Stub> stub_;
};

}  // namespace controller
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_ORCHESTRATOR_SERVICE_CLIENT_H_
