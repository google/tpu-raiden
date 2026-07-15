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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_CONTROLLER_RAIDEN_ORCHESTRATOR_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_CONTROLLER_RAIDEN_ORCHESTRATOR_H_

#include <string>

#include "grpcpp/grpcpp.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "tpu_raiden/proto/orchestrator_service.grpc.pb.h"
#include "tpu_raiden/proto/orchestrator_service.pb.h"

namespace tpu_raiden {

// Central orchestrator that acts as a directory service for Raiden Controllers.
// It maintains a mapping from RaidenId to Controller physical addresses.
// It serves this via gRPC.
class RaidenOrchestrator final : public proto::OrchestratorService::Service {
 public:
  RaidenOrchestrator() = default;
  ~RaidenOrchestrator() override = default;

  RaidenOrchestrator(const RaidenOrchestrator&) = delete;
  RaidenOrchestrator& operator=(const RaidenOrchestrator&) = delete;

  // Registers a controller address for a given RaidenId.
  ::grpc::Status RegisterController(
      ::grpc::ServerContext* context,
      const proto::RegisterControllerRequest* request,
      proto::RegisterControllerResponse* response) override;

  // Resolves the controller address for a given RaidenId.
  ::grpc::Status ResolveController(
      ::grpc::ServerContext* context,
      const proto::ResolveControllerRequest* request,
      proto::ResolveControllerResponse* response) override;

 private:
  absl::Mutex mutex_;
  // Maps serialized RaidenIdProto to Controller Address (e.g., "ip:port")
  absl::flat_hash_map<std::string, std::string> registry_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_CONTROLLER_RAIDEN_ORCHESTRATOR_H_
