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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_CONTROLLER_CLIENT_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_CONTROLLER_CLIENT_H_

#include <grpcpp/grpcpp.h>

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "grpcpp/channel.h"
#include "tpu_raiden/proto/controller_service.grpc.pb.h"
#include "tpu_raiden/proto/controller_service.pb.h"

namespace tpu_raiden {
namespace core {
namespace controller {

class RaidenControllerClient {
 public:
  explicit RaidenControllerClient(std::shared_ptr<grpc::Channel> channel);
  explicit RaidenControllerClient(absl::string_view endpoint);

  // Registers a worker with the controller.
  absl::Status RegisterWorker(absl::string_view worker_id,
                              absl::string_view raiden_worker_endpoint,
                              absl::string_view raiden_transfer_endpoint);

  // Alias for snake_case callers.
  absl::Status register_worker(absl::string_view worker_id,
                               absl::string_view raiden_worker_endpoint,
                               absl::string_view raiden_transfer_endpoint) {
    return RegisterWorker(worker_id, raiden_worker_endpoint,
                          raiden_transfer_endpoint);
  }

 private:
  std::unique_ptr<
      ::tpu_raiden::tpu_raiden::proto::RaidenControllerService::Stub>
      stub_;
};

}  // namespace controller
}  // namespace core
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_CONTROLLER_CLIENT_H_
