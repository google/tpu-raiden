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

#ifndef THIRD_PARTY_TPU_RAIDEN_RPC_COORDINATION_CLIENT_H_
#define THIRD_PARTY_TPU_RAIDEN_RPC_COORDINATION_CLIENT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "grpcpp/channel.h"
#include "tpu_raiden/rpc/coordination.grpc.pb.h"
#include "tpu_raiden/rpc/coordination.pb.h"

namespace tpu_raiden {
namespace rpc {

class CoordinationClient {
 public:
  explicit CoordinationClient(std::shared_ptr<grpc::Channel> channel);

  absl::StatusOr<std::vector<ReplicaInfoEntry>> CollectReplicaInfo(
      int32_t device_id, int32_t expected_count,
      const std::vector<int32_t>& info);

 private:
  std::unique_ptr<CoordinationService::Stub> stub_;
};

}  // namespace rpc
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_RPC_COORDINATION_CLIENT_H_
