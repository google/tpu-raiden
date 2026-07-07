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

#include "tpu_raiden/rpc/coordination_client.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "grpcpp/client_context.h"
#include "grpcpp/support/status.h"
#include "tpu_raiden/rpc/coordination.grpc.pb.h"

namespace tpu_raiden {
namespace rpc {

CoordinationClient::CoordinationClient(std::shared_ptr<grpc::Channel> channel)
    : stub_(CoordinationService::NewStub(channel)) {}

absl::StatusOr<std::vector<ReplicaInfoEntry>>
CoordinationClient::CollectReplicaInfo(int32_t device_id,
                                       int32_t expected_count,
                                       const std::vector<int32_t>& info) {
  CollectReplicaInfoRequest request;
  request.set_device_id(device_id);
  request.set_expected_count(expected_count);
  for (int32_t v : info) {
    request.add_info(v);
  }

  CollectReplicaInfoResponse response;
  grpc::ClientContext context;

  grpc::Status status = stub_->CollectReplicaInfo(&context, request, &response);

  if (!status.ok()) {
    return absl::InternalError(
        absl::StrCat("RPC failed: ", status.error_message()));
  }

  std::vector<ReplicaInfoEntry> result;
  result.reserve(response.entries_size());
  for (const auto& entry : response.entries()) {
    result.push_back(entry);
  }

  return result;
}

}  // namespace rpc
}  // namespace tpu_raiden
