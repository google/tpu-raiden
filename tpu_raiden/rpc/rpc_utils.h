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

#ifndef THIRD_PARTY_TPU_RAIDEN_RPC_RPC_UTILS_H_
#define THIRD_PARTY_TPU_RAIDEN_RPC_RPC_UTILS_H_

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace rpc {

// Helper to establish a simple TCP connection to a peer address (host:port).
// Supports both IPv4 and IPv6.
absl::StatusOr<int> SimpleConnect(absl::string_view peer);

// Helper to send a ControlRequest and receive a ControlResponse synchronously.
absl::Status SendRpcSync(absl::string_view address,
                         const rpc::ControlRequest& req,
                         rpc::ControlResponse& resp);

}  // namespace rpc
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_RPC_RPC_UTILS_H_
