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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_WORKER_SERVICE_CLIENT_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_WORKER_SERVICE_CLIENT_H_

#include <memory>

#include "absl/status/statusor.h"
#include "grpcpp/channel.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_raiden/proto/worker_service.grpc.pb.h"
#include "tpu_raiden/proto/worker_service.pb.h"

namespace tpu_raiden {
namespace controller {

// Client for interacting with the WorkerService gRPC endpoint on a transfer
// worker asynchronously.
class WorkerServiceClient {
 public:
  explicit WorkerServiceClient(std::shared_ptr<grpc::Channel> channel);

  // Allocates sharded buffers on the remote transfer worker asynchronously.
  tsl::Future<proto::CreateBuffersResponse> CreateBuffers(
      const proto::CreateBuffersRequest& request);

  // Deallocates sharded buffers on the remote transfer worker asynchronously.
  tsl::Future<proto::DeleteBuffersResponse> DeleteBuffers(
      const proto::DeleteBuffersRequest& request);

  // Transfers (copies) disjoint memory regions across memory spaces on the
  // remote transfer worker asynchronously. The transfer specification applies
  // uniformly across all buffers, i.e., all shards and major dimensions
  // (layers or blocks).
  tsl::Future<> TransferBuffers(const proto::TransferBuffersRequest& request);

 private:
  std::unique_ptr<proto::WorkerService::Stub> stub_;
};

}  // namespace controller
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_WORKER_SERVICE_CLIENT_H_
