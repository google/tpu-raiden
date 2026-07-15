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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_CONTROLLER_TEST_UTIL_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_CONTROLLER_TEST_UTIL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tpu_raiden/core/raw_transfer_core.h"
#include "grpcpp/channel.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "tpu_raiden/core/controller/controller_client.h"
#include "tpu_raiden/core/controller/controller_service.h"
#include "tpu_raiden/core/controller/worker_service_client.h"
#include "tpu_raiden/core/controller/worker_service_impl.h"

namespace tpu_raiden {
namespace controller {

struct MockTransferManager {
  int d2h_calls = 0;
  int h2d_calls = 0;
  int h2h_calls = 0;
  std::string last_peer;
  std::vector<int64_t> last_src_offsets;
  std::vector<int64_t> last_dst_offsets;
  std::vector<int64_t> last_copy_sizes;

  absl::StatusOr<raiden::PjRtCopyFuture> D2h(
      const std::vector<int64_t>& src_offsets,
      const std::vector<int64_t>& dst_offsets,
      const std::vector<int64_t>& copy_sizes) {
    d2h_calls++;
    last_src_offsets = src_offsets;
    last_dst_offsets = dst_offsets;
    last_copy_sizes = copy_sizes;
    return raiden::PjRtCopyFuture();
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2d(
      const std::vector<int64_t>& src_offsets,
      const std::vector<int64_t>& dst_offsets,
      const std::vector<int64_t>& copy_sizes) {
    h2d_calls++;
    last_src_offsets = src_offsets;
    last_dst_offsets = dst_offsets;
    last_copy_sizes = copy_sizes;
    return raiden::PjRtCopyFuture();
  }

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hWrite(
      std::string peer, const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids = {}, uint64_t uuid = 0,
      int layer_idx = -1) {
    h2h_calls++;
    last_peer = peer;
    last_src_offsets.assign(src_block_ids.begin(), src_block_ids.end());
    last_dst_offsets.assign(dst_block_ids.begin(), dst_block_ids.end());
    return std::make_pair(std::vector<int>{}, raiden::PjRtCopyFuture());
  }
};

// Helper struct wrapping an in-process gRPC test server and client for
// WorkerService.
struct TestWorkerServer {
  std::unique_ptr<WorkerServiceImpl> service;
  std::unique_ptr<grpc::Server> server;
  std::string server_address;
  std::shared_ptr<grpc::Channel> channel;
  std::unique_ptr<WorkerServiceClient> client;

  ~TestWorkerServer() {
    if (server) {
      server->Shutdown();
    }
  }
};

// Creates and starts an in-process gRPC TestWorkerServer hosting
// WorkerServiceImpl on an ephemeral port.
inline std::unique_ptr<TestWorkerServer> CreateTestWorkerServer() {
  auto test_server = std::make_unique<TestWorkerServer>();
  test_server->service = std::make_unique<WorkerServiceImpl>();

  grpc::ServerBuilder builder;
  int selected_port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &selected_port);
  builder.RegisterService(test_server->service.get());
  test_server->server = builder.BuildAndStart();

  test_server->server_address = absl::StrCat("localhost:", selected_port);
  test_server->channel = grpc::CreateChannel(
      test_server->server_address, grpc::InsecureChannelCredentials());
  test_server->client =
      std::make_unique<WorkerServiceClient>(test_server->channel);
  return test_server;
}

}  // namespace controller
}  // namespace tpu_raiden

namespace tpu_raiden {
namespace core {
namespace controller {

// Helper struct wrapping an in-process gRPC test server and client for
// RaidenControllerService.
struct TestControllerServer {
  std::unique_ptr<RaidenControllerServiceImpl> service;
  std::unique_ptr<grpc::Server> server;
  std::string server_address;
  std::shared_ptr<grpc::Channel> channel;
  std::unique_ptr<RaidenControllerClient> client;

  ~TestControllerServer() {
    if (server) {
      server->Shutdown();
    }
  }
};

// Creates and starts an in-process gRPC TestControllerServer hosting
// RaidenControllerServiceImpl on an ephemeral port.
inline std::unique_ptr<TestControllerServer> CreateTestControllerServer() {
  auto test_server = std::make_unique<TestControllerServer>();
  test_server->service =
      std::make_unique<RaidenControllerServiceImpl>();

  grpc::ServerBuilder builder;
  int selected_port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &selected_port);
  builder.RegisterService(test_server->service.get());
  test_server->server = builder.BuildAndStart();

  test_server->server_address = absl::StrCat("localhost:", selected_port);
  test_server->channel = grpc::CreateChannel(
      test_server->server_address, grpc::InsecureChannelCredentials());
  test_server->client =
      std::make_unique<RaidenControllerClient>(test_server->channel);
  return test_server;
}

}  // namespace controller
}  // namespace core
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_CONTROLLER_TEST_UTIL_H_
