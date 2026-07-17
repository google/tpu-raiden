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

#include "tpu_raiden/core/controller/controller_server.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "tpu_raiden/core/controller/controller_client.h"

namespace tpu_raiden {
namespace core {
namespace controller {
namespace {

using ::absl_testing::StatusIs;

TEST(ControllerServerTest, StartServerAndGetPortWorks) {
  auto server = ControllerServer::Create();
  ABSL_ASSERT_OK(server->StartServer(/*port=*/0));
  int port = server->GetGrpcPort();
  EXPECT_GT(port, 0);

  // Connect via client and verify gRPC communication works.
  std::string server_address = "localhost:" + std::to_string(port);
  RaidenControllerClient client(server_address);

  absl::Status status =
      client.RegisterWorker("worker_0", "localhost:10001", "localhost:10002");
  ABSL_ASSERT_OK(status);

  auto worker_or = server->GetWorkerRegistry()->GetWorker("worker_0");
  ABSL_ASSERT_OK(worker_or);
  EXPECT_EQ(worker_or->worker_id, "worker_0");
  EXPECT_EQ(worker_or->raiden_worker_endpoint, "localhost:10001");
  EXPECT_EQ(worker_or->raiden_transfer_endpoint, "localhost:10002");
}

TEST(ControllerServerTest, SingletonIsReused) {
  auto& server1 = ControllerServer::GetInstance();
  ABSL_ASSERT_OK(server1.StartServer(/*port=*/0));
  int port1 = server1.GetGrpcPort();

  auto& server2 = ControllerServer::GetInstance();
  ABSL_ASSERT_OK(server2.StartServer(/*port=*/0));
  int port2 = server2.GetGrpcPort();

  EXPECT_EQ(port1, port2);
  EXPECT_EQ(&server1, &server2);
}

TEST(ControllerServerTest, MultipleServersCanRunConcurrently) {
  std::unique_ptr<ControllerServer> server1 = ControllerServer::Create();
  ABSL_ASSERT_OK(server1->StartServer(/*port=*/0));
  int port1 = server1->GetGrpcPort();
  EXPECT_GT(port1, 0);

  std::unique_ptr<ControllerServer> server2 = ControllerServer::Create();
  ABSL_ASSERT_OK(server2->StartServer(/*port=*/0));
  int port2 = server2->GetGrpcPort();
  EXPECT_GT(port2, 0);

  EXPECT_NE(port1, port2);
}

TEST(ControllerServerTest, StartServerWithInvalidPortFails) {
  auto server = ControllerServer::Create();
  absl::Status status = server->StartServer(/*port=*/-1);
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ControllerServerTest, StartServerWithConflictingPortFails) {
  auto server = ControllerServer::Create();
  ABSL_ASSERT_OK(server->StartServer(/*port=*/0));
  int active_port = server->GetGrpcPort();
  int different_port = (active_port == 12345) ? 12346 : 12345;
  absl::Status status = server->StartServer(different_port);
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace controller
}  // namespace core
}  // namespace tpu_raiden
