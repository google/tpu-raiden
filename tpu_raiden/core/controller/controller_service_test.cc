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

#include "tpu_raiden/core/controller/controller_service.h"

#include <grpcpp/grpcpp.h>

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "tpu_raiden/core/controller/controller_client.h"

namespace tpu_raiden {
namespace core {
namespace controller {
namespace {

class RaidenControllerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(),
                             &port);
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    ASSERT_NE(server_, nullptr);
    ASSERT_GT(port, 0);

    server_address_ = "127.0.0.1:" + std::to_string(port);
    auto channel = grpc::CreateChannel(server_address_,
                                       grpc::InsecureChannelCredentials());
    client_ = std::make_unique<RaidenControllerClient>(channel);
  }

  void TearDown() override {
    if (server_) {
      server_->Shutdown();
    }
  }

  std::string server_address_;
  RaidenControllerServiceImpl service_;
  std::unique_ptr<grpc::Server> server_;
  std::unique_ptr<RaidenControllerClient> client_;
};

TEST_F(RaidenControllerTest, RegisterWorkerSuccessfully) {
  std::string transfer_addr = "10.0.0.1:8000";
  absl::Status status =
      client_->RegisterWorker("worker_0", "10.0.0.1:9000", transfer_addr);
  EXPECT_OK(status);

  auto workers = service_.GetRegisteredWorkers();
  ASSERT_EQ(workers.size(), 1);
  EXPECT_EQ(workers[0].worker_id, "worker_0");
  EXPECT_EQ(workers[0].raiden_worker_endpoint, "10.0.0.1:9000");
  EXPECT_EQ(workers[0].raiden_transfer_endpoint, transfer_addr);
}

TEST_F(RaidenControllerTest, RegisterWorkerAliasSnakeCase) {
  std::string transfer_addr = "10.0.0.2:8000";
  absl::Status status =
      client_->register_worker("worker_1", "10.0.0.2:9000", transfer_addr);
  EXPECT_OK(status);

  auto worker_or = service_.GetWorker("worker_1");
  ASSERT_OK(worker_or);
  EXPECT_EQ(worker_or->worker_id, "worker_1");
  EXPECT_EQ(worker_or->raiden_worker_endpoint, "10.0.0.2:9000");
  EXPECT_EQ(worker_or->raiden_transfer_endpoint, transfer_addr);
}

TEST_F(RaidenControllerTest, ConstructWithEndpointString) {
  RaidenControllerClient client(server_address_);
  absl::Status status = client.RegisterWorker("worker_endpoint_ctor",
                                              "10.0.0.1:9000", "10.0.0.1:8000");
  EXPECT_OK(status);

  auto workers = service_.GetRegisteredWorkers();
  ASSERT_EQ(workers.size(), 1);
  EXPECT_EQ(workers[0].worker_id, "worker_endpoint_ctor");
}

TEST_F(RaidenControllerTest, RegisterWorkerEmptyIdFails) {
  absl::Status status = client_->RegisterWorker("", "10.0.0.1:9000", {});
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
}

TEST_F(RaidenControllerTest, RegisterWorkerNoAddressesFails) {
  absl::Status status = client_->RegisterWorker("worker_no_addr", "", {});
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace controller
}  // namespace core
}  // namespace tpu_raiden
