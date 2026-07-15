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

#include "tpu_raiden/core/controller/raiden_orchestrator.h"

#include <memory>
#include <string>

#include "grpcpp/grpcpp.h"
#include <gtest/gtest.h>
#include "grpcpp/support/status.h"
#include "tpu_raiden/proto/orchestrator_service.grpc.pb.h"
#include "tpu_raiden/proto/orchestrator_service.pb.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace {

class RaidenOrchestratorTest : public ::testing::Test {
 protected:
  RaidenOrchestratorTest() {
    grpc::ServerBuilder builder;
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    channel_ = server_->InProcessChannel(grpc::ChannelArguments());
    stub_ = proto::OrchestratorService::NewStub(channel_);
  }

  void TearDown() override {
    server_->Shutdown();
    server_->Wait();
  }

  RaidenOrchestrator service_;
  std::unique_ptr<grpc::Server> server_;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<proto::OrchestratorService::Stub> stub_;
};

TEST_F(RaidenOrchestratorTest, RegisterAndResolveController) {
  // 1. Register a controller
  proto::RegisterControllerRequest reg_req;
  reg_req.mutable_raiden_id()->set_job_name("test_job");
  reg_req.mutable_raiden_id()->set_job_replica_id("1");
  reg_req.set_address("192.168.1.100:5000");

  proto::RegisterControllerResponse reg_res;
  grpc::ClientContext reg_ctx;
  grpc::Status reg_status =
      stub_->RegisterController(&reg_ctx, reg_req, &reg_res);

  EXPECT_TRUE(reg_status.ok());
  EXPECT_TRUE(reg_res.success());

  // 2. Resolve the same controller
  proto::ResolveControllerRequest res_req;
  res_req.mutable_raiden_id()->set_job_name("test_job");
  res_req.mutable_raiden_id()->set_job_replica_id("1");

  proto::ResolveControllerResponse res_res;
  grpc::ClientContext res_ctx;
  grpc::Status res_status =
      stub_->ResolveController(&res_ctx, res_req, &res_res);

  EXPECT_TRUE(res_status.ok());
  EXPECT_TRUE(res_res.success());
  EXPECT_EQ(res_res.address(), "192.168.1.100:5000");
}

TEST_F(RaidenOrchestratorTest, ResolveUnknownControllerFails) {
  proto::ResolveControllerRequest res_req;
  res_req.mutable_raiden_id()->set_job_name("unknown_job");
  res_req.mutable_raiden_id()->set_job_replica_id("99");

  proto::ResolveControllerResponse res_res;
  grpc::ClientContext res_ctx;
  grpc::Status res_status =
      stub_->ResolveController(&res_ctx, res_req, &res_res);

  EXPECT_FALSE(res_status.ok());
  EXPECT_EQ(res_status.error_code(), grpc::StatusCode::NOT_FOUND);
  EXPECT_FALSE(res_res.success());
  EXPECT_TRUE(res_res.address().empty());
}

}  // namespace
}  // namespace tpu_raiden
