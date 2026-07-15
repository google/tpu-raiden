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

#include "tpu_raiden/core/controller/raiden_controller.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "tpu_raiden/core/controller/controller_client.h"
#include "tpu_raiden/core/controller/raiden_orchestrator.h"
#include "tpu_raiden/core/controller/test_util.h"
#include "tpu_raiden/core/kv_manager_holder.h"
#include "tpu_raiden/proto/worker_service.pb.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace controller {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

class RaidenControllerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_server_ = CreateTestWorkerServer();
    unit_.set_job_name("test_job");
    unit_.set_job_replica_id("0");
    unit_.set_data_name("test_data");

    orchestrator_service_ = std::make_unique<RaidenOrchestrator>();
    grpc::ServerBuilder builder;
    int bound_port = 0;
    builder.AddListeningPort("0.0.0.0:0", grpc::InsecureServerCredentials(),
                             &bound_port);
    builder.RegisterService(orchestrator_service_.get());
    orchestrator_server_ = builder.BuildAndStart();
    orchestrator_address_ = absl::StrCat("localhost:", bound_port);
  }

  void TearDown() override {
    if (orchestrator_server_) {
      orchestrator_server_->Shutdown();
      orchestrator_server_->Wait();
    }
  }

  void RegisterAndInitWorker(RaidenController& controller,
                             const std::string& worker_id,
                             const std::string& worker_address) {
    std::string server_address =
        absl::StrCat("localhost:", controller.raiden_controller_port());
    core::controller::RaidenControllerClient client(server_address);
    auto status =
        client.RegisterWorker(worker_id, worker_address, worker_address);
    ASSERT_TRUE(status.ok()) << status.message();
  }

  rpc::RaidenIdProto unit_;
  std::unique_ptr<TestWorkerServer> test_server_;
  std::unique_ptr<RaidenOrchestrator> orchestrator_service_;
  std::unique_ptr<grpc::Server> orchestrator_server_;
  std::string orchestrator_address_;
};

TEST_F(RaidenControllerTest, AllocateAndDeallocateSuccess) {
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
  {
    RaidenController controller(
        unit_, std::vector<std::string>{test_server_->server_address},
        /*num_blocks=*/10, /*num_shards=*/2,
        /*shard_size_bytes=*/1024);

    EXPECT_EQ(test_server_->service->GetBufferCount(), 20);
    EXPECT_EQ(controller.block_manager()->num_locked_blocks(), 0);

    auto alloc_or = controller.Allocate(/*num_blocks=*/3);
    ASSERT_TRUE(alloc_or.ok());
    const auto& sharded_buffers = *alloc_or;
    ASSERT_EQ(sharded_buffers.size(), 3);

    EXPECT_EQ(test_server_->service->GetBufferCount(), 20);
    EXPECT_EQ(controller.block_manager()->num_locked_blocks(), 3);

    EXPECT_EQ(sharded_buffers[0].buffer_handles_size(), 2);
    EXPECT_NE(sharded_buffers[0].buffer_handles(0).handle(),
              sharded_buffers[0].buffer_handles(1).handle());

    ASSERT_TRUE(controller.Deallocate(sharded_buffers).ok());
    EXPECT_EQ(test_server_->service->GetBufferCount(), 20);
    EXPECT_EQ(controller.block_manager()->num_locked_blocks(), 0);
  }
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
}

TEST_F(RaidenControllerTest, RegisterWorkerSuccessfully) {
  RaidenController controller(unit_, /*num_blocks=*/5, /*num_shards=*/2,
                              /*shard_size_bytes=*/512);

  // Initial state shouldn't have the worker in registry
  EXPECT_FALSE(controller.worker_registry()->GetWorker("worker_0").ok());

  RegisterAndInitWorker(controller, "worker_0", test_server_->server_address);

  // Worker should be registered
  auto worker_or = controller.worker_registry()->GetWorker("worker_0");
  ASSERT_OK(worker_or);
  EXPECT_NE(worker_or->worker_service_client, nullptr);

  // Registration should have synchronously created the buffers
  EXPECT_EQ(test_server_->service->GetBufferCount(), 10);
}

TEST_F(RaidenControllerTest, ConstructWithServerAddressWorks) {
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
  {
    RaidenController controller(
        unit_, std::vector<std::string>{test_server_->server_address},
        /*num_blocks=*/5, /*num_shards=*/2,
        /*shard_size_bytes=*/512);
    EXPECT_EQ(test_server_->service->GetBufferCount(), 10);
  }
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
}

TEST_F(RaidenControllerTest, AllocateExceedingCapacityFails) {
  RaidenController controller(
      unit_, std::vector<std::string>{test_server_->server_address},
      /*num_blocks=*/5, /*num_shards=*/1, /*shard_size_bytes=*/512);

  auto alloc_or = controller.Allocate(/*num_blocks=*/10);
  EXPECT_FALSE(alloc_or.ok());
  EXPECT_EQ(controller.block_manager()->num_locked_blocks(), 0);
}

TEST_F(RaidenControllerTest, DeallocateNonExistentBufferFails) {
  RaidenController controller(
      unit_, std::vector<std::string>{test_server_->server_address},
      /*num_blocks=*/5, /*num_shards=*/1, /*shard_size_bytes=*/512);

  proto::BufferProto fake_buffer;
  fake_buffer.add_buffer_handles()->set_handle(9999);

  std::vector<proto::BufferProto> to_delete = {fake_buffer};
  EXPECT_FALSE(controller.Deallocate(to_delete).ok());
}

TEST_F(RaidenControllerTest, ConstructorThrowsOnBufferCreationFailure) {
  EXPECT_THROW(
      {
        RaidenController controller(
            unit_, std::vector<std::string>{"localhost:1"}, /*num_blocks=*/5,
            /*num_shards=*/1, /*shard_size_bytes=*/512);
      },
      std::runtime_error);
}

TEST_F(RaidenControllerTest, RegisterWorkerFailsOnBufferCreationFailure) {
  RaidenController controller(unit_, /*num_blocks=*/5,
                              /*num_shards=*/1, /*shard_size_bytes=*/512);

  std::string server_address =
      absl::StrCat("localhost:", controller.raiden_controller_port());
  core::controller::RaidenControllerClient client(server_address);
  auto status =
      client.RegisterWorker("worker_bad", "localhost:1", "localhost:1");
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.message(),
              HasSubstr("CreateBuffers RPC failed: failed to connect"));
}

TEST_F(RaidenControllerTest, TransferBuffersDelegatesToWorkerService) {
  RaidenController controller(
      unit_, std::vector<std::string>{test_server_->server_address},
      /*num_blocks=*/5, /*num_shards=*/1, /*shard_size_bytes=*/512);

  std::vector<int64_t> src_offsets = {10};
  std::vector<int64_t> dst_offsets = {20};

  auto status = controller.TransferBuffers("worker_0", rpc::MEMORY_TYPE_HBM,
                                            rpc::MEMORY_TYPE_DRAM, src_offsets,
                                            dst_offsets)
                          .Await();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Transfer manager is not configured on WorkerService"));
}

TEST_F(RaidenControllerTest, TransferBuffersValidationMismatchedOffsets) {
  RaidenController controller(
      unit_, std::vector<std::string>{test_server_->server_address},
      /*num_blocks=*/5, /*num_shards=*/1, /*shard_size_bytes=*/512);

  std::vector<int64_t> src_offsets = {10, 30};
  std::vector<int64_t> dst_offsets = {20};

  auto status = controller.TransferBuffers("worker_0", rpc::MEMORY_TYPE_HBM,
                                            rpc::MEMORY_TYPE_DRAM, src_offsets,
                                            dst_offsets)
                          .Await();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      status.message(),
      HasSubstr(
          "Source and destination offsets must have the same non-zero length"));
}

TEST_F(RaidenControllerTest, TransferBuffersValidationMismatchedCopySizes) {
  RaidenController controller(
      unit_, std::vector<std::string>{test_server_->server_address},
      /*num_blocks=*/5, /*num_shards=*/1, /*shard_size_bytes=*/512);

  std::vector<int64_t> src_offsets = {10, 20};
  std::vector<int64_t> dst_offsets = {20, 30};
  std::vector<int64_t> copy_sizes = {1};

  auto status =
      controller
          .TransferBuffers(rpc::MEMORY_TYPE_HBM, rpc::MEMORY_TYPE_DRAM,
                           src_offsets, dst_offsets, copy_sizes)
          .Await();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      status.message(),
      HasSubstr(
          "copy_sizes, if provided, must match the length of src_offsets"));
}

TEST_F(RaidenControllerTest, TransferBuffersValidationEmptyOffsets) {
  RaidenController controller(
      unit_, std::vector<std::string>{test_server_->server_address},
      /*num_blocks=*/5, /*num_shards=*/1, /*shard_size_bytes=*/512);

  std::vector<int64_t> src_offsets = {};
  std::vector<int64_t> dst_offsets = {};

  auto status =
      controller
          .TransferBuffers(rpc::MEMORY_TYPE_HBM, rpc::MEMORY_TYPE_DRAM,
                           src_offsets, dst_offsets)
          .Await();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      status.message(),
      HasSubstr(
          "Source and destination offsets must have the same non-zero length"));
}

TEST_F(RaidenControllerTest, MultiWorkerBroadcastSupport) {
  auto test_server2 = CreateTestWorkerServer();
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
  EXPECT_EQ(test_server2->service->GetBufferCount(), 0);

  std::vector<std::string> addresses = {test_server_->server_address,
                                        test_server2->server_address};

  {
    RaidenController controller(unit_, addresses, /*num_blocks=*/5,
                                /*num_shards=*/2, /*shard_size_bytes=*/512);

    // Buffers created on both worker servers.
    EXPECT_EQ(test_server_->service->GetBufferCount(), 10);
    EXPECT_EQ(test_server2->service->GetBufferCount(), 10);

    // Broadcast TransferBuffers.
    std::vector<int64_t> src_offsets = {10};
    std::vector<int64_t> dst_offsets = {20};
    auto status =
        controller
            .TransferBuffers(rpc::MEMORY_TYPE_HBM, rpc::MEMORY_TYPE_DRAM,
                             src_offsets, dst_offsets)
            .Await();
    EXPECT_FALSE(status.ok());
    EXPECT_THAT(
        status.message(),
        HasSubstr("Transfer manager is not configured on WorkerService"));
  }

  // Buffers cleaned up on both worker servers on destructor.
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
  EXPECT_EQ(test_server2->service->GetBufferCount(), 0);
}

TEST_F(RaidenControllerTest, TransferBuffersD2HSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  RaidenController controller(unit_, /*num_blocks=*/5, /*num_shards=*/1,
                              /*shard_size_bytes=*/512);
  RegisterAndInitWorker(controller, "worker_0", test_server_->server_address);

  std::vector<int64_t> src_offsets = {10, 30};
  std::vector<int64_t> dst_offsets = {20, 40};
  std::vector<int64_t> copy_sizes = {1, 2};

  auto status = controller
                    .TransferBuffers("worker_0", rpc::MEMORY_TYPE_HBM,
                                     rpc::MEMORY_TYPE_DRAM, src_offsets,
                                     dst_offsets, copy_sizes)
                    .Await();
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(mock_mgr.d2h_calls, 1);
  EXPECT_EQ(mock_mgr.h2d_calls, 0);
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(10, 30));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(20, 40));
  EXPECT_THAT(mock_mgr.last_copy_sizes, ElementsAre(1, 2));
}

TEST_F(RaidenControllerTest, TransferBuffersH2DSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  RaidenController controller(unit_, /*num_blocks=*/5, /*num_shards=*/1,
                              /*shard_size_bytes=*/512);
  RegisterAndInitWorker(controller, "worker_0", test_server_->server_address);

  std::vector<int64_t> src_offsets = {100};
  std::vector<int64_t> dst_offsets = {200};

  auto status =
      controller
          .TransferBuffers("worker_0", rpc::MEMORY_TYPE_DRAM,
                           rpc::MEMORY_TYPE_HBM, src_offsets, dst_offsets)
          .Await();
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(mock_mgr.d2h_calls, 0);
  EXPECT_EQ(mock_mgr.h2d_calls, 1);
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(100));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(200));
  EXPECT_THAT(mock_mgr.last_copy_sizes, ElementsAre(1));
}

TEST_F(RaidenControllerTest, TransferBuffersH2HSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  RaidenController controller(unit_, /*num_blocks=*/5, /*num_shards=*/1,
                              /*shard_size_bytes=*/512);
  RegisterAndInitWorker(controller, "worker_0", test_server_->server_address);

  std::vector<int64_t> src_offsets = {10};
  std::vector<int64_t> dst_offsets = {20};

  auto status =
      controller
          .TransferBuffers("worker_0", rpc::MEMORY_TYPE_DRAM,
                           rpc::MEMORY_TYPE_DRAM, src_offsets, dst_offsets,
                           /*copy_sizes=*/{}, "localhost:8080")
          .Await();
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(mock_mgr.d2h_calls, 0);
  EXPECT_EQ(mock_mgr.h2d_calls, 0);
  EXPECT_EQ(mock_mgr.h2h_calls, 1);
  EXPECT_EQ(mock_mgr.last_peer, "localhost:8080");
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(10));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(20));
}

TEST_F(RaidenControllerTest, ControllerServiceRegistrationAndResolution) {
  RaidenController controller(unit_, /*num_blocks=*/5,
                              /*num_shards=*/2, /*shard_size_bytes=*/512,
                              /*raiden_controller_port=*/0,
                              orchestrator_address_);

  int raiden_controller_port = controller.raiden_controller_port();
  EXPECT_GT(raiden_controller_port, 0);

  auto resolve_or = controller.ResolvePeerController(unit_);
  ASSERT_TRUE(resolve_or.ok());
  EXPECT_EQ(*resolve_or, absl::StrCat("localhost:", raiden_controller_port));

  rpc::RaidenIdProto fake_peer;
  fake_peer.set_job_name("fake_job");
  auto fail_or = controller.ResolvePeerController(fake_peer);
  EXPECT_FALSE(fail_or.ok());
  EXPECT_EQ(fail_or.status().code(), absl::StatusCode::kNotFound);
}

}  // namespace
}  // namespace controller
}  // namespace tpu_raiden
