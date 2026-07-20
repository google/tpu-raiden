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
#include <stdexcept>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_raiden/core/controller/controller_client.h"
#include "tpu_raiden/core/controller/orchestrator_service_client.h"
#include "tpu_raiden/core/controller/raiden_orchestrator.h"
#include "tpu_raiden/core/controller/test_util.h"
#include "tpu_raiden/core/kv_manager_holder.h"
#include "tpu_raiden/kv_cache/raiden_id.h"
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
    const auto& allocated_buffers = *alloc_or;
    ASSERT_EQ(allocated_buffers.size(), 3);

    EXPECT_EQ(test_server_->service->GetBufferCount(), 20);
    EXPECT_EQ(controller.block_manager()->num_locked_blocks(), 3);

    EXPECT_EQ(allocated_buffers[0].buffer_handles_size(), 2);
    EXPECT_NE(allocated_buffers[0].buffer_handles(0).handle(),
              allocated_buffers[0].buffer_handles(1).handle());

    ASSERT_TRUE(controller.Deallocate(allocated_buffers).ok());
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

TEST_F(RaidenControllerTest, DeallocateInvalidIndexFails) {
  RaidenController controller(
      unit_, std::vector<std::string>{test_server_->server_address},
      /*num_blocks=*/5, /*num_shards=*/1, /*shard_size_bytes=*/512);

  // 1. Missing index
  proto::BufferProto missing_index_buf;
  EXPECT_TRUE(
      absl::IsInvalidArgument(controller.Deallocate({missing_index_buf})));

  // 2. Negative index
  proto::BufferProto negative_index_buf;
  negative_index_buf.set_index(-1);
  EXPECT_TRUE(
      absl::IsInvalidArgument(controller.Deallocate({negative_index_buf})));

  // 3. Out-of-bounds index (e.g. index 10 when num_blocks is 5)
  proto::BufferProto oob_buf;
  oob_buf.set_index(10);
  EXPECT_TRUE(absl::IsInvalidArgument(controller.Deallocate({oob_buf})));
}

TEST_F(RaidenControllerTest, AllocateAndDeallocateBlockIdsSuccess) {
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
  {
    RaidenController controller(
        unit_, std::vector<std::string>{test_server_->server_address},
        /*num_blocks=*/10, /*num_shards=*/2,
        /*shard_size_bytes=*/1024);

    EXPECT_EQ(test_server_->service->GetBufferCount(), 20);
    EXPECT_EQ(controller.block_manager()->num_locked_blocks(), 0);

    auto alloc_or = controller.AllocateBlockIds(/*num_blocks=*/3);
    ASSERT_TRUE(alloc_or.ok());
    const auto& block_ids = *alloc_or;
    ASSERT_EQ(block_ids.size(), 3);

    EXPECT_EQ(test_server_->service->GetBufferCount(), 20);
    EXPECT_EQ(controller.block_manager()->num_locked_blocks(), 3);

    for (int block_id : block_ids) {
      EXPECT_GE(block_id, 0);
      EXPECT_LT(block_id, 10);
    }

    ASSERT_TRUE(controller.DeallocateBlockIds(block_ids).ok());
    EXPECT_EQ(test_server_->service->GetBufferCount(), 20);
    EXPECT_EQ(controller.block_manager()->num_locked_blocks(), 0);
  }
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
}

TEST_F(RaidenControllerTest, DeallocateNonExistentBlockIdFails) {
  RaidenController controller(
      unit_, std::vector<std::string>{test_server_->server_address},
      /*num_blocks=*/5, /*num_shards=*/1, /*shard_size_bytes=*/512);

  std::vector<int> to_delete = {9999};
  EXPECT_FALSE(controller.DeallocateBlockIds(to_delete).ok());
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

  Buffer src_buf(10, {}, std::nullopt, rpc::MEMORY_TYPE_HBM);
  Buffer dst_buf(20, {}, std::nullopt, rpc::MEMORY_TYPE_DRAM);

  auto status =
      controller.TransferBuffers("worker_0", {src_buf}, {dst_buf}).Await();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("Transfer manager is not configured on WorkerService"));
}

TEST_F(RaidenControllerTest, TransferBuffersValidationMismatchedOffsets) {
  RaidenController controller(
      unit_, std::vector<std::string>{test_server_->server_address},
      /*num_blocks=*/5, /*num_shards=*/1, /*shard_size_bytes=*/512);

  Buffer src_buf1(10, {}, std::nullopt, rpc::MEMORY_TYPE_HBM);
  Buffer src_buf2(30, {}, std::nullopt, rpc::MEMORY_TYPE_HBM);
  Buffer dst_buf1(20, {}, std::nullopt, rpc::MEMORY_TYPE_DRAM);

  auto status =
      controller.TransferBuffers("worker_0", {src_buf1, src_buf2}, {dst_buf1})
          .Await();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      status.message(),
      HasSubstr(
          "Source and destination buffers must have the same non-zero length"));
}

TEST_F(RaidenControllerTest, TransferBuffersValidationMismatchedCopySizes) {
  RaidenController controller(
      unit_, std::vector<std::string>{test_server_->server_address},
      /*num_blocks=*/5, /*num_shards=*/1, /*shard_size_bytes=*/512);

  Buffer src_buf1(10, {}, std::nullopt, rpc::MEMORY_TYPE_HBM);
  Buffer src_buf2(20, {}, std::nullopt, rpc::MEMORY_TYPE_HBM);
  Buffer dst_buf1(20, {}, std::nullopt, rpc::MEMORY_TYPE_DRAM);
  Buffer dst_buf2(30, {}, std::nullopt, rpc::MEMORY_TYPE_DRAM);
  std::vector<int64_t> copy_sizes = {1};

  auto status = controller
                    .TransferBuffers({src_buf1, src_buf2}, {dst_buf1, dst_buf2},
                                     copy_sizes)
                    .Await();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      status.message(),
      HasSubstr(
          "copy_sizes, if provided, must match the length of src_buffers"));
}

TEST_F(RaidenControllerTest, TransferBuffersValidationEmptyOffsets) {
  RaidenController controller(
      unit_, std::vector<std::string>{test_server_->server_address},
      /*num_blocks=*/5, /*num_shards=*/1, /*shard_size_bytes=*/512);

  auto status = controller.TransferBuffers({}, {}).Await();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      status.message(),
      HasSubstr(
          "Source and destination buffers must have the same non-zero length"));
}

TEST_F(RaidenControllerTest, TransferBuffersValidationInvalidIndex) {
  RaidenController controller(
      unit_, std::vector<std::string>{test_server_->server_address},
      /*num_blocks=*/5, /*num_shards=*/1, /*shard_size_bytes=*/512);

  // 1. Source buffer has invalid negative index
  Buffer src_buf_invalid(-1, {}, std::nullopt, rpc::MEMORY_TYPE_HBM);
  Buffer dst_buf_valid(2, {}, std::nullopt, rpc::MEMORY_TYPE_DRAM);

  auto status1 =
      controller.TransferBuffers("worker_0", {src_buf_invalid}, {dst_buf_valid})
          .Await();
  EXPECT_FALSE(status1.ok());
  EXPECT_EQ(status1.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status1.message(),
              HasSubstr("Source buffer has invalid negative index: -1"));

  // 2. Destination buffer has invalid negative index
  Buffer src_buf_valid(1, {}, std::nullopt, rpc::MEMORY_TYPE_HBM);
  Buffer dst_buf_invalid(-2, {}, std::nullopt, rpc::MEMORY_TYPE_DRAM);

  auto status2 =
      controller.TransferBuffers("worker_0", {src_buf_valid}, {dst_buf_invalid})
          .Await();
  EXPECT_FALSE(status2.ok());
  EXPECT_EQ(status2.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status2.message(),
              HasSubstr("Destination buffer has invalid negative index: -2"));
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
    Buffer src_buf(10, {}, std::nullopt, rpc::MEMORY_TYPE_HBM);
    Buffer dst_buf(20, {}, std::nullopt, rpc::MEMORY_TYPE_DRAM);
    auto status = controller.TransferBuffers({src_buf}, {dst_buf}).Await();
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

  Buffer src_buf1(10, {}, std::nullopt, rpc::MEMORY_TYPE_HBM);
  Buffer src_buf2(30, {}, std::nullopt, rpc::MEMORY_TYPE_HBM);
  Buffer dst_buf1(20, {}, std::nullopt, rpc::MEMORY_TYPE_DRAM);
  Buffer dst_buf2(40, {}, std::nullopt, rpc::MEMORY_TYPE_DRAM);
  std::vector<int64_t> copy_sizes = {1, 2};

  auto status = controller
                    .TransferBuffers("worker_0", {src_buf1, src_buf2},
                                     {dst_buf1, dst_buf2}, copy_sizes)
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

  Buffer src_buf(100, {}, std::nullopt, rpc::MEMORY_TYPE_DRAM);
  Buffer dst_buf(200, {}, std::nullopt, rpc::MEMORY_TYPE_HBM);

  auto status =
      controller.TransferBuffers("worker_0", {src_buf}, {dst_buf}).Await();
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

  Buffer src_buf(10, {}, std::nullopt, rpc::MEMORY_TYPE_DRAM);
  Buffer dst_buf(20, {}, "localhost:8080", rpc::MEMORY_TYPE_DRAM);

  auto status =
      controller.TransferBuffers("worker_0", {src_buf}, {dst_buf}).Await();
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

TEST_F(RaidenControllerTest, ReadRemoteSuccess) {
  auto test_server2 = CreateTestWorkerServer();
  auto src_controller_server = core::controller::CreateTestControllerServer();

  rpc::RaidenIdProto src_unit;
  src_unit.set_job_name("src_job");
  src_unit.set_job_replica_id("0");
  src_unit.set_data_name("src_data");
  src_unit.set_data_replica_idx(0);

  kv_cache::RaidenId src_raiden_id;
  src_raiden_id.job_name = "src_job";
  src_raiden_id.job_replica_id = "0";
  src_raiden_id.data_name = "src_data";
  src_raiden_id.data_replica_idx = 0;

  rpc::RaidenIdProto dest_unit;
  dest_unit.set_job_name("dest_job");
  dest_unit.set_job_replica_id("0");
  dest_unit.set_data_name("dest_data");
  dest_unit.set_data_replica_idx(0);

  OrchestratorServiceClient orchestrator_client(grpc::CreateChannel(
      orchestrator_address_, grpc::InsecureChannelCredentials()));
  auto register_status = orchestrator_client.RegisterController(
      src_unit, src_controller_server->server_address);
  ASSERT_TRUE(register_status.ok()) << register_status.message();

  RaidenController dest_controller(dest_unit, /*num_blocks=*/5,
                                   /*num_shards=*/2, /*shard_size_bytes=*/512,
                                   /*raiden_controller_port=*/0,
                                   orchestrator_address_);

  RegisterAndInitWorker(dest_controller, "worker_1",
                        test_server2->server_address);
  RegisterAndInitWorker(dest_controller, "worker_0",
                        test_server_->server_address);

  auto register_src_worker = [&](const std::string& worker_id,
                                 const std::string& worker_address,
                                 const std::string& transfer_endpoint) {
    auto status = src_controller_server->client->RegisterWorker(
        worker_id, worker_address, transfer_endpoint);
    ASSERT_TRUE(status.ok()) << status.message();
  };
  register_src_worker("worker_0", "src_worker_0_addr", "src_worker_0_transfer");
  register_src_worker("worker_1", "src_worker_1_addr", "src_worker_1_transfer");

  bool callback_triggered = false;
  std::vector<std::string> callback_peers;
  std::vector<int64_t> callback_src_offsets;
  std::vector<int64_t> callback_dst_offsets;

  src_controller_server->service->SetTransferBuffersCallback(
      [&](rpc::MemoryType src_mem_type, rpc::MemoryType dst_mem_type,
          absl::Span<const int64_t> src_offsets,
          absl::Span<const int64_t> dst_offsets,
          absl::Span<const int64_t> copy_sizes,
          absl::Span<const std::string> peers) {
        callback_triggered = true;
        EXPECT_EQ(src_mem_type, rpc::MemoryType::MEMORY_TYPE_DRAM);
        EXPECT_EQ(dst_mem_type, rpc::MemoryType::MEMORY_TYPE_DRAM);
        callback_src_offsets.assign(src_offsets.begin(), src_offsets.end());
        callback_dst_offsets.assign(dst_offsets.begin(), dst_offsets.end());
        callback_peers.assign(peers.begin(), peers.end());
        return tsl::Future<>(absl::OkStatus());
      });

  std::vector<int32_t> src_host_block_ids = {10, 11};
  std::vector<int32_t> dest_host_block_ids = {20, 21};

  auto read_status =
      dest_controller
          .ReadRemote(src_raiden_id, src_host_block_ids, dest_host_block_ids)
          .Await();
  ASSERT_TRUE(read_status.ok()) << read_status.message();

  EXPECT_TRUE(callback_triggered);
  EXPECT_THAT(callback_src_offsets, ElementsAre(10, 11));
  EXPECT_THAT(callback_dst_offsets, ElementsAre(20, 21));
  EXPECT_THAT(callback_peers, ElementsAre(test_server_->server_address,
                                          test_server2->server_address));
}

TEST_F(RaidenControllerTest, TransferBuffersLocalDramToRemoteHbmSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  RaidenController controller(unit_, /*num_blocks=*/5, /*num_shards=*/1,
                              /*shard_size_bytes=*/512);
  RegisterAndInitWorker(controller, "worker_0", test_server_->server_address);

  auto alloc_src = controller.AllocateBuffers(1);
  ASSERT_TRUE(alloc_src.ok());
  auto src_buffers = *alloc_src;
  src_buffers[0].set_memory_type(rpc::MEMORY_TYPE_DRAM);

  auto alloc_dst = controller.AllocateBuffers(1);
  ASSERT_TRUE(alloc_dst.ok());
  auto dst_buffers = *alloc_dst;
  dst_buffers[0].set_memory_type(rpc::MEMORY_TYPE_HBM);
  dst_buffers[0].set_remote_address("remote_host:9090");

  auto status = controller.TransferBuffers(src_buffers, dst_buffers).Await();
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(mock_mgr.h2d_write_calls, 1);
  EXPECT_EQ(mock_mgr.h2d_calls, 0);
  EXPECT_EQ(mock_mgr.last_peer, "remote_host:9090");
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(0));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(1));
}
}  // namespace
}  // namespace controller
}  // namespace tpu_raiden
