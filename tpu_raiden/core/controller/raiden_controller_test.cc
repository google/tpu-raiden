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
#include "tpu_raiden/core/controller/test_util.h"
#include "tpu_raiden/proto/worker_service.pb.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace controller {
namespace {

using ::testing::HasSubstr;

class RaidenControllerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_server_ = CreateTestServer();
    unit_.set_job_name("test_job");
    unit_.set_job_replica_id("0");
    unit_.set_data_name("test_data");
  }

  rpc::RaidenIdProto unit_;
  std::unique_ptr<TestServer> test_server_;
};

TEST_F(RaidenControllerTest, AllocateAndDeallocateSuccess) {
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
  {
    RaidenController controller(unit_, test_server_->channel, /*num_blocks=*/10,
                                /*num_shards=*/2, /*shard_size_bytes=*/1024);

    // 10 logical blocks * 2 shards = 20 buffer allocations created on worker in
    // constructor.
    EXPECT_EQ(test_server_->service->GetBufferCount(), 20);
    EXPECT_EQ(controller.block_manager()->num_locked_blocks(), 0);

    auto alloc_or = controller.Allocate(/*num_blocks=*/3);
    ASSERT_TRUE(alloc_or.ok());
    const auto& sharded_buffers = *alloc_or;
    ASSERT_EQ(sharded_buffers.size(), 3);

    // Buffers remain on worker; locked blocks increase locally.
    EXPECT_EQ(test_server_->service->GetBufferCount(), 20);
    EXPECT_EQ(controller.block_manager()->num_locked_blocks(), 3);

    // Verify structure of first sharded buffer.
    EXPECT_EQ(sharded_buffers[0].buffer_handles_size(), 2);
    EXPECT_NE(sharded_buffers[0].buffer_handles(0).handle(),
              sharded_buffers[0].buffer_handles(1).handle());

    // Deallocate the buffers locally.
    ASSERT_TRUE(controller.Deallocate(sharded_buffers).ok());
    EXPECT_EQ(test_server_->service->GetBufferCount(), 20);
    EXPECT_EQ(controller.block_manager()->num_locked_blocks(), 0);
  }
  // When controller goes out of scope, its destructor deletes all buffers on
  // worker.
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
}

TEST_F(RaidenControllerTest, ConstructWithServerAddressWorks) {
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
  {
    RaidenController controller(unit_, test_server_->server_address,
                                /*num_blocks=*/5, /*num_shards=*/2,
                                /*shard_size_bytes=*/512);

    EXPECT_EQ(test_server_->service->GetBufferCount(), 10);
    EXPECT_EQ(controller.block_manager()->num_locked_blocks(), 0);

    auto alloc_or = controller.Allocate(/*num_blocks=*/2);
    ASSERT_TRUE(alloc_or.ok());
    EXPECT_EQ(alloc_or->size(), 2);
    EXPECT_EQ(controller.block_manager()->num_locked_blocks(), 2);

    ASSERT_TRUE(controller.Deallocate(*alloc_or).ok());
    EXPECT_EQ(controller.block_manager()->num_locked_blocks(), 0);
  }
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
}

TEST_F(RaidenControllerTest, AllocateExceedingCapacityFails) {
  RaidenController controller(unit_, test_server_->channel, /*num_blocks=*/5,
                              /*num_shards=*/1, /*shard_size_bytes=*/512);

  auto alloc_or = controller.Allocate(/*num_blocks=*/10);
  EXPECT_FALSE(alloc_or.ok());
  EXPECT_EQ(controller.block_manager()->num_locked_blocks(), 0);
}

TEST_F(RaidenControllerTest, DeallocateNonExistentBufferFails) {
  RaidenController controller(unit_, test_server_->channel, /*num_blocks=*/5,
                              /*num_shards=*/1, /*shard_size_bytes=*/512);

  proto::BufferProto fake_buffer;
  fake_buffer.add_buffer_handles()->set_handle(9999);

  std::vector<proto::BufferProto> to_delete = {fake_buffer};
  EXPECT_FALSE(controller.Deallocate(to_delete).ok());
}

TEST_F(RaidenControllerTest, ConstructorThrowsOnBufferCreationFailure) {
  // Creating a controller with an invalid address fails to connect and create
  // buffers.
  EXPECT_THROW(
      {
        RaidenController controller(unit_, "localhost:1", /*num_blocks=*/5,
                                    /*num_shards=*/1, /*shard_size_bytes=*/512);
      },
      std::runtime_error);
}

TEST_F(RaidenControllerTest, TransferBuffersDelegatesToWorkerService) {
  RaidenController controller(unit_, test_server_->channel, /*num_blocks=*/5,
                              /*num_shards=*/1, /*shard_size_bytes=*/512);

  std::vector<int64_t> src_offsets = {10};
  std::vector<int64_t> dst_offsets = {20};

  auto resp_or = controller.TransferBuffers(
      rpc::MEMORY_TYPE_HBM, rpc::MEMORY_TYPE_DRAM, src_offsets, dst_offsets);
  ASSERT_TRUE(resp_or.ok());
  EXPECT_FALSE(resp_or->success());
  EXPECT_THAT(resp_or->message(),
              HasSubstr("Transfer manager is not configured on WorkerService"));
}

TEST_F(RaidenControllerTest, TransferBuffersValidationMismatchedOffsets) {
  RaidenController controller(unit_, test_server_->channel, /*num_blocks=*/5,
                              /*num_shards=*/1, /*shard_size_bytes=*/512);

  std::vector<int64_t> src_offsets = {10, 20};
  std::vector<int64_t> dst_offsets = {20};

  auto resp_or = controller.TransferBuffers(
      rpc::MEMORY_TYPE_HBM, rpc::MEMORY_TYPE_DRAM, src_offsets, dst_offsets);
  EXPECT_FALSE(resp_or.ok());
  EXPECT_EQ(resp_or.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      resp_or.status().message(),
      HasSubstr(
          "Source and destination offsets must have the same non-zero length"));
}

TEST_F(RaidenControllerTest, TransferBuffersValidationMismatchedCopySizes) {
  RaidenController controller(unit_, test_server_->channel, /*num_blocks=*/5,
                              /*num_shards=*/1, /*shard_size_bytes=*/512);

  std::vector<int64_t> src_offsets = {10, 20};
  std::vector<int64_t> dst_offsets = {20, 30};
  std::vector<int64_t> copy_sizes = {1};

  auto resp_or =
      controller.TransferBuffers(rpc::MEMORY_TYPE_HBM, rpc::MEMORY_TYPE_DRAM,
                                 src_offsets, dst_offsets, copy_sizes);
  EXPECT_FALSE(resp_or.ok());
  EXPECT_EQ(resp_or.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      resp_or.status().message(),
      HasSubstr(
          "copy_sizes, if provided, must match the length of src_offsets"));
}

TEST_F(RaidenControllerTest, TransferBuffersValidationEmptyOffsets) {
  RaidenController controller(unit_, test_server_->channel, /*num_blocks=*/5,
                              /*num_shards=*/1, /*shard_size_bytes=*/512);

  std::vector<int64_t> src_offsets = {};
  std::vector<int64_t> dst_offsets = {};

  auto resp_or = controller.TransferBuffers(
      rpc::MEMORY_TYPE_HBM, rpc::MEMORY_TYPE_DRAM, src_offsets, dst_offsets);
  EXPECT_FALSE(resp_or.ok());
  EXPECT_EQ(resp_or.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      resp_or.status().message(),
      HasSubstr(
          "Source and destination offsets must have the same non-zero length"));
}

}  // namespace
}  // namespace controller
}  // namespace tpu_raiden
