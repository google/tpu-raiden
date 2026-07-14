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

#include <stdint.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "tpu_raiden/core/controller/test_util.h"
#include "tpu_raiden/core/controller/worker_service_client.h"
#include "tpu_raiden/core/controller/worker_service_impl.h"
#include "tpu_raiden/core/kv_manager_holder.h"
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/proto/worker_service.pb.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace controller {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

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

class WorkerServiceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_server_ = CreateTestWorkerServer();
    unit_.set_job_name("test_job");
    unit_.set_job_replica_id("0");
    unit_.set_data_name("test_data");
  }

  rpc::RaidenIdProto unit_;
  std::unique_ptr<TestWorkerServer> test_server_;
};

TEST_F(WorkerServiceTest, CreateAndDeleteBuffersSuccess) {
  proto::CreateBuffersRequest create_req;
  *create_req.mutable_unit() = unit_;
  auto* spec1 = create_req.add_buffers();
  spec1->set_num_shards(2);
  spec1->set_size_bytes(1024);
  auto* spec2 = create_req.add_buffers();
  spec2->set_num_shards(2);
  spec2->set_size_bytes(1024);

  auto create_resp_or = test_server_->client->CreateBuffers(create_req);
  ASSERT_TRUE(create_resp_or.ok());
  EXPECT_TRUE(create_resp_or->success());
  ASSERT_EQ(create_resp_or->buffers_size(), 2);
  EXPECT_EQ(test_server_->service->GetBufferCount(), 4);

  const auto& buf1 = create_resp_or->buffers(0);
  const auto& buf2 = create_resp_or->buffers(1);
  ASSERT_EQ(buf1.buffer_handles_size(), 2);
  ASSERT_EQ(buf2.buffer_handles_size(), 2);

  uint64_t handle1 = buf1.buffer_handles(0).handle();
  uint64_t handle2 = buf1.buffer_handles(1).handle();
  EXPECT_NE(handle1, handle2);

  auto alloc_or = test_server_->service->GetBuffer(BufferHandle(handle1));
  ASSERT_TRUE(alloc_or.ok());
  EXPECT_EQ(alloc_or->size, 1024);
  EXPECT_NE(alloc_or->ptr, nullptr);

  proto::DeleteBuffersRequest delete_req;
  *delete_req.mutable_unit() = unit_;
  *delete_req.add_sharded_buffers() = buf1;
  *delete_req.add_sharded_buffers() = buf2;

  auto delete_resp_or = test_server_->client->DeleteBuffers(delete_req);
  ASSERT_TRUE(delete_resp_or.ok());
  EXPECT_TRUE(delete_resp_or->success());
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
}

TEST_F(WorkerServiceTest, CreateBuffersWithInvalidSpecFails) {
  proto::CreateBuffersRequest create_req;
  *create_req.mutable_unit() = unit_;
  auto* spec = create_req.add_buffers();
  spec->set_num_shards(0);
  spec->set_size_bytes(512);

  auto create_resp_or = test_server_->client->CreateBuffers(create_req);
  ASSERT_TRUE(create_resp_or.ok());
  EXPECT_FALSE(create_resp_or->success());
  EXPECT_THAT(create_resp_or->message(), HasSubstr("must be positive"));
}

TEST_F(WorkerServiceTest, DeleteNonExistentBufferFails) {
  proto::DeleteBuffersRequest delete_req;
  *delete_req.mutable_unit() = unit_;
  auto* sharded_buf = delete_req.add_sharded_buffers();
  sharded_buf->add_buffer_handles()->set_handle(9999);

  auto delete_resp_or = test_server_->client->DeleteBuffers(delete_req);
  ASSERT_TRUE(delete_resp_or.ok());
  EXPECT_FALSE(delete_resp_or->success());
  EXPECT_THAT(delete_resp_or->message(), HasSubstr("not found"));
}

TEST_F(WorkerServiceTest, TransferBuffersH2hSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(rpc::MEMORY_TYPE_DRAM);
  transfer->set_dst_mem_type(rpc::MEMORY_TYPE_DRAM);
  transfer->add_src_offsets(10);
  transfer->add_dst_offsets(20);
  transfer->set_peer("localhost:8080");

  auto transfer_resp_or = test_server_->client->TransferBuffers(transfer_req);
  ASSERT_TRUE(transfer_resp_or.ok());
  EXPECT_TRUE(transfer_resp_or->success());
  EXPECT_EQ(transfer_resp_or->message(), "Buffers transferred successfully");
  EXPECT_EQ(mock_mgr.d2h_calls, 0);
  EXPECT_EQ(mock_mgr.h2d_calls, 0);
  EXPECT_EQ(mock_mgr.h2h_calls, 1);
  EXPECT_EQ(mock_mgr.last_peer, "localhost:8080");
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(10));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(20));
}

TEST_F(WorkerServiceTest, TransferBuffersH2hMissingPeerFails) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(rpc::MEMORY_TYPE_DRAM);
  transfer->set_dst_mem_type(rpc::MEMORY_TYPE_DRAM);
  transfer->add_src_offsets(10);
  transfer->add_dst_offsets(20);

  auto transfer_resp_or = test_server_->client->TransferBuffers(transfer_req);
  ASSERT_TRUE(transfer_resp_or.ok());
  EXPECT_FALSE(transfer_resp_or->success());
  EXPECT_THAT(transfer_resp_or->message(),
              HasSubstr("Peer address must be provided"));
}

TEST_F(WorkerServiceTest, TransferBuffersH2hInvalidCopySizeFails) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(rpc::MEMORY_TYPE_DRAM);
  transfer->set_dst_mem_type(rpc::MEMORY_TYPE_DRAM);
  transfer->add_src_offsets(10);
  transfer->add_dst_offsets(20);
  transfer->add_copy_sizes(2);
  transfer->set_peer("localhost:8080");

  auto transfer_resp_or = test_server_->client->TransferBuffers(transfer_req);
  ASSERT_TRUE(transfer_resp_or.ok());
  EXPECT_FALSE(transfer_resp_or->success());
  EXPECT_THAT(transfer_resp_or->message(),
              HasSubstr("H2H transfers only support copy size of 1"));
}

TEST_F(WorkerServiceTest, TransferBuffersH2hOverflowFails) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(rpc::MEMORY_TYPE_DRAM);
  transfer->set_dst_mem_type(rpc::MEMORY_TYPE_DRAM);
  transfer->add_src_offsets(2147483648L);
  transfer->add_dst_offsets(20);
  transfer->set_peer("localhost:8080");

  auto transfer_resp_or = test_server_->client->TransferBuffers(transfer_req);
  ASSERT_TRUE(transfer_resp_or.ok());
  EXPECT_FALSE(transfer_resp_or->success());
  EXPECT_THAT(
      transfer_resp_or->message(),
      HasSubstr(
          "H2H transfer dispatch failed: Offset 2147483648 overflows int"));
}

TEST_F(WorkerServiceTest, TransferBuffersWithoutTransferManagerFails) {
  proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(rpc::MEMORY_TYPE_HBM);
  transfer->set_dst_mem_type(rpc::MEMORY_TYPE_DRAM);
  transfer->add_src_offsets(10);
  transfer->add_dst_offsets(20);

  auto transfer_resp_or = test_server_->client->TransferBuffers(transfer_req);
  ASSERT_TRUE(transfer_resp_or.ok());
  EXPECT_FALSE(transfer_resp_or->success());
  EXPECT_THAT(transfer_resp_or->message(),
              HasSubstr("Transfer manager is not configured on WorkerService"));
}

TEST_F(WorkerServiceTest, TransferBuffersOffsetsMismatchFails) {
  proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(rpc::MEMORY_TYPE_HBM);
  transfer->set_dst_mem_type(rpc::MEMORY_TYPE_DRAM);
  transfer->add_src_offsets(10);
  transfer->add_dst_offsets(20);
  transfer->add_dst_offsets(30);

  auto transfer_resp_or = test_server_->client->TransferBuffers(transfer_req);
  ASSERT_TRUE(transfer_resp_or.ok());
  EXPECT_FALSE(transfer_resp_or->success());
  EXPECT_THAT(transfer_resp_or->message(),
              HasSubstr("Source and destination offsets must have the same "
                        "non-zero length"));
}

TEST_F(WorkerServiceTest, TransferBuffersCopySizesMismatchFails) {
  proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(rpc::MEMORY_TYPE_HBM);
  transfer->set_dst_mem_type(rpc::MEMORY_TYPE_DRAM);
  transfer->add_src_offsets(10);
  transfer->add_dst_offsets(20);
  transfer->add_copy_sizes(1);
  transfer->add_copy_sizes(2);

  auto transfer_resp_or = test_server_->client->TransferBuffers(transfer_req);
  ASSERT_TRUE(transfer_resp_or.ok());
  EXPECT_FALSE(transfer_resp_or->success());
  EXPECT_THAT(
      transfer_resp_or->message(),
      HasSubstr(
          "copy_sizes, if provided, must match the length of src_offsets"));
}

TEST_F(WorkerServiceTest, TransferBuffersD2HSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(rpc::MEMORY_TYPE_HBM);
  transfer->set_dst_mem_type(rpc::MEMORY_TYPE_DRAM);
  transfer->add_src_offsets(10);
  transfer->add_src_offsets(30);
  transfer->add_dst_offsets(20);
  transfer->add_dst_offsets(40);
  transfer->add_copy_sizes(1);
  transfer->add_copy_sizes(2);

  auto transfer_resp_or = test_server_->client->TransferBuffers(transfer_req);
  ASSERT_TRUE(transfer_resp_or.ok());
  EXPECT_TRUE(transfer_resp_or->success());
  EXPECT_EQ(transfer_resp_or->message(), "Buffers transferred successfully");
  EXPECT_EQ(mock_mgr.d2h_calls, 1);
  EXPECT_EQ(mock_mgr.h2d_calls, 0);
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(10, 30));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(20, 40));
  EXPECT_THAT(mock_mgr.last_copy_sizes, ElementsAre(1, 2));
}

TEST_F(WorkerServiceTest, TransferBuffersH2DSuccess) {
  MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(KVManagerHolder(&mock_mgr));

  proto::TransferBuffersRequest transfer_req;
  auto* transfer = transfer_req.mutable_transfer();
  transfer->set_src_mem_type(rpc::MEMORY_TYPE_DRAM);
  transfer->set_dst_mem_type(rpc::MEMORY_TYPE_HBM);
  transfer->add_src_offsets(100);
  transfer->add_dst_offsets(200);

  auto transfer_resp_or = test_server_->client->TransferBuffers(transfer_req);
  ASSERT_TRUE(transfer_resp_or.ok());
  EXPECT_TRUE(transfer_resp_or->success());
  EXPECT_EQ(transfer_resp_or->message(), "Buffers transferred successfully");
  EXPECT_EQ(mock_mgr.d2h_calls, 0);
  EXPECT_EQ(mock_mgr.h2d_calls, 1);
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(100));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(200));
  EXPECT_THAT(mock_mgr.last_copy_sizes, ElementsAre(1));
}

}  // namespace
}  // namespace controller
}  // namespace tpu_raiden
