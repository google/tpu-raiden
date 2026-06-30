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

#include "tpu_raiden/core/kv_cache_manager_with_transfer.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/core/tpu_pjrt_manager.h"
#include "tpu_raiden/kv_cache/kv_cache_manager_base.h"

namespace tpu_raiden {
namespace {

using ::absl_testing::IsOk;

class TestKVCacheManagerWithTransfer : public KVCacheManagerWithTransfer {
 public:
  using KVCacheManagerWithTransfer::KVCacheManagerWithTransfer;

  void SetMockLocalIps(const std::vector<std::string>& ips) {
    mock_local_ips_ = ips;
  }

  std::vector<std::string> local_ips() const override {
    if (mock_local_ips_.has_value()) {
      return *mock_local_ips_;
    }
    return KVCacheManagerWithTransfer::local_ips();
  }

 private:
  std::optional<std::vector<std::string>> mock_local_ips_;
};

TEST(KVCacheManagerWithTransferTest, LocalOrchestratedTransfer) {
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * pjrt_manager,
                          TpuPjrtManager::GetDefault());

  // Rank 3 shape: {2, 32, 32} of float.
  // 2 slices, each slice is 32*32 = 1024 elements.
  std::vector<int64_t> shape_dims = {2, 32, 32};
  int64_t elements_per_slice = 32 * 32;
  int64_t total_elements = 2 * elements_per_slice;

  // Initialize buffer with distinct values:
  // Slice 0 (Block 0): 0, 1, 2, ...
  // Slice 1 (Block 1): 1024, 1025, ...
  std::vector<float> host_data(total_elements);
  for (int i = 0; i < total_elements; ++i) {
    host_data[i] = static_cast<float>(i);
  }

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      pjrt_manager->BufferFromHost(host_data.data(), xla::F32, shape_dims));

  ASSERT_THAT(buffer->GetReadyFuture().Await(), IsOk());

  // Create KVCacheManagerWithTransfer
  std::vector<std::vector<xla::PjRtBuffer*>> layer_buffers = {{buffer.get()}};
  auto engine = std::make_unique<KVCacheManagerWithTransfer>(
      layer_buffers,
      /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/std::nullopt,
      /*unsafe_skip_buffer_lock=*/true,
      /*parallelism=*/1,
      /*host_allocator=*/nullptr,
      /*node_id=*/0,
      /*local_control_port=*/0,
      /*max_blocks=*/2,
      /*num_slots=*/2,
      /*timeout_s=*/10.0);

  // Configure staging slots: 2 slots, max 2 blocks per slot
  ASSERT_THAT(engine->ConfigureHostStagingSlots(2, 2), IsOk());

  // Register Block 0 as ready for read on ourselves (producer)
  uint64_t uuid = 12345;
  std::string req_id = "local_test_req";
  engine->NotifyForRead(req_id, uuid, /*block_ids=*/{0});

  // Get our own control port
  int port = engine->local_control_port();
  ASSERT_GT(port, 0);
  std::string remote_endpoint = "127.0.0.1:" + std::to_string(port);

  // Start read: pull Block 0 (remote) and store it in Block 1 (local)
  engine->StartRead(req_id, uuid, remote_endpoint,
                    /*remote_block_ids=*/{0},
                    /*local_block_ids=*/{1});

  // Wait for transfer to complete by polling CompleteReadRaw
  bool done = false;
  for (int i = 0; i < 100; ++i) {
    auto [done_sending, done_recving, failed_recving] =
        engine->CompleteReadRaw();
    if (std::find(failed_recving.begin(), failed_recving.end(), req_id) !=
        failed_recving.end()) {
      FAIL() << "Transfer failed";
    }
    if (std::find(done_recving.begin(), done_recving.end(), req_id) !=
        done_recving.end()) {
      done = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  ASSERT_TRUE(done) << "Transfer timed out";

  // Verify device buffer: Block 1 should now contain Block 0's original data.
  // i.e. elements 1024..2047 should be equal to 0..1023.
  TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer->ToLiteral().Await());
  auto read_back = literal->data<float>();
  ASSERT_EQ(read_back.size(), total_elements);

  // Block 0 (unchanged)
  for (int i = 0; i < elements_per_slice; ++i) {
    EXPECT_EQ(read_back[i], static_cast<float>(i));
  }
  // Block 1 (overwritten with Block 0's data)
  for (int i = 0; i < elements_per_slice; ++i) {
    EXPECT_EQ(read_back[elements_per_slice + i], static_cast<float>(i));
  }
}

TEST(KVCacheManagerWithTransferTest, StartReadAcceptsParallelism) {
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * pjrt_manager,
                          TpuPjrtManager::GetDefault());
  std::vector<int64_t> shape_dims = {2, 32, 32};
  std::vector<float> host_data(2 * 32 * 32, 0.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      pjrt_manager->BufferFromHost(host_data.data(), xla::F32, shape_dims));
  ASSERT_THAT(buffer->GetReadyFuture().Await(), IsOk());

  std::vector<std::vector<xla::PjRtBuffer*>> layer_buffers = {{buffer.get()}};
  auto engine = std::make_unique<KVCacheManagerWithTransfer>(
      layer_buffers,
      /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/std::nullopt,
      /*unsafe_skip_buffer_lock=*/true,
      /*parallelism=*/1,
      /*host_allocator=*/nullptr,
      /*node_id=*/0,
      /*local_control_port=*/0,
      /*max_blocks=*/2,
      /*num_slots=*/2,
      /*timeout_s=*/10.0);
  ASSERT_THAT(engine->ConfigureHostStagingSlots(2, 2), IsOk());

  // Calling StartRead with a non-existent port throws or returns an op that
  // fails
  engine->StartRead("req_parallel", 99999, "127.0.0.1:8888", {0}, {0},
                    /*parallelism=*/2);
}
TEST(KVCacheManagerWithTransferTest,
     LocalOrchestratedTransferToCustomHostBlock) {
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * pjrt_manager,
                          TpuPjrtManager::GetDefault());

  // Rank 3 shape: {2, 32, 32} of float.
  // 2 slices, each slice is 32*32 = 1024 elements.
  std::vector<int64_t> shape_dims = {2, 32, 32};
  int64_t elements_per_slice = 32 * 32;
  int64_t total_elements = 2 * elements_per_slice;

  // Initialize buffer with distinct values:
  // Slice 0 (Block 0): 0, 1, 2, ...
  // Slice 1 (Block 1): 1024, 1025, ...
  std::vector<float> host_data(total_elements);
  for (int i = 0; i < total_elements; ++i) {
    host_data[i] = static_cast<float>(i);
  }

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      pjrt_manager->BufferFromHost(host_data.data(), xla::F32, shape_dims));

  ASSERT_THAT(buffer->GetReadyFuture().Await(), IsOk());

  // Create KVCacheManagerWithTransfer
  std::vector<std::vector<xla::PjRtBuffer*>> layer_buffers = {{buffer.get()}};
  auto engine = std::make_unique<KVCacheManagerWithTransfer>(
      layer_buffers,
      /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/6,  // Allocate 6 blocks in host
      /*unsafe_skip_buffer_lock=*/true,
      /*parallelism=*/1,
      /*host_allocator=*/nullptr,
      /*node_id=*/0,
      /*local_control_port=*/0,
      /*max_blocks=*/2,
      /*num_slots=*/2,
      /*timeout_s=*/10.0);

  // Configure staging slots: 2 slots, max 2 blocks per slot
  ASSERT_THAT(engine->ConfigureHostStagingSlots(2, 2), IsOk());

  // Register Block 0 as ready for read on ourselves (producer)
  uint64_t uuid = 12346;
  std::string req_id = "local_test_custom_host_req";
  engine->NotifyForRead(req_id, uuid, /*block_ids=*/{0});

  // Get our own control port
  int port = engine->local_control_port();
  ASSERT_GT(port, 0);
  std::string remote_endpoint = "127.0.0.1:" + std::to_string(port);

  // Start read: pull Remote Block 0 -> Local Device Block 1,
  // but target Local Host Block 0 as the staging/host block.
  engine->StartRead(req_id, uuid, remote_endpoint,
                    /*remote_block_ids=*/{0},
                    /*local_block_ids=*/{1},
                    /*parallelism=*/1,
                    /*local_host_block_ids=*/std::vector<int64_t>{4});

  // Wait for transfer to complete
  bool done = false;
  for (int i = 0; i < 100; ++i) {
    auto [done_sending, done_recving, failed_recving] =
        engine->CompleteReadRaw();
    if (std::find(failed_recving.begin(), failed_recving.end(), req_id) !=
        failed_recving.end()) {
      FAIL() << "Transfer failed";
    }
    if (std::find(done_recving.begin(), done_recving.end(), req_id) !=
        done_recving.end()) {
      done = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  ASSERT_TRUE(done) << "Transfer timed out";

  // Verify device buffer: Block 1 should now contain Block 0's original data.
  TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer->ToLiteral().Await());
  auto read_back = literal->data<float>();
  ASSERT_EQ(read_back.size(), total_elements);

  // Block 1 (overwritten with Block 0's data: 0..1023)
  for (int i = 0; i < elements_per_slice; ++i) {
    EXPECT_EQ(read_back[elements_per_slice + i], static_cast<float>(i));
  }

  // Verify host buffer:
  uint8_t* host_block_4 = engine->GetBlockHostPointer(0, 0, 4);
  uint8_t* host_block_5 = engine->GetBlockHostPointer(0, 0, 5);

  float* host_block_4_float = reinterpret_cast<float*>(host_block_4);
  float* host_block_5_float = reinterpret_cast<float*>(host_block_5);

  // Host Block 4 should contain the transferred data in tiled layout.
  // Tile width is 128 floats (512 bytes) due to TPU alignment.
  const int tile_width = 128;
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 32; ++c) {
      int physical_idx = r * tile_width + c;
      int logical_val = r * 32 + c;
      EXPECT_EQ(host_block_4_float[physical_idx],
                static_cast<float>(logical_val));
    }
  }

  // Host Block 5 should still be 0 (untouched, since we targeted Host Block 4)
  for (int i = 0; i < elements_per_slice; ++i) {
    EXPECT_EQ(host_block_5_float[i], 0.0f);
  }
}

using ::tpu_raiden::kv_cache::KVCacheManagerBase;

TEST(KVCacheManagerBaseTest, E2eRemoteD2DBlockWrite) {
  TF_ASSERT_OK_AND_ASSIGN(tpu_raiden::TpuPjrtManager * manager,
                          tpu_raiden::TpuPjrtManager::GetDefault());
  xla::PjRtClient* client = manager->client();

  std::vector<float> host_data = {1.1f, 2.2f, 3.3f, 4.4f};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> src_buffer,
      manager->BufferFromHost(host_data.data(), xla::F32, {4}));
  ASSERT_THAT(src_buffer->GetReadyFuture().Await(), IsOk());

  TF_ASSERT_OK_AND_ASSIGN(int64_t physical_size,
                          src_buffer->GetOnDeviceSizeInBytes());

  std::vector<uint8_t> receiver_buffer(physical_size, 0);

  KVCacheManagerBase receiver(1, 1, physical_size, std::nullopt, 10);
  receiver.SetExternalHostPointers({receiver_buffer.data()},
                                   {static_cast<size_t>(physical_size)});
  ASSERT_TRUE(receiver.local_port().has_value());
  int receiver_port = receiver.local_port().value();
  std::string receiver_address = absl::StrCat("127.0.0.1:", receiver_port);

  std::vector<float> zero_data(4, 0.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> dst_buffer,
      manager->BufferFromHost(zero_data.data(), xla::F32, {4}));
  ASSERT_THAT(dst_buffer->GetReadyFuture().Await(), IsOk());

  KVCacheManagerBase sender(1, 1, physical_size, std::nullopt, 10);

  BlockMetadata src;
  src.block_id = 0;
  src.data_ptr = src_buffer.get();
  src.address = "127.0.0.1:0";
  src.pjrt_client = client;

  BlockMetadata dst;
  dst.block_id = 0;
  dst.data_ptr = dst_buffer.get();
  dst.address = receiver_address;
  dst.pjrt_client = client;

  TF_ASSERT_OK_AND_ASSIGN(
      auto hold, raiden::BufferHoldAndAlias::Acquire(dst_buffer.get()));
  xla::Future<> recv_future = receiver.RemoteD2DBlockReceive(
      dst.block_id, std::move(hold), physical_size);

  xla::Future<> transfer_future =
      sender.RemoteD2DBlockWrite(src, dst, physical_size);

  absl::Status status = transfer_future.Await();
  ASSERT_THAT(status, IsOk());

  status = recv_future.Await();
  ASSERT_THAT(status, IsOk());

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> literal,
                          dst_buffer->ToLiteral().Await());
  auto read_data = literal->data<float>();
  ASSERT_EQ(read_data.size(), 4);
  EXPECT_EQ(read_data[0], 1.1f);
  EXPECT_EQ(read_data[1], 2.2f);
  EXPECT_EQ(read_data[2], 3.3f);
  EXPECT_EQ(read_data[3], 4.4f);
}

TEST(KVCacheManagerBaseTest, E2eRemoteD2DBlockWriteConcurrent) {
  TF_ASSERT_OK_AND_ASSIGN(tpu_raiden::TpuPjrtManager * manager,
                          tpu_raiden::TpuPjrtManager::GetDefault());
  xla::PjRtClient* client = manager->client();

  std::vector<std::unique_ptr<xla::PjRtBuffer>> src_buffers;
  src_buffers.reserve(10);
  for (int i = 0; i < 10; ++i) {
    std::vector<float> host_data = {i + 1.0f, i + 2.0f, i + 3.0f, i + 4.0f};
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<xla::PjRtBuffer> src_buffer,
        manager->BufferFromHost(host_data.data(), xla::F32, {4}));
    ASSERT_THAT(src_buffer->GetReadyFuture().Await(), IsOk());
    src_buffers.push_back(std::move(src_buffer));
  }

  TF_ASSERT_OK_AND_ASSIGN(int64_t single_buffer_size,
                          src_buffers[0]->GetOnDeviceSizeInBytes());

  std::vector<uint8_t> receiver_buffer(10 * single_buffer_size, 0);

  KVCacheManagerBase receiver(1, 1, single_buffer_size, std::nullopt, 10);
  receiver.SetExternalHostPointers(
      {receiver_buffer.data()}, {static_cast<size_t>(receiver_buffer.size())});
  ASSERT_TRUE(receiver.local_port().has_value());
  int receiver_port = receiver.local_port().value();
  std::string receiver_address = absl::StrCat("127.0.0.1:", receiver_port);

  size_t total_floats = (10 * single_buffer_size) / sizeof(float);
  std::vector<float> zero_data(total_floats, 0.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> dst_buffer,
      manager->BufferFromHost(zero_data.data(), xla::F32,
                              {static_cast<int64_t>(total_floats)}));
  ASSERT_THAT(dst_buffer->GetReadyFuture().Await(), IsOk());

  KVCacheManagerBase sender(1, 1, single_buffer_size, std::nullopt, 10);

  std::vector<xla::Future<>> recv_futures;
  recv_futures.reserve(10);
  for (int i = 0; i < 10; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto hold, raiden::BufferHoldAndAlias::Acquire(dst_buffer.get()));
    recv_futures.push_back(
        receiver.RemoteD2DBlockReceive(i, std::move(hold), single_buffer_size));
  }

  std::vector<xla::Future<>> futures;
  futures.reserve(10);
  for (int i = 0; i < 10; ++i) {
    BlockMetadata src;
    src.block_id = 0;
    src.data_ptr = src_buffers[i].get();
    src.address = "127.0.0.1:0";
    src.pjrt_client = client;

    BlockMetadata dst;
    dst.block_id = i;
    dst.data_ptr = dst_buffer.get();
    dst.address = receiver_address;
    dst.pjrt_client = client;

    futures.push_back(sender.RemoteD2DBlockWrite(src, dst, single_buffer_size));
  }

  absl::Status status = xla::JoinFutures(absl::MakeSpan(futures)).Await();
  ASSERT_THAT(status, IsOk());

  status = xla::JoinFutures(absl::MakeSpan(recv_futures)).Await();
  ASSERT_THAT(status, IsOk());

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> literal,
                          dst_buffer->ToLiteral().Await());
  auto read_data = literal->data<float>();
  ASSERT_EQ(read_data.size(), total_floats);
  size_t floats_per_block = single_buffer_size / sizeof(float);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(read_data[i * floats_per_block + 0], i + 1.0f);
    EXPECT_EQ(read_data[i * floats_per_block + 1], i + 2.0f);
    EXPECT_EQ(read_data[i * floats_per_block + 2], i + 3.0f);
    EXPECT_EQ(read_data[i * floats_per_block + 3], i + 4.0f);
  }
}

TEST(KVCacheManagerWithTransferTest, MultiIpOrchestratedTransfer) {
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * pjrt_manager,
                          TpuPjrtManager::GetDefault());

  std::vector<int64_t> shape_dims = {2, 32, 32};
  int64_t elements_per_slice = 32 * 32;
  int64_t total_elements = 2 * elements_per_slice;

  std::vector<float> host_data(total_elements);
  for (int i = 0; i < total_elements; ++i) {
    host_data[i] = static_cast<float>(i);
  }

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      pjrt_manager->BufferFromHost(host_data.data(), xla::F32, shape_dims));

  ASSERT_THAT(buffer->GetReadyFuture().Await(), IsOk());

  std::vector<std::vector<xla::PjRtBuffer*>> layer_buffers = {{buffer.get()}};

  auto engine = std::make_unique<TestKVCacheManagerWithTransfer>(
      layer_buffers,
      /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/std::nullopt,
      /*unsafe_skip_buffer_lock=*/true,
      /*parallelism=*/2,
      /*host_allocator=*/nullptr,
      /*node_id=*/0,
      /*local_control_port=*/0,
      /*max_blocks=*/2,
      /*num_slots=*/2,
      /*timeout_s=*/10.0);

  engine->SetMockLocalIps({"127.0.0.1", "127.0.0.2"});

  ASSERT_THAT(engine->ConfigureHostStagingSlots(2, 2), IsOk());

  uint64_t uuid = 12347;
  std::string req_id = "multi_ip_test_req";
  engine->NotifyForRead(req_id, uuid, /*block_ids=*/{0});

  int port = engine->local_control_port();
  ASSERT_GT(port, 0);
  std::string remote_endpoint = "127.0.0.1:" + std::to_string(port);

  engine->StartRead(req_id, uuid, remote_endpoint,
                    /*remote_block_ids=*/{0},
                    /*local_block_ids=*/{1},
                    /*parallelism=*/2);

  bool done = false;
  for (int i = 0; i < 100; ++i) {
    auto [done_sending, done_recving, failed_recving] =
        engine->CompleteReadRaw();
    if (std::find(failed_recving.begin(), failed_recving.end(), req_id) !=
        failed_recving.end()) {
      FAIL() << "Transfer failed";
    }
    if (std::find(done_recving.begin(), done_recving.end(), req_id) !=
        done_recving.end()) {
      done = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  ASSERT_TRUE(done) << "Transfer timed out";

  TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer->ToLiteral().Await());
  auto read_back = literal->data<float>();
  ASSERT_EQ(read_back.size(), total_elements);

  for (int i = 0; i < elements_per_slice; ++i) {
    EXPECT_EQ(read_back[i], static_cast<float>(i));
  }
  for (int i = 0; i < elements_per_slice; ++i) {
    EXPECT_EQ(read_back[elements_per_slice + i], static_cast<float>(i));
  }
}

}  // namespace
}  // namespace tpu_raiden
