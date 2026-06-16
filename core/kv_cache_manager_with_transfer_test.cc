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

#include "core/kv_cache_manager_with_transfer.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "core/tpu_pjrt_manager.h"

namespace tpu_raiden {
namespace {

using ::absl_testing::IsOk;

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
      /*external_host_ptrs=*/std::nullopt,
      /*unsafe_skip_buffer_lock=*/true,
      /*parallelism=*/1,
      /*host_allocator=*/nullptr,
      /*tp_rank=*/0,
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
      /*external_host_ptrs=*/std::nullopt,
      /*unsafe_skip_buffer_lock=*/true,
      /*parallelism=*/1,
      /*host_allocator=*/nullptr,
      /*tp_rank=*/0,
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
TEST(KVCacheManagerWithTransferTest, LocalOrchestratedTransferToCustomHostBlock) {
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
      /*host_blocks_to_allocate=*/2,  // Allocate 2 blocks in host
      /*external_host_ptrs=*/std::nullopt,
      /*unsafe_skip_buffer_lock=*/true,
      /*parallelism=*/1,
      /*host_allocator=*/nullptr,
      /*tp_rank=*/0,
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
                    /*local_host_block_ids=*/std::vector<int64_t>{0});

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
  uint8_t* host_block_0 = engine->GetBlockHostPointer(0, 0, 0);
  uint8_t* host_block_1 = engine->GetBlockHostPointer(0, 0, 1);

  float* host_block_0_float = reinterpret_cast<float*>(host_block_0);
  float* host_block_1_float = reinterpret_cast<float*>(host_block_1);







  // Host Block 0 should contain the transferred data in tiled layout.
  // Tile width is 128 floats (512 bytes) due to TPU alignment.
  const int tile_width = 128;
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 32; ++c) {
      int physical_idx = r * tile_width + c;
      int logical_val = r * 32 + c;
      EXPECT_EQ(host_block_0_float[physical_idx],
                static_cast<float>(logical_val));
    }
  }

  // Host Block 1 should still be 0 (untouched, since we targeted Host Block 0)
  for (int i = 0; i < elements_per_slice; ++i) {
    EXPECT_EQ(host_block_1_float[i], 0.0f);
  }
}

}  // namespace
}  // namespace tpu_raiden
