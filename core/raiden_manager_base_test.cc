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

#include "core/raiden_manager_base.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/future.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "core/host_memory_allocator.h"
#include "core/raw_transfer_core.h"
#include "core/tpu_pjrt_manager.h"
#include "transport/block_transport.h"

namespace tpu_raiden {
namespace {

// Test subclass to populate protected layers_ and implement AllocateBlocks
class TestRaidenManager : public RaidenManagerBase {
 public:
  TestRaidenManager(size_t num_layers, size_t num_shards,
                    size_t slice_byte_size, int block_size = 1,
                    std::optional<int> local_port = std::nullopt,
                    int parallelism = 1)
      : RaidenManagerBase(num_layers, num_shards, slice_byte_size, block_size,
                          local_port, parallelism) {
    layers_.resize(num_layers);
    for (size_t l = 0; l < num_layers; ++l) {
      layers_[l].shards.resize(num_shards);
    }
  }

  absl::StatusOr<std::vector<int>> AllocateBlocks(size_t num_blocks,
                                                  int64_t entity_id) override {
    std::vector<int> ids(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) {
      ids[i] = i;
    }
    return ids;
  }
};

TEST(RaidenManagerBaseTest, LifecycleAndConfig) {
  TestRaidenManager manager(/*num_layers=*/2, /*num_shards=*/4,
                            /*slice_byte_size=*/1024, /*block_size=*/1);

  EXPECT_EQ(manager.num_layers(), 2);
  EXPECT_EQ(manager.num_shards(), 4);
  EXPECT_EQ(manager.slice_byte_size(), 1024);
  EXPECT_EQ(manager.block_size(), 1);
  EXPECT_TRUE(manager.local_port().has_value());
  EXPECT_GT(manager.local_port().value(), 0);
}

TEST(RaidenManagerBaseTest, SetExternalHostPointersWithPinnedAllocator) {
  // Setup TPU PJRT and Pinned Host Allocator
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * pjrt_manager,
                          TpuPjrtManager::GetDefault());
  TF_ASSERT_OK_AND_ASSIGN(auto allocator,
                          HostMemoryAllocator::Create(pjrt_manager->client()));

  // Setup Manager
  size_t num_layers = 2;
  size_t num_shards = 2;
  size_t slice_size = 1024;
  TestRaidenManager manager(num_layers, num_shards, slice_size);

  // Allocate and bind host pointers
  std::vector<HostBufferAllocation> allocations;
  std::vector<const uint8_t*> host_ptrs;
  std::vector<size_t> host_sizes;

  size_t total_buffers = num_layers * num_shards;
  allocations.reserve(total_buffers);
  host_ptrs.reserve(total_buffers);
  host_sizes.reserve(total_buffers);

  for (size_t i = 0; i < total_buffers; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc,
                            allocator->Allocate(slice_size));
    allocations.push_back(std::move(alloc));
    host_ptrs.push_back(allocations.back().ptr);
    host_sizes.push_back(slice_size);
  }

  manager.SetExternalHostPointers(host_ptrs, host_sizes);

  // Verify
  size_t idx = 0;
  for (size_t l = 0; l < num_layers; ++l) {
    for (size_t sh = 0; sh < num_shards; ++sh) {
      EXPECT_EQ(manager.GetHostPointer(l, sh), host_ptrs[idx]);
      EXPECT_EQ(manager.GetHostSize(l, sh), slice_size);
      idx++;
    }
  }
}

TEST(RaidenManagerBaseTest, E2eLoopbackTransferH2h) {
  // Setup TPU PJRT and Allocator
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * pjrt_manager,
                          TpuPjrtManager::GetDefault());
  TF_ASSERT_OK_AND_ASSIGN(auto allocator,
                          HostMemoryAllocator::Create(pjrt_manager->client()));

  size_t num_layers = 1;
  size_t num_shards = 1;
  size_t slice_size = 1024;  // 1KB

  // Instantiate Sender and Receiver on ephemeral ports
  TestRaidenManager sender(num_layers, num_shards, slice_size);
  TestRaidenManager receiver(num_layers, num_shards, slice_size);

  ASSERT_TRUE(sender.local_port().has_value());
  ASSERT_TRUE(receiver.local_port().has_value());

  // Allocate pinned buffers for sender and receiver
  TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation send_alloc,
                          allocator->Allocate(slice_size));
  TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation recv_alloc,
                          allocator->Allocate(slice_size));

  sender.SetExternalHostPointers({send_alloc.ptr}, {slice_size});
  receiver.SetExternalHostPointers({recv_alloc.ptr}, {slice_size});

  // Write a test pattern to sender's buffer
  std::memset(send_alloc.ptr, 0x5A, slice_size);
  // Zero out receiver's buffer
  std::memset(recv_alloc.ptr, 0, slice_size);

  // H2H Write (Push) from Sender to Receiver
  std::string peer_address =
      absl::StrCat("127.0.0.1:", receiver.local_port().value());

  // Sender pushes block 0 to Receiver. Receiver will store it using the
  // allocated block ID (0).
  TF_ASSERT_OK_AND_ASSIGN(std::vector<int> pushed_ids,
                          sender.H2hWriteDirect(peer_address, {0}));
  EXPECT_EQ(pushed_ids.size(), 1);
  EXPECT_EQ(pushed_ids[0], 0);

  // Give a brief moment for socket transfer to complete and trigger ACKs
  absl::SleepFor(absl::Milliseconds(50));

  // Verify Receiver's buffer got populated!
  for (size_t i = 0; i < slice_size; ++i) {
    ASSERT_EQ(recv_alloc.ptr[i], 0x5A)
        << "Mismatch at index " << i << ", expected 0x5A, got "
        << static_cast<int>(recv_alloc.ptr[i]);
  }

  // Now let's test Pull. Zero out Sender's buffer, and fill Receiver's buffer
  // with a new pattern.
  std::memset(send_alloc.ptr, 0, slice_size);
  std::memset(recv_alloc.ptr, 0x3C, slice_size);

  // Sender pulls block 0 from Receiver
  TF_ASSERT_OK_AND_ASSIGN(std::vector<int> pulled_ids,
                          sender.H2hReadDirect(peer_address, {0}));
  EXPECT_EQ(pulled_ids.size(), 1);
  EXPECT_EQ(pulled_ids[0], 0);

  absl::SleepFor(absl::Milliseconds(50));

  // Verify Sender's buffer got populated with Receiver's new pattern!
  for (size_t i = 0; i < slice_size; ++i) {
    ASSERT_EQ(send_alloc.ptr[i], 0x3C)
        << "Mismatch at index " << i << ", expected 0x3C, got "
        << static_cast<int>(send_alloc.ptr[i]);
  }
}

TEST(RaidenManagerBaseTest, E2eRemoteD2DBlockWrite) {
  // 1. Initialize TPUPjRtManager to obtain the PjRtClient.
  TF_ASSERT_OK_AND_ASSIGN(tpu_raiden::TpuPjrtManager * manager,
                          tpu_raiden::TpuPjrtManager::GetDefault());
  xla::PjRtClient* client = manager->client();

  // 2. Create source device buffer containing mock data (representing data
  //    on the TPU).
  std::vector<float> host_data = {1.1f, 2.2f, 3.3f, 4.4f};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> src_buffer,
      manager->BufferFromHost(host_data.data(), xla::F32, {4}));
  ASSERT_THAT(src_buffer->GetReadyFuture().Await(), absl_testing::IsOk());

  TF_ASSERT_OK_AND_ASSIGN(int64_t physical_size,
                          src_buffer->GetOnDeviceSizeInBytes());

  // 3. Allocate a receiver buffer in host memory based on the physical size.
  std::vector<uint8_t> receiver_buffer(physical_size, 0);

  // 4. Start the receiver transport on a local port using TestRaidenManager.
  TestRaidenManager receiver(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/physical_size);
  receiver.SetExternalHostPointers({receiver_buffer.data()},
                                   {static_cast<size_t>(physical_size)});
  ASSERT_TRUE(receiver.local_port().has_value());
  int receiver_port = receiver.local_port().value();
  std::string receiver_address = absl::StrCat("127.0.0.1:", receiver_port);

  // Allocate destination device buffer on receiver
  std::vector<float> zero_data(4, 0.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> dst_buffer,
      manager->BufferFromHost(zero_data.data(), xla::F32, {4}));
  ASSERT_THAT(dst_buffer->GetReadyFuture().Await(), absl_testing::IsOk());

  // 5. Create the RaidenManagerBase sender instance.
  RaidenManagerBase sender(/*num_layers=*/1, /*num_shards=*/1,
                           /*slice_byte_size=*/physical_size,
                           /*block_size=*/1,
                           /*local_port=*/std::nullopt,
                           /*parallelism=*/1, /*max_staging_blocks=*/4);

  // 6. Prepare the source (TPU buffer) and destination (receiver TPU buffer)
  // metadata.
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

  // Register receive on receiver
  TF_ASSERT_OK_AND_ASSIGN(
      auto hold, raiden::BufferHoldAndAlias::Acquire(dst_buffer.get()));
  xla::Future<> recv_future = receiver.RemoteD2DBlockReceive(
      dst.block_id, std::move(hold), physical_size);

  // 7. Trigger the remote Device-to-Device (D2D) write transfer.
  xla::Future<> transfer_future =
      sender.RemoteD2DBlockWrite(src, dst, physical_size);

  // 8. Await completion of the transfer and receive.
  absl::Status status = transfer_future.Await();
  ASSERT_THAT(status, absl_testing::IsOk());

  status = recv_future.Await();
  ASSERT_THAT(status, absl_testing::IsOk());

  // 9. Verify data on receiver's device buffer
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> literal,
                          dst_buffer->ToLiteral().Await());
  auto read_data = literal->data<float>();
  ASSERT_EQ(read_data.size(), 4);
  EXPECT_EQ(read_data[0], 1.1f);
  EXPECT_EQ(read_data[1], 2.2f);
  EXPECT_EQ(read_data[2], 3.3f);
  EXPECT_EQ(read_data[3], 4.4f);
}

TEST(RaidenManagerBaseTest, E2eRemoteD2DBlockWriteConcurrent) {
  // 1. Initialize TPUPjRtManager to obtain the PjRtClient.
  TF_ASSERT_OK_AND_ASSIGN(tpu_raiden::TpuPjrtManager * manager,
                          tpu_raiden::TpuPjrtManager::GetDefault());
  xla::PjRtClient* client = manager->client();

  // 2. Create 10 mock source buffers on the TPU, each containing unique data.
  std::vector<std::unique_ptr<xla::PjRtBuffer>> src_buffers;
  src_buffers.reserve(10);
  for (int i = 0; i < 10; ++i) {
    std::vector<float> host_data = {i + 1.0f, i + 2.0f, i + 3.0f, i + 4.0f};
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<xla::PjRtBuffer> src_buffer,
        manager->BufferFromHost(host_data.data(), xla::F32, {4}));
    ASSERT_THAT(src_buffer->GetReadyFuture().Await(), absl_testing::IsOk());
    src_buffers.push_back(std::move(src_buffer));
  }

  TF_ASSERT_OK_AND_ASSIGN(int64_t single_buffer_size,
                          src_buffers[0]->GetOnDeviceSizeInBytes());

  // 3. Allocate a large receiver buffer in host memory for 10 blocks.
  std::vector<uint8_t> receiver_buffer(10 * single_buffer_size, 0);

  // 4. Start the receiver transport using TestRaidenManager.
  TestRaidenManager receiver(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/single_buffer_size);
  receiver.SetExternalHostPointers(
      {receiver_buffer.data()}, {static_cast<size_t>(receiver_buffer.size())});
  ASSERT_TRUE(receiver.local_port().has_value());
  int receiver_port = receiver.local_port().value();
  std::string receiver_address = absl::StrCat("127.0.0.1:", receiver_port);

  // Allocate a single large destination device buffer on receiver.
  size_t total_floats = (10 * single_buffer_size) / sizeof(float);
  std::vector<float> zero_data(total_floats, 0.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> dst_buffer,
      manager->BufferFromHost(zero_data.data(), xla::F32,
                              {static_cast<int64_t>(total_floats)}));
  ASSERT_THAT(dst_buffer->GetReadyFuture().Await(), absl_testing::IsOk());

  // 5. Create the RaidenManagerBase sender instance.
  RaidenManagerBase sender(/*num_layers=*/1, /*num_shards=*/1,
                           /*slice_byte_size=*/single_buffer_size,
                           /*block_size=*/1,
                           /*local_port=*/std::nullopt,
                           /*parallelism=*/1, /*max_staging_blocks=*/4);

  // Register receive on receiver for all 10 blocks using the same dst_buffer.
  std::vector<xla::Future<>> recv_futures;
  recv_futures.reserve(10);
  for (int i = 0; i < 10; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto hold, raiden::BufferHoldAndAlias::Acquire(dst_buffer.get()));
    recv_futures.push_back(
        receiver.RemoteD2DBlockReceive(i, std::move(hold), single_buffer_size));
  }

  // 6. Prepare metadata and trigger all 10 async transfers concurrently.
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

  // 7. Join all 10 futures and await completion of transfers and receives.
  absl::Status status = xla::JoinFutures(futures).Await();
  ASSERT_THAT(status, absl_testing::IsOk());

  status = xla::JoinFutures(recv_futures).Await();
  ASSERT_THAT(status, absl_testing::IsOk());

  // 8. Verify that the destination device buffer contains the correct values.
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

}  // namespace
}  // namespace tpu_raiden
