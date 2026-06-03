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

#include "transport/block_transport.h"

#include <chrono>  // NOLINT
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace tpu_raiden {
namespace transport {
namespace {

class MockDelegate : public BlockTransportDelegate {
 public:
  MockDelegate(size_t slice_size, int max_blocks = 1, size_t num_layers = 1,
               size_t num_shards = 1)
      : slice_size_(slice_size),
        max_blocks_(max_blocks),
        num_layers_(num_layers),
        num_shards_(num_shards) {
    buffers_.resize(num_layers_ * num_shards_);
    for (auto& buffer : buffers_) {
      buffer.resize(slice_size_ * max_blocks_, 0);
    }
  }

  absl::StatusOr<std::vector<int>> AllocateBlocks(size_t num_blocks,
                                                  int64_t entity_id) override {
    std::vector<int> ids;
    for (size_t i = 0; i < num_blocks; ++i) {
      ids.push_back(i % max_blocks_);
    }
    return ids;
  }

  absl::Status OnDataReceived() override {
    on_data_received_called_ = true;
    return absl::OkStatus();
  }

  absl::Status OnSingleBlockReceived(int block_id, size_t size_bytes) override {
    on_single_block_received_called_ = true;
    received_block_id_ = block_id;
    received_size_bytes_ = size_bytes;
    return OnDataReceived();
  }

  bool on_data_received_called() const { return on_data_received_called_; }
  void reset_data_received() { on_data_received_called_ = false; }

  bool on_single_block_received_called() const {
    return on_single_block_received_called_;
  }
  int received_block_id() const { return received_block_id_; }
  size_t received_size_bytes() const { return received_size_bytes_; }

  void reset_single_block_received() {
    on_single_block_received_called_ = false;
    received_block_id_ = -1;
    received_size_bytes_ = 0;
  }

  uint8_t* GetHostPointer(size_t layer_idx, size_t shard_idx) override {
    return buffers_[BufferIndex(layer_idx, shard_idx)].data();
  }

  size_t GetHostSize(size_t layer_idx, size_t shard_idx) override {
    return buffers_[BufferIndex(layer_idx, shard_idx)].size();
  }

  int GetRemoteReadBlockId(int base_remote_id, int chunk_k) override {
    return base_remote_id + chunk_k;
  }

  size_t num_layers() const override { return num_layers_; }
  size_t num_shards() const override { return num_shards_; }
  size_t slice_byte_size() const override { return slice_size_; }
  int block_size() const override { return 1; }
  size_t shard_factor() const override { return 1; }

  uint8_t* data(size_t layer_idx = 0, size_t shard_idx = 0) {
    return buffers_[BufferIndex(layer_idx, shard_idx)].data();
  }
  uint8_t* block_data(int block_id, size_t layer_idx = 0,
                      size_t shard_idx = 0) {
    return data(layer_idx, shard_idx) + block_id * slice_size_;
  }

 private:
  size_t BufferIndex(size_t layer_idx, size_t shard_idx) const {
    return layer_idx * num_shards_ + shard_idx;
  }

  size_t slice_size_;
  int max_blocks_;
  size_t num_layers_;
  size_t num_shards_;
  std::vector<std::vector<uint8_t>> buffers_;
  bool on_data_received_called_ = false;
  bool on_single_block_received_called_ = false;
  int received_block_id_ = -1;
  size_t received_size_bytes_ = 0;
};

TEST(BlockTransportTest, PushAndPullCorrectness) {
  size_t size = 1024;
  MockDelegate delegate1(size);
  MockDelegate delegate2(size);

  // Populate source with custom pattern
  std::memset(delegate1.data(), 0xAB, size);
  std::memset(delegate2.data(), 0x00, size);

  BlockTransport transport1(&delegate1, 0);
  BlockTransport transport2(&delegate2, 0);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::string peer2 = "localhost:" + std::to_string(transport2.local_port());

  // Push block 0 from transport1 to transport2
  auto push_res = transport1.Push(peer2, {0});
  ASSERT_TRUE(push_res.ok()) << push_res.status().message();

  // Verify push parity
  EXPECT_EQ(delegate2.data()[0], 0xAB);
  EXPECT_EQ(delegate2.data()[size - 1], 0xAB);

  // Reset dest to 0
  std::memset(delegate2.data(), 0x00, size);

  // Pull block 0 from transport1 using transport2
  std::string peer1 = "localhost:" + std::to_string(transport1.local_port());
  auto pull_res = transport2.Pull(peer1, {0});
  ASSERT_TRUE(pull_res.ok()) << pull_res.status().message();

  // Verify pull parity
  EXPECT_EQ(delegate2.data()[0], 0xAB);
  EXPECT_EQ(delegate2.data()[size - 1], 0xAB);
}

TEST(BlockTransportTest, PullWeightsChunk) {
  size_t size = 4096;
  MockDelegate delegate1(size);
  MockDelegate delegate2(size);

  std::memset(delegate1.data(), 0xEF, size);
  std::memset(delegate2.data(), 0x00, size);

  BlockTransport transport1(&delegate1, 0);
  BlockTransport transport2(&delegate2, 0);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::string peer1 = "localhost:" + std::to_string(transport1.local_port());

  // Pull 1024 bytes starting from offset 512 in remote buffer, into offset 1024
  // in local buffer!
  auto pull_res = transport2.PullWeightsChunk(
      peer1, /*src_shard_idx=*/0, /*src_offset_bytes=*/512,
      /*dst_shard_idx=*/0, /*dst_offset_bytes=*/1024, /*size_bytes=*/1024);
  ASSERT_TRUE(pull_res.ok()) << pull_res.message();

  // Verify correct byte-range copy
  EXPECT_EQ(delegate2.data()[1023], 0x00);
  EXPECT_EQ(delegate2.data()[1024], 0xEF);
  EXPECT_EQ(delegate2.data()[2047], 0xEF);
  EXPECT_EQ(delegate2.data()[2048], 0x00);
}

TEST(BlockTransportTest, PullNonContiguous) {
  size_t size = 1024;
  // Delegate 1 has 3 blocks capacity
  MockDelegate delegate1(size, 3);
  // Delegate 2 has 2 blocks capacity (we want to pull 2 blocks)
  MockDelegate delegate2(size, 2);

  // Populate source blocks with different patterns
  std::memset(delegate1.block_data(0), 0xAA, size);
  std::memset(delegate1.block_data(1), 0xBB, size);
  std::memset(delegate1.block_data(2), 0xCC, size);

  std::memset(delegate2.block_data(0), 0x00, size);
  std::memset(delegate2.block_data(1), 0x00, size);

  BlockTransport transport1(&delegate1, 0);
  BlockTransport transport2(&delegate2, 0);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::string peer1 = "localhost:" + std::to_string(transport1.local_port());

  // Pull block 0 and 2 (non-contiguous) from transport1 using transport2
  // We expect they will be written to local block 0 and 1 respectively.
  auto pull_res = transport2.Pull(peer1, {0, 2});
  ASSERT_TRUE(pull_res.ok()) << pull_res.status().message();

  // Verify pull parity
  // Local Block 0 should have 0xAA (from remote Block 0)
  EXPECT_EQ(delegate2.block_data(0)[0], 0xAA);
  EXPECT_EQ(delegate2.block_data(0)[size - 1], 0xAA);

  // Local Block 1 should have 0xCC (from remote Block 2)
  EXPECT_EQ(delegate2.block_data(1)[0], 0xCC);
  EXPECT_EQ(delegate2.block_data(1)[size - 1], 0xCC);
}

TEST(BlockTransportTest, PullExplicitDestPtrsMultiLayerUnevenParallelism) {
  constexpr size_t kSliceSize = 16;
  constexpr int kNumBlocks = 3;
  constexpr size_t kNumLayers = 2;
  MockDelegate source(kSliceSize, kNumBlocks, kNumLayers);
  MockDelegate receiver(kSliceSize, kNumBlocks, kNumLayers);

  for (size_t layer = 0; layer < kNumLayers; ++layer) {
    for (int block = 0; block < kNumBlocks; ++block) {
      std::memset(source.block_data(block, layer),
                  static_cast<int>(0x10 + layer * 0x10 + block), kSliceSize);
    }
  }

  std::vector<uint8_t> layer0(kSliceSize * kNumBlocks, 0);
  std::vector<uint8_t> layer1(kSliceSize * kNumBlocks, 0);
  std::vector<uint8_t*> explicit_dst_ptrs = {layer0.data(), layer1.data()};

  BlockTransport source_transport(&source, 0);
  BlockTransport receiver_transport(&receiver, 0);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::string source_peer =
      "localhost:" + std::to_string(source_transport.local_port());
  auto pull_res = receiver_transport.Pull(source_peer, {0, 1, 2}, {0, 1, 2},
                                          explicit_dst_ptrs,
                                          /*parallelism=*/2);
  ASSERT_TRUE(pull_res.ok()) << pull_res.status().message();
  EXPECT_EQ(*pull_res, std::vector<int>({0, 1, 2}));

  for (int block = 0; block < kNumBlocks; ++block) {
    EXPECT_EQ(layer0[block * kSliceSize], 0x10 + block);
    EXPECT_EQ(layer0[(block + 1) * kSliceSize - 1], 0x10 + block);
    EXPECT_EQ(layer1[block * kSliceSize], 0x20 + block);
    EXPECT_EQ(layer1[(block + 1) * kSliceSize - 1], 0x20 + block);
  }
}

TEST(BlockTransportTest, PullRejectsOutOfBoundsRemoteBlock) {
  constexpr size_t kSliceSize = 16;
  constexpr int kNumBlocks = 1;
  MockDelegate source(kSliceSize, kNumBlocks);
  MockDelegate receiver(kSliceSize, kNumBlocks);

  BlockTransport source_transport(&source, 0);
  BlockTransport receiver_transport(&receiver, 0);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::string source_peer =
      "localhost:" + std::to_string(source_transport.local_port());
  auto pull_res = receiver_transport.Pull(source_peer, {1}, {0});
  EXPECT_FALSE(pull_res.ok());
}

TEST(BlockTransportTest, WriteBlockDirectCorrectness) {
  size_t size = 1024;
  MockDelegate delegate1(size);
  MockDelegate delegate2(size);

  std::vector<uint8_t> src_data(size);
  for (size_t i = 0; i < size; ++i) {
    src_data[i] = static_cast<uint8_t>(i % 256);
  }
  std::memset(delegate2.data(), 0x00, size);

  BlockTransport transport1(&delegate1, 0);
  BlockTransport transport2(&delegate2, 0);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::string peer2 = "localhost:" + std::to_string(transport2.local_port());

  // Write block 0 directly from src_data to transport2
  auto write_res = transport1.WriteBlockDirect(peer2, /*remote_block_id=*/0,
                                               src_data.data(), size);
  ASSERT_TRUE(write_res.ok()) << write_res.message();

  // Verify the data was received correctly in delegate2
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(delegate2.data()[i], src_data[i]);
  }

  // Verify OnDataReceived was called
  EXPECT_TRUE(delegate2.on_data_received_called());

  // Verify OnSingleBlockReceived was called with correct arguments
  EXPECT_TRUE(delegate2.on_single_block_received_called());
  EXPECT_EQ(delegate2.received_block_id(), 0);
  EXPECT_EQ(delegate2.received_size_bytes(), size);
}

}  // namespace
}  // namespace transport
}  // namespace tpu_raiden
