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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"

namespace tpu_raiden {
namespace transport {
namespace {

class MockDelegate : public BlockTransportDelegate {
 public:
  MockDelegate(size_t slice_size, size_t num_blocks = 1,
               size_t num_layers = 1, size_t num_shards = 1)
      : slice_size_(slice_size),
        num_blocks_(num_blocks),
        num_layers_(num_layers),
        num_shards_(num_shards) {
    buffers_.resize(num_layers_ * num_shards_);
    for (auto& buffer : buffers_) {
      buffer.resize(slice_size_ * num_blocks_, 0);
    }
  }

  absl::StatusOr<std::vector<int>> AllocateBlocks(size_t num_blocks,
                                                  int64_t entity_id) override {
    std::vector<int> ids;
    for (size_t i = 0; i < num_blocks; ++i) {
      ids.push_back(static_cast<int>(i));
    }
    return ids;
  }

  absl::Status OnDataReceived() override { return absl::OkStatus(); }

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

  size_t bytes() const { return slice_size_ * num_blocks_; }

 private:
  size_t BufferIndex(size_t layer_idx, size_t shard_idx) const {
    return layer_idx * num_shards_ + shard_idx;
  }

  size_t slice_size_;
  size_t num_blocks_;
  size_t num_layers_;
  size_t num_shards_;
  std::vector<std::vector<uint8_t>> buffers_;
};

TEST(BlockTransportTest, PushAndPullCorrectness) {
  size_t size = 1024;
  MockDelegate delegate1(size);
  MockDelegate delegate2(size);

  // Populate source with custom pattern
  std::memset(delegate1.data(), 0xAB, delegate1.bytes());
  std::memset(delegate2.data(), 0x00, delegate2.bytes());

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
  std::memset(delegate2.data(), 0x00, delegate2.bytes());

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

  std::memset(delegate1.data(), 0xEF, delegate1.bytes());
  std::memset(delegate2.data(), 0x00, delegate2.bytes());

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

TEST(BlockTransportTest, PullExplicitDestPtrsMultiLayerUnevenParallelism) {
  constexpr size_t kSliceSize = 16;
  constexpr size_t kNumBlocks = 3;
  constexpr size_t kNumLayers = 2;
  MockDelegate source(kSliceSize, kNumBlocks, kNumLayers);
  MockDelegate receiver(kSliceSize, kNumBlocks, kNumLayers);

  for (size_t layer = 0; layer < kNumLayers; ++layer) {
    for (size_t block = 0; block < kNumBlocks; ++block) {
      std::memset(source.data(layer) + block * kSliceSize,
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
  auto pull_res = receiver_transport.Pull(
      source_peer, {0, 1, 2}, {0, 1, 2}, explicit_dst_ptrs,
      /*parallelism=*/2);
  ASSERT_TRUE(pull_res.ok()) << pull_res.status().message();
  EXPECT_EQ(*pull_res, std::vector<int>({0, 1, 2}));

  for (size_t block = 0; block < kNumBlocks; ++block) {
    EXPECT_EQ(layer0[block * kSliceSize], 0x10 + block);
    EXPECT_EQ(layer0[(block + 1) * kSliceSize - 1], 0x10 + block);
    EXPECT_EQ(layer1[block * kSliceSize], 0x20 + block);
    EXPECT_EQ(layer1[(block + 1) * kSliceSize - 1], 0x20 + block);
  }
}

TEST(BlockTransportTest, PullRejectsOutOfBoundsRemoteBlock) {
  constexpr size_t kSliceSize = 16;
  constexpr size_t kNumBlocks = 1;
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

}  // namespace
}  // namespace transport
}  // namespace tpu_raiden
