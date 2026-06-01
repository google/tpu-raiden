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
#include <cstring>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/status/status.h"

namespace tpu_raiden {
namespace transport {
namespace {

class MockDelegate : public BlockTransportDelegate {
 public:
  MockDelegate(size_t slice_size, int max_blocks = 1)
      : slice_size_(slice_size), max_blocks_(max_blocks) {
    buffer_.resize(slice_size_ * max_blocks_, 0);
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
    return buffer_.data();
  }

  size_t GetHostSize(size_t layer_idx, size_t shard_idx) override {
    return buffer_.size();
  }

  int GetRemoteReadBlockId(int base_remote_id, int chunk_k) override {
    return base_remote_id + chunk_k;
  }

  size_t num_layers() const override { return 1; }
  size_t num_shards() const override { return 1; }
  size_t slice_byte_size() const override { return slice_size_; }
  int block_size() const override { return 1; }
  size_t shard_factor() const override { return 1; }

  uint8_t* data() { return buffer_.data(); }
  uint8_t* block_data(int block_id) {
    return buffer_.data() + block_id * slice_size_;
  }

 private:
  size_t slice_size_;
  int max_blocks_;
  std::vector<uint8_t> buffer_;
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

TEST(BlockTransportTest, PullNonContiguousCoalesced) {
  // Mix a consecutive run with a gap: pull remote blocks {0, 1, 3}. The H2H
  // Read worker must coalesce {0, 1} into one consecutive read and issue a
  // separate read for {3}, while still landing them in local blocks {0, 1, 2}.
  size_t size = 1024;
  MockDelegate delegate1(size, 4);  // remote has 4 blocks
  MockDelegate delegate2(size, 3);  // local pulls 3 blocks

  std::memset(delegate1.block_data(0), 0xA0, size);
  std::memset(delegate1.block_data(1), 0xA1, size);
  std::memset(delegate1.block_data(2), 0xA2, size);
  std::memset(delegate1.block_data(3), 0xA3, size);

  std::memset(delegate2.block_data(0), 0x00, size);
  std::memset(delegate2.block_data(1), 0x00, size);
  std::memset(delegate2.block_data(2), 0x00, size);

  BlockTransport transport1(&delegate1, 0);
  BlockTransport transport2(&delegate2, 0);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::string peer1 = "localhost:" + std::to_string(transport1.local_port());
  auto pull_res = transport2.Pull(peer1, {0, 1, 3});
  ASSERT_TRUE(pull_res.ok()) << pull_res.status().message();

  // Local 0 <- remote 0, local 1 <- remote 1, local 2 <- remote 3.
  EXPECT_EQ(delegate2.block_data(0)[0], 0xA0);
  EXPECT_EQ(delegate2.block_data(0)[size - 1], 0xA0);
  EXPECT_EQ(delegate2.block_data(1)[0], 0xA1);
  EXPECT_EQ(delegate2.block_data(1)[size - 1], 0xA1);
  EXPECT_EQ(delegate2.block_data(2)[0], 0xA3);
  EXPECT_EQ(delegate2.block_data(2)[size - 1], 0xA3);
}

TEST(BlockTransportTest, PullNonContiguousParallel) {
  // Non-contiguous pull striped over 2 TCP streams: pull remote {0,1,3,4} with
  // parallelism=2. The worker partitions the 4 local blocks into two streams
  // (local [0,1] and [2,3]), and each stream coalesces its own remote slice
  // ({0,1} and {3,4}) into one consecutive read. The gap (remote block 2)
  // falls between the streams.
  size_t size = 1024;
  MockDelegate delegate1(size, 5);  // remote has 5 blocks (ids 0..4)
  MockDelegate delegate2(size, 4);  // local pulls 4 blocks

  std::memset(delegate1.block_data(0), 0xB0, size);
  std::memset(delegate1.block_data(1), 0xB1, size);
  std::memset(delegate1.block_data(2), 0xB2, size);  // skipped
  std::memset(delegate1.block_data(3), 0xB3, size);
  std::memset(delegate1.block_data(4), 0xB4, size);
  for (int b = 0; b < 4; ++b) std::memset(delegate2.block_data(b), 0x00, size);

  BlockTransport transport1(&delegate1, 0);
  BlockTransport transport2(&delegate2, 0);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::string peer1 = "localhost:" + std::to_string(transport1.local_port());
  auto pull_res = transport2.Pull(peer1, {0, 1, 3, 4}, /*parallelism=*/2);
  ASSERT_TRUE(pull_res.ok()) << pull_res.status().message();

  // local {0,1,2,3} <- remote {0,1,3,4}.
  EXPECT_EQ(delegate2.block_data(0)[0], 0xB0);
  EXPECT_EQ(delegate2.block_data(1)[0], 0xB1);
  EXPECT_EQ(delegate2.block_data(2)[0], 0xB3);
  EXPECT_EQ(delegate2.block_data(3)[0], 0xB4);
  EXPECT_EQ(delegate2.block_data(3)[size - 1], 0xB4);
}


}  // namespace
}  // namespace transport
}  // namespace tpu_raiden
