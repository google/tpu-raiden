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

#include "tpu_raiden/transport/block_transport.h"

#include <algorithm>
#include <chrono>  // NOLINT
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>  // NOLINT
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

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

  absl::StatusOr<std::vector<int>> AllocateBlocks(
      size_t num_blocks, uint64_t uuid = 0) override {
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

  void RegisterBlockReadinessCallback(size_t layer_idx, size_t shard_idx,
                                      int block_id, uint64_t uuid,
                                      HostBlockReadyCallback cb) override {
    {
      absl::MutexLock lock(wait_events_mu_);
      wait_events_.push_back(std::make_tuple(layer_idx, shard_idx, block_id));
    }
    cb(absl::OkStatus());
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

  BlockChunkRegionValidationMode block_chunk_region_validation_mode()
      const override {
    return region_validation_mode_;
  }

  absl::Status ValidateBlockChunksInRegions(
      size_t layer_idx, size_t shard_idx,
      const std::vector<BlockChunk>& chunks) override {
    ++region_validation_calls_;
    return region_validation_status_;
  }

  void set_region_validation(BlockChunkRegionValidationMode mode,
                             absl::Status status) {
    region_validation_mode_ = mode;
    region_validation_status_ = std::move(status);
    region_validation_calls_ = 0;
  }

  int region_validation_calls() const { return region_validation_calls_; }

  int GetRemoteReadBlockId(int base_remote_id, int chunk_k) override {
    return base_remote_id + chunk_k;
  }

  size_t num_layers() const override { return num_layers_; }
  size_t num_shards() const override { return num_shards_; }
  size_t slice_byte_size() const override { return slice_size_; }
  size_t shard_factor() const override { return 1; }

  uint8_t* data(size_t layer_idx = 0, size_t shard_idx = 0) {
    return buffers_[BufferIndex(layer_idx, shard_idx)].data();
  }
  uint8_t* block_data(int block_id, size_t layer_idx = 0,
                      size_t shard_idx = 0) {
    return data(layer_idx, shard_idx) + block_id * slice_size_;
  }
  std::vector<std::tuple<size_t, size_t, int>> wait_events() {
    absl::MutexLock lock(wait_events_mu_);
    return wait_events_;
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
  BlockChunkRegionValidationMode region_validation_mode_ =
      BlockChunkRegionValidationMode::kDisabled;
  absl::Status region_validation_status_;
  int region_validation_calls_ = 0;
  absl::Mutex wait_events_mu_;
  std::vector<std::tuple<size_t, size_t, int>> wait_events_;
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

TEST(BlockTransportTest, RegionValidationFailModeRejectsPushChunks) {
  size_t size = 1024;
  MockDelegate delegate1(size);
  MockDelegate delegate2(size);
  delegate1.set_region_validation(
      BlockChunkRegionValidationMode::kFail,
      absl::FailedPreconditionError("chunk crosses padding"));

  BlockTransport transport1(&delegate1, 0);
  BlockTransport transport2(&delegate2, 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::string peer2 = "localhost:" + std::to_string(transport2.local_port());
  auto push_res = transport1.Push(peer2, {0});

  ASSERT_FALSE(push_res.ok());
  EXPECT_EQ(push_res.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(delegate1.region_validation_calls(), 1);
}

TEST(BlockTransportTest, RegionValidationWarnModeAllowsPushChunks) {
  size_t size = 1024;
  MockDelegate delegate1(size);
  MockDelegate delegate2(size);
  std::memset(delegate1.data(), 0xCD, size);
  delegate1.set_region_validation(
      BlockChunkRegionValidationMode::kWarn,
      absl::FailedPreconditionError("chunk crosses padding"));

  BlockTransport transport1(&delegate1, 0);
  BlockTransport transport2(&delegate2, 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::string peer2 = "localhost:" + std::to_string(transport2.local_port());
  auto push_res = transport1.Push(peer2, {0});

  ASSERT_TRUE(push_res.ok()) << push_res.status().message();
  EXPECT_EQ(delegate1.region_validation_calls(), 1);
  EXPECT_EQ(delegate2.data()[0], 0xCD);
  EXPECT_EQ(delegate2.data()[size - 1], 0xCD);
}

TEST(BlockTransportTest, PushAndPullWithoutConnectionPool) {
  size_t size = 1024;
  MockDelegate delegate1(size);
  MockDelegate delegate2(size);

  // Populate source with custom pattern
  std::memset(delegate1.data(), 0xAB, size);
  std::memset(delegate2.data(), 0x00, size);

  // Disable connection pooling
  BlockTransport transport1(&delegate1, 0, /*enable_conn_pool=*/false);
  BlockTransport transport2(&delegate2, 0, /*enable_conn_pool=*/false);

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

TEST(BlockTransportTest, PullSupportsBlockMajorOrder) {
  constexpr size_t kSliceSize = 16;
  constexpr int kNumBlocks = 2;
  constexpr size_t kNumLayers = 2;
  MockDelegate source(kSliceSize, kNumBlocks, kNumLayers);
  MockDelegate receiver(kSliceSize, kNumBlocks, kNumLayers);

  for (size_t layer = 0; layer < kNumLayers; ++layer) {
    for (int block = 0; block < kNumBlocks; ++block) {
      std::memset(source.block_data(block, layer),
                  static_cast<int>(0x10 + layer * 0x10 + block), kSliceSize);
    }
  }

  BlockTransport source_transport(&source, 0);
  BlockTransport receiver_transport(&receiver, 0);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::string source_peer =
      "localhost:" + std::to_string(source_transport.local_port());
  auto pull_res = receiver_transport.Pull(
      source_peer, {0, 1}, {0, 1}, {}, /*parallelism=*/1,
      MajorOrder::kBlockMajor);
  ASSERT_TRUE(pull_res.ok()) << pull_res.status().message();

  for (int block = 0; block < kNumBlocks; ++block) {
    EXPECT_EQ(receiver.block_data(block, 0)[0], 0x10 + block);
    EXPECT_EQ(receiver.block_data(block, 1)[0], 0x20 + block);
  }

  EXPECT_EQ(source.wait_events(),
            (std::vector<std::tuple<size_t, size_t, int>>{
                std::make_tuple(0, 0, 0),
                std::make_tuple(1, 0, 0),
                std::make_tuple(0, 0, 1),
                std::make_tuple(1, 0, 1),
            }));
}

class MockBlockTransport : public BlockTransport {
 public:
  struct CallRecord {
    std::string peer;
    std::string local_ip;
  };

  MockBlockTransport(BlockTransportDelegate* delegate, int local_port,
                     const std::vector<std::string>& local_ips)
      : BlockTransport(delegate, local_port, true, local_ips) {}

  absl::StatusOr<int> AcquireConnection(absl::string_view peer,
                                        absl::string_view local_ip) override {
    absl::MutexLock lock(mock_mu_);
    acquire_calls_.push_back({std::string(peer), std::string(local_ip)});
    return absl::InternalError("mock_connection_halt");
  }

  std::vector<CallRecord> acquire_calls() {
    absl::MutexLock lock(mock_mu_);
    return acquire_calls_;
  }

 private:
  absl::Mutex mock_mu_;
  std::vector<CallRecord> acquire_calls_ ABSL_GUARDED_BY(mock_mu_);
};

TEST(BlockTransportTest, RoundRobinDistribution) {
  constexpr size_t kSliceSize = 16;
  MockDelegate delegate(kSliceSize);

  std::vector<std::string> local_ips = {"10.0.0.1", "10.0.0.2"};
  std::vector<std::string> peers = {"10.0.0.3:1234", "10.0.0.4:1234",
                                    "10.0.0.5:1234"};

  MockBlockTransport transport(&delegate, 0, local_ips);

  std::vector<int> src_blocks = {0, 1, 2, 3, 4, 5};
  auto res = transport.Push(peers, src_blocks, {}, /*parallelism=*/6);

  EXPECT_FALSE(res.ok());
  EXPECT_EQ(res.status().message(), "mock_connection_halt");

  auto calls = transport.acquire_calls();
  ASSERT_EQ(calls.size(), 6);

  std::vector<MockBlockTransport::CallRecord> expected = {
      {"10.0.0.3:1234", "10.0.0.1"}, {"10.0.0.4:1234", "10.0.0.2"},
      {"10.0.0.5:1234", "10.0.0.1"}, {"10.0.0.3:1234", "10.0.0.2"},
      {"10.0.0.4:1234", "10.0.0.1"}, {"10.0.0.5:1234", "10.0.0.2"}};

  auto compare = [](const MockBlockTransport::CallRecord& a,
                    const MockBlockTransport::CallRecord& b) {
    if (a.peer != b.peer) return a.peer < b.peer;
    return a.local_ip < b.local_ip;
  };

  std::sort(calls.begin(), calls.end(), compare);
  std::sort(expected.begin(), expected.end(), compare);

  for (size_t i = 0; i < 6; ++i) {
    EXPECT_EQ(calls[i].peer, expected[i].peer);
    EXPECT_EQ(calls[i].local_ip, expected[i].local_ip);
  }
}

// Force warnings check
}  // namespace
}  // namespace transport
}  // namespace tpu_raiden
