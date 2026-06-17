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

#include <sys/uio.h>

#include <chrono>  // NOLINT
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>  // NOLINT
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
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

  absl::Status WaitForBlockRead(size_t layer_idx, size_t shard_idx,
                                int block_id) override {
    absl::MutexLock lock(wait_events_mu_);
    wait_events_.push_back(std::make_tuple(layer_idx, shard_idx, block_id));
    return absl::OkStatus();
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

  int port1 = 0;
  int port2 = 0;
  BlockTransport transport1(&delegate1, "127.0.0.1", port1);
  BlockTransport transport2(&delegate2, "127.0.0.1", port2);

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

TEST(BlockTransportTest, PushAndPullWithoutConnectionPool) {
  size_t size = 1024;
  MockDelegate delegate1(size);
  MockDelegate delegate2(size);

  // Populate source with custom pattern
  std::memset(delegate1.data(), 0xAB, size);
  std::memset(delegate2.data(), 0x00, size);

  // Disable connection pooling
  int port1 = 0;
  int port2 = 0;
  BlockTransport transport1(&delegate1, "127.0.0.1", port1,
                            /*enable_conn_pool=*/false);
  BlockTransport transport2(&delegate2, "127.0.0.1", port2,
                            /*enable_conn_pool=*/false);

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

TEST(BlockTransportTest, PullBuffer) {
  size_t size = 4096;
  MockDelegate delegate1(size);
  MockDelegate delegate2(size);

  std::memset(delegate1.data(), 0xEF, size);
  std::memset(delegate2.data(), 0x00, size);

  int port1 = 0;
  int port2 = 0;
  BlockTransport transport1(&delegate1, "127.0.0.1", port1);
  BlockTransport transport2(&delegate2, "127.0.0.1", port2);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::string peer1 = "localhost:" + std::to_string(transport1.local_port());

  // Pull 1024 bytes starting from offset 512 in remote buffer, into offset 1024
  // in local buffer!
  auto pull_res = transport2.PullBuffer(
      peer1, /*buffer_id=*/0, /*src_shard_idx=*/0, /*src_offset_bytes=*/512,
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

  int port1 = 0;
  int port2 = 0;
  BlockTransport transport1(&delegate1, "127.0.0.1", port1);
  BlockTransport transport2(&delegate2, "127.0.0.1", port2);

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

  int sport = 0;
  int rport = 0;
  BlockTransport source_transport(&source, "127.0.0.1", sport);
  BlockTransport receiver_transport(&receiver, "127.0.0.1", rport);

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

  int sport = 0;
  int rport = 0;
  BlockTransport source_transport(&source, "127.0.0.1", sport);
  BlockTransport receiver_transport(&receiver, "127.0.0.1", rport);

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

  int sport = 0;
  int rport = 0;
  BlockTransport source_transport(&source, "127.0.0.1", sport);
  BlockTransport receiver_transport(&receiver, "127.0.0.1", rport);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::string source_peer =
      "localhost:" + std::to_string(source_transport.local_port());
  auto pull_res =
      receiver_transport.Pull(source_peer, {0, 1}, {0, 1}, {},
                              /*parallelism=*/1, MajorOrder::kBlockMajor);
  ASSERT_TRUE(pull_res.ok()) << pull_res.status().message();

  for (int block = 0; block < kNumBlocks; ++block) {
    EXPECT_EQ(receiver.block_data(block, 0)[0], 0x10 + block);
    EXPECT_EQ(receiver.block_data(block, 1)[0], 0x20 + block);
  }

  EXPECT_EQ(source.wait_events(), (std::vector<std::tuple<size_t, size_t, int>>{
                                      std::make_tuple(0, 0, 0),
                                      std::make_tuple(1, 0, 0),
                                      std::make_tuple(0, 0, 1),
                                      std::make_tuple(1, 0, 1),
                                  }));
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

  int port1 = 0;
  int port2 = 0;
  BlockTransport transport1(&delegate1, "127.0.0.1", port1);
  BlockTransport transport2(&delegate2, "127.0.0.1", port2);

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

TEST(BlockTransportTest, PortHuntingTest) {
  MockDelegate delegate1(1024);
  MockDelegate delegate2(1024);

  int port1 = 15000;
  BlockTransport transport1(&delegate1, "127.0.0.1", port1);
  EXPECT_EQ(port1, 15000);
  EXPECT_EQ(transport1.local_port(), 15000);

  int port2 = 15000;  // Same target port!
  BlockTransport transport2(&delegate2, "127.0.0.1", port2);
  // It should have hunted to 15001
  EXPECT_EQ(port2, 15001);
  EXPECT_EQ(transport2.local_port(), 15001);

  // Verify communication works on the hunted port
  std::memset(delegate1.data(), 0xAB, 1024);
  std::string peer2 = "127.0.0.1:" + std::to_string(transport2.local_port());
  auto push_res = transport1.Push(peer2, {0});
  ASSERT_TRUE(push_res.ok()) << push_res.status().message();
  EXPECT_EQ(delegate2.data()[0], 0xAB);
}
TEST(BlockTransportTest, EphemeralPortTest) {
  MockDelegate delegate(1024);
  int port = 0;
  BlockTransport transport(&delegate, "127.0.0.1", port);
  EXPECT_GT(port, 0);
  EXPECT_EQ(transport.local_port(), port);
  EXPECT_GE(port, 1024);
}

TEST(BlockTransportTest, PortWrappingTest) {
  MockDelegate delegate1(1024);
  MockDelegate delegate2(1024);

  int port1 = 65535;
  BlockTransport transport1(&delegate1, "127.0.0.1", port1);
  EXPECT_EQ(port1, 65535);

  int port2 = 65535;
  BlockTransport transport2(&delegate2, "127.0.0.1", port2);

  // Under the new wrapping logic, 65536 should wrap to 2048.
  // We expect it to bind to 2048 (if free) or hunt further.
  // The old code would truncate to 0 and get an OS ephemeral port (typically >=
  // 32768). So asserting < 30000 will fail on the old code and pass on the new
  // code.
  EXPECT_GE(port2, 1024);
  EXPECT_LT(port2, 30000);
  EXPECT_NE(port2, 65535);
}

TEST(BlockTransportTest, WritevExactSuccess) {
  std::vector<char> buf1(100, 'a');
  std::vector<char> buf2(200, 'b');
  std::vector<struct iovec> iov = {{buf1.data(), buf1.size()},
                                   {buf2.data(), buf2.size()}};

  int call_count = 0;
  auto mock_writev = [&](int fd, const struct iovec* iov_ptr,
                         int iovcnt) -> ssize_t {
    call_count++;
    EXPECT_EQ(iovcnt, 2);
    EXPECT_EQ(iov_ptr[0].iov_len, 100);
    EXPECT_EQ(iov_ptr[1].iov_len, 200);
    return 300;  // Full write
  };

  auto status = WritevExact(1, iov.data(), iov.size(), mock_writev);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(call_count, 1);
}

TEST(BlockTransportTest, WritevExactPartialAndResume) {
  std::vector<char> buf1(100, 'a');
  std::vector<char> buf2(200, 'b');
  std::vector<struct iovec> iov = {{buf1.data(), buf1.size()},
                                   {buf2.data(), buf2.size()}};

  int call_count = 0;
  auto mock_writev = [&](int fd, const struct iovec* iov_ptr,
                         int iovcnt) -> ssize_t {
    call_count++;
    if (call_count == 1) {
      EXPECT_EQ(iovcnt, 2);
      EXPECT_EQ(iov_ptr[0].iov_len, 100);
      return 50;  // Partial write of first iovec
    } else if (call_count == 2) {
      EXPECT_EQ(iovcnt, 2);
      EXPECT_EQ(iov_ptr[0].iov_len, 50);  // Remaining of first
      EXPECT_EQ(iov_ptr[1].iov_len, 200);
      return 150;  // Writes remaining 50 of first, and 100 of second
    } else if (call_count == 3) {
      EXPECT_EQ(iovcnt, 1);
      EXPECT_EQ(iov_ptr[0].iov_len, 100);  // Remaining of second
      return 100;                          // Full write of remaining
    }
    return -1;
  };

  auto status = WritevExact(1, iov.data(), iov.size(), mock_writev);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(call_count, 3);
}

TEST(BlockTransportTest, WritevExactHandlesEintr) {
  std::vector<char> buf1(100, 'a');
  std::vector<struct iovec> iov = {{buf1.data(), buf1.size()}};

  int call_count = 0;
  auto mock_writev = [&](int fd, const struct iovec* iov_ptr,
                         int iovcnt) -> ssize_t {
    call_count++;
    if (call_count == 1) {
      errno = EINTR;
      return -1;
    }
    return 100;
  };

  auto status = WritevExact(1, iov.data(), iov.size(), mock_writev);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(call_count, 2);
}

TEST(BlockTransportTest, WritevExactHandlesClosedSocket) {
  std::vector<char> buf1(100, 'a');
  std::vector<struct iovec> iov = {{buf1.data(), buf1.size()}};

  auto mock_writev = [&](int fd, const struct iovec* iov_ptr,
                         int iovcnt) -> ssize_t {
    return 0;  // Connection closed
  };

  auto status = WritevExact(1, iov.data(), iov.size(), mock_writev);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.message(), "Socket closed unexpectedly during writev");
}

}  // namespace
}  // namespace transport
}  // namespace tpu_raiden
