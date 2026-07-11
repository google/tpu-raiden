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

#include "tpu_raiden/transport/raw_buffer_transport.h"

#include <signal.h>

#include <chrono>  // NOLINT
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace tpu_raiden {
namespace transport {
namespace {

class RawMockDelegate : public RawBufferTransportDelegate {
 public:
  explicit RawMockDelegate(size_t buffer_size) {
    buffer_.resize(buffer_size, 0);
  }

  uint8_t* GetHostPointer(size_t buffer_id, size_t shard_idx) override {
    return buffer_.data();
  }

  size_t GetHostSize(size_t buffer_id, size_t shard_idx) override {
    return buffer_.size();
  }

  absl::Status OnDataReceived() override {
    absl::MutexLock lock( mu_ );
    on_data_received_called_ = true;
    return absl::OkStatus();
  }

  uint8_t* data() { return buffer_.data(); }
  bool on_data_received() const {
    absl::MutexLock lock( mu_ );
    return on_data_received_called_;
  }

 private:
  std::vector<uint8_t> buffer_;
  mutable absl::Mutex mu_;
  bool on_data_received_called_ ABSL_GUARDED_BY(mu_) = false;
};

TEST(RawBufferTransportTest, PullBufferCorrectness) {
  size_t size = 4096;
  RawMockDelegate delegate1(size);
  RawMockDelegate delegate2(size);

  std::memset(delegate1.data(), 0xEF, size);
  std::memset(delegate2.data(), 0x00, size);

  RawBufferTransport transport1(&delegate1, 0);
  RawBufferTransport transport2(&delegate2, 0);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::string peer1 = "localhost:" + std::to_string(transport1.local_port());

  auto pull_res = transport2.PullBuffer(
      peer1, /*buffer_id=*/0, /*src_shard_idx=*/0, /*src_offset_bytes=*/512,
      /*dst_shard_idx=*/0, /*dst_offset_bytes=*/1024, /*size_bytes=*/1024);
  ASSERT_TRUE(pull_res.ok()) << pull_res.message();

  EXPECT_EQ(delegate2.data()[1023], 0x00);
  EXPECT_EQ(delegate2.data()[1024], 0xEF);
  EXPECT_EQ(delegate2.data()[2047], 0xEF);
  EXPECT_EQ(delegate2.data()[2048], 0x00);
}

TEST(RawBufferTransportTest, PushBufferCorrectness) {
  size_t size = 4096;
  RawMockDelegate delegate1(size);
  RawMockDelegate delegate2(size);

  std::vector<uint8_t> push_payload(1024, 0xAB);
  std::memset(delegate2.data(), 0x00, size);

  RawBufferTransport transport1(&delegate1, 0);
  RawBufferTransport transport2(&delegate2, 0);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::string peer2 = "localhost:" + std::to_string(transport2.local_port());

  auto push_res = transport1.PushBuffer(
      peer2, /*buffer_id=*/0, /*dst_shard_idx=*/0, /*dst_offset_bytes=*/512,
      push_payload.data(), push_payload.size());
  ASSERT_TRUE(push_res.ok()) << push_res.message();

  EXPECT_EQ(delegate2.data()[511], 0x00);
  EXPECT_EQ(delegate2.data()[512], 0xAB);
  EXPECT_EQ(delegate2.data()[1535], 0xAB);
  EXPECT_EQ(delegate2.data()[1536], 0x00);
}

TEST(RawBufferTransportTest, PollEINTRIsBenign) {
  size_t size = 4096;
  RawMockDelegate delegate1(size);
  RawMockDelegate delegate2(size);

  std::memset(delegate2.data(), 0x00, size);

  RawBufferTransport transport1(&delegate1, 0);
  RawBufferTransport transport2(&delegate2, 0);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::string peer2 = "localhost:" + std::to_string(transport2.local_port());

  // Register a dummy signal handler
  signal(SIGUSR1, [](int) {});

  // Send a signal to the process, which will interrupt some poll() calls with
  // EINTR.
  kill(getpid(), SIGUSR1);

  // Perform a push to verify the connection worker didn't die.
  std::vector<uint8_t> push_payload(1024, 0xAB);
  auto push_res = transport1.PushBuffer(
      peer2, /*buffer_id=*/0, /*dst_shard_idx=*/0, /*dst_offset_bytes=*/512,
      push_payload.data(), push_payload.size());
  ASSERT_TRUE(push_res.ok()) << push_res.message();
}

TEST(RawBufferTransportTest, RejectsOutOfBounds) {
  size_t size = 1024;
  RawMockDelegate delegate1(size);
  RawMockDelegate delegate2(size);

  RawBufferTransport transport1(&delegate1, 0);
  RawBufferTransport transport2(&delegate2, 0);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::string peer1 = "localhost:" + std::to_string(transport1.local_port());

  auto pull_res = transport2.PullBuffer(
      peer1, /*buffer_id=*/0, /*src_shard_idx=*/0, /*src_offset_bytes=*/0,
      /*dst_shard_idx=*/0, /*dst_offset_bytes=*/512, /*size_bytes=*/1024);
  EXPECT_FALSE(pull_res.ok());
}

class TestRawBufferTransport : public RawBufferTransport {
 public:
  using RawBufferTransport::AcquireConnection;
  using RawBufferTransport::RawBufferTransport;
  using RawBufferTransport::ReleaseConnection;
};

TEST(RawBufferTransportTest, MultiIpPoolingIsolation) {
  size_t size = 1024;
  RawMockDelegate delegate1(size);
  RawMockDelegate delegate2(size);

  RawBufferTransport transport1(&delegate1, 0);
  TestRawBufferTransport transport2(&delegate2, 0);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::string peer1 = "127.0.0.1:" + std::to_string(transport1.local_port());

  // 1. Acquire connection with local_ip = "127.0.0.1"
  auto fd1_or = transport2.AcquireConnection(peer1, "127.0.0.1");
  ASSERT_TRUE(fd1_or.ok()) << fd1_or.status().message();
  int fd1 = fd1_or.value();

  // Release it. It should be pooled under "127.0.0.1->peer1".
  transport2.ReleaseConnection(peer1, fd1, "127.0.0.1");

  // 2. Acquire connection with local_ip = "127.0.0.2"
  // This should NOT reuse fd1 because it's a different local IP.
  auto fd2_or = transport2.AcquireConnection(peer1, "127.0.0.2");
  ASSERT_TRUE(fd2_or.ok()) << fd2_or.status().message();
  int fd2 = fd2_or.value();

  EXPECT_NE(fd1, fd2);

  // Release it. It should be pooled under "127.0.0.2->peer1".
  transport2.ReleaseConnection(peer1, fd2, "127.0.0.2");

  // 3. Acquire connection with local_ip = "127.0.0.1" again.
  // This SHOULD reuse fd1.
  auto fd3_or = transport2.AcquireConnection(peer1, "127.0.0.1");
  ASSERT_TRUE(fd3_or.ok()) << fd3_or.status().message();
  int fd3 = fd3_or.value();

  EXPECT_EQ(fd1, fd3);
  transport2.ReleaseConnection(peer1, fd3, "127.0.0.1");

  // 4. Acquire connection with local_ip = "127.0.0.2" again.
  // This SHOULD reuse fd2.
  auto fd4_or = transport2.AcquireConnection(peer1, "127.0.0.2");
  ASSERT_TRUE(fd4_or.ok()) << fd4_or.status().message();
  int fd4 = fd4_or.value();

  EXPECT_EQ(fd2, fd4);
  transport2.ReleaseConnection(peer1, fd4, "127.0.0.2");
}

// Force warnings check
}  // namespace
}  // namespace transport
}  // namespace tpu_raiden
