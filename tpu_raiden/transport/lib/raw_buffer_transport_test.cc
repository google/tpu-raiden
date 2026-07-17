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

#include "tpu_raiden/transport/lib/raw_buffer_transport.h"

#include <signal.h>

#include <chrono>  // NOLINT
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_raiden/transport/lib/util/util.h"

namespace tpu_raiden::transport::lib::testing {
namespace {

using ::testing::Each;
using ::testing::Eq;
using ::testing::Ne;
using ::testing::Pointwise;

constexpr int kLocalPort = 0;
constexpr size_t kBufferId = 0;
constexpr size_t kSrcShardIdx = 0;
constexpr size_t kDstShardIdx = 0;

std::string GetIpPort(const RawBufferTransport& transport) {
  return "localhost:" + std::to_string(transport.local_port());
}

class RawMockDelegate : public RawBufferTransportDelegate {
 public:
  explicit RawMockDelegate(size_t buffer_size) : buffer_(buffer_size, 0) {
    DCHECK(AllZero(buffer_));
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
  absl::Span<uint8_t> DataSpan() { return absl::MakeSpan(buffer_); }
  absl::Span<const uint8_t> DataSpan(size_t offset, size_t length) {
    return absl::MakeConstSpan(buffer_.data() + offset, length);
  }

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
  // Set up src/dst buffers.
  constexpr size_t size = 64 * 1024;
  RawMockDelegate src(size);
  RawMockDelegate dst(size);
  RandomNonZero(src.DataSpan());

  // Pre-condition: all the dst bytes are not equal to the src.
  ASSERT_THAT(dst.DataSpan(), Pointwise(Ne(), src.DataSpan()));

  // Create two transports.
  RawBufferTransport src_transport(&src, kLocalPort);
  RawBufferTransport dst_transport(&dst, kLocalPort);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Pull a buffer segment from src to dst.
  const std::string src_addr = GetIpPort(src_transport);
  constexpr size_t kLen = 62 * 1024;
  constexpr size_t kSrcOffset = 512;
  constexpr size_t kDstOffset = 1024;
  const auto pull_res =
      dst_transport.PullBuffer(src_addr, kBufferId, kSrcShardIdx, kSrcOffset,
                               kDstShardIdx, kDstOffset, kLen);
  ASSERT_OK(pull_res) << pull_res.message();

  // Post-condition: only the copied dst bytes are equal to the src.
  EXPECT_THAT(dst.DataSpan(0, kDstOffset), Each(Eq(0)));
  EXPECT_THAT(dst.DataSpan(kDstOffset, kLen),
              Pointwise(Eq(), src.DataSpan(kSrcOffset, kLen)));
  EXPECT_THAT(dst.DataSpan(kDstOffset + kLen, size - kDstOffset - kLen),
              Each(Eq(0)));
}

TEST(RawBufferTransportTest, PushBufferCorrectness) {
  // Set up src/dst buffers.
  constexpr size_t size = 64 * 1024;
  RawMockDelegate src(size);
  RawMockDelegate dst(size);
  RandomNonZero(src.DataSpan());

  // Pre-condition: all the dst bytes are not equal to the src.
  ASSERT_THAT(dst.DataSpan(), Pointwise(Ne(), src.DataSpan()));

  // Create two transports.
  RawBufferTransport src_transport(&src, kLocalPort);
  RawBufferTransport dst_transport(&dst, kLocalPort);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Push a buffer segment from src to dst.
  constexpr size_t kLen = 62 * 1024;
  constexpr size_t kDstOffset = 512;
  std::vector<uint8_t> push_payload(kLen);
  RandomNonZero(absl::MakeSpan(push_payload));
  const std::string dst_addr = GetIpPort(dst_transport);
  const auto push_res =
      src_transport.PushBuffer(dst_addr, kBufferId, kDstShardIdx, kDstOffset,
                               push_payload.data(), push_payload.size());
  EXPECT_OK(push_res) << push_res.message();

  // Post-condition: only the copied dst bytes are equal to the src.
  EXPECT_THAT(dst.DataSpan(0, kDstOffset), Each(Eq(0)));
  EXPECT_THAT(dst.DataSpan(kDstOffset, kLen),
              Pointwise(Eq(), absl::MakeConstSpan(push_payload)));
  EXPECT_THAT(dst.DataSpan(kDstOffset + kLen, size - kDstOffset - kLen),
              Each(Eq(0)));
}

TEST(RawBufferTransportTest, PollEINTRIsBenign) {
  // Set up src/dst buffers.
  constexpr size_t size = 4096;
  RawMockDelegate src(size);
  RawMockDelegate dst(size);

  // Create two transports.
  RawBufferTransport src_transport(&src, 0);
  RawBufferTransport dst_transport(&dst, 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Register a dummy signal handler.
  signal(SIGUSR1, [](int) {});
  // Send a signal to the process to interrupt some poll() calls with EINTR.
  kill(getpid(), SIGUSR1);

  // Perform a push to verify the connection worker didn't die.
  const std::string dst_addr = GetIpPort(dst_transport);
  const std::vector<uint8_t> push_payload(1024, 0xAB);
  constexpr size_t kDstOffset = 512;
  const auto push_res =
      src_transport.PushBuffer(dst_addr, kBufferId, kDstShardIdx, kDstOffset,
                               push_payload.data(), push_payload.size());
  EXPECT_OK(push_res) << push_res.message();
}

TEST(RawBufferTransportTest, RejectsOutOfBounds) {
  // Set up src/dst buffers.
  constexpr size_t size = 1024;
  RawMockDelegate src(size);
  RawMockDelegate dst(size);

  // Create two transports.
  RawBufferTransport src_transport(&src, 0);
  RawBufferTransport dst_transport(&dst, 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Pulling an out-of-bounds buffer segment from src to dst should fail.
  const std::string src_addr = GetIpPort(src_transport);
  constexpr size_t kSrcOffset = 0;
  constexpr size_t kDstOffset = size / 2;
  constexpr size_t kLen = size / 2 + 1;
  static_assert(kDstOffset + kLen > size);
  const auto pull_res =
      dst_transport.PullBuffer(src_addr, kBufferId, kSrcShardIdx, kSrcOffset,
                               kDstShardIdx, kDstOffset, kLen);
  EXPECT_FALSE(pull_res.ok()) << pull_res.message();
}

class TestRawBufferTransport : public RawBufferTransport {
 public:
  using RawBufferTransport::AcquireConnection;
  using RawBufferTransport::RawBufferTransport;
  using RawBufferTransport::ReleaseConnection;
};

TEST(RawBufferTransportTest, MultiIpPoolingIsolation) {
  // Set up src/dst buffers.
  constexpr size_t size = 1024;
  RawMockDelegate src(size);
  RawMockDelegate dst(size);

  // Create two transports.
  RawBufferTransport src_transport(&src, 0);
  TestRawBufferTransport dst_transport(&dst, 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // 1. Acquire connection with local_ip = "127.0.0.1"
  const std::string src_addr = GetIpPort(src_transport);
  const auto fd1_or = dst_transport.AcquireConnection(src_addr, "127.0.0.1");
  ASSERT_OK(fd1_or) << fd1_or.status().message();
  const int fd1 = fd1_or.value();

  // Release it. It should be pooled under "127.0.0.1->peer1".
  dst_transport.ReleaseConnection(src_addr, fd1, "127.0.0.1");

  // 2. Acquire connection with local_ip = "127.0.0.2"
  // This should NOT reuse fd1 because it's a different local IP.
  const auto fd2_or = dst_transport.AcquireConnection(src_addr, "127.0.0.2");
  ASSERT_OK(fd2_or) << fd2_or.status().message();
  const int fd2 = fd2_or.value();

  EXPECT_NE(fd1, fd2);

  // Release it. It should be pooled under "127.0.0.2->peer1".
  dst_transport.ReleaseConnection(src_addr, fd2, "127.0.0.2");

  // 3. Acquire connection with local_ip = "127.0.0.1" again.
  // This SHOULD reuse fd1.
  const auto fd3_or = dst_transport.AcquireConnection(src_addr, "127.0.0.1");
  ASSERT_OK(fd3_or) << fd3_or.status().message();
  const int fd3 = fd3_or.value();

  EXPECT_EQ(fd1, fd3);
  dst_transport.ReleaseConnection(src_addr, fd3, "127.0.0.1");

  // 4. Acquire connection with local_ip = "127.0.0.2" again.
  // This SHOULD reuse fd2.
  const auto fd4_or = dst_transport.AcquireConnection(src_addr, "127.0.0.2");
  ASSERT_OK(fd4_or) << fd4_or.status().message();
  const int fd4 = fd4_or.value();

  EXPECT_EQ(fd2, fd4);
  dst_transport.ReleaseConnection(src_addr, fd4, "127.0.0.2");
}

}  // namespace
}  // namespace tpu_raiden::transport::lib::testing
