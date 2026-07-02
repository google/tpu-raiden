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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"

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


// Force warnings check
}  // namespace
}  // namespace transport
}  // namespace tpu_raiden
