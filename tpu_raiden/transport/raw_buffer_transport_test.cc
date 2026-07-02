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

struct TestCommandMetadata {
  uint32_t sequence;
  char action_name[16];
};

TEST(RawBufferTransportTest, IssueCommandCorrectness) {
  RawMockDelegate delegate1(1024);
  RawMockDelegate delegate2(1024);

  RawBufferTransport transport1(&delegate1, 0);
  RawBufferTransport transport2(&delegate2, 0);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::string peer2 = "localhost:" + std::to_string(transport2.local_port());

  TestCommandMetadata expected_meta = {1337, "test-action"};
  std::string serialized_meta(reinterpret_cast<const char*>(&expected_meta),
                              sizeof(TestCommandMetadata));

  bool callback_executed = false;
  absl::Notification done;
  auto register_status = transport2.RegisterCommand(
      200,
      [&](RawBufferTransport::ConnectionCloser closer,
          absl::string_view command_meta) -> absl::Status {
        if (command_meta.size() != sizeof(TestCommandMetadata)) {
          return absl::InvalidArgumentError("Received invalid metadata size");
        }
        const auto* received_meta =
            reinterpret_cast<const TestCommandMetadata*>(command_meta.data());
        EXPECT_EQ(received_meta->sequence, 1337);
        EXPECT_STREQ(received_meta->action_name, "test-action");
        callback_executed = true;
        closer.close_fn(true);
        done.Notify();
        return absl::OkStatus();
      });
  ASSERT_TRUE(register_status.ok()) << register_status.message();

  auto send_res = transport1.IssueCommand(peer2, 200, serialized_meta);
  ASSERT_TRUE(send_res.ok()) << send_res.status().message();

  done.WaitForNotification();
  send_res->close_fn(true);

  EXPECT_TRUE(callback_executed);
}

TEST(RawBufferTransportTest, E2EPushCommandAndBufferTransfer) {
  struct PushCommand {
    char source_peer[128];
    size_t offset_bytes;
    size_t size_bytes;
  };

  size_t size = 4096;
  RawMockDelegate delegate1(size);
  RawMockDelegate delegate2(size);

  std::vector<uint8_t> payload_data(256, 0x55);
  std::vector<uint8_t> payload_data2(256, 0x66);
  std::vector<uint8_t> e2e_pull_buf(256, 0);
  std::vector<uint8_t> e2e_pull_buf2(256, 0);

  RawBufferTransport transport1(&delegate1, 0);
  RawBufferTransport transport2(&delegate2, 0);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::string peer1 = "localhost:" + std::to_string(transport1.local_port());
  std::string peer2 = "localhost:" + std::to_string(transport2.local_port());

  // 1. Receiver (transport2) registers command 100.
  // The callback will read the metadata and directly call Receive (reusing fd)
  // to read the incoming data stream from the sender.
  bool callback_executed = false;
  absl::Notification done;
  auto register_status = transport2.RegisterCommand(
      100,
      [&](RawBufferTransport::ConnectionCloser closer,
          absl::string_view command_meta) -> absl::Status {
        if (command_meta.size() < sizeof(PushCommand)) {
          return absl::InvalidArgumentError("Invalid PushCommand size");
        }
        PushCommand cmd =
            *reinterpret_cast<const PushCommand*>(command_meta.data());

        // First Receive
        ReceiveSpec receive_spec;
        receive_spec.fd = closer.fd;  // Reuse connection!
        receive_spec.slices = {{e2e_pull_buf.data(), cmd.size_bytes}};
        absl::Status status = transport2.Receive(receive_spec);
        EXPECT_TRUE(status.ok()) << status.message();

        // Second Receive
        ReceiveSpec receive_spec2;
        receive_spec2.fd = closer.fd;
        receive_spec2.slices = {{e2e_pull_buf2.data(), cmd.size_bytes}};
        status = transport2.Receive(receive_spec2);
        EXPECT_TRUE(status.ok()) << status.message();

        callback_executed = true;
        closer.close_fn(true);
        done.Notify();
        return status;
      });
  ASSERT_TRUE(register_status.ok());

  // 2. Sender (transport1) issues command 100 with peer1 address in metadata.
  PushCommand cmd = {};
  std::strncpy(cmd.source_peer, peer1.c_str(), sizeof(cmd.source_peer) - 1);
  cmd.offset_bytes = 0;
  cmd.size_bytes = 256;

  std::string serialized_cmd(reinterpret_cast<const char*>(&cmd), sizeof(cmd));

  auto send_cmd_status = transport1.IssueCommand(peer2, 100, serialized_cmd);
  ASSERT_TRUE(send_cmd_status.ok()) << send_cmd_status.status().message();
  auto closer = std::move(send_cmd_status.value());

  // 3. After IssueCommand is acked, sender (transport1) pushes the data
  // using Send (Active Sender) reusing the same connection.
  SendSpec send_spec;
  send_spec.fd = closer.fd;
  send_spec.slices = {
      {payload_data.data(), payload_data.size()},
  };
  auto send_status = transport1.Send(send_spec);
  ASSERT_TRUE(send_status.ok()) << send_status.message();

  // Second Send
  SendSpec send_spec2;
  send_spec2.fd = closer.fd;
  send_spec2.slices = {
      {payload_data2.data(), payload_data2.size()},
  };
  send_status = transport1.Send(send_spec2);
  ASSERT_TRUE(send_status.ok()) << send_status.message();

  // Wait for the passive receive thread to complete.
  done.WaitForNotification();

  // Close connection
  closer.close_fn(true);

  EXPECT_TRUE(callback_executed);
  EXPECT_EQ(e2e_pull_buf[0], 0x55);
  EXPECT_EQ(e2e_pull_buf[255], 0x55);
  EXPECT_EQ(e2e_pull_buf2[0], 0x66);
  EXPECT_EQ(e2e_pull_buf2[255], 0x66);
}

TEST(RawBufferTransportTest, E2EPullCommandAndBufferTransfer) {
  struct PullCommand {
    char dest_peer[128];
    size_t offset_bytes;
    size_t size_bytes;
  };

  size_t size = 4096;
  RawMockDelegate delegate1(size);
  RawMockDelegate delegate2(size);

  std::vector<uint8_t> payload_data(256, 0x77);
  std::vector<uint8_t> e2e_pull_buf(256, 0);

  RawBufferTransport transport1(&delegate1, 0);  // Active Puller (A)
  RawBufferTransport transport2(&delegate2, 0);  // Passive Provider (B)

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::string peer2 = "localhost:" + std::to_string(transport2.local_port());

  // 1. Provider (transport2) registers command 101.
  // The callback will read the metadata and call Send (reusing fd)
  // to stream the requested local data back to the puller.
  bool callback_executed = false;
  absl::Notification done;
  auto register_status = transport2.RegisterCommand(
      101,
      [&](RawBufferTransport::ConnectionCloser closer,
          absl::string_view command_meta) -> absl::Status {
        if (command_meta.size() < sizeof(PullCommand)) {
          return absl::InvalidArgumentError("Invalid PullCommand size");
        }
        PullCommand cmd =
            *reinterpret_cast<const PullCommand*>(command_meta.data());

        SendSpec send_spec;
        send_spec.fd = closer.fd;  // Reuse connection!
        send_spec.slices = {{payload_data.data(), cmd.size_bytes}};

        absl::Status status = transport2.Send(send_spec);
        EXPECT_TRUE(status.ok()) << status.message();
        callback_executed = true;
        closer.close_fn(true);
        done.Notify();
        return status;
      });
  ASSERT_TRUE(register_status.ok());

  // 2. Active Puller (transport1) issues command 101.
  PullCommand cmd = {};
  cmd.offset_bytes = 0;
  cmd.size_bytes = 256;

  std::string serialized_cmd(reinterpret_cast<const char*>(&cmd), sizeof(cmd));

  auto send_cmd_status = transport1.IssueCommand(peer2, 101, serialized_cmd);
  ASSERT_TRUE(send_cmd_status.ok()) << send_cmd_status.status().message();
  auto closer = std::move(send_cmd_status.value());

  // 3. After IssueCommand is acked, active puller (transport1) receives the
  // data using Receive reusing the same connection.
  ReceiveSpec receive_spec;
  receive_spec.fd = closer.fd;
  receive_spec.slices = {{e2e_pull_buf.data(), cmd.size_bytes}};

  auto receive_status = transport1.Receive(receive_spec);
  ASSERT_TRUE(receive_status.ok()) << receive_status.message();

  // Wait for the provider thread callback to complete.
  done.WaitForNotification();

  // Close connection
  closer.close_fn(true);

  EXPECT_TRUE(callback_executed);
  EXPECT_EQ(e2e_pull_buf[0], 0x77);
  EXPECT_EQ(e2e_pull_buf[255], 0x77);
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
