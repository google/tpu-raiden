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

#include "transport/socket_transport.h"

#include <chrono>  // NOLINT
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "third_party/peregrine/src/api/transport.h"
#include "third_party/peregrine/src/api/types.h"

namespace tpu_raiden {
namespace transport {
namespace {

// Helper to poll completion with a bounding timeout.
bool PollUntilDone(peregrine::Transport* transport, peregrine::Handle handle,
                   int timeout_ms = 5000) {
  auto start = std::chrono::steady_clock::now();
  while (true) {
    auto status_or_s = transport->Poll(handle);
    if (status_or_s.ok() && peregrine::IsCompleted(status_or_s.value())) {
      return status_or_s.value() == peregrine::Status::kSuccess;
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - start)
                       .count();
    if (elapsed > timeout_ms) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

TEST(SocketTransportTest, PointToPointWriteTransfer) {
  // Instantiate two endpoints on separate ports.
  int port1 = 23456;
  int port2 = 23457;
  auto transport1 = std::make_unique<SocketTransport>(port1);
  auto transport2 = std::make_unique<SocketTransport>(port2);

  // Give background server listeners a moment to initialize listening sockets.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Allocate source and destination memory payloads.
  std::string src_payload =
      "Hello Distributed KV Cache Transfer via POSIX TCP!";
  std::vector<uint8_t> dst_payload(src_payload.size(), 0);

  // Prepare peregrine Request struct mapping remote memory pointers cleanly.
  peregrine::Request request;
  request.op = peregrine::Op::kWrite;
  request.laddr = reinterpret_cast<uint8_t*>(src_payload.data());
  request.raddr = dst_payload.data();
  request.len = src_payload.size();

  // Post write transfer from transport1 to transport2 endpoint string.
  std::string peer2 = absl::StrCat("127.0.0.1:", port2);
  auto status_or_handle = transport1->Post(peer2, request);
  ASSERT_TRUE(status_or_handle.ok()) << status_or_handle.status().message();

  // Poll completion.
  EXPECT_TRUE(PollUntilDone(transport1.get(), status_or_handle.value()));

  // Assert direct destination memory structure matching.
  EXPECT_EQ(
      std::memcmp(dst_payload.data(), src_payload.data(), src_payload.size()),
      0);
}

TEST(SocketTransportTest, PointToPointReadTransfer) {
  int port1 = 23458;
  int port2 = 23459;
  auto transport1 = std::make_unique<SocketTransport>(port1);
  auto transport2 = std::make_unique<SocketTransport>(port2);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  std::string remote_payload = "Remote data sequence ready to pull!";
  std::vector<uint8_t> local_payload(remote_payload.size(), 0);

  peregrine::Request request;
  request.op = peregrine::Op::kRead;
  request.laddr = local_payload.data();
  request.raddr = reinterpret_cast<uint8_t*>(remote_payload.data());
  request.len = remote_payload.size();

  std::string peer2 = absl::StrCat("127.0.0.1:", port2);
  auto status_or_handle = transport1->Post(peer2, request);
  ASSERT_TRUE(status_or_handle.ok()) << status_or_handle.status().message();

  EXPECT_TRUE(PollUntilDone(transport1.get(), status_or_handle.value()));

  // Assert successful read copy.
  EXPECT_EQ(std::memcmp(local_payload.data(), remote_payload.data(),
                        remote_payload.size()),
            0);
}

}  // namespace
}  // namespace transport
}  // namespace tpu_raiden
