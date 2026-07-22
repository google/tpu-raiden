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

#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_tcp.h"

#include <sys/socket.h>
#include <sys/types.h>

#include <cstring>
#include <memory>
#include <thread>  // NOLINT
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/synchronization/notification.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/endpoint.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/types.h"
#include "tpu_raiden/transport/peregrine/src/internal/util/test_util.h"

namespace peregrine::internal::testing {
namespace {

template <int kFamily>
class TcpSocketTest : public ::testing::Test {
  static_assert(kFamily == AF_INET || kFamily == AF_INET6);

 protected:
  TcpSocketTest()
      : local_(kFamily == AF_INET ? IPv4Localhost() : IPv6Localhost(),
               TestOnly_FindFreeTcpPort(kFamily)),
        listener_(TestOnly_CreateTcpSocket(kFamily)),
        connector_(TestOnly_CreateTcpSocket(kFamily)) {
    DCHECK(listener_->IsValid());
    DCHECK(connector_->IsValid());
    DCHECK(!listener_->IsConnected());
    DCHECK(!connector_->IsConnected());
    DCHECK_NE(listener_->fd(), connector_->fd());
  }

 protected:
  const Endpoint local_;
  const std::unique_ptr<TcpSocket> listener_;
  const std::unique_ptr<TcpSocket> connector_;
};

using TcpIPv4SocketTest = TcpSocketTest<AF_INET>;
using TcpIPv6SocketTest = TcpSocketTest<AF_INET6>;

TEST_F(TcpIPv4SocketTest, SmallMessage) {
  // Create a small send message and a recv buffer.
  const std::vector<Byte> message = {'h', 'e', 'l', 'l', 'o'};
  const size_t kMsgSize = message.size();
  std::vector<Byte> recv_buf(kMsgSize, 0);
  ASSERT_NE(recv_buf, message);

  // First, create a server thread.
  absl::Notification server_ready;
  std::thread server([&]() {
    CHECK(listener_->Listen(local_));
    server_ready.Notify();
    DCHECK(listener_->IsBlocking());
    const fd_t new_fd = listener_->Accept();

    CHECK_GE(new_fd.value(), 0);
    auto new_socket = TcpSocket::Create(new_fd, AF_INET);
    DCHECK(new_socket->IsBlocking());
    DCHECK(new_socket->IsConnected());
    CHECK_EQ(new_socket->Recv(recv_buf.data(), kMsgSize), kMsgSize);
    CHECK_OK(TcpSocket::Recv(new_socket->fd(), recv_buf.data(), kMsgSize));
  });

  // Second, create a client thread.
  std::thread client([&]() {
    server_ready.WaitForNotification();
    CHECK(connector_->Connect(local_));
    DCHECK(connector_->IsBlocking());
    DCHECK(connector_->IsConnected());
    CHECK_EQ(connector_->Send(message.data(), kMsgSize), kMsgSize);
    CHECK_OK(TcpSocket::Send(connector_->fd(), message.data(), kMsgSize));
  });

  // Wait for both threads to finish.
  client.join();
  server.join();

  // Check that the server got the client's message.
  EXPECT_EQ(recv_buf, message);
}

TEST_F(TcpIPv6SocketTest, BigData) {
  // Create a big chunk of data and a recv buffer.
  constexpr size_t kDataSize = 16UL << 20;
  std::vector<Byte> send_buf(kDataSize, 0x01);
  std::vector<Byte> recv_buf(kDataSize, 0x02);
  ASSERT_NE(recv_buf, send_buf);

  // First, create a server thread.
  absl::Notification server_ready;
  std::thread server([&]() {
    CHECK(listener_->Listen(local_));
    server_ready.Notify();
    DCHECK(listener_->IsBlocking());
    const fd_t new_fd = listener_->Accept();

    CHECK_GE(new_fd.value(), 0);
    auto new_socket = TcpSocket::Create(new_fd, AF_INET6);
    DCHECK(new_socket->IsBlocking());
    DCHECK(new_socket->IsConnected());
    CHECK_EQ(new_socket->Recv(recv_buf.data(), kDataSize), kDataSize);
    CHECK_OK(TcpSocket::Recv(new_socket->fd(), recv_buf.data(), kDataSize));
  });

  // Second, create a client thread.
  std::thread client([&]() {
    server_ready.WaitForNotification();
    CHECK(connector_->Connect(local_));
    DCHECK(connector_->IsBlocking());
    DCHECK(connector_->IsConnected());
    CHECK_EQ(connector_->Send(send_buf.data(), kDataSize), kDataSize);
    CHECK_OK(TcpSocket::Send(connector_->fd(), send_buf.data(), kDataSize));
  });

  // Wait for both threads to finish.
  client.join();
  server.join();

  // Check that the server got the client's data.
  EXPECT_EQ(recv_buf, send_buf);
}

}  // namespace
}  // namespace peregrine::internal::testing
