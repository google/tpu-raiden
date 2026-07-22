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

#include "tpu_raiden/transport/peregrine/src/api/socket_util.h"

#include <sys/socket.h>
#include <sys/types.h>

#include <cstring>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "tpu_raiden/transport/peregrine/src/api/transport_types.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/endpoint.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/types.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_tcp.h"
#include "tpu_raiden/transport/peregrine/src/internal/util/test_util.h"
#include "tpu_raiden/transport/peregrine/src/util/util.h"

namespace peregrine::testing {
namespace {

using ::peregrine::internal::Endpoint;
using ::peregrine::internal::TcpSocket;
using ::peregrine::internal::testing::IPv4Localhost;
using ::peregrine::internal::testing::IPv6Localhost;
using ::peregrine::internal::testing::TestOnly_CreateTcpSocket;
using ::peregrine::internal::testing::TestOnly_FindFreeTcpPort;
using ::peregrine::util::RandomNonZero;
using ::testing::Combine;
using ::testing::Eq;
using ::testing::Ne;
using ::testing::Pointwise;
using ::testing::TestParamInfo;
using ::testing::Values;

using Param = std::tuple</*family=*/int, /*riov=*/bool, /*wiov=*/bool>;

std::string ToString(const TestParamInfo<Param>& info) {
  const int family = std::get<0>(info.param);
  const bool read_iovec = std::get<1>(info.param);
  const bool write_iovec = std::get<2>(info.param);
  DCHECK(family == AF_INET || family == AF_INET6);
  return absl::StrFormat("IPv%d_Read%s_Write%s", family == AF_INET ? 4 : 6,
                         read_iovec ? "V" : "", write_iovec ? "V" : "");
}

class SocketUtilTest : public ::testing::TestWithParam<Param> {
 protected:
  SocketUtilTest()
      : family_(std::get<0>(GetParam())),
        read_iovec_(std::get<1>(GetParam())),
        write_iovec_(std::get<2>(GetParam())),
        local_(family_ == AF_INET ? IPv4Localhost() : IPv6Localhost(),
               TestOnly_FindFreeTcpPort(family_)),
        peer_(local_),
        listener_(TestOnly_CreateTcpSocket(family_)),
        connector_(TestOnly_CreateTcpSocket(family_)) {
    DCHECK(listener_->IsValid());
    DCHECK(connector_->IsValid());
    DCHECK(!listener_->IsConnected());
    DCHECK(!connector_->IsConnected());
    DCHECK_NE(listener_->fd(), connector_->fd());
  }

 protected:
  const int family_;
  const bool read_iovec_;
  const bool write_iovec_;
  const Endpoint local_;
  const Endpoint peer_;
  const std::unique_ptr<TcpSocket> listener_;
  const std::unique_ptr<TcpSocket> connector_;
};

INSTANTIATE_TEST_SUITE_P(, SocketUtilTest,
                         Combine(/*family=*/Values(AF_INET, AF_INET6),
                                 /*riov=*/Values(false, true),
                                 /*wiov=*/Values(false, true)),
                         ToString);

TEST_P(SocketUtilTest, ReadWrite) {
  // Create a big chunk of send/recv buffers with random data.
  constexpr size_t kSize = 64UL << 20;
  std::vector<Byte> send_buf(kSize, 0x01);
  std::vector<Byte> recv_buf(kSize, 0x00);
  RandomNonZero(absl::MakeSpan(send_buf));
  ASSERT_THAT(recv_buf, Pointwise(Ne(), send_buf));

  // First, create a server thread.
  absl::Notification server_ready;
  std::thread server([&]() {
    CHECK(listener_->Listen(local_));
    server_ready.Notify();
    DCHECK(listener_->IsBlocking());
    const internal::fd_t new_fd = listener_->Accept();

    CHECK_GE(new_fd.value(), 0);
    auto new_socket = TcpSocket::Create(new_fd, family_);
    DCHECK(new_socket->IsBlocking());
    DCHECK(new_socket->IsConnected());

    if (read_iovec_) {
      std::vector<struct iovec> iovs;
      const size_t partial = kSize / 3;
      iovs.push_back({recv_buf.data(), partial});
      iovs.push_back({recv_buf.data() + partial, partial});
      iovs.push_back({recv_buf.data() + partial * 2, kSize - partial * 2});
      CHECK_OK(ReadVExact(new_socket->fd().value(), iovs));
    } else {
      CHECK_OK(ReadExact(new_socket->fd().value(), recv_buf.data(), kSize));
    }
  });

  // Second, create a client thread.
  std::thread client([&]() {
    server_ready.WaitForNotification();
    CHECK(connector_->Connect(peer_));
    DCHECK(connector_->IsBlocking());
    DCHECK(connector_->IsConnected());

    if (write_iovec_) {
      std::vector<struct iovec> iovs;
      const size_t partial = kSize / 2;
      iovs.push_back({send_buf.data(), partial});
      iovs.push_back({send_buf.data() + partial, kSize - partial});
      CHECK_OK(WriteVExact(connector_->fd().value(), iovs));
    } else {
      CHECK_OK(WriteExact(connector_->fd().value(), send_buf.data(), kSize));
    }
  });

  // Wait for both threads to finish.
  client.join();
  server.join();

  // Check that the recv buffer has the same data as the send.
  ASSERT_THAT(recv_buf, Pointwise(Eq(), send_buf));
}

}  // namespace
}  // namespace peregrine::testing
