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

#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_udp.h"

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
class UdpSocketTest : public ::testing::Test {
 protected:
  UdpSocketTest()
      : sndr_(kFamily == AF_INET ? IPv4Localhost() : IPv6Localhost(),
              TestOnly_FindFreeUdpPort(kFamily)),
        rcvr_(kFamily == AF_INET ? IPv4Localhost() : IPv6Localhost(),
              TestOnly_FindFreeUdpPort(kFamily)),
        sskt_(TestOnly_CreateUdpSocket(kFamily)),
        rskt_(TestOnly_CreateUdpSocket(kFamily)) {
    CHECK_NE(sndr_.Port(), rcvr_.Port());
    DCHECK(sskt_->IsValid());
    DCHECK(rskt_->IsValid());
    DCHECK(!sskt_->IsConnected());
    DCHECK(!rskt_->IsConnected());
    DCHECK_NE(sskt_->fd(), rskt_->fd());
  }

 protected:
  const Endpoint sndr_;
  const Endpoint rcvr_;
  const std::unique_ptr<UdpSocket> sskt_;
  const std::unique_ptr<UdpSocket> rskt_;
};

using UdpSocketIPv4Test = UdpSocketTest<AF_INET>;
using UdpSocketIPv6Test = UdpSocketTest<AF_INET6>;

TEST_F(UdpSocketIPv4Test, SendRecv) {
  // Create a small send message and a recv buffer.
  const std::vector<Byte> message = {'h', 'e', 'l', 'l', 'o'};
  const size_t kMsgSize = message.size();
  std::vector<Byte> recv_buf(kMsgSize, 0);
  ASSERT_NE(recv_buf, message);

  // First, create a receiver thread.
  absl::Notification rcvr_ready;
  std::thread receiver([&]() {
    CHECK(rskt_->Bind(rcvr_));
    CHECK(rskt_->Connect(sndr_));
    DCHECK(rskt_->IsBlocking());
    DCHECK(rskt_->IsConnected());
    rcvr_ready.Notify();
    const ssize_t n = rskt_->Recv(recv_buf.data(), kMsgSize);
    CHECK_GT(n, 0);
    CHECK_LE(n, kMsgSize);
  });

  // Second, create a sender thread.
  std::thread sender([&]() {
    rcvr_ready.WaitForNotification();
    CHECK(sskt_->Bind(sndr_));
    CHECK(sskt_->Connect(rcvr_));
    DCHECK(sskt_->IsBlocking());
    DCHECK(sskt_->IsConnected());
    CHECK_EQ(sskt_->Send(message.data(), kMsgSize), kMsgSize);
  });

  // Wait for both threads to finish.
  sender.join();
  receiver.join();

  // Check that the server got the client's message.
  EXPECT_EQ(recv_buf, message);
}

TEST_F(UdpSocketIPv6Test, ScatterGather) {
  // Create a small send message and a recv buffer.
  const std::vector<Byte> message = {'h', 'e', 'l', 'l', 'o'};
  const size_t kMsgSize = message.size();
  std::vector<Byte> recv_buf(kMsgSize, 0);
  ASSERT_NE(recv_buf, message);

  // First, create a receiver thread.
  absl::Notification rcvr_ready;
  std::thread receiver([&]() {
    CHECK(rskt_->Bind(rcvr_));
    CHECK(rskt_->Connect(sndr_));
    DCHECK(rskt_->IsBlocking());
    DCHECK(rskt_->IsConnected());
    constexpr int kRN = 2;
    const struct iovec recv_iov[kRN] = {
        {.iov_base = (void*)recv_buf.data(), .iov_len = 2},
        {.iov_base = (void*)(recv_buf.data() + 2), .iov_len = kMsgSize - 2},
    };
    rcvr_ready.Notify();
    const ssize_t n = rskt_->RecvV(recv_iov, kRN, kMsgSize);
    CHECK_GT(n, 0);
    CHECK_LE(n, kMsgSize);
  });

  // Second, create a sender thread.
  std::thread sender([&]() {
    rcvr_ready.WaitForNotification();
    CHECK(sskt_->Bind(sndr_));
    CHECK(sskt_->Connect(rcvr_));
    DCHECK(sskt_->IsBlocking());
    DCHECK(sskt_->IsConnected());
    constexpr int kSN = 2;
    const struct iovec send_iov[kSN] = {
        {.iov_base = (void*)message.data(), .iov_len = 1},
        {.iov_base = (void*)(message.data() + 1), .iov_len = kMsgSize - 1},
    };
    CHECK_EQ(sskt_->SendV(send_iov, kSN, kMsgSize), kMsgSize);
  });

  // Wait for both threads to finish.
  sender.join();
  receiver.join();

  // Check that the server got the client's message.
  EXPECT_EQ(recv_buf, message);
}

}  // namespace
}  // namespace peregrine::internal::testing
