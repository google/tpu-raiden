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

#include "tpu_raiden/transport/peregrine/src/internal/socket/acceptor.h"

#include <sys/socket.h>

#include <memory>
#include <thread>  // NOLINT
#include <utility>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/endpoint.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_tcp.h"
#include "tpu_raiden/transport/peregrine/src/internal/util/test_util.h"

namespace peregrine::internal::testing {
namespace {

template <int kFamily>
class TcpAcceptorTest : public ::testing::Test {
 protected:
  TcpAcceptorTest()
      : local_(TestOnly_LocalEndpoint(kFamily, /*tcp=*/true)),
        acceptor_(TcpAcceptor::Create(local_)) {
    CHECK_NE(acceptor_, nullptr);
  }

  static void Accept(std::unique_ptr<TcpSocket> socket) {
    auto x = std::move(socket);
    CHECK_NE(x, nullptr);
  }

  static void ShortSleep() { absl::SleepFor(absl::Milliseconds(300)); }

 protected:
  const Endpoint local_;
  std::unique_ptr<TcpAcceptor> acceptor_;
};

using TcpAcceptorTestIPv4 = TcpAcceptorTest<AF_INET>;
using TcpAcceptorTestIPv6 = TcpAcceptorTest<AF_INET6>;

TEST_F(TcpAcceptorTestIPv4, StartThenStop) {
  std::jthread ta([&]() {
    DCHECK(acceptor_->Socket().IsBlocking());
    acceptor_->Start(Accept);
  });

  ShortSleep();
  acceptor_->Stop();
}

TEST_F(TcpAcceptorTestIPv6, StopThenStart) {
  acceptor_->Stop();

  std::jthread ta([&]() {
    DCHECK(acceptor_->Socket().IsBlocking());
    acceptor_->Start(Accept);
  });
}

}  // namespace
}  // namespace peregrine::internal::testing
