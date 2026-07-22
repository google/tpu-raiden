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

#include "tpu_raiden/transport/peregrine/src/internal/util/test_util.h"

#include <sys/socket.h>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/endpoint.h"

namespace peregrine::internal::testing {
namespace {

TEST(TestUtilTest, TcpPort) {
  for (const int family : {AF_INET, AF_INET6}) {
    EXPECT_NE(TestOnly_FindFreeTcpPort(family), 0);
  }
}

TEST(TestUtilTest, UdpPort) {
  for (const int family : {AF_INET, AF_INET6}) {
    EXPECT_NE(TestOnly_FindFreeUdpPort(family), 0);
  }
}

TEST(TestUtilTest, TcpSocket) {
  for (const int family : {AF_INET, AF_INET6}) {
    const auto socket = TestOnly_CreateTcpSocket(family);
    EXPECT_NE(socket, nullptr);
  }
}

TEST(TestUtilTest, UdpSocket) {
  for (const int family : {AF_INET, AF_INET6}) {
    const auto socket = TestOnly_CreateUdpSocket(family);
    EXPECT_NE(socket, nullptr);
  }
}

TEST(TestUtilTest, LocalEndpoint) {
  for (const int family : {AF_INET, AF_INET6}) {
    for (const bool tcp : {true, false}) {
      const Endpoint e = TestOnly_LocalEndpoint(family, tcp);
      EXPECT_TRUE(e.IsValid());
      LOG(INFO) << e;
    }
  }
}

}  // namespace
}  // namespace peregrine::internal::testing
