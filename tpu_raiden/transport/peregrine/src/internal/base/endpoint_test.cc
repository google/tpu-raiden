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

#include "tpu_raiden/transport/peregrine/src/internal/base/endpoint.h"

#include <arpa/inet.h>

#include <cstring>
#include <utility>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/ipaddr.h"

namespace peregrine::internal::testing {
namespace {

constexpr ipv4_t kIPv4{.s_addr = 0x0100007f};
constexpr ipv6_t kIPv6 = IN6ADDR_LOOPBACK_INIT;

TEST(EndpointTest, Create) {
  EXPECT_EQ(Endpoint::Create("127.0.0.1:12345"), Endpoint(kIPv4, 12345));
  EXPECT_EQ(Endpoint::Create("[::1]:54321"), Endpoint(kIPv6, 54321));

  EXPECT_FALSE(Endpoint::Create("?").IsValid());
  EXPECT_FALSE(Endpoint::Create("::1").IsValid());
  EXPECT_FALSE(Endpoint::Create("127.0.0.1:").IsValid());
}

TEST(EndpointTest, Validity) {
  EXPECT_FALSE(Endpoint().IsValid());
  EXPECT_FALSE(Endpoint(kIPv4, 0).IsValid());

  const Endpoint a(kIPv4, 23456);
  EXPECT_TRUE(a.IsValid());
  LOG(INFO) << "endpoint = " << a;
}

TEST(EndpointTest, Ctors) {
  const Endpoint a(kIPv4, 9999);
  const Endpoint b(a);
  const Endpoint c = a;
  LOG(INFO) << "a = " << a;
  LOG(INFO) << "b = " << b;
  LOG(INFO) << "c = " << c;

  const Endpoint d(std::move(a));
  const Endpoint e = std::move(b);
  LOG(INFO) << "d = " << d;
  LOG(INFO) << "e = " << e;

  const Endpoint f(IpAddr(kIPv4), 9999);
  EXPECT_EQ(f, c);
  LOG(INFO) << "f = " << f;

  const Endpoint g(IpAddr(kIPv6), 9999);
  const Endpoint h(kIPv6, 9999);
  EXPECT_EQ(g, h);
  LOG(INFO) << "g = " << g;
  LOG(INFO) << "h = " << h;
}

TEST(EndpointTest, Hash) {
  const Endpoint a(kIPv4, 9999);
  const Endpoint b(kIPv4, 9999);
  const Endpoint c(kIPv6, 9999);
  const Endpoint d(kIPv4, 7777);
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
  EXPECT_NE(a, d);

  EXPECT_EQ(a.Hash(), b.Hash());
  EXPECT_EQ(Endpoint::Hash(a), Endpoint::Hash(b));

  absl::flat_hash_set<Endpoint> set;
  set.insert(a);
  set.insert(b);
  set.insert(c);
  set.insert(d);
  EXPECT_EQ(set.size(), 3);
}

TEST(EndpointTest, BuildIPv4Sockaddr) {
  const Endpoint e = Endpoint::Create("127.0.0.1:34567");
  struct sockaddr_in sa = e.BuildIPv4Sockaddr();
  EXPECT_EQ(sa.sin_family, AF_INET);
  EXPECT_EQ(sa.sin_port, htons(34567));
  EXPECT_EQ(sa.sin_addr.s_addr, inet_addr("127.0.0.1"));
}

TEST(EndpointTest, BuildIPv6Sockaddr) {
  const Endpoint e = Endpoint::Create("[::1]:45678");
  struct sockaddr_in6 sa = e.BuildIPv6Sockaddr();
  EXPECT_EQ(sa.sin6_family, AF_INET6);
  EXPECT_EQ(sa.sin6_port, htons(45678));
  struct in6_addr addr2;
  ASSERT_EQ(inet_pton(AF_INET6, "::1", &addr2), 1);
  EXPECT_EQ(std::memcmp(&sa.sin6_addr, &addr2, sizeof(addr2)), 0);
}

}  // namespace
}  // namespace peregrine::internal::testing
