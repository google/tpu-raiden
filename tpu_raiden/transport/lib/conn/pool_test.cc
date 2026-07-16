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

#include "tpu_raiden/transport/lib/conn/pool.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tpu_raiden::transport::lib::testing {
namespace {

TEST(ConnPoolTest, Basic) {
  ConnPool pool;
  const std::string peer = "localhost:12345";

  const auto fd_or = pool.Borrow(peer);
  ASSERT_OK(fd_or) << fd_or.status().message();

  const int fd = fd_or.value();
  pool.Return(/*ok=*/true, fd, peer);

  pool.Close();
}

TEST(ConnPoolTest, MultiIpPoolingIsolation) {
  const std::string peer = "127.0.0.1:12345";
  ConnPool pool;

  // 1. Borrow connection with local_ip = "127.0.0.1"
  const auto fd1_or = pool.Borrow(peer, "127.0.0.1");
  ASSERT_OK(fd1_or) << fd1_or.status().message();
  const int fd1 = fd1_or.value();

  // Return it. It should be pooled under "127.0.0.1->peer".
  pool.Return(/*ok=*/true, fd1, peer, "127.0.0.1");

  // 2. Borrow connection with local_ip = "127.0.0.2"
  // This should NOT reuse fd1 because it's a different local IP.
  const auto fd2_or = pool.Borrow(peer, "127.0.0.2");
  ASSERT_OK(fd2_or) << fd2_or.status().message();
  const int fd2 = fd2_or.value();
  EXPECT_NE(fd1, fd2);

  // Return it. It should be pooled under "127.0.0.2->peer".
  pool.Return(/*ok=*/true, fd2, peer, "127.0.0.2");

  // 3. Borrow connection with local_ip = "127.0.0.1" again.
  // This SHOULD reuse fd1.
  const auto fd3_or = pool.Borrow(peer, "127.0.0.1");
  ASSERT_OK(fd3_or) << fd3_or.status().message();
  const int fd3 = fd3_or.value();
  EXPECT_EQ(fd3, fd1);
  pool.Return(/*ok=*/true, fd3, peer, "127.0.0.1");

  // 4. Borrow connection with local_ip = "127.0.0.2" again.
  // This SHOULD reuse fd2.
  const auto fd4_or = pool.Borrow(peer, "127.0.0.2");
  ASSERT_OK(fd4_or) << fd4_or.status().message();
  const int fd4 = fd4_or.value();
  EXPECT_EQ(fd4, fd2);
  pool.Return(/*ok=*/true, fd4, peer, "127.0.0.2");

  pool.Close();
}

}  // namespace
}  // namespace tpu_raiden::transport::lib::testing
