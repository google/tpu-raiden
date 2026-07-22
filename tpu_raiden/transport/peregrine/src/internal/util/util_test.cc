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

#include "tpu_raiden/transport/peregrine/src/internal/util/util.h"

#include <gtest/gtest.h>
#include "absl/log/log.h"

namespace peregrine::internal::testing {
namespace {

TEST(UtilTest, IsPowerOfTwo) {
  EXPECT_FALSE(IsPowerOfTwo(0));
  EXPECT_TRUE(IsPowerOfTwo(1));
  EXPECT_TRUE(IsPowerOfTwo(2));
  EXPECT_FALSE(IsPowerOfTwo(3));
  EXPECT_TRUE(IsPowerOfTwo(4));
  EXPECT_FALSE(IsPowerOfTwo(15));
  EXPECT_TRUE(IsPowerOfTwo(16));
}

TEST(UtilTest, IsValid) {
  void* nonnull = reinterpret_cast<void*>(0xffff);
  ASSERT_NE(nonnull, nullptr);

  EXPECT_TRUE(IsValid({.iov_base = nonnull, .iov_len = 1}));
  EXPECT_TRUE(IsValid({.iov_base = nonnull, .iov_len = 0}));

  EXPECT_TRUE(IsValid({.iov_base = nullptr, .iov_len = 0}));
  EXPECT_FALSE(IsValid({.iov_base = nullptr, .iov_len = 1}));
}

TEST(UtilTest, TotalLength) {
  const struct iovec iov[] = {
      {.iov_base = (void*)0xffff0000, .iov_len = 1024},
      {.iov_base = (void*)0xffff1000, .iov_len = 2048},
      {.iov_base = (void*)0xffff2000, .iov_len = 4096},
  };
  EXPECT_TRUE(IsValid(iov));
  EXPECT_EQ(TotalLength(iov, 3), 7168);
}

TEST(UtilTest, ThreadId) {
  EXPECT_EQ(ThreadId(), ThreadId());
  LOG(INFO) << "thread #" << ThreadId();
}

}  // namespace
}  // namespace peregrine::internal::testing
