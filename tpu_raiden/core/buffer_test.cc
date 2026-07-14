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

#include "tpu_raiden/core/buffer.h"

#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"

namespace tpu_raiden {
namespace {

TEST(BufferTest, ConstructAndAccessProperties) {
  std::vector<BufferShard> shards = {
      {/*handle=*/1, /*offset=*/0, /*size=*/1024},
      {/*handle=*/2, /*offset=*/1024, /*size=*/2048},
  };

  Buffer buffer(42, shards);

  EXPECT_EQ(buffer.index(), 42);
  EXPECT_EQ(buffer.shards().size(), 2);
  EXPECT_EQ(buffer.shards()[0].handle, 1);
  EXPECT_EQ(buffer.shards()[0].offset, 0);
  EXPECT_EQ(buffer.shards()[0].size, 1024);
  EXPECT_EQ(buffer.shards()[1].handle, 2);
  EXPECT_EQ(buffer.shards()[1].offset, 1024);
  EXPECT_EQ(buffer.shards()[1].size, 2048);

  EXPECT_FALSE(buffer.RemoteAddress().has_value());
}

TEST(BufferTest, ConstructWithRemoteAddress) {
  std::vector<BufferShard> shards = {
      {/*handle=*/1, /*offset=*/0, /*size=*/1024},
  };
  std::string remote_address = "localhost:12345";

  Buffer buffer(7, shards, remote_address);

  EXPECT_EQ(buffer.index(), 7);
  EXPECT_TRUE(buffer.RemoteAddress().has_value());
  EXPECT_EQ(*buffer.RemoteAddress(), remote_address);
}

}  // namespace
}  // namespace tpu_raiden
