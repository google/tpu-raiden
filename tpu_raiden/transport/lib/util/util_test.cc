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

#include "tpu_raiden/transport/lib/util/util.h"

#include <gtest/gtest.h>
#include "absl/random/random.h"

namespace tpu_raiden::transport::lib::testing {
namespace {

TEST(UtilTest, Random) {
  absl::BitGen gen;
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(Random(gen, 0, 0), 0);
    EXPECT_EQ(Random(gen, 1, 1), 1);
    const int n = Random(gen, 1, 10);
    EXPECT_LE(1, n);
    EXPECT_LE(n, 10);
  }
}

}  // namespace
}  // namespace tpu_raiden::transport::lib::testing
