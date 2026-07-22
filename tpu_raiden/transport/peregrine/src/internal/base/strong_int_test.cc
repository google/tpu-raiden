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

#include "tpu_raiden/transport/peregrine/src/internal/base/strong_int.h"

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"

namespace peregrine::internal::testing {
namespace {

TEST(StrongIntTest, Basic) {
  DEFINE_STRONG_INT_TYPE(T, int);

  const T x(1);
  EXPECT_EQ(x.value(), 1);
  EXPECT_EQ(x + x, T(2));
  EXPECT_EQ(x - x, T(0));

  absl::flat_hash_set<T> ts;
  ts.insert(x);
  EXPECT_TRUE(ts.contains(x));
  EXPECT_EQ(ts.size(), 1);
  ts.insert(x);
  EXPECT_EQ(ts.size(), 1);
  ts.erase(x);
  EXPECT_TRUE(ts.empty());
}

}  // namespace
}  // namespace peregrine::internal::testing
