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


#include "tpu_raiden/core/kv_manager_holder.h"

#include <vector>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/core/raiden_transfer_endpoint.h"
#include "tpu_raiden/core/controller/test_util.h"

namespace tpu_raiden {
namespace {

using ::testing::_;
using ::testing::Return;
using ::tpu_raiden::controller::MockTransferManager;
using ::tpu_raiden::controller::ShardAwareMockTransferManager;

TEST(KVManagerHolderTest, H2dReadVectorFallbackToString) {
  MockTransferManager mock;
  KVManagerHolder holder(&mock);

  std::vector<RaidenTransferEndpoint> eps = {{"peer_a", {}}};
  auto result = holder.H2dRead(eps, {1}, {2}, {3}, {4});
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(mock.h2d_read_calls, 1);
  EXPECT_EQ(mock.last_peer, "peer_a");
}

TEST(KVManagerHolderTest, H2dReadVectorPrefersVectorOverload) {
  ShardAwareMockTransferManager mock;
  KVManagerHolder holder(&mock);
  
  std::vector<RaidenTransferEndpoint> eps = {{"peer_a", {}}};
  auto result = holder.H2dRead(eps, {1}, {2}, {3}, {4});
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(mock.vector_h2d_read_calls, 1);
}

}  // namespace
}  // namespace tpu_raiden
