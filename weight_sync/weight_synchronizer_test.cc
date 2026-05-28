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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "weight_sync/weight_synchronizer_base.h"

namespace tpu_raiden {
namespace weight_sync {
namespace {

class WeightSynchronizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Unified physical test parameters representing 64KB weight buffer E2E!
    num_layers_ = 1;
    num_shards_ = 1;
    slice_byte_size_ = 65536;  // 64KB
  }

  size_t num_layers_;
  size_t num_shards_;
  size_t slice_byte_size_;
};

TEST_F(WeightSynchronizerTest, PushAndPullWeightsCorrectnessE2e) {
  // 1. Instantiate three independent CPU-only synchronizers locally E2E!
  // ws_source represents the active RL Trainer
  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers_, num_shards_, slice_byte_size_,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  // ws_dest1 and ws_dest2 represent inference server peers
  auto ws_dest1 = std::make_unique<WeightSynchronizerBase>(
      num_layers_, num_shards_, slice_byte_size_,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  auto ws_dest2 = std::make_unique<WeightSynchronizerBase>(
      num_layers_, num_shards_, slice_byte_size_,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest1->local_port().has_value());
  ASSERT_TRUE(ws_dest2->local_port().has_value());

  std::string source_peer =
      "localhost:" + std::to_string(*ws_source->local_port());
  std::string dest1_peer =
      "localhost:" + std::to_string(*ws_dest1->local_port());
  std::string dest2_peer =
      "localhost:" + std::to_string(*ws_dest2->local_port());

  LOG(INFO) << "Launched C++ Weight Syncers: Source=" << source_peer
            << ", Dest1=" << dest1_peer << ", Dest2=" << dest2_peer;

  // 2. Populate the Trainer source buffer with distinct byte pattern (0xAB)
  // E2E!
  uint8_t* src_host_ptr = const_cast<uint8_t*>(ws_source->GetHostPointer(0, 0));
  ASSERT_NE(src_host_ptr, nullptr);
  std::memset(src_host_ptr, 0xAB, slice_byte_size_);

  // Populate inference servers with zeros baseline
  uint8_t* dest1_host_ptr =
      const_cast<uint8_t*>(ws_dest1->GetHostPointer(0, 0));
  uint8_t* dest2_host_ptr =
      const_cast<uint8_t*>(ws_dest2->GetHostPointer(0, 0));
  ASSERT_NE(dest1_host_ptr, nullptr);
  ASSERT_NE(dest2_host_ptr, nullptr);
  std::memset(dest1_host_ptr, 0x00, slice_byte_size_);
  std::memset(dest2_host_ptr, 0x00, slice_byte_size_);

  // Assert baseline state (zeros)
  EXPECT_EQ(dest1_host_ptr[0], 0x00);
  EXPECT_EQ(dest2_host_ptr[0], 0x00);

  // ==========================================================================
  // Test Scenario 1: Push weights from Trainer ws_source to both inference
  // peers!
  // ==========================================================================
  absl::Status push_status = ws_source->PushWeights({dest1_peer, dest2_peer});
  ASSERT_TRUE(push_status.ok()) << push_status.message();

  // Assert successful E2E network sockets streaming and exact byte parity!
  for (size_t i = 0; i < slice_byte_size_; ++i) {
    EXPECT_EQ(dest1_host_ptr[i], 0xAB) << "Mismatch at byte " << i;
    EXPECT_EQ(dest2_host_ptr[i], 0xAB) << "Mismatch at byte " << i;
  }

  // ==========================================================================
  // Test Scenario 2: Pull weights from Trainer ws_source E2E!
  // ==========================================================================
  // Reset destinations back to zeros baseline
  std::memset(dest1_host_ptr, 0x00, slice_byte_size_);
  std::memset(dest2_host_ptr, 0x00, slice_byte_size_);

  // ws_dest1 and ws_dest2 pull current weights from the trainer source peer!
  absl::Status pull_status1 = ws_dest1->PullWeights(source_peer);
  absl::Status pull_status2 = ws_dest2->PullWeights(source_peer);
  ASSERT_TRUE(pull_status1.ok()) << pull_status1.message();
  ASSERT_TRUE(pull_status2.ok()) << pull_status2.message();

  // Assert successful pulls E2E and exact byte parity!
  for (size_t i = 0; i < slice_byte_size_; ++i) {
    EXPECT_EQ(dest1_host_ptr[i], 0xAB) << "Mismatch at byte " << i;
    EXPECT_EQ(dest2_host_ptr[i], 0xAB) << "Mismatch at byte " << i;
  }
}

}  // namespace
}  // namespace weight_sync
}  // namespace tpu_raiden
