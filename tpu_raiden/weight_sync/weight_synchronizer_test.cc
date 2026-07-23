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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"
#include "tpu_raiden/weight_sync/weight_synchronizer_base.h"

ABSL_DECLARE_FLAG(size_t, raiden_weight_sync_host_buffer_scratchpad_size);

namespace tpu_raiden {
namespace weight_sync {
namespace {

class WeightSynchronizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    absl::SetFlag(&FLAGS_raiden_weight_sync_host_buffer_scratchpad_size, 0);
    // Unified physical test parameters representing 64KB weight buffer E2E!
    num_layers_ = 1;
    num_shards_ = 1;
    slice_byte_size_ = 65536;  // 64KB
  }

  size_t num_layers_;
  size_t num_shards_;
  size_t slice_byte_size_;
};

TEST_F(WeightSynchronizerTest, PushWeightsCorrectnessE2e) {
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
}

TEST_F(WeightSynchronizerTest, PushWeightsHeterogeneousCorrectness) {
  size_t num_layers = 2;
  size_t num_shards = 1;
  std::vector<size_t> slice_byte_sizes = {16384, 32768};

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_sizes,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_sizes,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());

  std::string source_peer =
      "localhost:" + std::to_string(*ws_source->local_port());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  // Populate source layers
  uint8_t* src_l0_ptr = const_cast<uint8_t*>(ws_source->GetHostPointer(0, 0));
  uint8_t* src_l1_ptr = const_cast<uint8_t*>(ws_source->GetHostPointer(1, 0));
  ASSERT_NE(src_l0_ptr, nullptr);
  ASSERT_NE(src_l1_ptr, nullptr);
  std::memset(src_l0_ptr, 0xAA, slice_byte_sizes[0]);
  std::memset(src_l1_ptr, 0xBB, slice_byte_sizes[1]);

  // Populate dest layers with zeros
  uint8_t* dest_l0_ptr = const_cast<uint8_t*>(ws_dest->GetHostPointer(0, 0));
  uint8_t* dest_l1_ptr = const_cast<uint8_t*>(ws_dest->GetHostPointer(1, 0));
  ASSERT_NE(dest_l0_ptr, nullptr);
  ASSERT_NE(dest_l1_ptr, nullptr);
  std::memset(dest_l0_ptr, 0x00, slice_byte_sizes[0]);
  std::memset(dest_l1_ptr, 0x00, slice_byte_sizes[1]);

  absl::Status push_status = ws_source->PushWeights({dest_peer});
  ASSERT_TRUE(push_status.ok()) << push_status.message();

  for (size_t i = 0; i < slice_byte_sizes[0]; ++i) {
    EXPECT_EQ(dest_l0_ptr[i], 0xAA) << "Mismatch at layer 0 byte " << i;
  }
  for (size_t i = 0; i < slice_byte_sizes[1]; ++i) {
    EXPECT_EQ(dest_l1_ptr[i], 0xBB) << "Mismatch at layer 1 byte " << i;
  }
}

TEST_F(WeightSynchronizerTest, CustomLayerNamesInitialization) {
  size_t num_layers = 2;
  size_t num_shards = 1;
  std::vector<size_t> slice_byte_sizes = {16384, 32768};
  std::vector<std::string> custom_names = {"my_layer_0", "my_layer_1"};

  auto ws = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_sizes,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1,
      /*parallelism=*/1, /*listener_port=*/std::nullopt,
      /*bind_ip=*/std::nullopt, custom_names);

  EXPECT_EQ(ws->layer_names().size(), 2);
  EXPECT_EQ(ws->layer_names()[0], "my_layer_0");
  EXPECT_EQ(ws->layer_names()[1], "my_layer_1");
}

TEST_F(WeightSynchronizerTest, PushWeightsReshardedExactBoundary) {
  size_t num_layers = 1;
  size_t num_shards = 1;
  size_t slice_byte_size = 16384;

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  uint8_t* src_host_ptr = const_cast<uint8_t*>(ws_source->GetHostPointer(0, 0));
  uint8_t* dest_host_ptr = const_cast<uint8_t*>(ws_dest->GetHostPointer(0, 0));
  ASSERT_NE(src_host_ptr, nullptr);
  ASSERT_NE(dest_host_ptr, nullptr);
  std::memset(src_host_ptr, 0xAB, slice_byte_size);
  std::memset(dest_host_ptr, 0x00, slice_byte_size);

  tpu_raiden::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(12345);

  auto* schedules = request.mutable_shard_push_schedules();
  auto* entry = (*schedules)[0].add_entries();
  entry->set_dst_peer(dest_peer);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(slice_byte_size);
  entry->set_count(1);
  entry->set_layer_idx(0);

  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();

  for (size_t i = 0; i < slice_byte_size; ++i) {
    EXPECT_EQ(dest_host_ptr[i], 0xAB) << "Mismatch at byte " << i;
  }
}

TEST_F(WeightSynchronizerTest, PushWeightsReshardedZeroBytes) {
  size_t num_layers = 1;
  size_t num_shards = 1;
  size_t slice_byte_size = 16384;

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  uint8_t* src_host_ptr = const_cast<uint8_t*>(ws_source->GetHostPointer(0, 0));
  uint8_t* dest_host_ptr = const_cast<uint8_t*>(ws_dest->GetHostPointer(0, 0));
  ASSERT_NE(src_host_ptr, nullptr);
  ASSERT_NE(dest_host_ptr, nullptr);
  std::memset(src_host_ptr, 0xAB, slice_byte_size);
  std::memset(dest_host_ptr, 0x00, slice_byte_size);

  tpu_raiden::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(12345);

  auto* schedules = request.mutable_shard_push_schedules();
  auto* entry = (*schedules)[0].add_entries();
  entry->set_dst_peer(dest_peer);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(0);  // 0 bytes request
  entry->set_count(1);
  entry->set_layer_idx(0);

  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();

  for (size_t i = 0; i < slice_byte_size; ++i) {
    EXPECT_EQ(dest_host_ptr[i], 0x00) << "Mismatch at byte " << i;
  }
}

TEST_F(WeightSynchronizerTest, PushWeightsReshardedOutOfBoundsError) {
  size_t num_layers = 1;
  size_t num_shards = 1;
  size_t slice_byte_size = 1024;

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  tpu_raiden::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(12345);

  auto* schedules = request.mutable_shard_push_schedules();
  auto* entry = (*schedules)[0].add_entries();
  entry->set_dst_peer(dest_peer);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(500);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(600);  // 500 + 600 = 1100 > 1024
  entry->set_count(1);
  entry->set_layer_idx(0);

  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(WeightSynchronizerTest, PushWeightsReshardedInvalidLayerIndexError) {
  size_t num_layers = 2;
  size_t num_shards = 1;
  size_t slice_byte_size = 1024;

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  tpu_raiden::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(12345);

  auto* schedules = request.mutable_shard_push_schedules();
  auto* entry = (*schedules)[0].add_entries();
  entry->set_dst_peer(dest_peer);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(128);
  entry->set_count(1);
  entry->set_layer_idx(2);  // out of bounds (only 2 layers, indices 0 and 1)

  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(WeightSynchronizerTest, PushWeightsReshardedEmptySchedule) {
  size_t num_layers = 1;
  size_t num_shards = 1;
  size_t slice_byte_size = 1024;

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_size,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1);

  tpu_raiden::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(12345);
  // No schedules added

  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_F(WeightSynchronizerTest, PushWeightsReshardedFallbackByNameSuccess) {
  size_t num_layers = 2;
  size_t num_shards = 1;
  std::vector<size_t> slice_byte_sizes = {16384, 16384};
  std::vector<std::string> custom_names = {"layer_A", "layer_B"};

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_sizes,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1,
      /*parallelism=*/1, /*listener_port=*/std::nullopt,
      /*bind_ip=*/std::nullopt, custom_names);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_sizes,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1,
      /*parallelism=*/1, /*listener_port=*/std::nullopt,
      /*bind_ip=*/std::nullopt, custom_names);

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  // Populate source layer_B (index 1) with 0xAB
  uint8_t* src_l1_ptr = const_cast<uint8_t*>(ws_source->GetHostPointer(1, 0));
  ASSERT_NE(src_l1_ptr, nullptr);
  std::memset(src_l1_ptr, 0xAB, slice_byte_sizes[1]);

  // Populate dest layer_B with zeros
  uint8_t* dest_l1_ptr = const_cast<uint8_t*>(ws_dest->GetHostPointer(1, 0));
  ASSERT_NE(dest_l1_ptr, nullptr);
  std::memset(dest_l1_ptr, 0x00, slice_byte_sizes[1]);

  tpu_raiden::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(12345);

  auto* src_unit = request.add_src_units();
  src_unit->set_data_name("layer_B");

  auto* schedules = request.mutable_shard_push_schedules();
  auto* entry = (*schedules)[0].add_entries();
  entry->set_dst_peer(dest_peer);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(slice_byte_sizes[1]);
  entry->set_count(1);
  // Do not set layer_idx (forces fallback)

  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();

  for (size_t i = 0; i < slice_byte_sizes[1]; ++i) {
    EXPECT_EQ(dest_l1_ptr[i], 0xAB) << "Mismatch at byte " << i;
  }
}

TEST_F(WeightSynchronizerTest,
       PushWeightsReshardedFallbackByNameNotFoundError) {
  size_t num_layers = 2;
  size_t num_shards = 1;
  std::vector<size_t> slice_byte_sizes = {16384, 16384};
  std::vector<std::string> custom_names = {"layer_A", "layer_B"};

  auto ws_source = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_sizes,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1,
      /*parallelism=*/1, /*listener_port=*/std::nullopt,
      /*bind_ip=*/std::nullopt, custom_names);
  auto ws_dest = std::make_unique<WeightSynchronizerBase>(
      num_layers, num_shards, slice_byte_sizes,
      /*local_port=*/0, /*host_blocks_to_allocate=*/1,
      /*parallelism=*/1, /*listener_port=*/std::nullopt,
      /*bind_ip=*/std::nullopt, custom_names);

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  tpu_raiden::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(12345);

  auto* src_unit = request.add_src_units();
  src_unit->set_data_name("unknown_layer");  // Not registered

  auto* schedules = request.mutable_shard_push_schedules();
  auto* entry = (*schedules)[0].add_entries();
  entry->set_dst_peer(dest_peer);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(1024);
  entry->set_count(1);
  // Do not set layer_idx

  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);
}

TEST_F(WeightSynchronizerTest, PushWeightsReshardedSkipD2h) {
  auto client_status_or = xla::GetXlaPjrtCpuClient(xla::CpuClientOptions());
  ASSERT_TRUE(client_status_or.ok()) << client_status_or.status().message();
  auto client = std::move(client_status_or.value());

  size_t slice_byte_size = 1024;

  std::vector<uint8_t> src_device_data(slice_byte_size, 0xDD);
  auto memory_space_status_or =
      client->addressable_devices()[0]->default_memory_space();
  ASSERT_TRUE(memory_space_status_or.ok())
      << memory_space_status_or.status().message();
  xla::PjRtMemorySpace* memory_space = memory_space_status_or.value();

  auto src_buffer_status_or = client->BufferFromHostBuffer(
      src_device_data.data(), xla::U8, {static_cast<int64_t>(slice_byte_size)},
      /*byte_strides=*/std::nullopt,
      xla::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
      /*on_done_with_host_buffer=*/nullptr, memory_space,
      /*device_layout=*/nullptr);
  ASSERT_TRUE(src_buffer_status_or.ok())
      << src_buffer_status_or.status().message();
  auto src_pjrt_buffer = std::move(src_buffer_status_or.value());

  std::vector<uint8_t> dest_device_data(slice_byte_size, 0x00);
  auto dest_buffer_status_or = client->BufferFromHostBuffer(
      dest_device_data.data(), xla::U8, {static_cast<int64_t>(slice_byte_size)},
      /*byte_strides=*/std::nullopt,
      xla::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
      /*on_done_with_host_buffer=*/nullptr, memory_space,
      /*device_layout=*/nullptr);
  ASSERT_TRUE(dest_buffer_status_or.ok())
      << dest_buffer_status_or.status().message();
  auto dest_pjrt_buffer = std::move(dest_buffer_status_or.value());

  std::vector<std::vector<xla::PjRtBuffer*>> src_buffers = {
      {src_pjrt_buffer.get()}};
  std::vector<std::vector<xla::PjRtBuffer*>> dest_buffers = {
      {dest_pjrt_buffer.get()}};

  auto ws_source =
      std::make_unique<WeightSynchronizerBase>(src_buffers, /*local_port=*/0);
  auto ws_dest =
      std::make_unique<WeightSynchronizerBase>(dest_buffers, /*local_port=*/0);

  ASSERT_TRUE(ws_source->local_port().has_value());
  ASSERT_TRUE(ws_dest->local_port().has_value());
  std::string dest_peer = "localhost:" + std::to_string(*ws_dest->local_port());

  uint8_t* src_host_ptr = const_cast<uint8_t*>(ws_source->GetHostPointer(0, 0));
  ASSERT_NE(src_host_ptr, nullptr);
  std::memset(src_host_ptr, 0xAA, slice_byte_size);

  uint8_t* dest_host_ptr = const_cast<uint8_t*>(ws_dest->GetHostPointer(0, 0));
  ASSERT_NE(dest_host_ptr, nullptr);
  std::memset(dest_host_ptr, 0x00, slice_byte_size);

  tpu_raiden::rpc::StartTransferRequest request;
  request.set_skip_d2h(true);
  request.set_uuid(12345);

  auto* schedules = request.mutable_shard_push_schedules();
  auto* entry = (*schedules)[0].add_entries();
  entry->set_dst_peer(dest_peer);
  entry->set_dst_shard_idx(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(slice_byte_size);
  entry->set_count(1);
  entry->set_layer_idx(0);

  absl::Status status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();

  for (size_t i = 0; i < slice_byte_size; ++i) {
    EXPECT_EQ(dest_host_ptr[i], 0xAA) << "Mismatch at byte " << i;
  }

  std::memset(dest_host_ptr, 0x00, slice_byte_size);
  request.set_skip_d2h(false);
  status = ws_source->PushWeightsResharded(request);
  EXPECT_TRUE(status.ok()) << status.message();

  for (size_t i = 0; i < slice_byte_size; ++i) {
    EXPECT_EQ(dest_host_ptr[i], 0xDD) << "Mismatch at byte " << i;
  }
}

}  // namespace
}  // namespace weight_sync
}  // namespace tpu_raiden
