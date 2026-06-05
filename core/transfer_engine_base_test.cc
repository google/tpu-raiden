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

#include "core/transfer_engine_base.h"

#include "xla/tsl/platform/test.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/statusor.h"
#include "core/tpu_pjrt_manager.h"
#include "kv_cache/kv_cache_manager_base.h"

namespace tpu_raiden {
namespace {

using ::absl_testing::IsOk;

TEST(TransferEngineBaseTest, LocalD2HAndH2D) {
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * pjrt_manager,
                          TpuPjrtManager::GetDefault());

  // Rank 3 shape: {2, 32, 32} of float.
  std::vector<int64_t> shape_dims = {2, 32, 32};
  int64_t elements_per_slice = 32 * 32;
  int64_t total_elements = 2 * elements_per_slice;

  std::vector<float> host_data(total_elements);
  for (int i = 0; i < total_elements; ++i) {
    host_data[i] = static_cast<float>(i);
  }

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      pjrt_manager->BufferFromHost(host_data.data(), xla::F32, shape_dims));

  ASSERT_THAT(buffer->GetReadyFuture().Await(), IsOk());

  // Create KVCacheManagerBase
  std::vector<std::vector<xla::PjRtBuffer*>> layer_buffers = {{buffer.get()}};
  auto kv_manager = std::make_unique<kv_cache::KVCacheManagerBase>(
      layer_buffers, /*block_size=*/1, /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/4,
      /*external_host_ptrs=*/std::nullopt,
      /*unsafe_skip_buffer_lock=*/true);

  // Configure staging slots: 2 slots, max 2 blocks per slot
  ASSERT_THAT(kv_manager->ConfigureHostStagingSlots(2, 2), IsOk());

  // Create TransferEngineBase
  auto engine = std::make_unique<TransferEngineBase>(
      std::move(kv_manager), /*tp_rank=*/0, /*local_control_port=*/0,
      /*max_blocks=*/2, /*num_slots=*/2, /*timeout_s=*/10.0,
      /*unsafe_skip_buffer_lock=*/true);

  // 1. Test SubmitD2H (Full copy of 2 blocks to slot 0)
  int64_t op_id_d2h = engine->SubmitD2H(/*slot_idx=*/0, /*num_blocks=*/2,
                                        /*block_ids=*/{0, 1});
  engine->WaitTransfer(op_id_d2h);

  // Verify data in host span
  std::vector<kv_cache::KVCacheHostSpan> host_spans =
      engine->LayerSpans(/*slot_idx=*/0, /*num_blocks=*/2);
  ASSERT_EQ(host_spans.size(), 1);

  // Since it's tiled, we don't easily check float values directly unless we
  // detile. But we can check that it is not all zeros.
  float* host_floats = reinterpret_cast<float*>(host_spans[0].ptr);
  bool all_zero = true;
  for (size_t i = 0; i < host_spans[0].nbytes / sizeof(float); ++i) {
    if (host_floats[i] != 0.0f) {
      all_zero = false;
      break;
    }
  }
  EXPECT_FALSE(all_zero);

  // 2. Test SubmitH2D
  // We will copy from slot 0 back to device, but first let's zero out the
  // device buffer by creating a new empty buffer and H2D copying from a zeroed
  // host span? Actually, we can just modify the host span and copy it back,
  // then verify device buffer changed. Let's fill the host span with a distinct
  // value (e.g. 42.0f). Wait, if we fill tiled host span with a constant, it
  // should be constant on device too (no matter tiling).
  size_t num_floats = host_spans[0].nbytes / sizeof(float);
  for (size_t i = 0; i < num_floats; ++i) {
    host_floats[i] = 42.0f;
  }

  int64_t op_id_h2d = engine->SubmitH2D(/*slot_idx=*/0, /*num_blocks=*/2,
                                        /*local_block_ids=*/{0, 1});
  engine->WaitTransfer(op_id_h2d);

  // Verify device buffer contains 42.0f
  TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer->ToLiteral().Await());
  auto read_back = literal->data<float>();
  ASSERT_EQ(read_back.size(), total_elements);
  for (int i = 0; i < total_elements; ++i) {
    EXPECT_EQ(read_back[i], 42.0f);
  }
}

TEST(TransferEngineBaseTest, StartReadAcceptsParallelism) {
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * pjrt_manager,
                          TpuPjrtManager::GetDefault());
  std::vector<int64_t> shape_dims = {2, 32, 32};
  std::vector<float> host_data(2 * 32 * 32, 0.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      pjrt_manager->BufferFromHost(host_data.data(), xla::F32, shape_dims));
  ASSERT_THAT(buffer->GetReadyFuture().Await(), IsOk());

  std::vector<std::vector<xla::PjRtBuffer*>> layer_buffers = {{buffer.get()}};
  auto kv_manager = std::make_unique<kv_cache::KVCacheManagerBase>(
      layer_buffers, /*block_size=*/1, /*local_port=*/std::nullopt,
      /*host_blocks_to_allocate=*/4, std::nullopt, true);
  ASSERT_THAT(kv_manager->ConfigureHostStagingSlots(2, 2), IsOk());

  auto engine = std::make_unique<TransferEngineBase>(std::move(kv_manager), 0,
                                                     0, 2, 2, 10.0, true);

  // Calling StartRead with a non-existent port throws or returns an op that
  // fails
  int64_t op_id = engine->StartRead("req_parallel", 99999, "127.0.0.1:8888",
                                    {0}, {0}, /*parallelism=*/2);
  EXPECT_GT(op_id, 0);
}

}  // namespace
}  // namespace tpu_raiden
