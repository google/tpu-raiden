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
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tpu_raiden/kv_cache/kv_cache_manager_base.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"
#include "tpu_raiden/transport/block_transport.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

class TestKVCacheManager : public KVCacheManagerBase {
 public:
  TestKVCacheManager(size_t num_layers, size_t num_shards,
                     size_t slice_byte_size, int host_blocks = 0)
      : KVCacheManagerBase(
            num_layers, num_shards, slice_byte_size, /*local_port=*/std::nullopt,
            host_blocks > 0 ? std::make_optional(host_blocks) : std::nullopt) {
    buffer_holds_.resize(num_layers);
    for (size_t l = 0; l < num_layers; ++l) {
      buffer_holds_[l].holds.resize(num_shards);
    }
  }

  void SetLayerPhysicalSizeForTest(size_t layer_idx, size_t physical_size,
                                   int64_t major_dim_size) {
    buffer_holds_[layer_idx].physical_size = physical_size;
    major_dim_size_ = major_dim_size;
  }
};

LayerBlockLayout FullAttentionLayout(int64_t slot_bytes) {
  return LayerBlockLayout{
      .kind = LayerKind::kFullAttention,
      .slot_bytes = slot_bytes,
      .regions = {RegionSpec{
          .name = "fa_payload",
          .offset_bytes = 0,
          .stride_bytes = 64,
          .unit_bytes = 32,
          .num_units = 2,
          .units_per_stride = 1,
      }},
  };
}

LayerBlockLayout MambaLayout(int64_t slot_bytes) {
  return LayerBlockLayout{
      .kind = LayerKind::kMambaState,
      .slot_bytes = slot_bytes,
      .regions = {RegionSpec{
          .name = "gdn_ssm",
          .offset_bytes = 0,
          .stride_bytes = 64,
          .unit_bytes = 64,
          .num_units = 2,
          .units_per_stride = 1,
      }},
  };
}

TEST(KVCacheManagerTest, CompilesAndLinksSuccessfully) {
  EXPECT_TRUE(true);
}

TEST(KVCacheManagerTest, UnregisterActivePlanAllowsUuidReuse) {
  TestKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128);
  tpu_raiden::rpc::StartTransferRequest request;
  request.set_uuid(112233);
  request.set_is_sender(true);

  absl::Status status = manager.UnregisterActivePlan(112233);
  EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);

  status = manager.RegisterActivePlan(112233, request, /*is_sender=*/true);
  EXPECT_TRUE(status.ok()) << status.ToString();

  status = manager.RegisterActivePlan(112233, request, /*is_sender=*/true);
  EXPECT_EQ(status.code(), absl::StatusCode::kAlreadyExists);

  status = manager.UnregisterActivePlan(112233);
  EXPECT_TRUE(status.ok()) << status.ToString();

  status = manager.RegisterActivePlan(112233, request, /*is_sender=*/true);
  EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(KVCacheManagerTest, D2hFailsWithMismatchedCopySpecLengths) {
  TestKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128);
  std::vector<int64_t> src_offsets = {0};
  std::vector<int64_t> dst_offsets = {0, 1};  // Mismatch
  std::vector<int64_t> sizes = {1};

  auto status = manager.D2h(src_offsets, dst_offsets, sizes);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.status().message(),
              testing::HasSubstr("must have the same length"));
}

TEST(KVCacheManagerTest, HybridLayoutProtoRoundTrip) {
  LayerBlockLayout layout = FullAttentionLayout(/*slot_bytes=*/128);
  tpu_raiden::rpc::LayerBlockLayoutProto proto = ToProto(layout);

  auto parsed = LayerBlockLayoutFromProto(proto);
  ASSERT_TRUE(parsed.ok()) << parsed.status().ToString();
  EXPECT_EQ(parsed->kind, LayerKind::kFullAttention);
  EXPECT_EQ(parsed->slot_bytes, 128);
  ASSERT_EQ(parsed->regions.size(), 1);
  EXPECT_EQ(parsed->regions[0].name, "fa_payload");
  EXPECT_EQ(parsed->regions[0].offset_bytes, 0);
  EXPECT_EQ(parsed->regions[0].stride_bytes, 64);
  EXPECT_EQ(parsed->regions[0].unit_bytes, 32);
  EXPECT_EQ(parsed->regions[0].num_units, 2);
  EXPECT_EQ(parsed->regions[0].units_per_stride, 1);
  EXPECT_TRUE(parsed->Validate(/*manager_slot_bytes=*/128).ok());
}

TEST(KVCacheManagerTest, HybridLayoutValidationRejectsBadRegions) {
  LayerBlockLayout overflow = FullAttentionLayout(/*slot_bytes=*/128);
  overflow.regions[0].offset_bytes = 96;
  overflow.regions[0].num_units = 2;
  overflow.regions[0].stride_bytes = 64;
  absl::Status status = overflow.Validate(/*manager_slot_bytes=*/128);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), testing::HasSubstr("exceeds slot bytes"));

  LayerBlockLayout wrong_slot = FullAttentionLayout(/*slot_bytes=*/64);
  status = wrong_slot.Validate(/*manager_slot_bytes=*/128);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), testing::HasSubstr("slot_bytes"));

  LayerBlockLayout compact_mamba = MambaLayout(/*slot_bytes=*/160);
  status = compact_mamba.Validate(/*manager_slot_bytes=*/128);
  EXPECT_TRUE(status.ok()) << status.ToString();

  status = compact_mamba.Validate(/*manager_slot_bytes=*/96);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), testing::HasSubstr("exceeds slot bytes"));

  status = compact_mamba.Validate(/*manager_slot_bytes=*/192);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), testing::HasSubstr("slot_bytes"));

  LayerBlockLayout negative = FullAttentionLayout(/*slot_bytes=*/128);
  negative.regions[0].offset_bytes = -1;
  status = negative.Validate(/*manager_slot_bytes=*/128);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), testing::HasSubstr("offset_bytes"));

  negative = FullAttentionLayout(/*slot_bytes=*/128);
  negative.regions[0].num_units = -1;
  status = negative.Validate(/*manager_slot_bytes=*/128);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), testing::HasSubstr("num_units"));
}

TEST(KVCacheManagerTest, SetBlockLayoutsAndGetHybridBlockRef) {
  TestKVCacheManager manager(/*num_layers=*/2, /*num_shards=*/1,
                             /*slice_byte_size=*/128, /*host_blocks=*/2);
  std::vector<LayerBlockLayout> layouts = {
      MambaLayout(/*slot_bytes=*/128),
      FullAttentionLayout(/*slot_bytes=*/128),
  };

  absl::Status status = manager.SetBlockLayouts(layouts);
  ASSERT_TRUE(status.ok()) << status.ToString();

  std::vector<size_t> fa_layers =
      manager.LayerIndicesOfKind(LayerKind::kFullAttention);
  EXPECT_THAT(fa_layers, testing::ElementsAre(1));

  const LayerBlockLayout* layer0 = manager.block_layout(0);
  ASSERT_NE(layer0, nullptr);
  EXPECT_EQ(layer0->kind, LayerKind::kMambaState);

  auto ref = manager.GetHybridBlockRef(/*layer_idx=*/1, /*shard_idx=*/0,
                                       /*block_id=*/1);
  ASSERT_TRUE(ref.ok()) << ref.status().ToString();
  EXPECT_EQ(ref->slot_bytes, 128);
  EXPECT_EQ(ref->layout->kind, LayerKind::kFullAttention);
  EXPECT_EQ(ref->layer_idx, 1);
  EXPECT_EQ(ref->shard_idx, 0);
  EXPECT_EQ(ref->block_id, 1);
  EXPECT_EQ(ref->ptr, manager.GetHostPointer(/*layer_idx=*/1, /*shard_idx=*/0) +
                      128);

  auto out_of_range = manager.GetHybridBlockRef(/*layer_idx=*/1,
                                                /*shard_idx=*/0,
                                                /*block_id=*/2);
  EXPECT_EQ(out_of_range.status().code(), absl::StatusCode::kOutOfRange);
}

TEST(KVCacheManagerTest, HybridBlockRefUsesHostStrideAndReportsLayerSlot) {
  TestKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/256, /*host_blocks=*/2);
  manager.SetLayerPhysicalSizeForTest(/*layer_idx=*/0,
                                      /*physical_size=*/128,
                                      /*major_dim_size=*/1);

  ASSERT_TRUE(
      manager.SetBlockLayouts({FullAttentionLayout(/*slot_bytes=*/128)}).ok());
  EXPECT_EQ(manager.bytes_per_block(), 256);
  EXPECT_EQ(manager.LayerBlockByteSize(/*layer_idx=*/0), 128);

  auto ref = manager.GetHybridBlockRef(/*layer_idx=*/0, /*shard_idx=*/0,
                                       /*block_id=*/1);
  ASSERT_TRUE(ref.ok()) << ref.status().ToString();
  EXPECT_EQ(ref->slot_bytes, 128);
  EXPECT_EQ(ref->ptr, manager.GetHostPointer(/*layer_idx=*/0,
                                             /*shard_idx=*/0) +
                      256);
}

TEST(KVCacheManagerTest, SetBlockLayoutsFailsAfterActivePlanRegistered) {
  TestKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128, /*host_blocks=*/1);
  tpu_raiden::rpc::StartTransferRequest request;
  request.set_uuid(445566);
  request.set_is_sender(true);

  absl::Status status =
      manager.RegisterActivePlan(445566, request, /*is_sender=*/true);
  ASSERT_TRUE(status.ok()) << status.ToString();

  status = manager.SetBlockLayouts({FullAttentionLayout(/*slot_bytes=*/128)});
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.message(), testing::HasSubstr("active plans"));
}

TEST(KVCacheManagerTest, RegionAwareChunkValidationAcceptsStridedLiveChunks) {
  TestKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128, /*host_blocks=*/2);
  ASSERT_TRUE(
      manager.SetBlockLayouts({FullAttentionLayout(/*slot_bytes=*/128)}).ok());
  manager.SetBlockChunkRegionValidation(
      transport::BlockChunkRegionValidationMode::kFail);
  EXPECT_EQ(manager.block_chunk_region_validation_mode(),
            transport::BlockChunkRegionValidationMode::kFail);

  uint8_t* base = manager.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  std::vector<transport::BlockChunk> chunks = {
      {.ptr = base, .size = 32},
      {.ptr = base + 64, .size = 32},
      {.ptr = base + 128 + 64, .size = 32},
  };

  absl::Status status = manager.ValidateBlockChunksInRegions(
      /*layer_idx=*/0, /*shard_idx=*/0, chunks);
  EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(KVCacheManagerTest, RegionAwareChunkValidationRejectsPaddingChunks) {
  TestKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128, /*host_blocks=*/1);
  ASSERT_TRUE(
      manager.SetBlockLayouts({FullAttentionLayout(/*slot_bytes=*/128)}).ok());
  manager.SetBlockChunkRegionValidation(
      transport::BlockChunkRegionValidationMode::kFail);

  uint8_t* base = manager.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  std::vector<transport::BlockChunk> tail_padding = {
      {.ptr = base + 96, .size = 16},
  };
  absl::Status status = manager.ValidateBlockChunksInRegions(
      /*layer_idx=*/0, /*shard_idx=*/0, tail_padding);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.message(), testing::HasSubstr("non-live bytes"));

  std::vector<transport::BlockChunk> crosses_stride_gap = {
      {.ptr = base, .size = 96},
  };
  status = manager.ValidateBlockChunksInRegions(
      /*layer_idx=*/0, /*shard_idx=*/0, crosses_stride_gap);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.message(), testing::HasSubstr("non-live bytes"));
}

TEST(KVCacheManagerTest, H2dFailsWithNegativeOffsets) {
  TestKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128);
  std::vector<int64_t> src_offsets = {-1};  // Negative
  std::vector<int64_t> dst_offsets = {0};
  std::vector<int64_t> sizes = {1};

  auto status = manager.H2d(src_offsets, dst_offsets, sizes);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.status().message(),
              testing::HasSubstr("must be non-negative"));
}

TEST(KVCacheManagerTest, D2hFailsWithCpuOnlyManager) {
  // Use the base class directly to test CPU-only behavior
  KVCacheManagerBase manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128);
  std::vector<int64_t> src_offsets = {0};
  std::vector<int64_t> dst_offsets = {0};
  std::vector<int64_t> sizes = {1};

  auto status = manager.D2h(src_offsets, dst_offsets, sizes);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.status().message(),
              testing::HasSubstr("requires a device-backed"));
}

TEST(KVCacheManagerTest, H2dFailsWithOutOfRangeLayer) {
  TestKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128);
  std::vector<int64_t> src_offsets;
  std::vector<int64_t> dst_offsets;
  std::vector<int64_t> sizes;

  auto status = manager.H2d(src_offsets, dst_offsets, sizes,
                            /*slot_idx=*/std::nullopt,
                            /*layer_idx=*/1, /*shard_idx=*/0);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_THAT(status.status().message(),
              testing::HasSubstr("layer or shard index out of range"));
}

TEST(KVCacheManagerTest, H2hReadExplicitAcceptsParallelism) {
  KVCacheManagerBase manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128);
  std::vector<uint8_t*> ptrs = {nullptr};
  auto status = manager.H2hReadExplicit("127.0.0.1:8080", {0}, {0}, ptrs,
                                        /*parallelism=*/2);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kUnavailable);
  EXPECT_THAT(status.status().message(),
              testing::HasSubstr("Failed to connect to peer"));
}

TEST(KVCacheManagerTest, AsymmetricBlockSizesGetBlockChunks) {
  // 1. Sender (Block size: 256 bytes)
  TestKVCacheManager sender(/*num_layers=*/1, /*num_shards=*/1,
                            /*slice_byte_size=*/256);
  std::vector<uint8_t> sender_buffer(256 * 10, 0);
  std::vector<const uint8_t*> sender_ptrs = {sender_buffer.data()};
  std::vector<size_t> sender_sizes = {sender_buffer.size()};
  sender.SetExternalHostPointers(sender_ptrs, sender_sizes);

  // 2. Receiver (Block size: 512 bytes)
  TestKVCacheManager receiver(/*num_layers=*/1, /*num_shards=*/1,
                              /*slice_byte_size=*/512);
  std::vector<uint8_t> receiver_buffer(512 * 10, 0);
  std::vector<const uint8_t*> receiver_ptrs = {receiver_buffer.data()};
  std::vector<size_t> receiver_sizes = {receiver_buffer.size()};
  receiver.SetExternalHostPointers(receiver_ptrs, receiver_sizes);

  // Set up StartTransferRequest with schedules
  tpu_raiden::rpc::StartTransferRequest request;
  request.set_uuid(112233);
  request.set_is_sender(true);

  // Schedule:
  // Sender (256B block 0) has shard entry:
  //   - Pushes src_block_id=0, src_offset_bytes=64, size_bytes=128
  //   - To dst_shard_idx=0, dst_block_id=0, dst_offset_bytes=192 on receiver
  auto* schedules = request.mutable_shard_push_schedules();
  auto* src_schedule = &(*schedules)[0];  // shard 0
  auto* entry = src_schedule->add_entries();
  entry->set_dst_peer("127.0.0.1:20025");
  entry->set_dst_shard_idx(0);
  entry->set_dst_offset_bytes(192);
  entry->set_src_offset_bytes(64);
  entry->set_size_bytes(128);
  entry->set_src_block_id(0);
  entry->set_dst_block_id(0);

  // Register active plan on both sides
  absl::Status status =
      sender.RegisterActivePlan(112233, request, /*is_sender=*/true);
  ASSERT_TRUE(status.ok()) << status.ToString();

  // Receiver schedule should be the same request but marked as is_sender=false
  request.set_is_sender(false);
  status = receiver.RegisterActivePlan(112233, request, /*is_sender=*/false);
  ASSERT_TRUE(status.ok()) << status.ToString();

  // 3. Resolve chunks on Sender (should return offset 64 from block 0 base)
  std::vector<int64_t> src_block_ids = {0};
  std::vector<transport::BlockChunk> sender_chunks = sender.GetBlockChunks(
      /*layer_idx=*/0, /*shard_idx=*/0, src_block_ids, /*total_bytes=*/128,
      /*uuid=*/112233, /*sender_node_id=*/-1, /*peer=*/"127.0.0.1:20025");

  ASSERT_EQ(sender_chunks.size(), 1);
  EXPECT_EQ(sender_chunks[0].size, 128);
  // Verify pointer offset
  uint8_t* sender_base =
      sender.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  EXPECT_EQ(sender_chunks[0].ptr, sender_base + 64);

  // 4. Resolve chunks on Receiver (should return offset 192 from block 0 base)
  std::vector<int64_t> dst_block_ids = {0};
  std::vector<transport::BlockChunk> receiver_chunks = receiver.GetBlockChunks(
      /*layer_idx=*/0, /*shard_idx=*/0, dst_block_ids, /*total_bytes=*/128,
      /*uuid=*/112233, /*sender_node_id=*/0, /*peer=*/"127.0.0.1:20025");

  ASSERT_EQ(receiver_chunks.size(), 1);
  EXPECT_EQ(receiver_chunks[0].size, 128);
  // Verify pointer offset
  uint8_t* receiver_base =
      receiver.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  EXPECT_EQ(receiver_chunks[0].ptr, receiver_base + 192);
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
