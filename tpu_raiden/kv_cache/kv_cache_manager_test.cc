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
            num_layers, num_shards, slice_byte_size,
            /*local_port=*/std::nullopt,
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

// Pool with one strided live region per block: live [0, 32) and [64, 96)
// within each block, everything else non-live.
PoolSpec StridedPool(const std::string& tag, size_t storage_index,
                     int64_t base_offset, int64_t stride, int64_t num_blocks) {
  return PoolSpec{
      .tag = tag,
      .storage_index = storage_index,
      .base_offset_bytes = base_offset,
      .block_stride_bytes = stride,
      .num_blocks = num_blocks,
      .regions = {RegionSpec{
          .name = "payload",
          .offset_bytes = 0,
          .stride_bytes = 64,
          .unit_bytes = 32,
          .num_units = 2,
          .units_per_stride = 1,
      }},
      .dtype_tag = "dtype_a",
  };
}

// Pool whose single region covers every byte of the block.
PoolSpec DensePool(const std::string& tag, size_t storage_index,
                   int64_t base_offset, int64_t stride, int64_t num_blocks) {
  return PoolSpec{
      .tag = tag,
      .storage_index = storage_index,
      .base_offset_bytes = base_offset,
      .block_stride_bytes = stride,
      .num_blocks = num_blocks,
      .regions = {RegionSpec{
          .name = "block",
          .offset_bytes = 0,
          .stride_bytes = stride,
          .unit_bytes = stride,
          .num_units = 1,
          .units_per_stride = 1,
      }},
      .dtype_tag = "dtype_a",
  };
}

TEST(KVCacheManagerTest, CompilesAndLinksSuccessfully) {
  EXPECT_TRUE(true);
}

TEST(KVCacheManagerTest, RegisterPoolsValidatesAgainstStorage) {
  TestKVCacheManager manager(/*num_layers=*/2, /*num_shards=*/1,
                             /*slice_byte_size=*/128, /*host_blocks=*/2);

  // Overlapping pools on one storage are allowed (aliased-raw pattern).
  absl::Status status = manager.RegisterPools({
      DensePool("kind_a", 0, 0, 128, 2),
      StridedPool("kind_b", 0, 0, 128, 2),
      DensePool("kind_a", 1, 64, 64, 3),
  });
  ASSERT_TRUE(status.ok()) << status.ToString();
  EXPECT_TRUE(manager.has_explicit_pools());
  EXPECT_EQ(manager.num_pools(), 3);

  // storage_index out of range.
  status = manager.RegisterPools({DensePool("kind_a", 5, 0, 128, 2)});
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), testing::HasSubstr("storage_index"));

  // Pool exceeding the storage: 2 blocks x 128 + base 64 > 256.
  status = manager.RegisterPools({DensePool("kind_a", 0, 64, 128, 2)});
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), testing::HasSubstr("exceeds storage bytes"));

  // The device physical size takes precedence over the host mirror size.
  TestKVCacheManager device_backed(/*num_layers=*/1, /*num_shards=*/1,
                                   /*slice_byte_size=*/256, /*host_blocks=*/2);
  device_backed.SetLayerPhysicalSizeForTest(/*layer_idx=*/0,
                                            /*physical_size=*/128,
                                            /*major_dim_size=*/1);
  EXPECT_TRUE(
      device_backed.RegisterPools({DensePool("kind_a", 0, 0, 128, 1)}).ok());
  status = device_backed.RegisterPools({DensePool("kind_a", 0, 0, 128, 2)});
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), testing::HasSubstr("exceeds storage bytes"));
}

// Pool block pointer math with interior base offsets and per-pool strides.
TEST(KVCacheManagerTest, GetPoolBlockRefPointerMath) {
  TestKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128, /*host_blocks=*/4);
  ASSERT_TRUE(manager
                  .RegisterPools({
                      DensePool("kind_a", 0, 0, 128, 2),
                      DensePool("kind_b", 0, 256, 64, 4),
                  })
                  .ok());

  uint8_t* base = manager.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);

  auto ref = manager.GetPoolBlockRef(/*pool_idx=*/0, /*shard_idx=*/0,
                                     /*block_id=*/1);
  ASSERT_TRUE(ref.ok()) << ref.status().ToString();
  EXPECT_EQ(ref->ptr, base + 128);
  EXPECT_EQ(ref->block_stride_bytes, 128);
  EXPECT_EQ(ref->pool->tag, "kind_a");
  EXPECT_EQ(ref->pool_idx, 0);
  EXPECT_EQ(ref->block_id, 1);

  ref = manager.GetPoolBlockRef(/*pool_idx=*/1, /*shard_idx=*/0,
                                /*block_id=*/0);
  ASSERT_TRUE(ref.ok()) << ref.status().ToString();
  EXPECT_EQ(ref->ptr, base + 256);
  ref = manager.GetPoolBlockRef(/*pool_idx=*/1, /*shard_idx=*/0,
                                /*block_id=*/3);
  ASSERT_TRUE(ref.ok()) << ref.status().ToString();
  EXPECT_EQ(ref->ptr, base + 256 + 3 * 64);

  EXPECT_EQ(manager.GetPoolBlockRef(0, 0, 2).status().code(),
            absl::StatusCode::kOutOfRange);
  EXPECT_EQ(manager.GetPoolBlockRef(1, 0, 4).status().code(),
            absl::StatusCode::kOutOfRange);
  EXPECT_EQ(manager.GetPoolBlockRef(2, 0, 0).status().code(),
            absl::StatusCode::kOutOfRange);
  EXPECT_EQ(manager.GetPoolBlockRef(0, 1, 0).status().code(),
            absl::StatusCode::kOutOfRange);
}

TEST(KVCacheManagerTest, RegisterPoolsGrowsDeviceBackedHostMirror) {
  TestKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/64, /*host_blocks=*/1);
  uint8_t* original = manager.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  ASSERT_NE(original, nullptr);
  for (size_t i = 0; i < 64; ++i) {
    original[i] = static_cast<uint8_t>(i);
  }

  manager.SetLayerPhysicalSizeForTest(/*layer_idx=*/0,
                                      /*physical_size=*/256,
                                      /*major_dim_size=*/1);
  absl::Status status =
      manager.RegisterPools({DensePool("kind_a", 0, 0, 64, 4)});
  ASSERT_TRUE(status.ok()) << status.ToString();
  EXPECT_GE(manager.GetHostSize(/*layer_idx=*/0, /*shard_idx=*/0), 256);

  uint8_t* grown = manager.GetHostPointer(/*layer_idx=*/0, /*shard_idx=*/0);
  ASSERT_NE(grown, nullptr);
  for (size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(grown[i], static_cast<uint8_t>(i));
  }
  auto last_ref = manager.GetPoolBlockRef(/*pool_idx=*/0, /*shard_idx=*/0,
                                          /*block_id=*/3);
  ASSERT_TRUE(last_ref.ok()) << last_ref.status().ToString();
  EXPECT_EQ(last_ref->ptr, grown + 3 * 64);
}

// The pool table is frozen while plans are active.
TEST(KVCacheManagerTest, RegisterPoolsFailsAfterActivePlanRegistered) {
  TestKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128, /*host_blocks=*/1);
  tpu_raiden::rpc::StartTransferRequest request;
  request.set_uuid(445566);
  request.set_is_sender(true);

  absl::Status status =
      manager.RegisterActivePlan(445566, request, /*is_sender=*/true);
  ASSERT_TRUE(status.ok()) << status.ToString();

  status = manager.RegisterPools({DensePool("kind_a", 0, 0, 128, 1)});
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.message(), testing::HasSubstr("active plans"));
}

TEST(KVCacheManagerTest, PoolIndicesWithTag) {
  TestKVCacheManager manager(/*num_layers=*/2, /*num_shards=*/1,
                             /*slice_byte_size=*/128, /*host_blocks=*/2);
  PoolSpec pool_b = DensePool("kind_b", 1, 0, 128, 2);
  pool_b.dtype_tag = "dtype_b";
  ASSERT_TRUE(manager
                  .RegisterPools({
                      DensePool("kind_a", 0, 0, 128, 2),
                      pool_b,
                      StridedPool("kind_a", 1, 0, 128, 2),
                  })
                  .ok());

  EXPECT_THAT(manager.PoolIndicesWithTag("kind_a"), testing::ElementsAre(0, 2));
  EXPECT_THAT(manager.PoolIndicesWithTag("kind_b"), testing::ElementsAre(1));
  EXPECT_THAT(manager.PoolIndicesWithTag("missing"), testing::IsEmpty());
}

// Without RegisterPools the manager exposes one implicit Opaque pool per
// storage and reports no explicit pools.
TEST(KVCacheManagerTest, ImplicitPoolsMirrorStorages) {
  TestKVCacheManager manager(/*num_layers=*/2, /*num_shards=*/1,
                             /*slice_byte_size=*/128, /*host_blocks=*/2);
  EXPECT_FALSE(manager.has_explicit_pools());
  EXPECT_EQ(manager.num_pools(), 2);

  const PoolSpec* pool = manager.pool(0);
  ASSERT_NE(pool, nullptr);
  EXPECT_EQ(pool->tag, "opaque");
  EXPECT_EQ(pool->storage_index, 0);
  EXPECT_EQ(pool->base_offset_bytes, 0);
  EXPECT_EQ(pool->block_stride_bytes, 128);
  EXPECT_EQ(pool->num_blocks, 2);

  auto ref = manager.GetPoolBlockRef(/*pool_idx=*/1, /*shard_idx=*/0,
                                     /*block_id=*/1);
  ASSERT_TRUE(ref.ok()) << ref.status().ToString();
  EXPECT_EQ(ref->ptr, manager.GetHostPointer(/*layer_idx=*/1,
                                             /*shard_idx=*/0) +
                          128);
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
