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

// Deviceless executor unit tier (X1/X4): the ValidatePoolReshardPlan
// accept/reject table, the device-only rejection at the public entry points,
// and the tag-neutral skip summary. The executor byte path and the pool
// receive lifecycle are device-only by design and are exercised on real chips
// by the D-series harness.

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tpu_raiden/core/kv_cache_manager_with_transfer.h"
#include "tpu_raiden/kv_cache/pool_layout.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace {

class TestManager : public KVCacheManagerWithTransfer {
 public:
  explicit TestManager(double timeout_s = 30.0)
      : KVCacheManagerWithTransfer(
            /*num_layers=*/1, /*num_shards=*/1,
            /*slice_byte_size=*/128,
            /*local_port=*/std::nullopt,
            /*host_blocks_to_allocate=*/std::make_optional(4),
            /*parallelism=*/1, /*node_id=*/0,
            /*local_control_port=*/-1, /*max_blocks=*/0, /*num_slots=*/0,
            timeout_s) {}

  using KVCacheManagerWithTransfer::BuildPoolReshardSkipSummary;
  using KVCacheManagerWithTransfer::ValidatePoolReshardPlan;
};

kv_cache::PoolSpec DensePool(std::string tag, int64_t block_stride = 128,
                             std::string dtype_tag = "bf16") {
  return kv_cache::PoolSpec{
      .tag = std::move(tag),
      .storage_index = 0,
      .base_offset_bytes = 0,
      .block_stride_bytes = block_stride,
      .num_blocks = 4,
      .regions = {kv_cache::RegionSpec{
          .name = "block",
          .offset_bytes = 0,
          .stride_bytes = block_stride,
          .unit_bytes = block_stride,
          .num_units = 1,
          .units_per_stride = 1,
      }},
      .dtype_tag = std::move(dtype_tag),
  };
}

rpc::StartTransferRequest ValidPlan(
    int64_t uuid, const std::vector<std::string>& dtype_tags = {"bf16"},
    const std::vector<int32_t>& transferred_pools = {0}) {
  rpc::StartTransferRequest plan;
  plan.set_uuid(uuid);
  plan.set_req_id("pool_reshard_req_" + std::to_string(uuid));
  plan.set_dst_mem_type(rpc::MEMORY_TYPE_HBM);
  plan.set_use_block_chunks(true);
  plan.set_expected_pushes_per_pool(1);
  for (int32_t pool_idx : transferred_pools) {
    plan.add_transfer_pool_indices(pool_idx);
  }
  for (const std::string& tag : dtype_tags) {
    plan.add_pool_dtype_tags(tag);
  }
  auto* entry = (*plan.mutable_shard_push_schedules())[0].add_entries();
  entry->set_dst_peer("127.0.0.1:1");
  entry->set_dst_shard_idx(0);
  entry->set_src_block_id(0);
  entry->set_dst_block_id(0);
  entry->set_src_offset_bytes(0);
  entry->set_dst_offset_bytes(0);
  entry->set_size_bytes(16);
  entry->set_src_stride_bytes(0);
  entry->set_dst_stride_bytes(0);
  entry->set_count(1);
  return plan;
}

void ExpectInvalid(const absl::Status& status, const std::string& fragment) {
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument)
      << status.ToString();
  EXPECT_NE(std::string(status.message()).find(fragment), std::string::npos)
      << status.ToString();
}

TEST(PoolReshardValidationTest, AcceptsCanonicalPlanOnExplicitPools) {
  TestManager manager;
  ASSERT_TRUE(manager.RegisterPools({DensePool("fa")}).ok());
  rpc::StartTransferRequest plan = ValidPlan(/*uuid=*/1001);

  EXPECT_TRUE(manager
                  .ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                           /*is_sender=*/false)
                  .ok());
  EXPECT_TRUE(manager
                  .ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                           /*is_sender=*/true)
                  .ok());
}

TEST(PoolReshardValidationTest, AcceptsImplicitPools) {
  // A manager that never registered explicit pools exposes one implicit
  // opaque pool per storage (dtype_tag ""); a legacy whole-manager transfer
  // is expressible against it (N5).
  TestManager manager;
  rpc::StartTransferRequest plan =
      ValidPlan(/*uuid=*/1002, /*dtype_tags=*/{""});

  EXPECT_TRUE(manager
                  .ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                           /*is_sender=*/false)
                  .ok());
}

TEST(PoolReshardValidationTest, HasNoTagPolicy) {
  // Pool selection is request data resolved by the controller. The executor
  // validates geometry and consistency for whatever pool set the plan
  // declares; no tag value is special-cased.
  TestManager manager;
  ASSERT_TRUE(manager.RegisterPools({DensePool("gdn.conv")}).ok());
  rpc::StartTransferRequest plan = ValidPlan(/*uuid=*/1003);

  EXPECT_TRUE(manager
                  .ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                           /*is_sender=*/false)
                  .ok());
}

TEST(PoolReshardValidationTest, DeviceOnlyRejectionAtPublicEntryPoints) {
  TestManager manager;
  ASSERT_TRUE(manager.RegisterPools({DensePool("fa")}).ok());
  rpc::StartTransferRequest plan = ValidPlan(/*uuid=*/1004);

  const absl::Status recv_status =
      manager.PoolReshardRegisterRecv(plan, std::vector<int64_t>{0});
  EXPECT_EQ(recv_status.code(), absl::StatusCode::kFailedPrecondition)
      << recv_status.ToString();
  EXPECT_NE(std::string(recv_status.message()).find("device-attached"),
            std::string::npos);

  const absl::Status push_status =
      manager.PoolReshardPush(plan, std::vector<int64_t>{0});
  EXPECT_EQ(push_status.code(), absl::StatusCode::kFailedPrecondition)
      << push_status.ToString();

  // A refused arm publishes no native accounting.
  EXPECT_EQ(manager.GetPoolReshardSkipSummary(plan.req_id()).status().code(),
            absl::StatusCode::kNotFound);
}

TEST(PoolReshardValidationTest, RejectsMissingIdentityAndPoolFields) {
  TestManager manager;
  ASSERT_TRUE(manager.RegisterPools({DensePool("fa")}).ok());

  rpc::StartTransferRequest plan = ValidPlan(/*uuid=*/1005);
  plan.clear_req_id();
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "req_id");

  plan = ValidPlan(/*uuid=*/0);
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "uuid must be positive");

  plan = ValidPlan(/*uuid=*/1006);
  plan.clear_transfer_pool_indices();
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "transfer_pool_indices");

  plan = ValidPlan(/*uuid=*/1007);
  plan.set_expected_pushes_per_pool(0);
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "expected_pushes_per_pool");

  plan = ValidPlan(/*uuid=*/1008);
  plan.set_use_block_chunks(false);
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "use_block_chunks");
}

TEST(PoolReshardValidationTest, RejectsOutOfRangeDuplicateAndDtypeMismatch) {
  TestManager manager;
  ASSERT_TRUE(manager.RegisterPools({DensePool("fa")}).ok());

  rpc::StartTransferRequest plan =
      ValidPlan(/*uuid=*/1009, /*dtype_tags=*/{"bf16"},
                /*transferred_pools=*/{1});
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "out of range");

  plan = ValidPlan(/*uuid=*/1010, /*dtype_tags=*/{"bf16"},
                   /*transferred_pools=*/{0, 0});
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "duplicate transfer pool index");

  plan = ValidPlan(/*uuid=*/1011, /*dtype_tags=*/{"fp8"});
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "dtype tag mismatch");

  plan = ValidPlan(/*uuid=*/1012, /*dtype_tags=*/{"bf16", "bf16"});
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "one dtype tag per pool");
}

TEST(PoolReshardValidationTest, ChecksEveryPoolSpanAndDestinationZeroCover) {
  TestManager manager;
  ASSERT_TRUE(
      manager.RegisterPools({DensePool("fa", 128), DensePool("fa", 64)}).ok());
  rpc::StartTransferRequest plan =
      ValidPlan(/*uuid=*/1013, /*dtype_tags=*/{"bf16", "bf16"},
                /*transferred_pools=*/{0, 1});
  auto* entry = plan.mutable_shard_push_schedules()->at(0).mutable_entries(0);
  entry->set_dst_offset_bytes(48);
  entry->set_size_bytes(32);
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "destination span exceeds declared pool 1");

  plan = ValidPlan(/*uuid=*/1014, /*dtype_tags=*/{"bf16", "bf16"},
                   /*transferred_pools=*/{0, 1});
  entry = plan.mutable_shard_push_schedules()->at(0).mutable_entries(0);
  entry->set_dst_offset_bytes(16);
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "no transfer entry starting at offset 0");

  plan = ValidPlan(/*uuid=*/1015, /*dtype_tags=*/{"bf16", "bf16"},
                   /*transferred_pools=*/{0, 1});
  plan.mutable_shard_push_schedules()->at(0).clear_entries();
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/false),
                "no entries");
}

TEST(PoolReshardValidationTest, RejectsOverflowingSenderSpan) {
  TestManager manager;
  ASSERT_TRUE(manager.RegisterPools({DensePool("fa")}).ok());
  rpc::StartTransferRequest plan = ValidPlan(/*uuid=*/1016);
  auto* entry = plan.mutable_shard_push_schedules()->at(0).mutable_entries(0);
  entry->set_src_offset_bytes(96);
  entry->set_src_stride_bytes(std::numeric_limits<int64_t>::max());
  entry->set_dst_stride_bytes(1);
  entry->set_size_bytes(16);
  entry->set_count(2);

  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0},
                                                /*is_sender=*/true),
                "source span exceeds declared pool 0");
}

TEST(PoolReshardValidationTest, RejectsBlockIdsOutsideDeclaredPool) {
  TestManager manager;
  ASSERT_TRUE(manager.RegisterPools({DensePool("fa")}).ok());
  rpc::StartTransferRequest plan = ValidPlan(/*uuid=*/1017);

  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{9},
                                                /*is_sender=*/false),
                "out of range for pool");
  ExpectInvalid(
      manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{0, 0},
                                      /*is_sender=*/false),
      "unique");
  ExpectInvalid(manager.ValidatePoolReshardPlan(plan, std::vector<int64_t>{},
                                                /*is_sender=*/false),
                "must not be empty");
}

TEST(PoolReshardSkipSummaryTest, CountsUnselectedPoolsByOpaqueTag) {
  TestManager manager;
  ASSERT_TRUE(manager
                  .RegisterPools({DensePool("fa"), DensePool("fa"),
                                  DensePool("gdn.conv"), DensePool("gdn.ssm")})
                  .ok());
  rpc::StartTransferRequest plan =
      ValidPlan(/*uuid=*/1018, /*dtype_tags=*/{"bf16", "bf16", "bf16", "bf16"},
                /*transferred_pools=*/{0, 1});

  const auto summary = manager.BuildPoolReshardSkipSummary(plan);
  EXPECT_EQ(summary.transferred_pools, 2);
  ASSERT_EQ(summary.skipped_pool_counts.size(), 2u);
  EXPECT_EQ(summary.skipped_pool_counts.at("gdn.conv"), 1);
  EXPECT_EQ(summary.skipped_pool_counts.at("gdn.ssm"), 1);
}

}  // namespace
}  // namespace tpu_raiden
