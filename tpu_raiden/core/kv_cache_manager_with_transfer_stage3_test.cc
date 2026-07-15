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

#include "tpu_raiden/core/kv_cache_manager_with_transfer.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
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

  using KVCacheManagerWithTransfer::OnPoolReceived;
};

kv_cache::PoolSpec DensePool(std::string tag, int64_t block_stride = 128) {
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
      .dtype_tag = "bf16",
  };
}

rpc::StartTransferRequest ValidPlan(
    int64_t uuid, int pool_count = 1,
    const std::vector<int32_t>& transferred_pools = {0}) {
  rpc::StartTransferRequest plan;
  plan.set_uuid(uuid);
  plan.set_req_id("stage3_req_" + std::to_string(uuid));
  plan.set_dst_mem_type(rpc::MEMORY_TYPE_HBM);
  plan.set_use_block_chunks(true);
  plan.set_expected_pushes_per_pool(1);
  for (int32_t pool_idx : transferred_pools) {
    plan.add_transfer_pool_indices(pool_idx);
  }
  for (int i = 0; i < pool_count; ++i) {
    plan.add_pool_dtype_tags("bf16");
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

TEST(KVCacheManagerWithTransferStage3Test, ValidateAcceptsCanonicalFaPlan) {
  TestManager manager;
  ASSERT_TRUE(manager.RegisterPools({DensePool("fa")}).ok());
  rpc::StartTransferRequest plan = ValidPlan(/*uuid=*/1001);

  ASSERT_TRUE(manager.ReshardRegisterRecv(plan, std::vector<int64_t>{0}).ok());
  EXPECT_TRUE(manager.OnPoolReceived(/*pool_idx=*/0, plan.uuid()).ok());
  auto [done_sending, done_recving, failed_recving] = manager.CompleteReadRaw();
  EXPECT_TRUE(done_sending.empty());
  EXPECT_EQ(done_recving, std::vector<std::string>({plan.req_id()}));
  EXPECT_TRUE(failed_recving.empty());
}

TEST(KVCacheManagerWithTransferStage3Test, ValidateRejectsImplicitPools) {
  TestManager manager;
  const absl::Status status = manager.ReshardRegisterRecv(
      ValidPlan(/*uuid=*/1002), std::vector<int64_t>{0});
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_NE(std::string(status.message()).find("explicitly registered pools"),
            std::string::npos);
}

TEST(KVCacheManagerWithTransferStage3Test,
     ValidateRejectsMissingIdentityAndPoolFields) {
  TestManager manager;
  ASSERT_TRUE(manager.RegisterPools({DensePool("fa")}).ok());

  rpc::StartTransferRequest plan = ValidPlan(/*uuid=*/1003);
  plan.clear_req_id();
  ExpectInvalid(manager.ReshardRegisterRecv(plan, std::vector<int64_t>{0}),
                "req_id");

  plan = ValidPlan(/*uuid=*/0);
  ExpectInvalid(manager.ReshardRegisterRecv(plan, std::vector<int64_t>{0}),
                "uuid must be positive");

  plan = ValidPlan(/*uuid=*/1004);
  plan.clear_transfer_pool_indices();
  ExpectInvalid(manager.ReshardRegisterRecv(plan, std::vector<int64_t>{0}),
                "transfer_pool_indices");
}

TEST(KVCacheManagerWithTransferStage3Test,
     ValidateRejectsUndeclaredAndNonFaPools) {
  {
    TestManager manager;
    ASSERT_TRUE(manager.RegisterPools({DensePool("fa")}).ok());
    rpc::StartTransferRequest plan = ValidPlan(
        /*uuid=*/1005, /*pool_count=*/1, /*transferred_pools=*/{1});
    ExpectInvalid(manager.ReshardRegisterRecv(plan, std::vector<int64_t>{0}),
                  "out of range");
  }
  {
    TestManager manager;
    ASSERT_TRUE(manager.RegisterPools({DensePool("gdn.conv")}).ok());
    rpc::StartTransferRequest plan = ValidPlan(/*uuid=*/1006);
    ExpectInvalid(manager.ReshardRegisterRecv(plan, std::vector<int64_t>{0}),
                  "non-FA pool");
  }
}

TEST(KVCacheManagerWithTransferStage3Test,
     ValidateRejectsIncompleteFaCoverageAndZeroEntries) {
  TestManager manager;
  ASSERT_TRUE(
      manager.RegisterPools({DensePool("fa"), DensePool("fa")}).ok());
  rpc::StartTransferRequest plan =
      ValidPlan(/*uuid=*/1007, /*pool_count=*/2,
                /*transferred_pools=*/{0});
  ExpectInvalid(manager.ReshardRegisterRecv(plan, std::vector<int64_t>{0}),
                "every registered FA pool");

  plan = ValidPlan(/*uuid=*/1008, /*pool_count=*/2,
                   /*transferred_pools=*/{0, 1});
  plan.mutable_shard_push_schedules()->at(0).clear_entries();
  ExpectInvalid(manager.ReshardRegisterRecv(plan, std::vector<int64_t>{0}),
                "no entries");
}

TEST(KVCacheManagerWithTransferStage3Test,
     ValidateChecksEveryPoolSpanAndDestinationOffsetZeroCover) {
  TestManager manager;
  ASSERT_TRUE(
      manager.RegisterPools({DensePool("fa", 128), DensePool("fa", 64)})
          .ok());
  rpc::StartTransferRequest plan =
      ValidPlan(/*uuid=*/1009, /*pool_count=*/2,
                /*transferred_pools=*/{0, 1});
  auto* entry =
      plan.mutable_shard_push_schedules()->at(0).mutable_entries(0);
  entry->set_dst_offset_bytes(48);
  entry->set_size_bytes(32);
  ExpectInvalid(manager.ReshardRegisterRecv(plan, std::vector<int64_t>{0}),
                "destination span exceeds declared pool 1");

  plan = ValidPlan(/*uuid=*/1010, /*pool_count=*/2,
                   /*transferred_pools=*/{0, 1});
  entry = plan.mutable_shard_push_schedules()->at(0).mutable_entries(0);
  entry->set_dst_offset_bytes(16);
  ExpectInvalid(manager.ReshardRegisterRecv(plan, std::vector<int64_t>{0}),
                "no transfer entry starting at offset 0");
}

TEST(KVCacheManagerWithTransferStage3Test,
     ValidateRejectsOverflowingSenderSpanBeforeRegistration) {
  TestManager manager;
  ASSERT_TRUE(manager.RegisterPools({DensePool("fa")}).ok());
  rpc::StartTransferRequest plan = ValidPlan(/*uuid=*/1011);
  auto* entry =
      plan.mutable_shard_push_schedules()->at(0).mutable_entries(0);
  entry->set_src_offset_bytes(96);
  entry->set_src_stride_bytes(std::numeric_limits<int64_t>::max());
  entry->set_dst_stride_bytes(1);
  entry->set_size_bytes(16);
  entry->set_count(2);

  ExpectInvalid(manager.ReshardPush(plan, std::vector<int64_t>{0}),
                "source span exceeds declared pool 0");
  EXPECT_EQ(manager.GetReshardSkipSummary(plan.req_id()).status().code(),
            absl::StatusCode::kNotFound)
      << "rejected plans must not publish native accounting";
}

TEST(KVCacheManagerWithTransferStage3Test,
     PoolLifecycleRejectsUndeclaredAndDuplicateAndWaitsForAllPools) {
  constexpr int64_t kUuid = 1012;
  TestManager manager;
  ASSERT_TRUE(
      manager.RegisterPools({DensePool("fa"), DensePool("fa")}).ok());
  rpc::StartTransferRequest plan =
      ValidPlan(kUuid, /*pool_count=*/2, /*transferred_pools=*/{0, 1});
  ASSERT_TRUE(manager.ReshardRegisterRecv(plan, std::vector<int64_t>{0}).ok());

  EXPECT_EQ(manager.OnPoolReceived(/*pool_idx=*/2, kUuid).code(),
            absl::StatusCode::kInvalidArgument);
  ASSERT_TRUE(manager.OnPoolReceived(/*pool_idx=*/0, kUuid).ok());
  EXPECT_EQ(manager.OnPoolReceived(/*pool_idx=*/0, kUuid).code(),
            absl::StatusCode::kAlreadyExists);

  auto [partial_send, partial_recv, partial_failed] =
      manager.CompleteReadRaw();
  EXPECT_TRUE(partial_send.empty());
  EXPECT_TRUE(partial_recv.empty());
  EXPECT_TRUE(partial_failed.empty());

  ASSERT_TRUE(manager.OnPoolReceived(/*pool_idx=*/1, kUuid).ok());
  EXPECT_EQ(manager.OnPoolReceived(/*pool_idx=*/1, kUuid).code(),
            absl::StatusCode::kAlreadyExists);
  auto [done_send, done_recv, failed_recv] = manager.CompleteReadRaw();
  EXPECT_TRUE(done_send.empty());
  EXPECT_EQ(done_recv, std::vector<std::string>({plan.req_id()}));
  EXPECT_TRUE(failed_recv.empty());

  // Normal completion unregisters both manager and transport state. Polling
  // retires the completed RecvEntry, so a fresh generation can reuse UUID.
  plan.set_req_id("stage3_reused_after_completion");
  ASSERT_TRUE(manager.ReshardRegisterRecv(plan, std::vector<int64_t>{0}).ok());
  ASSERT_TRUE(manager.OnPoolReceived(/*pool_idx=*/0, kUuid).ok());
  ASSERT_TRUE(manager.OnPoolReceived(/*pool_idx=*/1, kUuid).ok());
  (void)manager.CompleteReadRaw();
}

TEST(KVCacheManagerWithTransferStage3Test,
     TimeoutSurfacesFailureUnregistersPlanAndAllowsUuidReuse) {
  constexpr int64_t kUuid = 1013;
  TestManager manager(/*timeout_s=*/0.0);
  ASSERT_TRUE(manager.RegisterPools({DensePool("fa")}).ok());
  rpc::StartTransferRequest plan = ValidPlan(kUuid);
  ASSERT_TRUE(manager.ReshardRegisterRecv(plan, std::vector<int64_t>{0}).ok());

  auto [done_send, done_recv, failed_recv] = manager.CompleteReadRaw();
  EXPECT_TRUE(done_send.empty());
  EXPECT_TRUE(done_recv.empty());
  EXPECT_EQ(failed_recv, std::vector<std::string>({plan.req_id()}));

  plan.set_req_id("stage3_reused_after_timeout");
  ASSERT_TRUE(manager.ReshardRegisterRecv(plan, std::vector<int64_t>{0}).ok());
  ASSERT_TRUE(manager.OnPoolReceived(/*pool_idx=*/0, kUuid).ok());
  (void)manager.CompleteReadRaw();
}

TEST(KVCacheManagerWithTransferStage3Test,
     NativeSkipSummaryExistsOnlyAfterSuccessfulRegistration) {
  TestManager manager;
  ASSERT_TRUE(manager
                  .RegisterPools({DensePool("fa"), DensePool("fa"),
                                  DensePool("gdn.conv"),
                                  DensePool("gdn.ssm")})
                  .ok());
  rpc::StartTransferRequest plan =
      ValidPlan(/*uuid=*/1014, /*pool_count=*/4,
                /*transferred_pools=*/{0, 1});

  EXPECT_EQ(manager.GetReshardSkipSummary(plan.req_id()).status().code(),
            absl::StatusCode::kNotFound);
  ASSERT_TRUE(manager.ReshardRegisterRecv(plan, std::vector<int64_t>{0}).ok());
  auto summary = manager.GetReshardSkipSummary(plan.req_id());
  ASSERT_TRUE(summary.ok()) << summary.status();
  EXPECT_EQ(summary->transferred_fa_pools, 2);
  EXPECT_EQ(summary->skipped_gdn_conv_pools, 1);
  EXPECT_EQ(summary->skipped_gdn_ssm_pools, 1);
  EXPECT_EQ(summary->reason, "stage3_fa_only");

  ASSERT_TRUE(manager.OnPoolReceived(/*pool_idx=*/0, plan.uuid()).ok());
  ASSERT_TRUE(manager.OnPoolReceived(/*pool_idx=*/1, plan.uuid()).ok());
  (void)manager.CompleteReadRaw();
}

TEST(KVCacheManagerWithTransferStage3Test,
     ReshardPushRunsPoolTransportAndPublishesNativeSummary) {
  constexpr int64_t kUuid = 1015;
  TestManager sender;
  TestManager receiver;
  ASSERT_TRUE(sender.RegisterPools({DensePool("fa")}).ok());
  ASSERT_TRUE(receiver.RegisterPools({DensePool("fa")}).ok());

  auto src_ref = sender.GetPoolBlockRef(/*pool_idx=*/0, /*shard_idx=*/0,
                                        /*block_id=*/0);
  auto dst_ref = receiver.GetPoolBlockRef(/*pool_idx=*/0, /*shard_idx=*/0,
                                          /*block_id=*/0);
  ASSERT_TRUE(src_ref.ok()) << src_ref.status();
  ASSERT_TRUE(dst_ref.ok()) << dst_ref.status();
  std::memset(src_ref->ptr, 0xA5, src_ref->block_stride_bytes);
  std::memset(dst_ref->ptr, 0x00, dst_ref->block_stride_bytes);

  rpc::StartTransferRequest plan = ValidPlan(kUuid);
  ASSERT_TRUE(receiver.local_port().has_value());
  plan.mutable_shard_push_schedules()->at(0).mutable_entries(0)->set_dst_peer(
      "localhost:" + std::to_string(*receiver.local_port()));
  ASSERT_TRUE(
      receiver.ReshardRegisterRecv(plan, std::vector<int64_t>{0}).ok());
  ASSERT_TRUE(sender.ReshardPush(plan, std::vector<int64_t>{0},
                                 /*parallelism=*/1)
                  .ok());

  auto sender_summary = sender.GetReshardSkipSummary(plan.req_id());
  ASSERT_TRUE(sender_summary.ok()) << sender_summary.status();
  EXPECT_EQ(sender_summary->transferred_fa_pools, 1);

  bool send_complete = false;
  bool recv_complete = false;
  for (int attempt = 0; attempt < 200 &&
                        (!send_complete || !recv_complete);
       ++attempt) {
    auto [done_sending, ignored_recv, ignored_failed] =
        sender.CompleteReadRaw();
    send_complete = send_complete || !done_sending.empty();
    auto [ignored_send, done_recving, failed_recving] =
        receiver.CompleteReadRaw();
    ASSERT_TRUE(failed_recving.empty());
    recv_complete = recv_complete || !done_recving.empty();
    if (!send_complete || !recv_complete) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  EXPECT_TRUE(send_complete);
  EXPECT_TRUE(recv_complete);
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(dst_ref->ptr[i], 0xA5);
  }
  for (int i = 16; i < dst_ref->block_stride_bytes; ++i) {
    EXPECT_EQ(dst_ref->ptr[i], 0x00);
  }
}

}  // namespace
}  // namespace tpu_raiden
