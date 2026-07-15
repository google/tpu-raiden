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

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/test.h"
// clang-format off
#include "tpu_raiden/core/kv_cache_manager_with_transfer.h"
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/frameworks/jax/kv_cache_manager.h"
#include "tpu_raiden/core/controller/controller_service.h"
#include "tpu_raiden/core/controller/test_util.h"
#include "tpu_raiden/core/controller/raiden_controller.h"
// clang-format on
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {
namespace jax {
namespace {

struct StartReadCall {
  std::string req_id;
  uint64_t uuid;
  std::string remote_endpoint;
  std::vector<int64_t> remote_block_ids;
  std::vector<int64_t> local_block_ids;
  std::optional<std::vector<int64_t>> local_host_block_ids;
};

class MockSubManager : public KVCacheManagerWithTransfer {
 public:
  MockSubManager()
      : KVCacheManagerWithTransfer({}, std::nullopt, std::nullopt, false, 1,
                                   nullptr) {}

  std::vector<std::string> sending, recving, failed;
  std::string started_ep;
  std::vector<int64_t> started_remote_blocks, started_local_blocks;
  std::vector<StartReadCall> start_read_calls;

  int d2h_calls = 0;
  int h2d_calls = 0;
  std::vector<int64_t> last_d2h_src_offsets;
  std::vector<int64_t> last_d2h_dst_offsets;
  std::vector<int64_t> last_d2h_copy_sizes;
  std::vector<int64_t> last_h2d_src_offsets;
  std::vector<int64_t> last_h2d_dst_offsets;
  std::vector<int64_t> last_h2d_copy_sizes;

  void StartRead(const std::string& req_id, uint64_t uuid,
                 const std::string& remote_endpoint,
                 const std::vector<int64_t>& remote_block_ids,
                 const std::vector<int64_t>& local_block_ids,
                 int parallelism = 1,
                 std::optional<std::vector<int64_t>> local_host_block_ids =
                     std::nullopt) override {
    started_ep = remote_endpoint;
    started_remote_blocks = remote_block_ids;
    started_local_blocks = local_block_ids;
    start_read_calls.push_back({req_id, uuid, remote_endpoint, remote_block_ids,
                                local_block_ids, local_host_block_ids});
  }

  std::tuple<std::vector<std::string>, std::vector<std::string>,
             std::vector<std::string>>
  CompleteReadRaw() override {
    auto res = std::make_tuple(sending, recving, failed);
    sending.clear();
    recving.clear();
    failed.clear();
    return res;
  }

  absl::StatusOr<raiden::PjRtCopyFuture> D2h(
      const std::vector<int64_t>& src_offsets_major_dim = {},
      const std::vector<int64_t>& dst_offsets_major_dim = {},
      const std::vector<int64_t>& copy_sizes_major_dim = {},
      std::optional<int64_t> slot_idx = std::nullopt,
      std::optional<size_t> layer_idx = std::nullopt,
      std::optional<size_t> shard_idx = std::nullopt) override {
    d2h_calls++;
    last_d2h_src_offsets = src_offsets_major_dim;
    last_d2h_dst_offsets = dst_offsets_major_dim;
    last_d2h_copy_sizes = copy_sizes_major_dim;
    return raiden::PjRtCopyFuture();
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2d(
      const std::vector<int64_t>& src_offsets_major_dim = {},
      const std::vector<int64_t>& dst_offsets_major_dim = {},
      const std::vector<int64_t>& copy_sizes_major_dim = {},
      std::optional<int64_t> slot_idx = std::nullopt,
      std::optional<size_t> layer_idx = std::nullopt,
      std::optional<size_t> shard_idx = std::nullopt) override {
    h2d_calls++;
    last_h2d_src_offsets = src_offsets_major_dim;
    last_h2d_dst_offsets = dst_offsets_major_dim;
    last_h2d_copy_sizes = copy_sizes_major_dim;
    return raiden::PjRtCopyFuture();
  }
};

TEST(KVCacheManagerWrapperTest,
     VotingConsensusPromotesDoneRecvingOnlyWhenAllAgree) {
  auto sub0 = std::make_unique<MockSubManager>();
  auto sub1 = std::make_unique<MockSubManager>();
  MockSubManager* ptr0 = sub0.get();
  MockSubManager* ptr1 = sub1.get();

  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs;
  subs.push_back(std::move(sub0));
  subs.push_back(std::move(sub1));

  KVCacheManager mgr(std::move(subs));

  // 1. Only sub0 finishes recving for req_123
  ptr0->recving.push_back("req_123");
  auto [s1, r1, f1] = mgr.CompleteReadRaw();
  EXPECT_TRUE(s1.empty());
  EXPECT_TRUE(r1.empty());
  EXPECT_TRUE(f1.empty());

  // 2. Now sub1 finishes recving for req_123
  ptr1->recving.push_back("req_123");
  auto [s2, r2, f2] = mgr.CompleteReadRaw();
  EXPECT_TRUE(s2.empty());
  EXPECT_EQ(r2, std::vector<std::string>{"req_123"});
  EXPECT_TRUE(f2.empty());
}

TEST(KVCacheManagerWrapperTest,
     AnyFailureImmediatelyTriggersFailedRecvingAndSuppressesFutureAcks) {
  auto sub0 = std::make_unique<MockSubManager>();
  auto sub1 = std::make_unique<MockSubManager>();
  MockSubManager* ptr0 = sub0.get();
  MockSubManager* ptr1 = sub1.get();

  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs;
  subs.push_back(std::move(sub0));
  subs.push_back(std::move(sub1));

  KVCacheManager mgr(std::move(subs));

  // 1. sub0 reports failure for req_fail
  ptr0->failed.push_back("req_fail");
  auto [s1, r1, f1] = mgr.CompleteReadRaw();
  EXPECT_TRUE(s1.empty());
  EXPECT_TRUE(r1.empty());
  EXPECT_EQ(f1, std::vector<std::string>{"req_fail"});

  // 2. Later sub1 finishes recving for req_fail (should be suppressed)
  ptr1->recving.push_back("req_fail");
  auto [s2, r2, f2] = mgr.CompleteReadRaw();
  EXPECT_TRUE(s2.empty());
  EXPECT_TRUE(r2.empty());
  EXPECT_TRUE(f2.empty());
}

TEST(KVCacheManagerWrapperTest,
     VotingConsensusPromotesDoneSendingOnlyWhenAllAgree) {
  auto sub0 = std::make_unique<MockSubManager>();
  auto sub1 = std::make_unique<MockSubManager>();
  MockSubManager* ptr0 = sub0.get();
  MockSubManager* ptr1 = sub1.get();

  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs;
  subs.push_back(std::move(sub0));
  subs.push_back(std::move(sub1));

  KVCacheManager mgr(std::move(subs));

  ptr0->sending.push_back("req_send");
  EXPECT_TRUE(std::get<0>(mgr.CompleteReadRaw()).empty());

  ptr1->sending.push_back("req_send");
  EXPECT_EQ(std::get<0>(mgr.CompleteReadRaw()),
            std::vector<std::string>{"req_send"});
}

TEST(KVCacheManagerWrapperTest,
     StartReadMatchesRemoteEndpointsByShardIntersection) {
  auto sub0 = std::make_unique<MockSubManager>();
  auto sub1 = std::make_unique<MockSubManager>();
  MockSubManager* ptr0 = sub0.get();
  MockSubManager* ptr1 = sub1.get();

  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs;
  subs.push_back(std::move(sub0));
  subs.push_back(std::move(sub1));

  KVCacheManager mgr(std::move(subs));

  // Local sub0 holds shards [4, 5, 6, 7], sub1 holds [0, 1, 2, 3]
  mgr.SetSubmanagerShardsForTesting({{4, 5, 6, 7}, {0, 1, 2, 3}});

  std::vector<EndpointDescriptor> remote_descs = {
      {"10.0.0.1:45000", {0, 1, 2, 3}}, {"10.0.0.2:45000", {4, 5, 6, 7}}};

  std::vector<int64_t> remote_blocks = {10, 20};
  std::vector<int64_t> local_blocks = {30, 40};

  mgr.StartRead("req_test", 999, remote_descs, remote_blocks, local_blocks);

  EXPECT_EQ(ptr0->started_ep, "10.0.0.2:45000");
  EXPECT_EQ(ptr0->started_remote_blocks, remote_blocks);
  EXPECT_EQ(ptr0->started_local_blocks, local_blocks);

  EXPECT_EQ(ptr1->started_ep, "10.0.0.1:45000");
  EXPECT_EQ(ptr1->started_remote_blocks, remote_blocks);
  EXPECT_EQ(ptr1->started_local_blocks, local_blocks);
}

TEST(KVCacheManagerWrapperTest, StartReadUnifiedMultiEndpointMultiSubManager) {
  auto sub0 = std::make_unique<MockSubManager>();
  auto sub1 = std::make_unique<MockSubManager>();
  MockSubManager* ptr0 = sub0.get();
  MockSubManager* ptr1 = sub1.get();

  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs;
  subs.push_back(std::move(sub0));
  subs.push_back(std::move(sub1));

  KVCacheManager mgr(std::move(subs));
  // Local sub0 holds shards [0, 1], sub1 holds [2, 3]
  mgr.SetSubmanagerShardsForTesting({{0, 1}, {2, 3}});

  std::vector<EndpointDescriptor> remote_descs = {{"10.0.0.1:45000", {0, 1}},
                                                  {"10.0.0.2:45000", {0, 1}},
                                                  {"10.0.0.3:45000", {2, 3}},
                                                  {"10.0.0.4:45000", {2, 3}}};

  std::vector<int64_t> remote_blocks = {10, 20, 30, 40, 50};
  std::vector<int64_t> local_blocks = {100, 200, 300, 400, 500};
  std::vector<int64_t> host_blocks = {1000, 2000, 3000, 4000, 5000};
  uint64_t uuid = 1000;

  mgr.StartRead("req_unified", uuid, remote_descs, remote_blocks, local_blocks,
                1, host_blocks);

  // In unified architecture, full block lists are passed to each submanager
  // with original req_id (no _ep0/_ep1 slicing)
  ASSERT_EQ(ptr0->start_read_calls.size(), 1);
  const auto& call0 = ptr0->start_read_calls[0];
  EXPECT_EQ(call0.req_id, "req_unified");
  EXPECT_EQ(call0.uuid, uuid);
  EXPECT_EQ(call0.remote_block_ids, remote_blocks);
  EXPECT_EQ(call0.local_block_ids, local_blocks);
  ASSERT_TRUE(call0.local_host_block_ids.has_value());
  EXPECT_EQ(call0.local_host_block_ids.value(), host_blocks);

  ASSERT_EQ(ptr1->start_read_calls.size(), 1);
  const auto& call1 = ptr1->start_read_calls[0];
  EXPECT_EQ(call1.req_id, "req_unified");
  EXPECT_EQ(call1.uuid, uuid);
  EXPECT_EQ(call1.remote_block_ids, remote_blocks);
  EXPECT_EQ(call1.local_block_ids, local_blocks);
  ASSERT_TRUE(call1.local_host_block_ids.has_value());
  EXPECT_EQ(call1.local_host_block_ids.value(), host_blocks);

  // Completion Tracking Aggregation:
  // 1. sub0 completes first -> parent req_unified should NOT be returned yet
  // (needs sub1 too)
  ptr0->recving.push_back("req_unified");
  auto [s1, r1, f1] = mgr.CompleteReadRaw();
  EXPECT_TRUE(r1.empty());

  // 2. sub1 completes -> parent req_unified SHOULD be returned in done_recving
  ptr1->recving.push_back("req_unified");
  auto [s2, r2, f2] = mgr.CompleteReadRaw();
  EXPECT_EQ(r2, std::vector<std::string>{"req_unified"});
}

TEST(KVCacheManagerWrapperTest, StartReadInputVectorValidation) {
  auto sub0 = std::make_unique<MockSubManager>();
  MockSubManager* ptr0 = sub0.get();

  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs;
  subs.push_back(std::move(sub0));

  KVCacheManager mgr(std::move(subs));

  std::vector<EndpointDescriptor> remote_descs = {{"10.0.0.1:45000", {0}}};
  std::vector<int64_t> remote_blocks = {10, 20};
  std::vector<int64_t> local_blocks = {100};  // Size mismatch!

  mgr.StartRead("req_mismatch", 100, remote_descs, remote_blocks, local_blocks);
  EXPECT_TRUE(ptr0->start_read_calls.empty());
}

TEST(KVCacheManagerWrapperTest, GrpcServerOptionalAndOffByDefault) {
  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs1;
  subs1.push_back(std::make_unique<MockSubManager>());
  KVCacheManager mgr_default(std::move(subs1));
  EXPECT_EQ(mgr_default.GetRaidenWorkerPort(), 0);

  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs2;
  subs2.push_back(std::make_unique<MockSubManager>());
  KVCacheManager mgr_explicit_off(std::move(subs2), /*raiden_worker_port=*/0,
                                  /*raiden_controller_address=*/std::nullopt);
  EXPECT_EQ(mgr_explicit_off.GetRaidenWorkerPort(), 0);

  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs3;
  subs3.push_back(std::make_unique<MockSubManager>());
  KVCacheManager mgr_started(std::move(subs3), /*raiden_worker_port=*/0,
                             /*raiden_controller_address=*/"localhost:12345");
  EXPECT_GT(mgr_started.GetRaidenWorkerPort(), 0);
}

TEST(KVCacheManagerWrapperTest, RaidenControllerTransferBuffersIntegration) {
  auto sub0 = std::make_unique<MockSubManager>();
  MockSubManager* ptr0 = sub0.get();

  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs;
  subs.push_back(std::move(sub0));

  KVCacheManager mgr(std::move(subs), /*raiden_worker_port=*/0,
                     /*raiden_controller_address=*/"localhost:12345");
  int port = mgr.GetRaidenWorkerPort();
  ASSERT_GT(port, 0);

  rpc::RaidenIdProto unit;
  unit.set_job_name("test_job");
  unit.set_job_replica_id("0");
  unit.set_data_name("test_data");

  std::string server_address = absl::StrCat("localhost:", port);

  controller::RaidenController controller(
      unit, std::vector<std::string>{server_address},
      /*num_blocks=*/5, /*num_shards=*/1,
      /*shard_size_bytes=*/512);

  std::vector<int64_t> src_offsets = {10, 30};
  std::vector<int64_t> dst_offsets = {20, 40};
  std::vector<int64_t> copy_sizes = {1, 2};

  auto status_d2h = controller.TransferBuffers(
                                "worker_0", rpc::MEMORY_TYPE_HBM,
                                rpc::MEMORY_TYPE_DRAM, src_offsets,
                                dst_offsets, copy_sizes)
                            .Await();
  ASSERT_TRUE(status_d2h.ok());
  EXPECT_EQ(ptr0->d2h_calls, 1);
  EXPECT_EQ(ptr0->h2d_calls, 0);
  EXPECT_EQ(ptr0->last_d2h_src_offsets, src_offsets);
  EXPECT_EQ(ptr0->last_d2h_dst_offsets, dst_offsets);
  EXPECT_EQ(ptr0->last_d2h_copy_sizes, copy_sizes);

  auto status_h2d = controller.TransferBuffers(
                                "worker_0", rpc::MEMORY_TYPE_DRAM,
                                rpc::MEMORY_TYPE_HBM, src_offsets,
                                dst_offsets, copy_sizes)
                            .Await();
  ASSERT_TRUE(status_h2d.ok());
  EXPECT_EQ(ptr0->d2h_calls, 1);
  EXPECT_EQ(ptr0->h2d_calls, 1);
  EXPECT_EQ(ptr0->last_h2d_src_offsets, src_offsets);
  EXPECT_EQ(ptr0->last_h2d_dst_offsets, dst_offsets);
  EXPECT_EQ(ptr0->last_h2d_copy_sizes, copy_sizes);
}

TEST(KVCacheManagerWrapperTest, WorkerSelfRegistrationWithControllerSuccess) {
  auto test_server = core::controller::CreateTestControllerServer();
  ASSERT_NE(test_server, nullptr);

  std::string raiden_controller_address = test_server->server_address;

  auto sub0 = std::make_unique<MockSubManager>();
  std::vector<std::unique_ptr<KVCacheManagerWithTransfer>> subs;
  subs.push_back(std::move(sub0));

  KVCacheManager mgr(std::move(subs), /*raiden_worker_port=*/0,
                     raiden_controller_address, "test_worker_node");

  auto workers =
      test_server->service->worker_registry()->GetRegisteredWorkers();
  ASSERT_EQ(workers.size(), 1);
  EXPECT_EQ(workers[0].worker_id, "test_worker_node");
  EXPECT_NE(
      workers[0].raiden_worker_endpoint.find(std::to_string(mgr.GetRaidenWorkerPort())),
      std::string::npos);
}

}  // namespace
}  // namespace jax
}  // namespace kv_cache
}  // namespace tpu_raiden
