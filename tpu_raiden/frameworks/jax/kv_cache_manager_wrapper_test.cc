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

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "xla/tsl/platform/test.h"
#include "tpu_raiden/core/kv_cache_manager_with_transfer.h"
#include "tpu_raiden/frameworks/jax/kv_cache_manager.h"

namespace tpu_raiden {
namespace kv_cache {
namespace jax {
namespace {

class MockSubManager : public KVCacheManagerWithTransfer {
 public:
  MockSubManager()
      : KVCacheManagerWithTransfer({}, std::nullopt, std::nullopt, false, 1,
                                   nullptr) {}

  std::vector<std::string> sending, recving, failed;
  std::string started_ep;
  std::vector<int64_t> started_remote_blocks, started_local_blocks;

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

}  // namespace
}  // namespace jax
}  // namespace kv_cache
}  // namespace tpu_raiden
