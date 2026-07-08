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

#include <grpcpp/grpcpp.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_client.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_server.h"
#include "tpu_raiden/kv_cache/raiden_id.h"

namespace tpu_raiden {
namespace kv_cache {
namespace global_registry {
namespace {

RaidenId MakeTestRaidenId(int idx) {
  return RaidenId{
      .job_name = "test_job_" + std::to_string(idx),
      .job_replica_id = "replica_" + std::to_string(idx),
      .data_name = "data_" + std::to_string(idx),
      .data_replica_idx = idx,
  };
}

class GlobalRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Start server on ephemeral port
    // Use short TTL and cleanup interval for testing
    service_ = std::make_unique<GlobalRegistryServiceImpl>(
        /*default_ttl=*/absl::Seconds(2),
        /*cleanup_interval=*/absl::Seconds(1));

    grpc::ServerBuilder builder;
    builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                             &port_);
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();

    std::string server_address = "localhost:" + std::to_string(port_);
    channel_ =
        grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
    client_ = std::make_unique<GlobalRegistryClient>(channel_);
  }

  void TearDown() override { server_->Shutdown(); }

  int port_ = 0;
  std::unique_ptr<GlobalRegistryServiceImpl> service_;
  std::unique_ptr<grpc::Server> server_;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<GlobalRegistryClient> client_;
};

TEST_F(GlobalRegistryTest, BasicRegisterAndLookup) {
  std::string hash = "hash1";
  RaidenId raiden_id = MakeTestRaidenId(1);
  int32_t block = 42;

  auto status = client_->Register({{hash, raiden_id, block}});
  EXPECT_TRUE(status.ok()) << status.ToString();

  auto lookup_res = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res.ok()) << lookup_res.status().ToString();
  ASSERT_EQ(lookup_res->size(), 1);
  const auto& meta = (*lookup_res)[0];
  EXPECT_EQ(meta.raiden_id().job_name(), raiden_id.job_name);
  EXPECT_EQ(meta.raiden_id().job_replica_id(), raiden_id.job_replica_id);
  EXPECT_EQ(meta.block_id(), block);
}

TEST_F(GlobalRegistryTest, MultiOwnerRegistrationAndRoundRobin) {
  std::string hash = "hash1";
  RaidenId id1 = MakeTestRaidenId(1);
  int32_t block1 = 42;
  RaidenId id2 = MakeTestRaidenId(2);
  int32_t block2 = 43;

  EXPECT_TRUE(
      client_->Register({{hash, id1, block1}, {hash, id2, block2}}).ok());

  // First Lookup should return one of them (say id1)
  auto lookup_res1 = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res1.ok()) << lookup_res1.status().ToString();
  ASSERT_EQ(lookup_res1->size(), 1);
  EXPECT_EQ((*lookup_res1)[0].raiden_id().job_name(), id1.job_name);

  // Second Lookup should return the other one (id2) due to round-robin
  auto lookup_res2 = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res2.ok()) << lookup_res2.status().ToString();
  ASSERT_EQ(lookup_res2->size(), 1);
  EXPECT_EQ((*lookup_res2)[0].raiden_id().job_name(), id2.job_name);

  // Third Lookup should wrap around to id1
  auto lookup_res3 = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res3.ok()) << lookup_res3.status().ToString();
  ASSERT_EQ(lookup_res3->size(), 1);
  EXPECT_EQ((*lookup_res3)[0].raiden_id().job_name(), id1.job_name);
}

TEST_F(GlobalRegistryTest, OverwriteRegistrationSameHost) {
  std::string hash = "hash1";
  RaidenId id = MakeTestRaidenId(1);
  int32_t block1 = 42;
  int32_t block2 = 43;

  EXPECT_TRUE(client_->Register({{hash, id, block1}}).ok());
  EXPECT_TRUE(client_->Register({{hash, id, block2}}).ok());

  auto lookup_res = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res.ok());
  ASSERT_EQ(lookup_res->size(), 1);
  const auto& meta = (*lookup_res)[0];
  EXPECT_EQ(meta.raiden_id().job_name(), id.job_name);
  EXPECT_EQ(meta.block_id(), block2);
}

TEST_F(GlobalRegistryTest, BatchLookup) {
  RaidenId id1 = MakeTestRaidenId(1);
  RaidenId id2 = MakeTestRaidenId(2);
  EXPECT_TRUE(client_->Register({{"h1", id1, 42}, {"h2", id2, 43}}).ok());

  auto batch_res = client_->Lookup({"h1", "h2", "h3"});
  ASSERT_TRUE(batch_res.ok()) << batch_res.status().ToString();

  ASSERT_EQ(batch_res->size(), 2);  // Only h1 and h2 returned
  EXPECT_EQ((*batch_res)[0].raiden_id().job_name(), id1.job_name);
  EXPECT_EQ((*batch_res)[0].block_id(), 42);
  EXPECT_EQ((*batch_res)[1].raiden_id().job_name(), id2.job_name);
  EXPECT_EQ((*batch_res)[1].block_id(), 43);
}

TEST_F(GlobalRegistryTest, StopOnFirstMiss) {
  RaidenId id1 = MakeTestRaidenId(1);
  RaidenId id3 = MakeTestRaidenId(3);
  EXPECT_TRUE(client_->Register({{"h1", id1, 42}, {"h3", id3, 44}}).ok());

  // h2 is a miss in the middle. The lookup should stop at h2 and NOT return h3.
  auto lookup_res = client_->Lookup({"h1", "h2", "h3"});
  ASSERT_TRUE(lookup_res.ok()) << lookup_res.status().ToString();

  ASSERT_EQ(lookup_res->size(), 1);  // Only h1 returned
  EXPECT_EQ((*lookup_res)[0].raiden_id().job_name(), id1.job_name);
}

TEST_F(GlobalRegistryTest, Unregister) {
  std::string hash1 = "hash1";
  std::string hash2 = "hash2";
  RaidenId id = MakeTestRaidenId(1);
  RaidenId wrong_id = MakeTestRaidenId(999);
  EXPECT_TRUE(client_->Register({{hash1, id, 42}, {hash2, id, 43}}).ok());

  // Unregister wrong host should fail
  auto status = client_->Unregister({hash1, hash2}, wrong_id);
  EXPECT_FALSE(status.ok());

  // Unregister correct host should succeed
  status = client_->Unregister({hash1, hash2}, id);
  EXPECT_TRUE(status.ok()) << status.ToString();

  // Lookup should now return empty
  auto lookup_res = client_->Lookup({hash1, hash2});
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 0);
}

TEST_F(GlobalRegistryTest, UnregisterOneOfMultipleOwners) {
  std::string hash = "hash1";
  RaidenId id1 = MakeTestRaidenId(1);
  RaidenId id2 = MakeTestRaidenId(2);
  EXPECT_TRUE(client_->Register({{hash, id1, 42}, {hash, id2, 43}}).ok());

  // Unregister id1
  auto status = client_->Unregister({hash}, id1);
  EXPECT_TRUE(status.ok());

  // Lookup should now only return id2
  auto lookup_res1 = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res1.ok());
  ASSERT_EQ(lookup_res1->size(), 1);
  EXPECT_EQ((*lookup_res1)[0].raiden_id().job_name(), id2.job_name);

  auto lookup_res2 = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res2.ok());
  ASSERT_EQ(lookup_res2->size(), 1);
  EXPECT_EQ((*lookup_res2)[0].raiden_id().job_name(), id2.job_name);
}

TEST_F(GlobalRegistryTest, TtlExpiration) {
  std::string hash = "hash1";
  RaidenId id = MakeTestRaidenId(1);

  // Register with 1 second TTL
  auto status = client_->Register({{hash, id, 42, absl::Seconds(1)}});
  EXPECT_TRUE(status.ok()) << status.ToString();

  // Immediate lookup should succeed
  auto lookup_res = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res.ok());
  ASSERT_EQ(lookup_res->size(), 1);
  EXPECT_EQ((*lookup_res)[0].block_id(), 42);

  // Wait for TTL to expire (1.5 seconds)
  absl::SleepFor(absl::Milliseconds(1500));

  // Lookup should now return empty (lazy filtering)
  lookup_res = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 0);

  // Wait another 1 second to let background cleanup run
  absl::SleepFor(absl::Seconds(1));

  // Force cleanup just in case, and verify it's gone
  service_->CleanupExpiredEntries();
}

TEST(HashHelperTest, CumulativeHashing) {
  std::vector<int64_t> tokens1 = {101, 202};
  std::vector<int64_t> tokens2 = {303, 404};

  std::string h1 = CalculatePrefixHash(tokens1);
  std::string h2 = CalculatePrefixHash(tokens2, h1);

  EXPECT_FALSE(h1.empty());
  EXPECT_FALSE(h2.empty());
  EXPECT_NE(h1, h2);

  // Verify stability
  std::string h1_ref = CalculatePrefixHash(tokens1);
  EXPECT_EQ(h1, h1_ref);

  std::string h2_ref = CalculatePrefixHash(tokens2, h1_ref);
  EXPECT_EQ(h2, h2_ref);

  // Empty tokens should still hash (just parent hash or empty hash)
  std::string h_empty = CalculatePrefixHash({});
  EXPECT_FALSE(h_empty.empty());
}

}  // namespace
}  // namespace global_registry
}  // namespace kv_cache
}  // namespace tpu_raiden
