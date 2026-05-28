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
#include "third_party/grpc/include/grpcpp/channel.h"
#include "third_party/grpc/include/grpcpp/create_channel.h"
#include "third_party/grpc/include/grpcpp/security/credentials.h"
#include "third_party/grpc/include/grpcpp/security/server_credentials.h"
#include "third_party/grpc/include/grpcpp/server.h"
#include "third_party/grpc/include/grpcpp/server_builder.h"
#include "kv_cache/global_registry/global_registry_client.h"
#include "kv_cache/global_registry/global_registry_server.h"

namespace tpu_raiden {
namespace kv_cache {
namespace global_registry {
namespace {

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
  std::string host = "10.0.0.1:1234";
  int32_t block = 42;

  auto status = client_->Register({{hash, host, block}});
  EXPECT_TRUE(status.ok()) << status.ToString();

  auto lookup_res = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res.ok()) << lookup_res.status().ToString();
  ASSERT_EQ(lookup_res->size(), 1);
  const auto& meta = (*lookup_res)[0];
  EXPECT_EQ(meta.host_address(), host);
  EXPECT_EQ(meta.block_id(), block);
}

TEST_F(GlobalRegistryTest, OverwriteRegistrationDifferentHosts) {
  std::string hash = "hash1";
  std::string host1 = "10.0.0.1:1234";
  int32_t block1 = 42;
  std::string host2 = "10.0.0.2:1234";
  int32_t block2 = 43;

  EXPECT_TRUE(
      client_->Register({{hash, host1, block1}, {hash, host2, block2}}).ok());

  auto lookup_res = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res.ok()) << lookup_res.status().ToString();
  ASSERT_EQ(lookup_res->size(), 1);
  const auto& meta = (*lookup_res)[0];
  EXPECT_EQ(meta.host_address(), host2);
  EXPECT_EQ(meta.block_id(), block2);
}

TEST_F(GlobalRegistryTest, OverwriteRegistration) {
  std::string hash = "hash1";
  std::string host = "10.0.0.1:1234";
  int32_t block1 = 42;
  int32_t block2 = 43;

  EXPECT_TRUE(client_->Register({{hash, host, block1}}).ok());
  EXPECT_TRUE(client_->Register({{hash, host, block2}}).ok());

  auto lookup_res = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res.ok()) << lookup_res.status().ToString();
  ASSERT_EQ(lookup_res->size(), 1);
  const auto& meta = (*lookup_res)[0];
  EXPECT_EQ(meta.host_address(), host);
  EXPECT_EQ(meta.block_id(), block2);
}

TEST_F(GlobalRegistryTest, BatchLookup) {
  EXPECT_TRUE(
      client_->Register({{"h1", "host1", 42}, {"h2", "host2", 43}}).ok());

  auto batch_res = client_->Lookup({"h1", "h2", "h3"});
  ASSERT_TRUE(batch_res.ok()) << batch_res.status().ToString();

  ASSERT_EQ(batch_res->size(), 2);  // Only h1 and h2 returned
  EXPECT_EQ((*batch_res)[0].host_address(), "host1");
  EXPECT_EQ((*batch_res)[0].block_id(), 42);
  EXPECT_EQ((*batch_res)[1].host_address(), "host2");
  EXPECT_EQ((*batch_res)[1].block_id(), 43);
}

TEST_F(GlobalRegistryTest, StopOnFirstMiss) {
  EXPECT_TRUE(
      client_->Register({{"h1", "host1", 42}, {"h3", "host3", 44}}).ok());

  // h2 is a miss in the middle. The lookup should stop at h2 and NOT return h3.
  auto lookup_res = client_->Lookup({"h1", "h2", "h3"});
  ASSERT_TRUE(lookup_res.ok()) << lookup_res.status().ToString();

  ASSERT_EQ(lookup_res->size(), 1);  // Only h1 returned
  EXPECT_EQ((*lookup_res)[0].host_address(), "host1");
}

TEST_F(GlobalRegistryTest, Unregister) {
  std::string hash1 = "hash1";
  std::string hash2 = "hash2";
  std::string host = "10.0.0.1:1234";
  EXPECT_TRUE(client_->Register({{hash1, host, 42}, {hash2, host, 43}}).ok());

  // Unregister wrong host should fail
  auto status = client_->Unregister({hash1, hash2}, "wrong_host");
  EXPECT_FALSE(status.ok());

  // Unregister correct host should succeed
  status = client_->Unregister({hash1, hash2}, host);
  EXPECT_TRUE(status.ok()) << status.ToString();

  // Lookup should now return empty
  auto lookup_res = client_->Lookup({hash1, hash2});
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 0);
}

TEST_F(GlobalRegistryTest, TtlExpiration) {
  std::string hash = "hash1";
  std::string host = "10.0.0.1:1234";

  // Register with 1 second TTL
  auto status = client_->Register({{hash, host, 42, absl::Seconds(1)}});
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
  // (Internal map should be empty, but we can only verify via lookup which
  // is already empty)
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
