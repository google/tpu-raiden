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
  RaidenId host = {"job1", "replica1", "data1", 0};
  int32_t block = 42;

  auto status = client_->Register({{hash, host, block}});
  EXPECT_TRUE(status.ok()) << status.ToString();

  auto lookup_res = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res.ok()) << lookup_res.status().ToString();
  ASSERT_EQ(lookup_res->size(), 1);
  const auto& meta = (*lookup_res)[0];
  EXPECT_EQ(meta.raiden_id().job_name(), host.job_name);
  EXPECT_EQ(meta.raiden_id().job_replica_id(), host.job_replica_id);
  EXPECT_EQ(meta.raiden_id().data_name(), host.data_name);
  EXPECT_EQ(meta.raiden_id().data_replica_idx(), host.data_replica_idx);
  EXPECT_EQ(meta.block_id(), block);
}

TEST_F(GlobalRegistryTest, MultiRegistrationAndRoundRobinLookup) {
  std::string hash = "hash1";
  RaidenId host1 = {"job1", "replica1", "data1", 0};
  int32_t block1 = 42;
  RaidenId host2 = {"job1", "replica2", "data1", 1};
  int32_t block2 = 43;

  EXPECT_TRUE(
      client_->Register({{hash, host1, block1}, {hash, host2, block2}}).ok());

  // First lookup should return host1 or host2 depending on internal order.
  // The server implementation iterates over insertion order, so we expect host1
  // then host2.
  auto lookup_res1 = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res1.ok()) << lookup_res1.status().ToString();
  ASSERT_EQ(lookup_res1->size(), 1);
  EXPECT_EQ((*lookup_res1)[0].raiden_id().job_replica_id(),
            host1.job_replica_id);
  EXPECT_EQ((*lookup_res1)[0].block_id(), block1);

  // Second lookup should round-robin to host2.
  auto lookup_res2 = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res2.ok()) << lookup_res2.status().ToString();
  ASSERT_EQ(lookup_res2->size(), 1);
  EXPECT_EQ((*lookup_res2)[0].raiden_id().job_replica_id(),
            host2.job_replica_id);
  EXPECT_EQ((*lookup_res2)[0].block_id(), block2);

  // Third lookup should wrap around to host1.
  auto lookup_res3 = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res3.ok()) << lookup_res3.status().ToString();
  ASSERT_EQ(lookup_res3->size(), 1);
  EXPECT_EQ((*lookup_res3)[0].raiden_id().job_replica_id(),
            host1.job_replica_id);
}

TEST_F(GlobalRegistryTest, OverwriteRegistrationSameHost) {
  std::string hash = "hash1";
  RaidenId host = {"job1", "replica1", "data1", 0};
  int32_t block1 = 42;
  int32_t block2 = 43;

  EXPECT_TRUE(client_->Register({{hash, host, block1}}).ok());
  EXPECT_TRUE(client_->Register({{hash, host, block2}}).ok());

  auto lookup_res = client_->Lookup({hash});
  ASSERT_TRUE(lookup_res.ok()) << lookup_res.status().ToString();
  ASSERT_EQ(lookup_res->size(), 1);
  EXPECT_EQ((*lookup_res)[0].block_id(), block2);
}

TEST_F(GlobalRegistryTest, MultiLookupTerminatesOnFirstMiss) {
  EXPECT_TRUE(client_
                  ->Register({{"h1", {"j1", "r1", "d1", 0}, 42},
                              {"h2", {"j1", "r2", "d1", 0}, 43}})
                  .ok());

  // Sequential hits for "h1" and "h2"
  auto res1 = client_->Lookup({"h1", "h2"});
  ASSERT_TRUE(res1.ok());
  EXPECT_EQ(res1->size(), 2);

  // Miss on "h3" stops sequential lookup, "h2" is omitted
  auto res2 = client_->Lookup({"h1", "h3", "h2"});
  ASSERT_TRUE(res2.ok());
  EXPECT_EQ(res2->size(), 1);
  EXPECT_EQ((*res2)[0].block_id(), 42);
}

TEST_F(GlobalRegistryTest, UnregisterSuccess) {
  std::string hash1 = "hash1";
  std::string hash2 = "hash2";
  RaidenId host = {"job1", "replica1", "data1", 0};

  EXPECT_TRUE(client_->Register({{hash1, host, 42}, {hash2, host, 43}}).ok());

  auto status = client_->Unregister({hash1}, host);
  EXPECT_TRUE(status.ok()) << status.ToString();

  auto res = client_->Lookup({hash1, hash2});
  ASSERT_TRUE(res.ok());
  EXPECT_EQ(res->size(), 0);  // Stops at first miss (hash1 is gone)

  auto res2 = client_->Lookup({hash2});
  ASSERT_TRUE(res2.ok());
  EXPECT_EQ(res2->size(), 1);
  EXPECT_EQ((*res2)[0].block_id(), 43);
}

TEST_F(GlobalRegistryTest, ExpirationTtlFilter) {
  std::string hash = "hash1";
  RaidenId host = {"job1", "replica1", "data1", 0};

  auto status = client_->Register({{hash, host, 42, absl::Seconds(1)}});
  EXPECT_TRUE(status.ok());

  absl::SleepFor(absl::Milliseconds(1500));

  auto res = client_->Lookup({hash});
  ASSERT_TRUE(res.ok());
  EXPECT_EQ(res->size(), 0);
}

}  // namespace
}  // namespace global_registry
}  // namespace kv_cache
}  // namespace tpu_raiden
