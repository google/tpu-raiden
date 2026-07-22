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
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/support/status.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry.grpc.pb.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry.pb.h"
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

  // Calls PullOwned via the raw generated stub and collects all streamed
  // entries. Reports the number of streamed messages via `num_messages` and
  // the final stream status via `status` when provided.
  static std::vector<PullOwnedEntry> PullOwnedOn(
      const std::shared_ptr<grpc::Channel>& channel, const RaidenId& raiden_id,
      grpc::Status* status = nullptr, int* num_messages = nullptr) {
    auto stub = GlobalRegistryService::NewStub(channel);
    grpc::ClientContext context;
    PullOwnedRequest request;
    auto* id_proto = request.mutable_raiden_id();
    id_proto->set_job_name(raiden_id.job_name);
    id_proto->set_job_replica_id(raiden_id.job_replica_id);
    id_proto->set_data_name(raiden_id.data_name);
    id_proto->set_data_replica_idx(raiden_id.data_replica_idx);

    auto reader = stub->PullOwned(&context, request);
    std::vector<PullOwnedEntry> entries;
    PullOwnedResponse response;
    int messages = 0;
    while (reader->Read(&response)) {
      ++messages;
      for (const auto& entry : response.entries()) {
        entries.push_back(entry);
      }
    }
    grpc::Status finish_status = reader->Finish();
    if (status != nullptr) *status = finish_status;
    if (num_messages != nullptr) *num_messages = messages;
    return entries;
  }

  std::vector<PullOwnedEntry> PullOwned(const RaidenId& raiden_id,
                                        grpc::Status* status = nullptr,
                                        int* num_messages = nullptr) {
    return PullOwnedOn(channel_, raiden_id, status, num_messages);
  }

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

TEST_F(GlobalRegistryTest, PullOwnedReturnsOnlyOwnedEntries) {
  RaidenId owner_a = {"jobA", "r1", "d1", 0};
  RaidenId owner_b = {"jobB", "r1", "d1", 0};
  ASSERT_TRUE(client_
                  ->Register({{"h1", owner_a, 1, absl::Seconds(300)},
                              {"h2", owner_a, 2, absl::Seconds(300)},
                              {"h2", owner_b, 20, absl::Seconds(300)},
                              {"h3", owner_b, 30, absl::Seconds(300)}})
                  .ok());

  grpc::Status status;
  auto entries = PullOwned(owner_a, &status);
  EXPECT_TRUE(status.ok()) << status.error_message();
  ASSERT_EQ(entries.size(), 2);
  std::map<std::string, int32_t> blocks;
  for (const auto& entry : entries) {
    blocks[entry.prefix_hash()] = entry.block_id();
  }
  // h2 is registered by both owners; owner A must see its own block id.
  EXPECT_EQ(blocks.at("h1"), 1);
  EXPECT_EQ(blocks.at("h2"), 2);

  auto entries_b = PullOwned(owner_b);
  ASSERT_EQ(entries_b.size(), 2);
}

TEST_F(GlobalRegistryTest, PullOwnedEmptyForUnknownOwner) {
  grpc::Status status;
  int num_messages = 0;
  auto entries =
      PullOwned({"unknown", "r1", "d1", 0}, &status, &num_messages);
  EXPECT_TRUE(status.ok()) << status.error_message();
  EXPECT_TRUE(entries.empty());
  EXPECT_EQ(num_messages, 0);
}

TEST_F(GlobalRegistryTest, PullOwnedRequiresRaidenId) {
  grpc::Status status;
  auto entries = PullOwned(RaidenId{}, &status);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_TRUE(entries.empty());
}

TEST_F(GlobalRegistryTest, PullOwnedReflectsUnregisterAndIndexShrinks) {
  RaidenId owner = {"jobU", "r1", "d1", 0};
  ASSERT_TRUE(client_
                  ->Register({{"h1", owner, 1, absl::Seconds(300)},
                              {"h2", owner, 2, absl::Seconds(300)}})
                  .ok());
  EXPECT_EQ(service_->GetOwnerIndexSizeForTest(owner), 2);

  ASSERT_TRUE(client_->Unregister({"h1"}, owner).ok());
  EXPECT_EQ(service_->GetOwnerIndexSizeForTest(owner), 1);

  auto entries = PullOwned(owner);
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].prefix_hash(), "h2");

  ASSERT_TRUE(client_->Unregister({"h2"}, owner).ok());
  EXPECT_EQ(service_->GetOwnerIndexSizeForTest(owner), 0);
  EXPECT_TRUE(PullOwned(owner).empty());
}

TEST_F(GlobalRegistryTest, PullOwnedOmitsExpiredEntries) {
  RaidenId owner = {"jobE", "r1", "d1", 0};
  ASSERT_TRUE(client_
                  ->Register({{"h1", owner, 1, absl::Seconds(1)},
                              {"h2", owner, 2, absl::Seconds(300)}})
                  .ok());

  absl::SleepFor(absl::Milliseconds(1500));

  auto entries = PullOwned(owner);
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].prefix_hash(), "h2");
}

TEST_F(GlobalRegistryTest, CleanupShrinksOwnerIndex) {
  RaidenId owner = {"jobC", "r1", "d1", 0};
  ASSERT_TRUE(client_->Register({{"h1", owner, 1, absl::Seconds(1)}}).ok());
  EXPECT_EQ(service_->GetOwnerIndexSizeForTest(owner), 1);

  absl::SleepFor(absl::Milliseconds(1500));
  service_->CleanupExpiredEntries();

  EXPECT_EQ(service_->GetOwnerIndexSizeForTest(owner), 0);
  EXPECT_TRUE(PullOwned(owner).empty());
}

TEST_F(GlobalRegistryTest, PullOwnedUpsertKeepsSingleEntryWithLatestBlock) {
  RaidenId owner = {"jobO", "r1", "d1", 0};
  ASSERT_TRUE(client_->Register({{"h1", owner, 42, absl::Seconds(300)}}).ok());
  ASSERT_TRUE(client_->Register({{"h1", owner, 43, absl::Seconds(300)}}).ok());

  EXPECT_EQ(service_->GetOwnerIndexSizeForTest(owner), 1);
  auto entries = PullOwned(owner);
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].block_id(), 43);
}

TEST_F(GlobalRegistryTest, PullOwnedStreamsInServerConfiguredBatches) {
  GlobalRegistryServiceImpl service(
      /*default_ttl=*/absl::Seconds(300),
      /*cleanup_interval=*/absl::ZeroDuration(),
      /*pull_owned_batch_size=*/2);
  int port = 0;
  grpc::ServerBuilder builder;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();
  auto channel = grpc::CreateChannel("localhost:" + std::to_string(port),
                                     grpc::InsecureChannelCredentials());
  GlobalRegistryClient client(channel);

  RaidenId owner = {"jobB", "r1", "d1", 0};
  ASSERT_TRUE(client
                  .Register({{"h1", owner, 1},
                             {"h2", owner, 2},
                             {"h3", owner, 3},
                             {"h4", owner, 4},
                             {"h5", owner, 5}})
                  .ok());

  grpc::Status status;
  int num_messages = 0;
  auto entries = PullOwnedOn(channel, owner, &status, &num_messages);
  EXPECT_TRUE(status.ok()) << status.error_message();
  EXPECT_EQ(entries.size(), 5);
  EXPECT_EQ(num_messages, 3);  // ceil(5 entries / batch size 2)

  server->Shutdown();
}

TEST_F(GlobalRegistryTest, PullOwnedRemainingTtlForExpiringEntry) {
  RaidenId owner = {"jobT", "r1", "d1", 0};
  ASSERT_TRUE(client_->Register({{"h1", owner, 1, absl::Seconds(300)}}).ok());

  auto entries = PullOwned(owner);
  ASSERT_EQ(entries.size(), 1);
  // Wall-clock time elapses between Register and PullOwned, so the lower
  // bound is deliberately generous: it only needs to prove the value reflects
  // the registered TTL rather than the 0 infinite marker or the >= 1 clamp.
  // The upper bound is timing-safe (remaining TTL only decreases).
  EXPECT_GT(entries[0].remaining_ttl_seconds(), 200);
  EXPECT_LE(entries[0].remaining_ttl_seconds(), 300);
}

TEST_F(GlobalRegistryTest, ClientPullOwnedReturnsEntries) {
  RaidenId owner = {"jobW", "r1", "d1", 0};
  ASSERT_TRUE(client_
                  ->Register({{"h1", owner, 7, absl::Seconds(300)},
                              {"h2", owner, 8, absl::Seconds(300)}})
                  .ok());

  auto pulled_or = client_->PullOwned(owner);
  ASSERT_TRUE(pulled_or.ok()) << pulled_or.status().ToString();
  ASSERT_EQ(pulled_or->size(), 2);
  std::map<std::string, GlobalRegistryClient::PulledEntry> by_hash;
  for (const auto& entry : *pulled_or) {
    by_hash[entry.prefix_hash] = entry;
  }
  EXPECT_EQ(by_hash.at("h1").block_id, 7);
  EXPECT_EQ(by_hash.at("h2").block_id, 8);
  EXPECT_GT(by_hash.at("h1").remaining_ttl_seconds, 0);
}

TEST_F(GlobalRegistryTest, ClientPullOwnedEmptyForUnknownOwner) {
  auto pulled_or = client_->PullOwned({"nobody", "r1", "d1", 0});
  ASSERT_TRUE(pulled_or.ok()) << pulled_or.status().ToString();
  EXPECT_TRUE(pulled_or->empty());
}

TEST_F(GlobalRegistryTest, ClientPullOwnedRejectsEmptyRaidenId) {
  auto pulled_or = client_->PullOwned(RaidenId{});
  EXPECT_FALSE(pulled_or.ok());
}

TEST_F(GlobalRegistryTest, PullOwnedRemainingTtlZeroForInfiniteTtl) {
  GlobalRegistryServiceImpl service(
      /*default_ttl=*/absl::InfiniteDuration(),
      /*cleanup_interval=*/absl::ZeroDuration());
  int port = 0;
  grpc::ServerBuilder builder;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();
  auto channel = grpc::CreateChannel("localhost:" + std::to_string(port),
                                     grpc::InsecureChannelCredentials());
  GlobalRegistryClient client(channel);

  RaidenId owner = {"jobI", "r1", "d1", 0};
  // No explicit TTL: the server's default (infinite) applies.
  ASSERT_TRUE(client.Register({{"h1", owner, 1}}).ok());

  auto entries = PullOwnedOn(channel, owner);
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].remaining_ttl_seconds(), 0);

  server->Shutdown();
}

}  // namespace
}  // namespace global_registry
}  // namespace kv_cache
}  // namespace tpu_raiden
