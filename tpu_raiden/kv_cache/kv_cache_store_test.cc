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

#include "tpu_raiden/kv_cache/kv_cache_store.h"

#include <atomic>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "grpcpp/grpcpp.h"
#include "net/util/ports.h"
#include <csignal>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
// OSS shim: google3 gunit provides bare EXPECT_OK/ASSERT_OK; abseil ships them as
// ABSL_EXPECT_OK/ABSL_ASSERT_OK. Alias to abseil's IsOk() matcher.
#ifndef EXPECT_OK
#define EXPECT_OK(expr) EXPECT_THAT(expr, ::absl_testing::IsOk())
#define ASSERT_OK(expr) ASSERT_THAT(expr, ::absl_testing::IsOk())
#endif
// OSS: google3 InitGoogle ignores SIGPIPE; the standalone gtest_main binary does
// not, so a broken-pipe write during socket teardown would kill the process.
namespace {
const int kRaidenIgnoreSigpipe = [] {
  signal(SIGPIPE, SIG_IGN);
  return 0;
}();
}  // namespace
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_client.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_server.h"
#include "tpu_raiden/kv_cache/kv_cache_listener.h"
#include "tpu_raiden/kv_cache/kv_cache_manager_base.h"
#include "tpu_raiden/kv_cache/raiden_controller_embedded.h"
#include "tpu_raiden/kv_cache/raiden_orchestrator.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

int PickConsecutiveUnusedPortsOrDie(int count) {
  for (int attempt = 0; attempt < 100; ++attempt) {
    int base = net_util::PickUnusedPortOrDie();
    bool all_free = true;
    for (int i = 0; i < count; ++i) {
      int port = base + i;
      int fd = socket(AF_INET, SOCK_STREAM, 0);
      if (fd < 0) {
        all_free = false;
        break;
      }
      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_port = htons(port);
      addr.sin_addr.s_addr = INADDR_ANY;
      int opt = 1;
      setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
      if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(fd);
        all_free = false;
        break;
      }
      close(fd);
    }
    if (all_free) return base;
  }
  LOG(FATAL) << "Could not find " << count << " consecutive unused ports";
  return 0;
}

// Verifies basic insert, lookup (including partial and early misses), and
// delete functionality.
TEST(KVCacheStoreTest, BasicTests) {
  KVCacheStore controller(50);
  EXPECT_EQ(controller.capacity(), 50);

  std::vector<std::string> hashes = {"4001", "4002"};
  std::vector<RaidenBlockID> slices = {
      RaidenId{"inference_server", "0", "kv_cache", 0},
      RaidenId{"inference_server", "1", "kv_cache", 0}};

  // 1. Insert
  EXPECT_TRUE(controller.Insert(hashes, slices, true).first);
  EXPECT_FALSE(
      controller.Insert(hashes, slices, true).first);  // Already exists

  // 2. Lookup with a partial miss at the end
  std::vector<std::string> hashes_with_miss = {"4001", "4002", "4003"};
  auto lookup_res = controller.Lookup(hashes_with_miss);
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].first, "4001");
  EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::INIT);

  // Lookup with an early miss
  std::vector<std::string> hashes_early_miss = {"4001", "4003", "4002"};
  auto lookup_res_early = controller.Lookup(hashes_early_miss);
  ASSERT_TRUE(lookup_res_early.ok());
  EXPECT_EQ(lookup_res_early->size(), 1);
  EXPECT_EQ((*lookup_res_early)[0].first, "4001");

  // 3. Delete
  controller.Delete(hashes, slices);
  EXPECT_TRUE(
      controller.Insert(hashes, slices, true).first);  // Succesful again
}

// Verifies pinning blocks to prevent eviction and releasing them to allow
// eviction.
TEST(KVCacheStoreTest, PinAndRelease) {
  KVCacheStore controller(2);

  std::vector<std::string> hashes = {"101", "102"};
  std::vector<RaidenBlockID> slices = {
      RaidenId{"inference_server", "0", "kv_cache", 0},
      RaidenId{"inference_server", "1", "kv_cache", 0}};

  EXPECT_TRUE(controller.Insert(hashes, slices, true).first);

  // Pin both
  EXPECT_TRUE(controller.Pin(hashes));
  EXPECT_EQ(controller.GetPinCount("101"), 1);
  EXPECT_EQ(controller.GetPinCount("102"), 1);

  // Inserting a third element should fail to evict because both existing items
  // are pinned
  std::vector<std::string> hash_3 = {"103"};
  std::vector<RaidenBlockID> slice_3 = {
      RaidenId{"inference_server", "2", "kv_cache", 0}};
  controller.Insert(hash_3, slice_3, true);

  // Release 101
  controller.Release({"101"});
  EXPECT_EQ(controller.GetPinCount("101"), 0);

  // Now inserting a fourth element (104) should successfully evict 101
  std::vector<std::string> hash_4 = {"104"};
  std::vector<RaidenBlockID> slice_4 = {
      RaidenId{"inference_server", "3", "kv_cache", 0}};
  controller.Insert(hash_4, slice_4, true);

  // Lookup 101 should result in an immediate miss (return size 0 before 102)
  EXPECT_EQ(controller.Lookup({"101", "102"})->size(), 0);
  EXPECT_EQ(controller.Lookup({"102"})->size(), 1);
}

// Verifies that if pinning a set of blocks fails (e.g. due to a missing block),
// none of them remain pinned.
TEST(KVCacheStoreTest, PartialPinRollback) {
  KVCacheStore controller(2);

  std::vector<std::string> hashes = {"201", "202"};
  std::vector<RaidenBlockID> slices = {
      RaidenId{"inference_server", "0", "kv_cache", 0},
      RaidenId{"inference_server", "1", "kv_cache", 0}};

  EXPECT_TRUE(controller.Insert(hashes, slices, true).first);

  // Attempt to pin a sequence with a missing hash (203)
  EXPECT_FALSE(controller.Pin({"201", "202", "203"}));

  // Confirm that 201 and 202 were completely reverted (pin count 0)
  EXPECT_EQ(controller.GetPinCount("201"), 0);
  EXPECT_EQ(controller.GetPinCount("202"), 0);
}

// Verifies that inserting beyond capacity evicts the least recently used (LRU)
// block and returns it.
TEST(KVCacheStoreTest, EvictionTracking) {
  KVCacheStore controller(2);

  std::vector<std::string> hashes_1_2 = {"101", "102"};
  std::vector<RaidenBlockID> slices_1_2 = {
      RaidenId{"inference_server", "0", "kv_cache", 0},
      RaidenId{"inference_server", "1", "kv_cache", 1}};

  // 1. Insert 101 and 102. No evictions should occur.
  auto res_1_2 = controller.Insert(hashes_1_2, slices_1_2, true);
  EXPECT_TRUE(res_1_2.first);
  EXPECT_TRUE(res_1_2.second.empty());

  // 2. Insert 103. Since capacity is 2, this must evict the LRU block (101).
  std::vector<std::string> hash_3 = {"103"};
  std::vector<RaidenBlockID> slice_3 = {
      RaidenId{"inference_server", "2", "kv_cache", 2}};

  auto res_3 = controller.Insert(hash_3, slice_3, true);
  EXPECT_TRUE(res_3.first);
  ASSERT_EQ(res_3.second.size(), 1);
  EXPECT_EQ(res_3.second[0].first, "101");
  EXPECT_EQ(res_3.second[0].second.status, BlockStatus::INIT);
  EXPECT_EQ(res_3.second[0].second.raiden_id.job_name, "inference_server");
  EXPECT_EQ(res_3.second[0].second.raiden_id.data_replica_idx, 0);

  // 3. Verify that lookup for 101 now misses, but 102 and 103 are present.
  EXPECT_EQ(controller.Lookup({"101"})->size(), 0);
  EXPECT_EQ(controller.Lookup({"102"})->size(), 1);
  EXPECT_EQ(controller.Lookup({"103"})->size(), 1);
}

// Verifies looking up blocks in the global registry when they are not found
// locally.
TEST(KVCacheStoreTest, GlobalLookupFallback) {
  // 1. Start a local registry server
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(service.get());
  auto server = builder.BuildAndStart();
  std::string server_address = "localhost:" + std::to_string(port);

  // 2. Register some blocks in the registry
  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  global_registry::GlobalRegistryClient registry_client(channel);

  std::string hash1 = "global_hash_1";
  RaidenId host1{"job1", "0", "kv_cache", 0};
  int32_t block1 = 42;

  std::string hash2 = "global_hash_2";
  RaidenId host2{"job2", "0", "kv_cache", 0};
  int32_t block2 = 43;

  // For Case 2: Register a hash that will also be present locally, but with
  // a different remote address in the registry.
  std::string hash_shared = "shared_hash";
  RaidenId host_shared_remote{"job_shared", "0", "kv_cache", 0};
  int32_t block_shared_remote = 99;

  ASSERT_TRUE(
      registry_client
          .Register({{hash1, host1, block1},
                     {hash2, host2, block2},
                     {hash_shared, host_shared_remote, block_shared_remote}})
          .ok());

  // 3. Create KVCacheStore with the registry address
  KVCacheStore store(50, server_address);

  // Insert blocks locally
  std::vector<std::string> local_hashes = {"local_only_hash", "shared_hash"};
  std::vector<RaidenBlockID> local_slices = {
      RaidenId{"local_job", "0", "kv_cache", 0},
      RaidenId{"local_job", "0", "kv_cache", 1}};
  ASSERT_TRUE(store.Insert(local_hashes, local_slices, true).first);

  // Case 1: Full local hit, no global hit
  {
    auto lookup_res = store.Lookup({"local_only_hash"}, /*enable_global=*/true);
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 1);
    EXPECT_EQ((*lookup_res)[0].first, "local_only_hash");
    EXPECT_EQ((*lookup_res)[0].second.raiden_id.job_name, "local_job");
    EXPECT_EQ((*lookup_res)[0].second.raiden_id.data_replica_idx, 0);
  }

  // Case 2: Both local and global has the same hit, but we return local hit
  // results
  {
    auto lookup_res = store.Lookup({"shared_hash"}, /*enable_global=*/true);
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 1);
    EXPECT_EQ((*lookup_res)[0].first, "shared_hash");
    // Should return local info, not remote info from registry
    EXPECT_EQ((*lookup_res)[0].second.raiden_id.job_name, "local_job");
    EXPECT_EQ((*lookup_res)[0].second.raiden_id.data_replica_idx, 1);
  }

  // Case 3: No local hit, only global hits
  {
    auto lookup_res = store.Lookup({"global_hash_1", "global_hash_2"},
                                   /*enable_global=*/true);
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 2);

    EXPECT_EQ((*lookup_res)[0].first, "global_hash_1");
    EXPECT_EQ((*lookup_res)[0].second.raiden_id, host1);

    EXPECT_EQ((*lookup_res)[1].first, "global_hash_2");
    EXPECT_EQ((*lookup_res)[1].second.raiden_id, host2);
  }

  // 4. Lookup with enable_global = false
  // It should stop at the first miss (which is the global hash if we query it)
  // If we query {"local_only_hash", "global_hash_1"}, it should return
  // local_only_hash and stop.
  {
    auto lookup_res = store.Lookup({"local_only_hash", "global_hash_1"},
                                   /*enable_global=*/false);
    ASSERT_TRUE(lookup_res.ok());
    EXPECT_EQ(lookup_res->size(), 1);
    EXPECT_EQ((*lookup_res)[0].first, "local_only_hash");
  }

  // 5. Lookup with enable_global = true
  // It should return both local and global
  {
    auto lookup_res =
        store.Lookup({"local_only_hash", "global_hash_1", "global_hash_2"},
                     /*enable_global=*/true);
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 3);

    EXPECT_EQ((*lookup_res)[0].first, "local_only_hash");
    EXPECT_EQ((*lookup_res)[0].second.raiden_id.job_name, "local_job");

    EXPECT_EQ((*lookup_res)[1].first, "global_hash_1");
    EXPECT_EQ((*lookup_res)[1].second.raiden_id, host1);

    EXPECT_EQ((*lookup_res)[2].first, "global_hash_2");
    EXPECT_EQ((*lookup_res)[2].second.raiden_id, host2);
  }

  // 6. Lookup with enable_global = true, but registry has a miss
  // It should stop at the first miss in registry
  {
    auto lookup_res = store.Lookup(
        {"local_only_hash", "global_hash_1", "missing_hash", "global_hash_2"},
        /*enable_global=*/true);
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 2);  // local_only_hash, global_hash_1
    EXPECT_EQ((*lookup_res)[0].first, "local_only_hash");
    EXPECT_EQ((*lookup_res)[1].first, "global_hash_1");
  }

  server->Shutdown();
}

// Verifies that local lookup still works even if the global registry is
// unreachable.
TEST(KVCacheStoreTest, GlobalLookupRegistryDown) {
  // Create KVCacheStore with an unreachable registry address
  KVCacheStore store(50, "invalid.address:12345");

  // Insert one block locally
  std::vector<std::string> local_hashes = {"local_hash"};
  std::vector<RaidenBlockID> local_slices = {
      RaidenId{"local_job", "0", "kv_cache", 0}};
  ASSERT_TRUE(store.Insert(local_hashes, local_slices, true).first);

  // Lookup with enable_global = true.
  // It should NOT fail even though the registry is down. It should return the
  // local hit.
  auto lookup_res = store.Lookup({"local_hash", "missing_hash"},
                                 /*enable_global=*/true);
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 1);
  EXPECT_EQ((*lookup_res)[0].first, "local_hash");
  EXPECT_EQ((*lookup_res)[0].second.raiden_id.job_name, "local_job");
}

// Verifies that lookup returns at most the capacity of the store.
TEST(KVCacheStoreTest, LookupCapLimit) {
  KVCacheStore store(2);

  std::vector<std::string> hashes = {"101", "102"};
  std::vector<RaidenBlockID> slices = {
      RaidenId{"inference_server", "0", "kv_cache", 0},
      RaidenId{"inference_server", "1", "kv_cache", 1}};

  ASSERT_TRUE(store.Insert(hashes, slices, true).first);

  // Lookup 3 hashes, but capacity is 2. It should only return 2.
  std::vector<std::string> lookup_hashes = {"101", "102", "103"};
  auto lookup_res = store.Lookup(lookup_hashes);
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].first, "101");
  EXPECT_EQ((*lookup_res)[1].first, "102");
}

// Verifies that lookup returns at most the capacity of the store, even with
// global registry enabled.
TEST(KVCacheStoreTest, LookupCapLimitWithGlobal) {
  // 1. Start a local registry server
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(service.get());
  auto server = builder.BuildAndStart();
  std::string server_address = "localhost:" + std::to_string(port);

  // 2. Register some blocks in the registry
  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  global_registry::GlobalRegistryClient registry_client(channel);

  std::string hash1 = "global_hash_1";
  RaidenId host1{"job1", "0", "kv_cache", 0};
  int32_t block1 = 42;

  std::string hash2 = "global_hash_2";
  RaidenId host2{"job2", "0", "kv_cache", 0};
  int32_t block2 = 43;

  std::string hash3 = "global_hash_3";
  RaidenId host3{"job3", "0", "kv_cache", 0};
  int32_t block3 = 44;

  ASSERT_TRUE(registry_client
                  .Register({{hash1, host1, block1},
                             {hash2, host2, block2},
                             {hash3, host3, block3}})
                  .ok());

  // 3. Create KVCacheStore with capacity 2
  KVCacheStore store(2, server_address);

  // Lookup 3 hashes, but capacity is 2. It should only return 2.
  std::vector<std::string> lookup_hashes = {"global_hash_1", "global_hash_2",
                                            "global_hash_3"};
  auto lookup_res = store.Lookup(lookup_hashes, /*enable_global=*/true);
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].first, "global_hash_1");
  EXPECT_EQ((*lookup_res)[1].first, "global_hash_2");

  server->Shutdown();
}

// Verifies capacity limits with a mix of local and global hits.
TEST(KVCacheStoreTest, LookupCapLimitMixed) {
  // 1. Start a local registry server
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(service.get());
  auto server = builder.BuildAndStart();
  std::string server_address = "localhost:" + std::to_string(port);

  // 2. Register some blocks in the registry
  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  global_registry::GlobalRegistryClient registry_client(channel);

  std::string hash2 = "global_hash_2";
  RaidenId host2{"job2", "0", "kv_cache", 0};
  int32_t block2 = 43;

  std::string hash3 = "global_hash_3";
  RaidenId host3{"job3", "0", "kv_cache", 0};
  int32_t block3 = 44;

  ASSERT_TRUE(
      registry_client.Register({{hash2, host2, block2}, {hash3, host3, block3}})
          .ok());

  // 3. Create KVCacheStore with capacity 2
  KVCacheStore store(2, server_address);

  // Insert 1 block locally
  std::vector<std::string> local_hashes = {"local_hash_1"};
  std::vector<RaidenBlockID> local_slices = {
      RaidenId{"local_job", "0", "kv_cache", 0}};
  ASSERT_TRUE(store.Insert(local_hashes, local_slices, true).first);

  // Lookup 3 hashes, but capacity is 2. It should only return 2 (1 local, 1
  // global).
  std::vector<std::string> lookup_hashes = {"local_hash_1", "global_hash_2",
                                            "global_hash_3"};
  auto lookup_res = store.Lookup(lookup_hashes, /*enable_global=*/true);
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].first, "local_hash_1");
  EXPECT_EQ((*lookup_res)[1].first, "global_hash_2");

  server->Shutdown();
}

// Verifies that lookup is limited by available space (capacity - pinned
// blocks).
TEST(KVCacheStoreTest, LookupAvailableSpaceLimit) {
  KVCacheStore store(3);

  std::vector<std::string> hashes = {"101", "102", "103"};
  std::vector<RaidenBlockID> slices = {
      RaidenId{"inference_server", "0", "kv_cache", 0},
      RaidenId{"inference_server", "1", "kv_cache", 1},
      RaidenId{"inference_server", "2", "kv_cache", 2}};

  ASSERT_TRUE(store.Insert(hashes, slices, true).first);

  // Pin 101. Pinned count = 1. Available space = 3 - 1 = 2.
  EXPECT_TRUE(store.Pin({"101"}));

  // Lookup 4 hashes. Since available space is 2, it should only return the
  // first 2.
  std::vector<std::string> lookup_hashes = {"101", "102", "103", "104"};
  auto lookup_res = store.Lookup(lookup_hashes);
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].first, "101");
  EXPECT_EQ((*lookup_res)[1].first, "102");
}

// Verifies inserting blocks and pinning them atomically, and space limits.
TEST(KVCacheStoreTest, InsertAndPin) {
  KVCacheStore store(2);

  // Insert local block
  std::vector<std::string> local_hashes = {"local_1"};
  std::vector<RaidenBlockID> local_slices = {
      RaidenId{"local_job", "0", "kv_cache", 0}};
  ASSERT_TRUE(store.Insert(local_hashes, local_slices, true).first);

  // Execute InsertAndPin
  std::vector<RaidenBlockID> slices = {
      RaidenId{"local_job", "0", "kv_cache", 0},
      RaidenId{"remote_job", "0", "kv_cache", 42}};
  auto res = store.InsertAndPin({"local_1", "remote_1"}, slices, true);
  EXPECT_TRUE(res.first);
  EXPECT_TRUE(res.second.empty());
  EXPECT_EQ(store.GetPinCount("local_1"), 1);
  EXPECT_EQ(store.GetPinCount("remote_1"), 1);

  // Since capacity is 2 and both local_1 and remote_1 are pinned, available
  // space is 0. Attempting to InsertAndPin remote_2 should fail due to lack
  // of space.
  auto res_fail = store.InsertAndPin({"remote_2"}, {RaidenBlockID()}, true);
  EXPECT_FALSE(res_fail.first);
}

// Verifies PutBack puts items in the LRU position in the cache.
TEST(KVCacheStoreTest, LruCachePutBack) {
  LRUCache<std::string, int> cache(3);
  cache.Put("A", 1);
  cache.Put("B", 2);
  // MRU to LRU is: B, A.

  // PutBack("C", 3) should add C to the back (LRU position).
  cache.PutBack("C", 3);
  // Now MRU to LRU should be: B, A, C.
  // Let's verify by checking Evict(): first evicted item should be C!
  auto evicted1 = cache.Evict();
  ASSERT_TRUE(evicted1.has_value());
  EXPECT_EQ(evicted1->first, "C");

  auto evicted2 = cache.Evict();
  ASSERT_TRUE(evicted2.has_value());
  EXPECT_EQ(evicted2->first, "A");

  auto evicted3 = cache.Evict();
  ASSERT_TRUE(evicted3.has_value());
  EXPECT_EQ(evicted3->first, "B");
}

// Verifies releasing pinned blocks and deleting them if they are remote
// placeholders.
TEST(KVCacheStoreTest, ReleaseAndDelete) {
  KVCacheStore store(2);

  // Insert two local blocks (not remote)
  std::vector<std::string> local_hashes = {"local_1", "local_2"};
  std::vector<RaidenBlockID> local_slices = {
      RaidenBlockID(RaidenId{"local_job", "0", "kv_cache", 0}, -1,
                    BlockStatus::HOST),
      RaidenBlockID(RaidenId{"local_job", "0", "kv_cache", 1}, -1,
                    BlockStatus::HOST)};
  ASSERT_TRUE(store.Insert(local_hashes, local_slices, true).first);

  // Now InsertAndPin two remote blocks, which will evict local_1 and local_2.
  std::vector<std::string> remote_hashes = {"remote_1", "remote_2"};
  std::vector<RaidenBlockID> remote_slices = {
      RaidenBlockID(RaidenId{"remote_job", "0", "kv_cache", 0}, -1,
                    BlockStatus::REMOTE),
      RaidenBlockID(RaidenId{"remote_job", "0", "kv_cache", 1}, -1,
                    BlockStatus::REMOTE)};
  auto res = store.InsertAndPin(remote_hashes, remote_slices, true);
  ASSERT_TRUE(res.first);
  ASSERT_EQ(res.second.size(), 2);
  EXPECT_EQ(store.GetPinCount("remote_1"), 1);
  EXPECT_EQ(store.GetPinCount("remote_2"), 1);
  EXPECT_EQ(store.Lookup({"local_1"})->size(), 0);
  EXPECT_EQ(store.Lookup({"local_2"})->size(), 0);

  // Now call ReleaseAndDelete to revert InsertAndPin!
  auto release_res = store.ReleaseAndDelete(remote_hashes, res.second);
  EXPECT_EQ(release_res.first, 2);
  EXPECT_TRUE(release_res.second.empty());

  // remote_1 and remote_2 should be unpinned and deleted (since REMOTE)
  EXPECT_EQ(store.GetPinCount("remote_1"), 0);
  EXPECT_EQ(store.GetPinCount("remote_2"), 0);
  EXPECT_EQ(store.Lookup({"remote_1"})->size(), 0);
  EXPECT_EQ(store.Lookup({"remote_2"})->size(), 0);

  // local_1 and local_2 should be restored to the cache!
  auto lookup_res = store.Lookup({"local_1", "local_2"});
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 2);

  // Test non-remote block in ReleaseAndDelete: should unpin without deleting
  store.InsertAndPin({"local_1"}, {local_slices[0]}, true);
  EXPECT_EQ(store.GetPinCount("local_1"), 1);
  auto res_non_remote = store.ReleaseAndDelete({"local_1"});
  EXPECT_EQ(res_non_remote.first, 0);
  EXPECT_TRUE(res_non_remote.second.empty());
  EXPECT_EQ(store.GetPinCount("local_1"), 0);
  EXPECT_EQ(store.Lookup({"local_1"})->size(), 1);

  // Test remote block pinned twice: after one ReleaseAndDelete, pin count is 1
  // so it should NOT be deleted!
  store.InsertAndPin({"remote_1"}, {remote_slices[0]}, true);
  store.Pin({"remote_1"});  // pin count is now 2
  EXPECT_EQ(store.GetPinCount("remote_1"), 2);
  auto res_pinned = store.ReleaseAndDelete({"remote_1"});
  EXPECT_EQ(res_pinned.first, 0);  // 0 deleted because pin count was 2 -> 1
  EXPECT_EQ(store.GetPinCount("remote_1"), 1);
  EXPECT_EQ(store.Lookup({"remote_1"})->size(), 1);
  store.Release({"remote_1"});
  store.Delete({"remote_1"}, {remote_slices[0]});

  // Test partial restore: 1 deleted remote block with 2 evicted entries
  BlockSliceList mock_evicted = {{"evict_1", {local_slices[0]}},
                                 {"evict_2", {local_slices[1]}}};
  store.InsertAndPin({"remote_2"}, {remote_slices[1]}, true);
  auto res_partial = store.ReleaseAndDelete({"remote_2"}, mock_evicted);
  EXPECT_EQ(res_partial.first, 1);
  ASSERT_EQ(res_partial.second.size(), 1);
  EXPECT_EQ(res_partial.second[0].first, "evict_1");  // evict_2 was restored!
}

// --- ThreadSafeQueue Tests ---

// Verifies basic queue operations (Push and Pop).
TEST(ThreadSafeQueueTest, BasicPushPop) {
  ThreadSafeQueue<int> q;
  q.Push(1);
  q.Push(2);

  int val = 0;
  EXPECT_TRUE(q.Pop(val));
  EXPECT_EQ(val, 1);
  EXPECT_TRUE(q.Pop(val));
  EXPECT_EQ(val, 2);
  EXPECT_TRUE(q.Empty());
}

// Verifies blocking pop and stopping the queue unblocks it.
TEST(ThreadSafeQueueTest, BlockingPopAndStop) {
  ThreadSafeQueue<int> q;
  std::atomic<bool> popped{false};
  int popped_val = 0;

  std::thread pop_thread([&]() {
    if (q.Pop(popped_val)) {
      popped = true;
    }
  });

  // Give thread time to start and block
  absl::SleepFor(absl::Milliseconds(100));
  EXPECT_FALSE(popped);

  q.Push(42);
  pop_thread.join();

  EXPECT_TRUE(popped);
  EXPECT_EQ(popped_val, 42);
}

// Verifies stopping the queue unblocks waiting threads.
TEST(ThreadSafeQueueTest, StopUnblocks) {
  ThreadSafeQueue<int> q;
  std::atomic<bool> pop_result{true};

  std::thread pop_thread([&]() {
    int val = 0;
    pop_result = q.Pop(val);
  });

  absl::SleepFor(absl::Milliseconds(100));
  q.Stop();
  pop_thread.join();

  EXPECT_FALSE(pop_result);
}

// --- RemoteFetchInteractionTest ---

// Verifies controller registration with orchestrator.
TEST(RemoteFetchInteractionTest, ControllerRegistration) {
  int orch_port = net_util::PickUnusedPortOrDie();
  RaidenOrchestrator orchestrator(orch_port);

  std::string orch_addr = absl::StrCat("localhost:", orch_port);

  KVCacheStore store1(10, "", RaidenId{"job1", "0", "data1", 0});
  int ctrl_port1 = net_util::PickUnusedPortOrDie();
  int worker_port1 = net_util::PickUnusedPortOrDie();

  RaidenControllerEmbedded controller1(
      &store1, ctrl_port1, orch_addr, worker_port1, {"localhost:10000"},
      /*bytes_per_block=*/1024, /*num_shards=*/1);

  // Start controller (this should trigger registration)
  EXPECT_OK(controller1.Start());

  // Give time for async registration
  absl::SleepFor(absl::Milliseconds(500));

  // Verify registration in orchestrator by trying to resolve it.
  // We can use controller2 to resolve controller1.
  KVCacheStore store2(10, "", RaidenId{"job2", "0", "data2", 0});
  int ctrl_port2 = net_util::PickUnusedPortOrDie();
  int worker_port2 = net_util::PickUnusedPortOrDie();
  RaidenControllerEmbedded controller2(
      &store2, ctrl_port2, orch_addr, worker_port2, {"localhost:20000"},
      /*bytes_per_block=*/1024, /*num_shards=*/1);

  EXPECT_OK(controller2.Start());
  absl::SleepFor(absl::Milliseconds(500));

  // Now try to resolve store1's controller from controller2
  // We need to expose ResolveRemoteController or test it via the queue flow.
  // Since ResolveRemoteController is private/internal, testing via Queue is
  // better integration test.
}

// Verifies end-to-end fetch flow for a single block using PushFetchWork.
TEST(RemoteFetchInteractionTest, QueueFlow) {
  // Setup Orchestrator
  int orch_port = net_util::PickUnusedPortOrDie();
  RaidenOrchestrator orchestrator(orch_port);
  std::string orch_addr = absl::StrCat("localhost:", orch_port);

  // Setup Store 1 (Sender)
  RaidenId id1{"job1", "0", "data1", 0};
  KVCacheStore store1(10, "", id1);
  int ctrl_port1 = net_util::PickUnusedPortOrDie();
  int worker_data_port1 = net_util::PickUnusedPortOrDie();
  int worker_ctrl_port1 = net_util::PickUnusedPortOrDie();

  KVCacheManagerBase manager1(/*num_layers=*/1, /*num_shards=*/1,
                              /*slice_byte_size=*/1024, worker_data_port1,
                              /*host_blocks_to_allocate=*/10);
  KVCacheListener listener1(&manager1, worker_ctrl_port1);

  std::string worker_data_addr1 = absl::StrCat("localhost:", worker_data_port1);
  RaidenControllerEmbedded controller1(
      &store1, ctrl_port1, orch_addr, worker_ctrl_port1, {},
      /*bytes_per_block=*/1024, /*num_shards=*/1);
  EXPECT_OK(controller1.Start());

  // Setup Store 2 (Receiver/Fetcher)
  RaidenId id2{"job2", "0", "data2", 0};
  KVCacheStore store2(10, "", id2);
  int ctrl_port2 = net_util::PickUnusedPortOrDie();
  int worker_data_port2 = net_util::PickUnusedPortOrDie();
  int worker_ctrl_port2 = net_util::PickUnusedPortOrDie();

  KVCacheManagerBase manager2(/*num_layers=*/1, /*num_shards=*/1,
                              /*slice_byte_size=*/1024, worker_data_port2,
                              /*host_blocks_to_allocate=*/10);
  KVCacheListener listener2(&manager2, worker_ctrl_port2);

  std::string worker_data_addr2 = absl::StrCat("localhost:", worker_data_port2);
  RaidenControllerEmbedded controller2(
      &store2, ctrl_port2, orch_addr, worker_ctrl_port2, {},
      /*bytes_per_block=*/1024, /*num_shards=*/1);
  EXPECT_OK(controller2.Start());

  // Insert hash1 into store1 so negotiation finds it
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(id1, /*host_id=*/3, BlockStatus::HOST)};
  store1.Insert({"hash1"}, slices, /*on_host=*/true);

  // Fill Source Buffer with pattern
  uint8_t* src_ptr = manager1.GetHostPointer(0, 0);
  size_t block_size = manager1.bytes_per_block();
  std::memset(src_ptr + 3 * block_size, 'A', block_size);

  absl::SleepFor(absl::Milliseconds(500));  // Wait for registrations

  // Simulate KVCacheStore2 pushing work
  FetchRequest req;
  FetchRequestItem item;
  item.src_raiden_id = id1;  // Fetch from store1
  item.block_hashes.push_back("hash1");
  item.dst_block_ids.push_back(5);
  req.push_back(item);

  store2.PushFetchWork(req);

  // Wait for processing
  absl::SleepFor(
      absl::Seconds(2));  // Increased sleep to ensure transfer completes

  // Verify Destination Buffer
  uint8_t* dst_ptr = manager2.GetHostPointer(0, 0);
  for (size_t i = 0; i < block_size; ++i) {
    EXPECT_EQ(dst_ptr[5 * block_size + i], 'A') << "Mismatch at index " << i;
  }
}

// Verifies fetch flow with multiple layers and shards.
TEST(RemoteFetchInteractionTest, QueueFlowMultiLayerMultiShard) {
  // Setup Orchestrator
  int orch_port = net_util::PickUnusedPortOrDie();
  RaidenOrchestrator orchestrator(orch_port);
  std::string orch_addr = absl::StrCat("localhost:", orch_port);

  // Setup Store 1 (Sender)
  RaidenId id1{"job1", "0", "data1", 0};
  KVCacheStore store1(10, "", id1);
  int ctrl_port1 = net_util::PickUnusedPortOrDie();
  int worker_data_port1 = net_util::PickUnusedPortOrDie();
  int worker_ctrl_port1 = net_util::PickUnusedPortOrDie();

  KVCacheManagerBase manager1(/*num_layers=*/2, /*num_shards=*/2,
                              /*slice_byte_size=*/1024, worker_data_port1,
                              /*host_blocks_to_allocate=*/10);
  KVCacheListener listener1(&manager1, worker_ctrl_port1);

  std::string worker_data_addr1 = absl::StrCat("localhost:", worker_data_port1);
  // Pass same address for both shards
  RaidenControllerEmbedded controller1(
      &store1, ctrl_port1, orch_addr, worker_ctrl_port1,
      {worker_data_addr1, worker_data_addr1}, /*bytes_per_block=*/1024,
      /*num_shards=*/2);
  EXPECT_OK(controller1.Start());

  // Setup Store 2 (Receiver/Fetcher)
  RaidenId id2{"job2", "0", "data2", 0};
  KVCacheStore store2(10, "", id2);
  int ctrl_port2 = net_util::PickUnusedPortOrDie();
  int worker_data_port2 = net_util::PickUnusedPortOrDie();
  int worker_ctrl_port2 = net_util::PickUnusedPortOrDie();

  KVCacheManagerBase manager2(/*num_layers=*/2, /*num_shards=*/2,
                              /*slice_byte_size=*/1024, worker_data_port2,
                              /*host_blocks_to_allocate=*/10);
  KVCacheListener listener2(&manager2, worker_ctrl_port2);

  std::string worker_data_addr2 = absl::StrCat("localhost:", worker_data_port2);
  // Pass same address for both shards
  RaidenControllerEmbedded controller2(
      &store2, ctrl_port2, orch_addr, worker_ctrl_port2,
      {worker_data_addr2, worker_data_addr2}, /*bytes_per_block=*/1024,
      /*num_shards=*/2);
  EXPECT_OK(controller2.Start());

  // Insert hash1 into store1 so negotiation finds it
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(id1, /*host_id=*/3, BlockStatus::HOST)};
  store1.Insert({"hash1"}, slices, /*on_host=*/true);

  // Fill Source Buffer with pattern for all layers and shards
  size_t block_size = manager1.bytes_per_block();
  for (size_t l = 0; l < 2; ++l) {
    for (size_t sh = 0; sh < 2; ++sh) {
      uint8_t* src_ptr = manager1.GetHostPointer(l, sh);
      std::memset(src_ptr + 3 * block_size, 'A' + l * 2 + sh, block_size);
    }
  }

  absl::SleepFor(absl::Milliseconds(500));  // Wait for registrations

  // Simulate KVCacheStore2 pushing work
  FetchRequest req;
  FetchRequestItem item;
  item.src_raiden_id = id1;  // Fetch from store1
  item.block_hashes.push_back("hash1");
  item.dst_block_ids.push_back(5);
  req.push_back(item);

  store2.PushFetchWork(req);

  // Wait for processing
  absl::SleepFor(
      absl::Seconds(2));  // Increased sleep to ensure transfer completes

  // Verify Destination Buffer for all layers and shards
  for (size_t l = 0; l < 2; ++l) {
    for (size_t sh = 0; sh < 2; ++sh) {
      uint8_t* dst_ptr = manager2.GetHostPointer(l, sh);
      for (size_t i = 0; i < block_size; ++i) {
        EXPECT_EQ(dst_ptr[5 * block_size + i], 'A' + l * 2 + sh)
            << "Mismatch at layer " << l << " shard " << sh << " index " << i;
      }
    }
  }
}

// Verifies multiple concurrent fetch requests.
TEST(RemoteFetchInteractionTest, QueueFlowConcurrent) {
  // Setup Orchestrator
  int orch_port = net_util::PickUnusedPortOrDie();
  RaidenOrchestrator orchestrator(orch_port);
  std::string orch_addr = absl::StrCat("localhost:", orch_port);

  // Setup Store 1 (Sender)
  RaidenId id1{"job1", "0", "data1", 0};
  KVCacheStore store1(10, "", id1);
  int ctrl_port1 = net_util::PickUnusedPortOrDie();
  int worker_data_port1 = net_util::PickUnusedPortOrDie();
  int worker_ctrl_port1 = net_util::PickUnusedPortOrDie();

  KVCacheManagerBase manager1(/*num_layers=*/1, /*num_shards=*/1,
                              /*slice_byte_size=*/1024, worker_data_port1,
                              /*host_blocks_to_allocate=*/10);
  KVCacheListener listener1(&manager1, worker_ctrl_port1);

  std::string worker_data_addr1 = absl::StrCat("localhost:", worker_data_port1);
  RaidenControllerEmbedded controller1(
      &store1, ctrl_port1, orch_addr, worker_ctrl_port1, {worker_data_addr1},
      /*bytes_per_block=*/1024, /*num_shards=*/1);
  EXPECT_OK(controller1.Start());

  // Setup Store 2 (Receiver/Fetcher)
  RaidenId id2{"job2", "0", "data2", 0};
  KVCacheStore store2(10, "", id2);
  int ctrl_port2 = net_util::PickUnusedPortOrDie();
  int worker_data_port2 = net_util::PickUnusedPortOrDie();
  int worker_ctrl_port2 = net_util::PickUnusedPortOrDie();

  KVCacheManagerBase manager2(/*num_layers=*/1, /*num_shards=*/1,
                              /*slice_byte_size=*/1024, worker_data_port2,
                              /*host_blocks_to_allocate=*/10);
  KVCacheListener listener2(&manager2, worker_ctrl_port2);

  std::string worker_data_addr2 = absl::StrCat("localhost:", worker_data_port2);
  RaidenControllerEmbedded controller2(
      &store2, ctrl_port2, orch_addr, worker_ctrl_port2, {worker_data_addr2},
      /*bytes_per_block=*/1024, /*num_shards=*/1);
  EXPECT_OK(controller2.Start());

  // Insert hashes into store1
  store1.Insert({"hash1"}, {RaidenBlockID(id1, 3, BlockStatus::HOST)}, true);
  store1.Insert({"hash2"}, {RaidenBlockID(id1, 4, BlockStatus::HOST)}, true);

  // Fill Source Buffer with patterns
  uint8_t* src_ptr = manager1.GetHostPointer(0, 0);
  size_t block_size = manager1.bytes_per_block();
  std::memset(src_ptr + 3 * block_size, 'A', block_size);
  std::memset(src_ptr + 4 * block_size, 'B', block_size);

  absl::SleepFor(absl::Milliseconds(500));  // Wait for registrations

  // Simulate KVCacheStore2 pushing multiple works concurrently (batched)
  FetchRequest req;
  FetchRequestItem item;
  item.src_raiden_id = id1;
  item.block_hashes.push_back("hash1");
  item.dst_block_ids.push_back(5);
  item.block_hashes.push_back("hash2");
  item.dst_block_ids.push_back(6);
  req.push_back(item);

  store2.PushFetchWork(req);

  // Wait for processing
  absl::SleepFor(
      absl::Seconds(3));  // Give it more time for concurrent requests

  // Verify Destination Buffer
  uint8_t* dst_ptr = manager2.GetHostPointer(0, 0);
  for (size_t i = 0; i < block_size; ++i) {
    EXPECT_EQ(dst_ptr[5 * block_size + i], 'A')
        << "Mismatch at index " << i << " for hash1";
    EXPECT_EQ(dst_ptr[6 * block_size + i], 'B')
        << "Mismatch at index " << i << " for hash2";
  }
}

// Verifies high-level Fetch API with embedded controller, including multiple
// blocks, layers, and shards.
TEST(RemoteFetchInteractionTest, QueueFlowEmbedded) {
  // Setup Orchestrator
  int orch_port = net_util::PickUnusedPortOrDie();
  RaidenOrchestrator orchestrator(orch_port);
  std::string orch_addr = absl::StrCat("localhost:", orch_port);

  // Setup Store 1 (Sender)
  RaidenId id1{"job1", "0", "data1", 0};
  int ctrl_port1 = net_util::PickUnusedPortOrDie();
  int worker_data_port1 = net_util::PickUnusedPortOrDie();
  int worker_ctrl_port1 = net_util::PickUnusedPortOrDie();

  KVCacheManagerBase manager1(/*num_layers=*/2, /*num_shards=*/2,
                              /*slice_byte_size=*/1024, worker_data_port1,
                              /*host_blocks_to_allocate=*/10);
  KVCacheListener listener1(&manager1, worker_ctrl_port1, ctrl_port1);

  // Enable embedded controller for Store 1
  RemoteFetchConfig config1{
      .orchestrator_address = orch_addr,
      .controller_port = ctrl_port1,
      .local_worker_port = worker_ctrl_port1,
      .bytes_per_block = 1024,
      .num_shards = 2,
  };
  KVCacheStore store1(10, "", id1, config1);

  // Setup Store 2 (Receiver/Fetcher)
  RaidenId id2{"job2", "0", "data2", 0};
  int ctrl_port2 = net_util::PickUnusedPortOrDie();
  int worker_data_port2 = net_util::PickUnusedPortOrDie();
  int worker_ctrl_port2 = net_util::PickUnusedPortOrDie();

  KVCacheManagerBase manager2(/*num_layers=*/2, /*num_shards=*/2,
                              /*slice_byte_size=*/1024, worker_data_port2,
                              /*host_blocks_to_allocate=*/10);
  KVCacheListener listener2(&manager2, worker_ctrl_port2, ctrl_port2);

  // Enable embedded controller for Store 2
  RemoteFetchConfig config2{
      .orchestrator_address = orch_addr,
      .controller_port = ctrl_port2,
      .local_worker_port = worker_ctrl_port2,
      .bytes_per_block = 1024,
      .num_shards = 2,
  };
  KVCacheStore store2(10, "", id2, config2);

  // Insert hash1 and hash2 into store1 so negotiation finds them
  std::vector<RaidenBlockID> slices1_1 = {
      RaidenBlockID(id1, /*host_id=*/3, BlockStatus::HOST)};
  store1.Insert({"hash1"}, slices1_1, /*on_host=*/true);
  std::vector<RaidenBlockID> slices1_2 = {
      RaidenBlockID(id1, /*host_id=*/4, BlockStatus::HOST)};
  store1.Insert({"hash2"}, slices1_2, /*on_host=*/true);

  // Insert REMOTE placeholders into store2
  std::vector<RaidenBlockID> slices2_1 = {RaidenBlockID(
      id1, /*host_id=*/5, BlockStatus::REMOTE)};  // Destination block 5
  store2.Insert({"hash1"}, slices2_1, /*on_host=*/true);
  std::vector<RaidenBlockID> slices2_2 = {RaidenBlockID(
      id1, /*host_id=*/6, BlockStatus::REMOTE)};  // Destination block 6
  store2.Insert({"hash2"}, slices2_2, /*on_host=*/true);

  // Fill Source Buffer with pattern for all layers and shards
  size_t block_size = manager1.bytes_per_block();
  for (size_t l = 0; l < 2; ++l) {
    for (size_t sh = 0; sh < 2; ++sh) {
      uint8_t* src_ptr = manager1.GetHostPointer(l, sh);
      std::memset(src_ptr + 3 * block_size, 'A' + l * 2 + sh, block_size);
      std::memset(src_ptr + 4 * block_size, 'B' + l * 2 + sh, block_size);
    }
  }

  absl::SleepFor(absl::Milliseconds(500));  // Wait for registrations

  // Trigger Fetch
  auto futures = store2.FetchRemote({"hash1", "hash2"});
  ASSERT_EQ(futures.size(), 2);

  ASSERT_TRUE(futures.contains("hash1"));
  ASSERT_TRUE(futures.contains("hash2"));

  // Wait for completion using the futures
  EXPECT_OK(futures["hash1"].Await());
  EXPECT_OK(futures["hash2"].Await());

  // Verify Polling Status
  auto [done, failed, pending] = store2.PollFetchRemoteStatus();
  EXPECT_EQ(done.size(), 2);
  EXPECT_THAT(done, testing::UnorderedElementsAre("hash1", "hash2"));
  EXPECT_EQ(failed.size(), 0);
  EXPECT_EQ(pending.size(), 0);

  // Verify Destination Buffer for all layers and shards
  for (size_t l = 0; l < 2; ++l) {
    for (size_t sh = 0; sh < 2; ++sh) {
      uint8_t* dst_ptr = manager2.GetHostPointer(l, sh);
      for (size_t i = 0; i < block_size; ++i) {
        EXPECT_EQ(dst_ptr[5 * block_size + i], 'A' + l * 2 + sh)
            << "Mismatch at layer " << l << " shard " << sh << " index " << i
            << " for hash1";
        EXPECT_EQ(dst_ptr[6 * block_size + i], 'B' + l * 2 + sh)
            << "Mismatch at layer " << l << " shard " << sh << " index " << i
            << " for hash2";
      }
    }
  }

  // Verify LRU Status Upgrade in Store 2 for hash1
  auto lookup_res1 = store2.Lookup({"hash1"});
  ASSERT_TRUE(lookup_res1.ok());
  ASSERT_EQ(lookup_res1->size(), 1);
  const auto& block_list1 = (*lookup_res1)[0].second;
  EXPECT_EQ(block_list1.status, BlockStatus::HOST);
  EXPECT_EQ(block_list1.host_block_id, 5);
  EXPECT_EQ(block_list1.raiden_id.job_name, "job2");

  // Verify LRU Status Upgrade in Store 2 for hash2
  auto lookup_res2 = store2.Lookup({"hash2"});
  ASSERT_TRUE(lookup_res2.ok());
  ASSERT_EQ(lookup_res2->size(), 1);
  const auto& block_list2 = (*lookup_res2)[0].second;
  EXPECT_EQ(block_list2.status, BlockStatus::HOST);
  EXPECT_EQ(block_list2.host_block_id, 6);
  EXPECT_EQ(block_list2.raiden_id.job_name, "job2");
}
// Verifies the complete flow: Lookup (local + global), InsertAndPin (allocating
// local blocks for remote placeholders), and Fetch.
TEST(RemoteFetchInteractionTest, QueueFlowEndToEnd) {
  // 1. Setup Orchestrator
  int orch_port = net_util::PickUnusedPortOrDie();
  RaidenOrchestrator orchestrator(orch_port);
  std::string orch_addr = absl::StrCat("localhost:", orch_port);

  // 2. Setup Global Registry Server
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder builder;
  int registry_port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &registry_port);
  builder.RegisterService(service.get());
  auto registry_server = builder.BuildAndStart();
  std::string registry_addr = "localhost:" + std::to_string(registry_port);

  // 3. Setup Store 1 (Remote Sender)
  // MUST use "kv_cache" as data_name and 0 as data_replica_idx to match Lookup
  // hardcoding
  RaidenId id1{"job1", "0", "kv_cache", 0};
  int ctrl_port1 = net_util::PickUnusedPortOrDie();
  int worker_data_port1 = net_util::PickUnusedPortOrDie();
  int worker_ctrl_port1 = net_util::PickUnusedPortOrDie();

  KVCacheManagerBase manager1(/*num_layers=*/1, /*num_shards=*/1,
                              /*slice_byte_size=*/1024, worker_data_port1,
                              /*host_blocks_to_allocate=*/10);
  KVCacheListener listener1(&manager1, worker_ctrl_port1, ctrl_port1);

  RemoteFetchConfig config1{
      .orchestrator_address = orch_addr,
      .controller_port = ctrl_port1,
      .local_worker_port = worker_ctrl_port1,
      .bytes_per_block = 1024,
      .num_shards = 1,
  };
  KVCacheStore store1(10, registry_addr, id1, config1);

  // 4. Setup Store 2 (Local Receiver/Fetcher)
  RaidenId id2{"job2", "0", "data2", 0};
  int ctrl_port2 = net_util::PickUnusedPortOrDie();
  int worker_data_port2 = net_util::PickUnusedPortOrDie();
  int worker_ctrl_port2 = net_util::PickUnusedPortOrDie();

  KVCacheManagerBase manager2(/*num_layers=*/1, /*num_shards=*/1,
                              /*slice_byte_size=*/1024, worker_data_port2,
                              /*host_blocks_to_allocate=*/10);
  KVCacheListener listener2(&manager2, worker_ctrl_port2, ctrl_port2);

  RemoteFetchConfig config2{
      .orchestrator_address = orch_addr,
      .controller_port = ctrl_port2,
      .local_worker_port = worker_ctrl_port2,
      .bytes_per_block = 1024,
      .num_shards = 1,
  };
  KVCacheStore store2(10, registry_addr, id2, config2);

  // 5. Insert blocks into store1 and fill data
  store1.Insert({"remote_hash1"}, {RaidenBlockID(id1, 1, BlockStatus::HOST)},
                true);
  store1.Insert({"remote_hash2"}, {RaidenBlockID(id1, 2, BlockStatus::HOST)},
                true);
  store1.Insert({"remote_hash3"}, {RaidenBlockID(id1, 3, BlockStatus::HOST)},
                true);
  store1.Insert({"remote_hash4"}, {RaidenBlockID(id1, 4, BlockStatus::HOST)},
                true);

  size_t block_size = manager1.bytes_per_block();
  uint8_t* src_ptr = manager1.GetHostPointer(0, 0);
  std::memset(src_ptr + 1 * block_size, 'A', block_size);
  std::memset(src_ptr + 2 * block_size, 'B', block_size);
  std::memset(src_ptr + 3 * block_size, 'C', block_size);
  std::memset(src_ptr + 4 * block_size, 'D', block_size);

  // 6. Register Store 1 blocks in Global Registry
  auto channel =
      grpc::CreateChannel(registry_addr, grpc::InsecureChannelCredentials());
  global_registry::GlobalRegistryClient registry_client(channel);

  // Register Store 1 blocks in Global Registry with its RaidenId
  std::vector<global_registry::Registration> regs = {
      {"remote_hash1", id1, 1},
      {"remote_hash2", id1, 2},
      {"remote_hash3", id1, 3},
      {"remote_hash4", id1, 4},
  };
  ASSERT_OK(registry_client.Register(regs));

  // 7. Insert local blocks into store2
  store2.Insert({"local_hash1"}, {RaidenBlockID(id2, 7, BlockStatus::HOST)},
                true);
  store2.Insert({"local_hash2"}, {RaidenBlockID(id2, 8, BlockStatus::HOST)},
                true);

  absl::SleepFor(absl::Milliseconds(500));  // Wait for registrations

  // 8. Lookup multiple hashes in store2
  std::vector<std::string> lookup_hashes = {"local_hash1",  "local_hash2",
                                            "remote_hash1", "remote_hash2",
                                            "remote_hash3", "remote_hash4"};

  auto lookup_res_or = store2.Lookup(lookup_hashes, /*enable_global=*/true);
  ASSERT_TRUE(lookup_res_or.ok());
  auto lookup_res = lookup_res_or.value();
  ASSERT_EQ(lookup_res.size(), 6);

  // 9. Prepare InsertAndPin arguments
  std::vector<std::string> pin_hashes;
  std::vector<RaidenBlockID> pin_slices;

  // Use hardcoded destination block IDs as in other tests
  // Must NOT conflict with local blocks 7 and 8 in store2
  std::vector<int> allocated_blocks = {1, 2, 3, 4};

  size_t remote_count = 0;
  for (auto& pair : lookup_res) {
    pin_hashes.push_back(pair.first);
    if (pair.second.status == BlockStatus::REMOTE) {
      // Assign allocated local block ID
      pair.second.host_block_id = allocated_blocks[remote_count++];
    }
    pin_slices.push_back(pair.second);
  }

  // 10. Call InsertAndPin
  auto pin_res = store2.InsertAndPin(pin_hashes, pin_slices, true);
  EXPECT_TRUE(pin_res.first);
  EXPECT_TRUE(pin_res.second.empty());  // No evictions expected

  // Verify pins
  for (const auto& hash : pin_hashes) {
    EXPECT_EQ(store2.GetPinCount(hash), 1);
  }

  // 11. Call Fetch for remote hashes
  std::vector<std::string> remote_hashes = {"remote_hash1", "remote_hash2",
                                            "remote_hash3", "remote_hash4"};
  auto futures = store2.FetchRemote(remote_hashes);
  ASSERT_EQ(futures.size(), 4);

  // 12. Wait for completion
  for (const auto& hash : remote_hashes) {
    ASSERT_TRUE(futures.contains(hash));
    EXPECT_OK(futures[hash].Await());
  }

  // 13. Verify Destination Buffer
  uint8_t* dst_ptr = manager2.GetHostPointer(0, 0);
  EXPECT_EQ(dst_ptr[allocated_blocks[0] * block_size], 'A');
  EXPECT_EQ(dst_ptr[allocated_blocks[1] * block_size], 'B');
  EXPECT_EQ(dst_ptr[allocated_blocks[2] * block_size], 'C');
  EXPECT_EQ(dst_ptr[allocated_blocks[3] * block_size], 'D');

  // 14. Verify Status Upgrade in Store 2
  for (size_t i = 0; i < 4; ++i) {
    auto lookup_res_after = store2.Lookup({remote_hashes[i]});
    ASSERT_TRUE(lookup_res_after.ok());
    ASSERT_EQ(lookup_res_after->size(), 1);
    const auto& block_list = (*lookup_res_after)[0].second;
    EXPECT_EQ(block_list.status, BlockStatus::HOST);
    EXPECT_EQ(block_list.host_block_id, allocated_blocks[i]);
  }

  // 15. Verify Async Write-Through in Global Registry
  // Wait for async write-through to complete (should be quick)
  absl::SleepFor(absl::Milliseconds(500));

  for (size_t i = 0; i < 4; ++i) {
    const auto& hash = remote_hashes[i];
    bool found_id1 = false;
    bool found_id2 = false;
    for (int attempt = 0; attempt < 10; ++attempt) {
      auto lookup_res_registry = registry_client.Lookup({hash});
      ASSERT_TRUE(lookup_res_registry.ok());
      ASSERT_EQ(lookup_res_registry->size(), 1);
      const auto& proto_id = (*lookup_res_registry)[0].raiden_id();

      if (proto_id.job_name() == id1.job_name &&
          proto_id.job_replica_id() == id1.job_replica_id &&
          proto_id.data_name() == id1.data_name &&
          proto_id.data_replica_idx() == id1.data_replica_idx) {
        found_id1 = true;
      } else if (proto_id.job_name() == id2.job_name &&
                 proto_id.job_replica_id() == id2.job_replica_id &&
                 proto_id.data_name() == id2.data_name &&
                 proto_id.data_replica_idx() == id2.data_replica_idx) {
        found_id2 = true;
      }
      if (found_id1 && found_id2) break;
    }
    EXPECT_TRUE(found_id1) << "Original owner id1 not found in registry for "
                           << hash;
    EXPECT_TRUE(found_id2)
        << "New owner id2 (write-through) not found in registry for " << hash;
  }

  registry_server->Shutdown();
}

// Verifies the remote fetch flow where both src and dst have multiple
// listeners.
TEST(RemoteFetchInteractionTest, QueueFlowMultiListenerEndToEnd) {
  // 1. Setup Orchestrator
  int orch_port = net_util::PickUnusedPortOrDie();
  RaidenOrchestrator orchestrator(orch_port);
  std::string orch_addr = absl::StrCat("localhost:", orch_port);

  // 2. Setup Global Registry Server
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder builder;
  int registry_port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &registry_port);
  builder.RegisterService(service.get());
  auto registry_server = builder.BuildAndStart();
  std::string registry_addr = "localhost:" + std::to_string(registry_port);

  // 3. Setup Store 1 (Remote Sender) with 2 listeners
  RaidenId id1{"job1", "0", "kv_cache", 0};
  int ctrl_port1 = net_util::PickUnusedPortOrDie();
  int worker_data_port1a = net_util::PickUnusedPortOrDie();
  int worker_data_port1b = net_util::PickUnusedPortOrDie();
  int worker_ctrl_port1_base = PickConsecutiveUnusedPortsOrDie(2);
  int worker_ctrl_port1_b = worker_ctrl_port1_base + 1;

  // Each manager handles 1 shard
  KVCacheManagerBase manager1a(/*num_layers=*/1, /*num_shards=*/1,
                               /*slice_byte_size=*/1024, worker_data_port1a,
                               /*host_blocks_to_allocate=*/10);
  KVCacheManagerBase manager1b(/*num_layers=*/1, /*num_shards=*/1,
                               /*slice_byte_size=*/1024, worker_data_port1b,
                               /*host_blocks_to_allocate=*/10);

  KVCacheListener listener1a(&manager1a, worker_ctrl_port1_base, ctrl_port1);
  KVCacheListener listener1b(&manager1b, worker_ctrl_port1_b, ctrl_port1);

  RemoteFetchConfig config1{
      .orchestrator_address = orch_addr,
      .controller_port = ctrl_port1,
      .local_worker_port = worker_ctrl_port1_base,
      .bytes_per_block = 1024,
      .num_shards = 2,  // Total shards
      .num_listeners = 2,
  };
  KVCacheStore store1(10, registry_addr, id1, config1);

  // 4. Setup Store 2 (Local Receiver/Fetcher) with 2 listeners
  RaidenId id2{"job2", "0", "data2", 0};
  int ctrl_port2 = net_util::PickUnusedPortOrDie();
  int worker_data_port2a = net_util::PickUnusedPortOrDie();
  int worker_data_port2b = net_util::PickUnusedPortOrDie();
  int worker_ctrl_port2_base = PickConsecutiveUnusedPortsOrDie(2);
  int worker_ctrl_port2_b = worker_ctrl_port2_base + 1;

  KVCacheManagerBase manager2a(/*num_layers=*/1, /*num_shards=*/1,
                               /*slice_byte_size=*/1024, worker_data_port2a,
                               /*host_blocks_to_allocate=*/10);
  KVCacheManagerBase manager2b(/*num_layers=*/1, /*num_shards=*/1,
                               /*slice_byte_size=*/1024, worker_data_port2b,
                               /*host_blocks_to_allocate=*/10);

  KVCacheListener listener2a(&manager2a, worker_ctrl_port2_base, ctrl_port2);
  KVCacheListener listener2b(&manager2b, worker_ctrl_port2_b, ctrl_port2);

  RemoteFetchConfig config2{
      .orchestrator_address = orch_addr,
      .controller_port = ctrl_port2,
      .local_worker_port = worker_ctrl_port2_base,
      .bytes_per_block = 1024,
      .num_shards = 2,  // Total shards
      .num_listeners = 2,
  };
  KVCacheStore store2(10, registry_addr, id2, config2);

  // 5. Insert blocks into store1 and fill data
  // For simplicity, let's insert into both sub-managers if needed, or just test
  // routing. Wait, store1.Insert just records metadata in store1. The actual
  // data is in managers.
  store1.Insert({"remote_hash1"}, {RaidenBlockID(id1, 1, BlockStatus::HOST)},
                true);

  size_t block_size = 1024;
  // Fill data in manager1a (Shard 0)
  uint8_t* src_ptr_a = manager1a.GetHostPointer(0, 0);
  std::memset(src_ptr_a + 1 * block_size, 'A', block_size);

  // Fill data in manager1b (Shard 1)
  uint8_t* src_ptr_b = manager1b.GetHostPointer(0, 0);
  std::memset(src_ptr_b + 1 * block_size, 'B', block_size);

  // 6. Register Store 1 blocks in Global Registry
  auto channel =
      grpc::CreateChannel(registry_addr, grpc::InsecureChannelCredentials());
  global_registry::GlobalRegistryClient registry_client(channel);

  std::vector<global_registry::Registration> regs = {
      {"remote_hash1", id1, 1},
  };
  ASSERT_OK(registry_client.Register(regs));

  absl::SleepFor(absl::Milliseconds(500));  // Wait for registrations

  // 7. Lookup in store2
  auto lookup_res_or = store2.Lookup({"remote_hash1"}, /*enable_global=*/true);
  ASSERT_TRUE(lookup_res_or.ok());
  auto lookup_res = lookup_res_or.value();
  ASSERT_EQ(lookup_res.size(), 1);

  // 8. InsertAndPin in store2
  std::vector<std::string> pin_hashes = {"remote_hash1"};
  std::vector<RaidenBlockID> pin_slices;
  std::vector<int> allocated_blocks = {1};  // Local ID 1 in store2

  auto& pair = lookup_res[0];
  if (pair.second.status == BlockStatus::REMOTE) {
    pair.second.host_block_id = allocated_blocks[0];
  }
  pin_slices.push_back(pair.second);

  auto pin_res = store2.InsertAndPin(pin_hashes, pin_slices, true);
  EXPECT_TRUE(pin_res.first);

  // 9. Fetch Remote
  auto futures = store2.FetchRemote(pin_hashes);
  ASSERT_EQ(futures.size(), 1);

  // 10. Wait for completion
  // This might hang if there are bugs in multi-listener support!
  EXPECT_OK(futures["remote_hash1"].Await());

  // 11. Verify Destination Buffers
  uint8_t* dst_ptr_a = manager2a.GetHostPointer(0, 0);
  EXPECT_EQ(dst_ptr_a[allocated_blocks[0] * block_size], 'A');

  uint8_t* dst_ptr_b = manager2b.GetHostPointer(0, 0);
  EXPECT_EQ(dst_ptr_b[allocated_blocks[0] * block_size], 'B');

  registry_server->Shutdown();
}

// Verifies end-to-end fetch flow where blocks belong to multiple different
// remote RaidenIDs. This tests that requests are correctly grouped by
// destination and dispatched.
TEST(RemoteFetchInteractionTest, QueueFlowMultiRemoteEndToEnd) {
  // 1. Setup Orchestrator
  int orch_port = net_util::PickUnusedPortOrDie();
  RaidenOrchestrator orchestrator(orch_port);
  std::string orch_addr = absl::StrCat("localhost:", orch_port);

  // 2. Setup Global Registry Server
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder builder;
  int registry_port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &registry_port);
  builder.RegisterService(service.get());
  auto registry_server = builder.BuildAndStart();
  std::string registry_addr = "localhost:" + std::to_string(registry_port);

  // 3. Setup Store 1 (Remote Sender 1)
  RaidenId id1{"job1", "0", "kv_cache", 0};
  int ctrl_port1 = net_util::PickUnusedPortOrDie();
  int worker_data_port1 = net_util::PickUnusedPortOrDie();
  int worker_ctrl_port1 = net_util::PickUnusedPortOrDie();

  KVCacheManagerBase manager1(/*num_layers=*/1, /*num_shards=*/1,
                              /*slice_byte_size=*/1024, worker_data_port1,
                              /*host_blocks_to_allocate=*/10);
  KVCacheListener listener1(&manager1, worker_ctrl_port1, ctrl_port1);

  RemoteFetchConfig config1{
      .orchestrator_address = orch_addr,
      .controller_port = ctrl_port1,
      .local_worker_port = worker_ctrl_port1,
      .bytes_per_block = 1024,
      .num_shards = 1,
  };
  KVCacheStore store1(10, registry_addr, id1, config1);

  // 4. Setup Store 3 (Remote Sender 2)
  RaidenId id3{"job3", "0", "kv_cache", 0};
  int ctrl_port3 = net_util::PickUnusedPortOrDie();
  int worker_data_port3 = net_util::PickUnusedPortOrDie();
  int worker_ctrl_port3 = net_util::PickUnusedPortOrDie();

  KVCacheManagerBase manager3(/*num_layers=*/1, /*num_shards=*/1,
                              /*slice_byte_size=*/1024, worker_data_port3,
                              /*host_blocks_to_allocate=*/10);
  KVCacheListener listener3(&manager3, worker_ctrl_port3, ctrl_port3);

  RemoteFetchConfig config3{
      .orchestrator_address = orch_addr,
      .controller_port = ctrl_port3,
      .local_worker_port = worker_ctrl_port3,
      .bytes_per_block = 1024,
      .num_shards = 1,
  };
  KVCacheStore store3(10, registry_addr, id3, config3);

  // 5. Setup Store 2 (Local Receiver/Fetcher)
  RaidenId id2{"job2", "0", "data2", 0};
  int ctrl_port2 = net_util::PickUnusedPortOrDie();
  int worker_data_port2 = net_util::PickUnusedPortOrDie();
  int worker_ctrl_port2 = net_util::PickUnusedPortOrDie();

  KVCacheManagerBase manager2(/*num_layers=*/1, /*num_shards=*/1,
                              /*slice_byte_size=*/1024, worker_data_port2,
                              /*host_blocks_to_allocate=*/10);
  KVCacheListener listener2(&manager2, worker_ctrl_port2, ctrl_port2);

  RemoteFetchConfig config2{
      .orchestrator_address = orch_addr,
      .controller_port = ctrl_port2,
      .local_worker_port = worker_ctrl_port2,
      .bytes_per_block = 1024,
      .num_shards = 1,
  };
  KVCacheStore store2(10, registry_addr, id2, config2);

  // 6. Insert blocks into store1 and store3 and fill data
  store1.Insert({"remote_hash1"}, {RaidenBlockID(id1, 1, BlockStatus::HOST)},
                true);
  store1.Insert({"remote_hash2"}, {RaidenBlockID(id1, 2, BlockStatus::HOST)},
                true);
  store3.Insert({"remote_hash3"}, {RaidenBlockID(id3, 3, BlockStatus::HOST)},
                true);
  store3.Insert({"remote_hash4"}, {RaidenBlockID(id3, 4, BlockStatus::HOST)},
                true);

  size_t block_size = manager1.bytes_per_block();

  // Fill Store 1 data
  uint8_t* src_ptr1 = manager1.GetHostPointer(0, 0);
  std::memset(src_ptr1 + 1 * block_size, 'A', block_size);
  std::memset(src_ptr1 + 2 * block_size, 'B', block_size);

  // Fill Store 3 data
  uint8_t* src_ptr3 = manager3.GetHostPointer(0, 0);
  std::memset(src_ptr3 + 3 * block_size, 'C', block_size);
  std::memset(src_ptr3 + 4 * block_size, 'D', block_size);

  // 7. Register blocks in Global Registry
  auto channel =
      grpc::CreateChannel(registry_addr, grpc::InsecureChannelCredentials());
  global_registry::GlobalRegistryClient registry_client(channel);

  std::vector<global_registry::Registration> regs = {
      {"remote_hash1", id1, 1},
      {"remote_hash2", id1, 2},
      {"remote_hash3", id3, 3},
      {"remote_hash4", id3, 4},
  };
  ASSERT_OK(registry_client.Register(regs));

  absl::SleepFor(absl::Milliseconds(500));  // Wait for registrations

  // 8. Lookup multiple hashes in store2
  std::vector<std::string> lookup_hashes = {"remote_hash1", "remote_hash2",
                                            "remote_hash3", "remote_hash4"};

  auto lookup_res_or = store2.Lookup(lookup_hashes, /*enable_global=*/true);
  ASSERT_TRUE(lookup_res_or.ok());
  auto lookup_res = lookup_res_or.value();
  ASSERT_EQ(lookup_res.size(), 4);

  // 9. Prepare InsertAndPin arguments
  std::vector<std::string> pin_hashes;
  std::vector<RaidenBlockID> pin_slices;

  std::vector<int> allocated_blocks = {1, 2, 3, 4};

  size_t remote_count = 0;
  for (auto& pair : lookup_res) {
    pin_hashes.push_back(pair.first);
    if (pair.second.status == BlockStatus::REMOTE) {
      pair.second.host_block_id = allocated_blocks[remote_count++];
    }
    pin_slices.push_back(pair.second);
  }

  // 10. Call InsertAndPin
  auto pin_res = store2.InsertAndPin(pin_hashes, pin_slices, true);
  EXPECT_TRUE(pin_res.first);
  EXPECT_TRUE(pin_res.second.empty());

  // 11. Call Fetch for remote hashes
  auto futures = store2.FetchRemote(lookup_hashes);
  ASSERT_EQ(futures.size(), 4);

  // 12. Wait for completion
  for (const auto& hash : lookup_hashes) {
    ASSERT_TRUE(futures.contains(hash));
    EXPECT_OK(futures[hash].Await());
  }

  // 13. Verify Destination Buffer
  uint8_t* dst_ptr = manager2.GetHostPointer(0, 0);
  EXPECT_EQ(dst_ptr[allocated_blocks[0] * block_size], 'A');
  EXPECT_EQ(dst_ptr[allocated_blocks[1] * block_size], 'B');
  EXPECT_EQ(dst_ptr[allocated_blocks[2] * block_size], 'C');
  EXPECT_EQ(dst_ptr[allocated_blocks[3] * block_size], 'D');

  // 14. Verify Status Upgrade in Store 2
  for (size_t i = 0; i < 4; ++i) {
    auto lookup_res_after = store2.Lookup({lookup_hashes[i]});
    ASSERT_TRUE(lookup_res_after.ok());
    ASSERT_EQ(lookup_res_after->size(), 1);
    const auto& block_list = (*lookup_res_after)[0].second;
    EXPECT_EQ(block_list.status, BlockStatus::HOST);
    EXPECT_EQ(block_list.host_block_id, allocated_blocks[i]);
  }

  registry_server->Shutdown();
}

class MockKVCacheManager : public KVCacheManagerBase {
 public:
  using KVCacheManagerBase::KVCacheManagerBase;

  absl::StatusOr<raiden::PjRtCopyFuture> H2d(
      const std::vector<int64_t>& src_offsets_major_dim,
      const std::vector<int64_t>& dst_offsets_major_dim,
      const std::vector<int64_t>& copy_sizes_major_dim,
      std::optional<int64_t> slot_idx, std::optional<size_t> target_layer_idx,
      std::optional<size_t> target_shard_idx) override {
    absl::MutexLock lock(&mu_);
    h2d_calls_++;
    last_src_offsets_ = src_offsets_major_dim;
    last_dst_offsets_ = dst_offsets_major_dim;
    last_target_shard_ = target_shard_idx;

    return raiden::PjRtCopyFuture(xla::Future<>(absl::OkStatus()), {});
  }

  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
  D2hAutoAllocate(const std::vector<int64_t>& src_offsets_major_dim,
                  const std::vector<int64_t>& copy_sizes_major_dim) override {
    absl::MutexLock lock(&mu_);
    d2h_auto_alloc_calls_++;
    last_d2h_src_offsets_ = src_offsets_major_dim;

    std::vector<int> allocated_ids;
    for (size_t i = 0; i < src_offsets_major_dim.size(); ++i) {
      allocated_ids.push_back(next_allocated_host_id_++);
    }
    last_d2h_allocated_ids_ = allocated_ids;

    return std::make_pair(
        allocated_ids,
        raiden::PjRtCopyFuture(xla::Future<>(absl::OkStatus()), {}));
  }

  absl::Status UnlockBlocks(const std::vector<int>& block_ids) override {
    absl::MutexLock lock(&mu_);
    unlock_calls_++;
    last_unlocked_ids_ = block_ids;
    return absl::OkStatus();
  }

  int h2d_calls() const {
    absl::MutexLock lock(&mu_);
    return h2d_calls_;
  }

  std::vector<int64_t> last_src_offsets() const {
    absl::MutexLock lock(&mu_);
    return last_src_offsets_;
  }

  std::vector<int64_t> last_dst_offsets() const {
    absl::MutexLock lock(&mu_);
    return last_dst_offsets_;
  }

  std::optional<size_t> last_target_shard() const {
    absl::MutexLock lock(&mu_);
    return last_target_shard_;
  }

  int d2h_auto_alloc_calls() const {
    absl::MutexLock lock(&mu_);
    return d2h_auto_alloc_calls_;
  }

  std::vector<int64_t> last_d2h_src_offsets() const {
    absl::MutexLock lock(&mu_);
    return last_d2h_src_offsets_;
  }

  std::vector<int> last_d2h_allocated_ids() const {
    absl::MutexLock lock(&mu_);
    return last_d2h_allocated_ids_;
  }

  int unlock_calls() const {
    absl::MutexLock lock(&mu_);
    return unlock_calls_;
  }

  std::vector<int> last_unlocked_ids() const {
    absl::MutexLock lock(&mu_);
    return last_unlocked_ids_;
  }

 private:
  mutable absl::Mutex mu_;
  int h2d_calls_ = 0;
  std::vector<int64_t> last_src_offsets_;
  std::vector<int64_t> last_dst_offsets_;
  std::optional<size_t> last_target_shard_;

  int d2h_auto_alloc_calls_ = 0;
  std::vector<int64_t> last_d2h_src_offsets_;
  std::vector<int> last_d2h_allocated_ids_;
  int next_allocated_host_id_ = 100;
  int unlock_calls_ = 0;
  std::vector<int> last_unlocked_ids_;
};

TEST(RemoteFetchInteractionTest, QueueFlowLoad) {
  // Setup Store & Mock Manager
  RaidenId id{"job1", "0", "data1", 0};
  int ctrl_port = net_util::PickUnusedPortOrDie();
  int worker_data_port = net_util::PickUnusedPortOrDie();
  int worker_ctrl_port = net_util::PickUnusedPortOrDie();

  MockKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/2,
                             /*slice_byte_size=*/1024, worker_data_port,
                             /*host_blocks_to_allocate=*/10);
  KVCacheListener listener(&manager, worker_ctrl_port, ctrl_port);

  int orch_port = net_util::PickUnusedPortOrDie();
  RaidenOrchestrator orchestrator(orch_port);
  std::string orch_addr = absl::StrCat("localhost:", orch_port);

  RemoteFetchConfig config{
      .orchestrator_address = orch_addr,
      .controller_port = ctrl_port,
      .local_worker_port = worker_ctrl_port,
      .bytes_per_block = 1024,
      .num_shards = 2,
  };
  KVCacheStore store(10, "", id, config);

  // Insert blocks to DRAM (HOST status)
  std::vector<RaidenBlockID> slices_1 = {
      RaidenBlockID(id, /*host_id=*/3, BlockStatus::HOST)};
  store.Insert({"hash1"}, slices_1, /*on_host=*/true);

  std::vector<RaidenBlockID> slices_2 = {
      RaidenBlockID(id, /*host_id=*/4, BlockStatus::HOST)};
  store.Insert({"hash2"}, slices_2, /*on_host=*/true);

  absl::SleepFor(absl::Milliseconds(500));

  // Trigger Load to HBM block 5 and 6
  auto futures = store.Load({"hash1", "hash2"}, {5, 6});
  ASSERT_EQ(futures.size(), 2);
  ASSERT_TRUE(futures.contains("hash1"));
  ASSERT_TRUE(futures.contains("hash2"));

  // Wait for completion
  EXPECT_OK(futures["hash1"].Await());
  EXPECT_OK(futures["hash2"].Await());

  // Verify Polling Status
  auto [done, failed, pending] = store.PollLoadStatus();
  EXPECT_EQ(done.size(), 2);
  EXPECT_THAT(done, testing::UnorderedElementsAre("hash1", "hash2"));
  EXPECT_EQ(failed.size(), 0);
  EXPECT_EQ(pending.size(), 0);

  // Verify Mock Manager calls
  EXPECT_EQ(manager.h2d_calls(), 2);
  EXPECT_THAT(manager.last_src_offsets(), testing::ElementsAre(3, 4));
  EXPECT_THAT(manager.last_dst_offsets(), testing::ElementsAre(5, 6));

  // Verify LRU Status Upgrade in Store
  auto lookup_res1 = store.Lookup({"hash1"});
  ASSERT_TRUE(lookup_res1.ok());
  ASSERT_EQ(lookup_res1->size(), 1);
  const auto& block_list1 = (*lookup_res1)[0].second;
  EXPECT_EQ(block_list1.status, BlockStatus::HOST_AND_HBM);
  EXPECT_EQ(block_list1.hbm_block_id, 5);
  EXPECT_EQ(block_list1.host_block_id, 3);

  auto lookup_res2 = store.Lookup({"hash2"});
  ASSERT_TRUE(lookup_res2.ok());
  ASSERT_EQ(lookup_res2->size(), 1);
  const auto& block_list2 = (*lookup_res2)[0].second;
  EXPECT_EQ(block_list2.status, BlockStatus::HOST_AND_HBM);
  EXPECT_EQ(block_list2.hbm_block_id, 6);
  EXPECT_EQ(block_list2.host_block_id, 4);
}

TEST(RemoteFetchInteractionTest, QueueFlowSave) {
  // Setup Store & Mock Manager
  RaidenId id{"job1", "0", "data1", 0};
  int ctrl_port = net_util::PickUnusedPortOrDie();
  int worker_data_port = net_util::PickUnusedPortOrDie();
  int worker_ctrl_port = net_util::PickUnusedPortOrDie();

  MockKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/2,
                             /*slice_byte_size=*/1024, worker_data_port,
                             /*host_blocks_to_allocate=*/10);
  KVCacheListener listener(&manager, worker_ctrl_port, ctrl_port);

  int orch_port = net_util::PickUnusedPortOrDie();
  RaidenOrchestrator orchestrator(orch_port);
  std::string orch_addr = absl::StrCat("localhost:", orch_port);

  RemoteFetchConfig config{
      .orchestrator_address = orch_addr,
      .controller_port = ctrl_port,
      .local_worker_port = worker_ctrl_port,
      .bytes_per_block = 1024,
      .num_shards = 2,
  };
  KVCacheStore store(10, "", id, config);

  // Insert blocks to HBM (HBM status) and PIN them
  std::vector<RaidenBlockID> slices_1 = {
      RaidenBlockID(id, /*host_id=*/-1, /*hbm_id=*/3, BlockStatus::HBM)};
  store.Insert({"hash1"}, slices_1, /*on_host=*/false);
  ASSERT_TRUE(store.Pin({"hash1"}));

  std::vector<RaidenBlockID> slices_2 = {
      RaidenBlockID(id, /*host_id=*/-1, /*hbm_id=*/4, BlockStatus::HBM)};
  store.Insert({"hash2"}, slices_2, /*on_host=*/false);
  ASSERT_TRUE(store.Pin({"hash2"}));

  absl::SleepFor(absl::Milliseconds(500));

  // Trigger Save from HBM to DRAM
  auto futures = store.Save({"hash1", "hash2"}, {3, 4});
  ASSERT_EQ(futures.size(), 2);
  ASSERT_TRUE(futures.contains("hash1"));
  ASSERT_TRUE(futures.contains("hash2"));

  // Wait for completion
  EXPECT_OK(futures["hash1"].Await());
  EXPECT_OK(futures["hash2"].Await());

  // Verify Polling Status
  auto [done, failed, pending] = store.PollSaveStatus();
  EXPECT_EQ(done.size(), 2);
  EXPECT_THAT(done, testing::UnorderedElementsAre("hash1", "hash2"));
  EXPECT_EQ(failed.size(), 0);
  EXPECT_EQ(pending.size(), 0);

  // Verify Mock Manager calls
  EXPECT_EQ(manager.d2h_auto_alloc_calls(), 1);
  EXPECT_THAT(manager.last_d2h_src_offsets(), testing::ElementsAre(3, 4));
  EXPECT_THAT(manager.last_d2h_allocated_ids(), testing::ElementsAre(100, 101));

  // Verify LRU Status Upgrade in Store
  auto lookup_res1 = store.Lookup({"hash1"});
  ASSERT_TRUE(lookup_res1.ok());
  ASSERT_EQ(lookup_res1->size(), 1);
  const auto& block_list1 = (*lookup_res1)[0].second;
  EXPECT_EQ(block_list1.status, BlockStatus::HOST_AND_HBM);
  EXPECT_EQ(block_list1.host_block_id, 100);
  EXPECT_EQ(block_list1.hbm_block_id, 3);

  auto lookup_res2 = store.Lookup({"hash2"});
  ASSERT_TRUE(lookup_res2.ok());
  ASSERT_EQ(lookup_res2->size(), 1);
  const auto& block_list2 = (*lookup_res2)[0].second;
  EXPECT_EQ(block_list2.status, BlockStatus::HOST_AND_HBM);
  EXPECT_EQ(block_list2.host_block_id, 101);
  EXPECT_EQ(block_list2.hbm_block_id, 4);
}

TEST(RemoteFetchInteractionTest, QueueFlowEvict) {
  // 1. Start a local registry server
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder builder;
  int registry_port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &registry_port);
  builder.RegisterService(service.get());
  auto registry_server = builder.BuildAndStart();
  std::string registry_addr = "localhost:" + std::to_string(registry_port);

  // Setup Store & Mock Manager
  RaidenId id{"job1", "0", "data1", 0};
  int ctrl_port = net_util::PickUnusedPortOrDie();
  int worker_data_port = net_util::PickUnusedPortOrDie();
  int worker_ctrl_port = net_util::PickUnusedPortOrDie();

  MockKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/2,
                             /*slice_byte_size=*/1024, worker_data_port,
                             /*host_blocks_to_allocate=*/10);
  KVCacheListener listener(&manager, worker_ctrl_port, ctrl_port);

  int orch_port = net_util::PickUnusedPortOrDie();
  RaidenOrchestrator orchestrator(orch_port);
  std::string orch_addr = absl::StrCat("localhost:", orch_port);

  RemoteFetchConfig config{
      .orchestrator_address = orch_addr,
      .controller_port = ctrl_port,
      .local_worker_port = worker_ctrl_port,
      .bytes_per_block = 1024,
      .num_shards = 2,
  };
  KVCacheStore store(10, registry_addr, id, config);

  // 2. Insert blocks to store: hash1 as HOST, hash2 as HOST_AND_HBM
  std::vector<RaidenBlockID> slices_1 = {
      RaidenBlockID(id, /*host_id=*/3, BlockStatus::HOST)};
  store.Insert({"hash1"}, slices_1, /*on_host=*/true);

  std::vector<RaidenBlockID> slices_2 = {RaidenBlockID(
      id, /*host_id=*/4, /*hbm_id=*/0, BlockStatus::HOST_AND_HBM)};
  store.Insert({"hash2"}, slices_2, /*on_host=*/true);

  // Register hash2 in global registry to verify unregistration
  auto channel =
      grpc::CreateChannel(registry_addr, grpc::InsecureChannelCredentials());
  global_registry::GlobalRegistryClient registry_client(channel);
  ASSERT_TRUE(registry_client.Register({{"hash2", id, 4}}).ok());

  absl::SleepFor(absl::Milliseconds(500));

  // Verify hash2 is registered in global registry before evict
  {
    auto lookup_res = registry_client.Lookup({"hash2"});
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 1);
  }

  // 3. Trigger Evict
  auto futures = store.Evict({"hash1", "hash2"});
  ASSERT_EQ(futures.size(), 2);
  ASSERT_TRUE(futures.contains("hash1"));
  ASSERT_TRUE(futures.contains("hash2"));

  // Verify status changed IMMEDIATELY (HOST -> INIT, HOST_AND_HBM -> HBM)
  {
    auto lookup_res1 = store.Lookup({"hash1"});
    ASSERT_TRUE(lookup_res1.ok());
    ASSERT_EQ(lookup_res1->size(), 1);
    EXPECT_EQ((*lookup_res1)[0].second.status, BlockStatus::INIT);

    auto lookup_res2 = store.Lookup({"hash2"});
    ASSERT_TRUE(lookup_res2.ok());
    ASSERT_EQ(lookup_res2->size(), 1);
    EXPECT_EQ((*lookup_res2)[0].second.status, BlockStatus::HBM);
  }

  // 4. Wait for completion
  EXPECT_OK(futures["hash1"].Await());
  EXPECT_OK(futures["hash2"].Await());

  // Verify Polling Status
  auto [done, failed, pending] = store.PollEvictStatus();
  EXPECT_EQ(done.size(), 2);
  EXPECT_THAT(done, testing::UnorderedElementsAre("hash1", "hash2"));
  EXPECT_EQ(failed.size(), 0);
  EXPECT_EQ(pending.size(), 0);

  // Verify Mock Manager calls (UnlockBlocks should be called once with {3, 4})
  EXPECT_EQ(manager.unlock_calls(), 1);
  EXPECT_THAT(manager.last_unlocked_ids(), testing::UnorderedElementsAre(3, 4));

  // Verify final status in Store
  // hash1 (was HOST -> INIT) should be ERASED from LRU
  {
    auto lookup_res = store.Lookup({"hash1"});
    ASSERT_TRUE(lookup_res.ok());
    EXPECT_EQ(lookup_res->size(), 0);
  }

  // hash2 (was HOST_AND_HBM -> HBM) should remain in LRU as HBM with
  // host_block_id = -1
  {
    auto lookup_res = store.Lookup({"hash2"});
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 1);
    EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::HBM);
    EXPECT_EQ((*lookup_res)[0].second.host_block_id, -1);
  }

  // Verify unregistration in global registry (hash2 should not be found
  // anymore)
  {
    bool unregistered = false;
    for (int i = 0; i < 50; ++i) {
      auto lookup_res = registry_client.Lookup({"hash2"});
      ASSERT_TRUE(lookup_res.ok());
      if (lookup_res->size() == 0) {
        unregistered = true;
        break;
      }
      absl::SleepFor(absl::Milliseconds(10));
    }
    EXPECT_TRUE(unregistered);
  }

  registry_server->Shutdown();
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
