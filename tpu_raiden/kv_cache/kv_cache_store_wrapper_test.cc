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

#include "tpu_raiden/kv_cache/kv_cache_store_wrapper.h"

#include <fcntl.h>
#include <netinet/in.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "tpu_raiden/kv_cache/kv_cache_store.h"
#include "tpu_raiden/kv_cache/raiden_id.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

using ::testing::IsEmpty;

// Drives the wrapper through the same environment contract the serving stack
// uses: RAIDEN_SHM_KEY turns the metadata table on, RAIDEN_SHM_MODEL_UID
// names its owner, and RAIDEN_DISABLE_SINGLETON_WORKER isolates each
// wrapper's embedded controller server so tests do not share state.
class KVCacheStoreWrapperTest : public ::testing::Test {
 protected:
  void SetUp() override {
    shm_key_ = absl::StrCat("raiden_store_wrapper_test_", getpid());
    UnlinkSegments();
    setenv("RAIDEN_SHM_KEY", shm_key_.c_str(), /*overwrite=*/1);
    setenv("RAIDEN_SHM_MODEL_UID", "model_a", /*overwrite=*/1);
    setenv("RAIDEN_DISABLE_SINGLETON_WORKER", "1", /*overwrite=*/1);
    unsetenv("RAIDEN_SHM_SERVER_NAME");
  }

  void TearDown() override {
    UnlinkSegments();
    unsetenv("RAIDEN_SHM_KEY");
    unsetenv("RAIDEN_SHM_MODEL_UID");
    unsetenv("RAIDEN_DISABLE_SINGLETON_WORKER");
    unsetenv("RAIDEN_SHM_SERVER_NAME");
  }

  std::unique_ptr<KVCacheStoreWrapper> MakeWrapper(size_t capacity,
                                                   int num_shards) {
    RaidenId rid{"wrapper_test_job", "0", "wrapper_test_cache", 0};
    return std::make_unique<KVCacheStoreWrapper>(
        capacity, /*global_registry_address=*/"", rid, num_shards,
        /*shard_size_bytes=*/512, /*raiden_orchestrator_address=*/"",
        /*store_server_ip=*/"");
  }

  bool MetadataSegmentExists(const std::string& suffix) {
    std::string name = absl::StrCat("/", shm_key_, suffix);
    int fd = shm_open(name.c_str(), O_RDONLY, 0);
    if (fd < 0) {
      return false;
    }
    close(fd);
    return true;
  }

  void UnlinkSegments() {
    shm_unlink(absl::StrCat("/", shm_key_, "_metadata").c_str());
    shm_unlink(absl::StrCat("/", shm_key_, "_metadata_test_server").c_str());
  }

  // Inserts `hashes` as host-resident blocks 0..n-1, mirroring them into the
  // metadata table.
  void InsertHostBlocks(KVCacheStoreWrapper& wrapper,
                        const std::vector<std::string>& hashes) {
    RaidenId rid{"wrapper_test_job", "0", "wrapper_test_cache", 0};
    std::vector<RaidenBlockID> slices;
    slices.reserve(hashes.size());
    for (int i = 0; i < static_cast<int>(hashes.size()); ++i) {
      slices.push_back(RaidenBlockID(rid, i, BlockStatus::HOST));
    }
    ASSERT_TRUE(wrapper->Insert(hashes, slices, /*on_host=*/true).first);
  }

  std::string shm_key_;
};

TEST_F(KVCacheStoreWrapperTest, NoShmKeySkipsMetadataTable) {
  unsetenv("RAIDEN_SHM_KEY");
  auto wrapper = MakeWrapper(/*capacity=*/4, /*num_shards=*/1);
  EXPECT_FALSE(MetadataSegmentExists("_metadata"));
  EXPECT_EQ((*wrapper)->capacity(), 4);
}

TEST_F(KVCacheStoreWrapperTest, NumShardsZeroSkipsMetadataTable) {
  auto wrapper = MakeWrapper(/*capacity=*/4, /*num_shards=*/0);
  EXPECT_FALSE(MetadataSegmentExists("_metadata"));
  EXPECT_EQ((*wrapper)->capacity(), 4);
}

TEST_F(KVCacheStoreWrapperTest, ColdStartCreatesMetadataTable) {
  auto wrapper = MakeWrapper(/*capacity=*/4, /*num_shards=*/1);
  EXPECT_TRUE(MetadataSegmentExists("_metadata"));
  auto lookup_or = (*wrapper)->Lookup({"host_1"});
  ASSERT_TRUE(lookup_or.ok());
  EXPECT_THAT(*lookup_or, IsEmpty());
}

TEST_F(KVCacheStoreWrapperTest, ServerNameSuffixesMetadataSegment) {
  setenv("RAIDEN_SHM_SERVER_NAME", "test_server", /*overwrite=*/1);
  auto wrapper = MakeWrapper(/*capacity=*/4, /*num_shards=*/1);
  EXPECT_TRUE(MetadataSegmentExists("_metadata_test_server"));
  EXPECT_FALSE(MetadataSegmentExists("_metadata"));
}

TEST_F(KVCacheStoreWrapperTest, RecoversHostBlocksAfterRestart) {
  auto wrapper = MakeWrapper(/*capacity=*/4, /*num_shards=*/1);
  InsertHostBlocks(*wrapper, {"host_1", "host_2"});

  // Destroying the wrapper simulates the engine dying: the mapping goes
  // away, the segment survives. The next incarnation must rebuild the LRU
  // cache from the surviving table during construction.
  wrapper.reset();
  wrapper = MakeWrapper(/*capacity=*/4, /*num_shards=*/1);

  auto lookup_or = (*wrapper)->Lookup({"host_1", "host_2"});
  ASSERT_TRUE(lookup_or.ok());
  ASSERT_EQ(lookup_or->size(), 2);
  EXPECT_EQ((*lookup_or)[0].first, "host_1");
  EXPECT_EQ((*lookup_or)[0].second.status, BlockStatus::HOST);
  EXPECT_EQ((*lookup_or)[0].second.host_block_id, 0);
  EXPECT_EQ((*lookup_or)[1].first, "host_2");
  EXPECT_EQ((*lookup_or)[1].second.status, BlockStatus::HOST);
  EXPECT_EQ((*lookup_or)[1].second.host_block_id, 1);
}

TEST_F(KVCacheStoreWrapperTest, ModelUidMismatchColdStarts) {
  auto wrapper = MakeWrapper(/*capacity=*/4, /*num_shards=*/1);
  InsertHostBlocks(*wrapper, {"host_1"});
  wrapper.reset();

  // A restart under another model must not resurrect the surviving table.
  setenv("RAIDEN_SHM_MODEL_UID", "model_b", /*overwrite=*/1);
  wrapper = MakeWrapper(/*capacity=*/4, /*num_shards=*/1);

  auto lookup_or = (*wrapper)->Lookup({"host_1"});
  ASSERT_TRUE(lookup_or.ok());
  EXPECT_THAT(*lookup_or, IsEmpty());
}

TEST_F(KVCacheStoreWrapperTest, ControllerBindFailureThrows) {
  unsetenv("RAIDEN_SHM_KEY");

  // Squat a port with a plain TCP socket (no SO_REUSEPORT), so the embedded
  // controller's gRPC bind on the same port must fail with EADDRINUSE.
  int squatter = socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(squatter, 0);
  sockaddr_in addr = {};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  ASSERT_EQ(bind(squatter, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)),
            0);
  ASSERT_EQ(listen(squatter, 1), 0);
  socklen_t addr_len = sizeof(addr);
  ASSERT_EQ(
      getsockname(squatter, reinterpret_cast<sockaddr*>(&addr), &addr_len), 0);
  const int occupied_port = ntohs(addr.sin_port);

  RaidenId rid{"wrapper_test_job", "0", "wrapper_test_cache", 0};
  EXPECT_THROW(KVCacheStoreWrapper(
                   /*lru_capacity=*/4, /*global_registry_address=*/"", rid,
                   /*num_shards=*/1, /*shard_size_bytes=*/512,
                   /*raiden_orchestrator_address=*/"",
                   /*store_server_ip=*/"127.0.0.1",
                   /*raiden_controller_port=*/occupied_port),
               std::runtime_error);
  close(squatter);
}

TEST_F(KVCacheStoreWrapperTest, NumRegisteredWorkers) {
  unsetenv("RAIDEN_SHM_KEY");
  // No controller (num_shards=0): the accessor reports zero workers.
  auto no_controller = MakeWrapper(/*capacity=*/4, /*num_shards=*/0);
  EXPECT_EQ((*no_controller)->num_registered_workers(), 0);
  // Controller up, nobody registered yet: still zero, and the gate a caller
  // builds on this accessor stays closed.
  auto with_controller = MakeWrapper(/*capacity=*/4, /*num_shards=*/1);
  EXPECT_EQ((*with_controller)->num_registered_workers(), 0);
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
