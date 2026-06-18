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

#include "tpu_raiden/kv_cache/kv_cache_store_internal.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/core/tpu_pjrt_manager.h"
#include "tpu_raiden/kv_cache/kv_cache_manager_base.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

using ::absl_testing::IsOk;

TEST(KVCacheStoreInternalTest, LocalInsertAndLookup) {
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * pjrt_manager,
                          TpuPjrtManager::GetDefault());

  // Rank 3 shape: {2, 32, 32} of float.
  std::vector<int64_t> shape_dims = {2, 32, 32};
  int64_t elements_per_slice = 32 * 32;
  int64_t total_elements = 2 * elements_per_slice;

  // Initialize buffer: slice 0 has 0, 1, 2... slice 1 has -1.0f
  std::vector<float> host_data(total_elements, -1.0f);
  for (int i = 0; i < elements_per_slice; ++i) {
    host_data[i] = static_cast<float>(i);
  }

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      pjrt_manager->BufferFromHost(host_data.data(), xla::F32, shape_dims));

  ASSERT_THAT(buffer->GetReadyFuture().Await(), IsOk());

  auto make_manager = [&](xla::PjRtBuffer* buf) {
    std::vector<std::vector<xla::PjRtBuffer*>> layer_buffers = {{buf}};
    return std::make_unique<kv_cache::KVCacheManagerBase>(
        layer_buffers, /*local_port=*/std::nullopt,
        /*host_blocks_to_allocate=*/4,
        /*unsafe_skip_buffer_lock=*/true);
  };

  // Create KVCacheStoreInternal (local only, capacity = 2)
  auto store =
      std::make_unique<KVCacheStoreInternal>(/*capacity=*/2);

  // 1. Insert slice 0 (hash 100) into store using a fresh manager
  auto kv_manager1 = make_manager(buffer.get());
  ASSERT_THAT(store->Insert(/*block_hashes=*/{100}, *kv_manager1,
                            /*src_offsets_major_dim=*/{0},
                            /*copy_sizes_major_dim=*/{1}),
              IsOk());

  // 2. Lookup hash 100 and fetch it into slice 1 using a fresh manager
  auto kv_manager2 = make_manager(buffer.get());
  TF_ASSERT_OK_AND_ASSIGN(
      auto lookup_result,
      store->LookupAndFetch(/*block_hashes=*/{100}, *kv_manager2,
                            /*dst_offsets_major_dim=*/{1},
                            /*copy_sizes_major_dim=*/{1}));

  std::vector<bool> hits = lookup_result.first;
  raiden::PjRtCopyFuture future = std::move(lookup_result.second);

  ASSERT_EQ(hits.size(), 1);
  EXPECT_TRUE(hits[0]);

  // Await the fetch to complete
  ASSERT_OK(future.Await());

  // 3. Verify buffer content: slice 1 should now match slice 0 (0, 1, 2...)
  TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer->ToLiteral().Await());
  auto read_back = literal->data<float>();
  ASSERT_EQ(read_back.size(), total_elements);

  for (int i = 0; i < elements_per_slice; ++i) {
    EXPECT_EQ(read_back[i], static_cast<float>(i));
  }
  for (int i = elements_per_slice; i < total_elements; ++i) {
    EXPECT_EQ(read_back[i], static_cast<float>(i - elements_per_slice));
  }
}

TEST(KVCacheStoreInternalTest, LruEviction) {
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * pjrt_manager,
                          TpuPjrtManager::GetDefault());

  std::vector<int64_t> shape_dims = {2, 32, 32};
  int64_t elements_per_slice = 32 * 32;
  int64_t total_elements = 2 * elements_per_slice;

  std::vector<float> host_data(total_elements, 1.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      pjrt_manager->BufferFromHost(host_data.data(), xla::F32, shape_dims));

  ASSERT_THAT(buffer->GetReadyFuture().Await(), IsOk());

  auto make_manager = [&](xla::PjRtBuffer* buf) {
    std::vector<std::vector<xla::PjRtBuffer*>> layer_buffers = {{buf}};
    return std::make_unique<kv_cache::KVCacheManagerBase>(
        layer_buffers, /*local_port=*/std::nullopt,
        /*host_blocks_to_allocate=*/4,
        /*unsafe_skip_buffer_lock=*/true);
  };

  // Capacity = 2
  auto store =
      std::make_unique<KVCacheStoreInternal>(/*capacity=*/2);

  // Insert hash 101 (slice 0)
  auto kv_manager1 = make_manager(buffer.get());
  ASSERT_THAT(store->Insert({101}, *kv_manager1, {0}, {1}), IsOk());

  // Insert hash 102 (slice 1)
  auto kv_manager2 = make_manager(buffer.get());
  ASSERT_THAT(store->Insert({102}, *kv_manager2, {1}, {1}), IsOk());

  // Both should be in cache
  {
    auto kv_manager3 = make_manager(buffer.get());
    TF_ASSERT_OK_AND_ASSIGN(
        auto res, store->LookupAndFetch({101}, *kv_manager3, {0}, {1}));
    EXPECT_TRUE(res.first[0]);
    ASSERT_OK(res.second.Await());
  }
  {
    auto kv_manager4 = make_manager(buffer.get());
    TF_ASSERT_OK_AND_ASSIGN(
        auto res, store->LookupAndFetch({102}, *kv_manager4, {1}, {1}));
    EXPECT_TRUE(res.first[0]);
    ASSERT_OK(res.second.Await());
  }

  // Insert hash 103 (slice 0) -> should evict the least recently used (101)
  auto kv_manager5 = make_manager(buffer.get());
  ASSERT_THAT(store->Insert({103}, *kv_manager5, {0}, {1}), IsOk());

  // 101 should be miss, 102 and 103 should be hit
  {
    auto kv_manager6 = make_manager(buffer.get());
    TF_ASSERT_OK_AND_ASSIGN(
        auto res, store->LookupAndFetch({101}, *kv_manager6, {0}, {1}));
    EXPECT_FALSE(res.first[0]);
  }
  {
    auto kv_manager7 = make_manager(buffer.get());
    TF_ASSERT_OK_AND_ASSIGN(
        auto res, store->LookupAndFetch({102}, *kv_manager7, {1}, {1}));
    EXPECT_TRUE(res.first[0]);
    ASSERT_OK(res.second.Await());
  }
  {
    auto kv_manager8 = make_manager(buffer.get());
    TF_ASSERT_OK_AND_ASSIGN(
        auto res, store->LookupAndFetch({103}, *kv_manager8, {0}, {1}));
    EXPECT_TRUE(res.first[0]);
    ASSERT_OK(res.second.Await());
  }
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
