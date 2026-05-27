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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_H_

#include <cstdint>
#include <list>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include <nanobind/nanobind.h>
#include "kv_cache/kv_cache_manager.h"

namespace tpu_raiden {
namespace kv_cache {

class KVCacheStore {
 public:
  KVCacheStore(int block_size, int capacity);

  absl::StatusOr<std::pair<std::vector<bool>, KVCacheTransferFuture>>
  LookupAndFetch(const std::vector<uint64_t>& block_hashes,
                 nanobind::list device_arrays,
                 const std::vector<int>& dst_offsets_major_dim,
                 const std::vector<int>& copy_sizes_major_dim);

  absl::Status Insert(const std::vector<uint64_t>& block_hashes,
                      nanobind::list device_arrays,
                      const std::vector<int>& src_offsets_major_dim,
                      const std::vector<int>& copy_sizes_major_dim);

 private:
  struct CacheEntry {
    uint64_t block_hash;
    std::vector<int> internal_block_ids;
    std::shared_ptr<std::vector<std::vector<uint8_t>>> host_buffers;
    std::shared_ptr<KVCacheTransferFuture> insert_future;
  };

  absl::Mutex mutex_;

  int block_size_;
  int capacity_;

  std::list<CacheEntry> lru_list_;
  absl::flat_hash_map<uint64_t, std::list<CacheEntry>::iterator> cache_map_;
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif
