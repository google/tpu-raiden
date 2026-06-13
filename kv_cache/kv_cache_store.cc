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

#include "kv_cache/kv_cache_store.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "kv_cache/lru_cache.h"

namespace tpu_raiden {
namespace kv_cache {

KVCacheStore::KVCacheStore(size_t capacity) : lru_cache_(capacity) {}

KVCacheStore::~KVCacheStore() = default;

absl::StatusOr<std::vector<std::pair<int64_t, std::vector<RaidenId>>>>
KVCacheStore::Lookup(const std::vector<uint64_t>& block_hashes) {
  absl::MutexLock lock(mutex_);

  std::vector<std::pair<int64_t, std::vector<RaidenId>>> results;
  results.reserve(block_hashes.size());

  for (uint64_t hash : block_hashes) {
    std::vector<RaidenId>* existing = lru_cache_.Get(hash);
    if (!existing || existing->empty()) {
      break;
    }
    results.push_back(std::make_pair(hash, *existing));
  }

  return results;
}

bool KVCacheStore::Insert(const std::vector<uint64_t>& block_hashes,
                          const std::vector<std::vector<RaidenId>>& slices,
                          bool /*on_host*/) {
  absl::MutexLock lock(mutex_);

  bool all_inserted = true;
  for (size_t i = 0; i < block_hashes.size(); ++i) {
    uint64_t hash = block_hashes[i];
    if (lru_cache_.Contains(hash)) {
      all_inserted = false;
    } else {
      if (i < slices.size()) {
        lru_cache_.Put(hash, slices[i]);
      } else {
        lru_cache_.Put(hash, {});
      }
    }
  }

  return all_inserted;
}

void KVCacheStore::Delete(const std::vector<uint64_t>& block_hashes,
                          const std::vector<std::vector<RaidenId>>& slices) {
  absl::MutexLock lock(mutex_);
  for (uint64_t hash : block_hashes) {
    lru_cache_.Erase(hash);
  }
}

bool KVCacheStore::Pin(const std::vector<uint64_t>& block_hashes) {
  absl::MutexLock lock(mutex_);
  for (size_t i = 0; i < block_hashes.size(); ++i) {
    if (!lru_cache_.Pin(block_hashes[i])) {
      for (size_t j = 0; j < i; ++j) {
        lru_cache_.Unpin(block_hashes[j]);
      }
      return false;
    }
  }
  return true;
}

void KVCacheStore::Release(const std::vector<uint64_t>& block_hashes) {
  absl::MutexLock lock(mutex_);
  for (uint64_t hash : block_hashes) {
    lru_cache_.Unpin(hash);
  }
}

int KVCacheStore::GetPinCount(uint64_t hash) const {
  absl::MutexLock lock(mutex_);
  return lru_cache_.GetPinCount(hash);
}

size_t KVCacheStore::capacity() const {
  absl::MutexLock lock(mutex_);
  return lru_cache_.capacity();
}

}  // namespace kv_cache
}  // namespace tpu_raiden
