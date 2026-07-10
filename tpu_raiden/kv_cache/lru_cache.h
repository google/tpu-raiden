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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_LRU_CACHE_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_LRU_CACHE_H_

#include <cstddef>
#include <list>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_map.h"

namespace tpu_raiden {
namespace kv_cache {

// A highly optimized generic LRU cache tailored for managing Key-Value cache
// block eviction and promotion.
//
// Key features:
// - Supports explicit pinning (`Pin` / `Unpin`) to lock actively executing
//   blocks in memory during transformer attention passes so they cannot be
//   evicted.
// - Features exact MRU promotion on `Get` and fast LRU eviction on `Evict`.
// - Exposes `Peek` to read values without perturbing LRU order.
template <typename Key, typename Value>
class LRUCache {
 public:
  explicit LRUCache(size_t capacity) : capacity_(capacity) {}

  ~LRUCache() { Clear(); }

  LRUCache(const LRUCache&) = delete;
  LRUCache& operator=(const LRUCache&) = delete;

  LRUCache(LRUCache&& other) noexcept = default;
  LRUCache& operator=(LRUCache&& other) noexcept = default;

  // Returns the maximum capacity of the cache.
  size_t capacity() const { return capacity_; }

  // Returns the number of elements currently stored in the cache.
  size_t size() const { return map_.size(); }

  // Returns the available space (free + evictable capacity).
  size_t available_space() const { return capacity_ - pinned_list_.size(); }

  // Returns true if the cache is completely empty.
  bool empty() const { return map_.empty(); }

  // Returns true if the key is present in the cache.
  bool Contains(const Key& key) const { return map_.contains(key); }

  // Clears all elements from the cache.
  void Clear() {
    map_.clear();
    lru_list_.clear();
    pinned_list_.clear();
  }

  // Inserts or updates a key-value pair.
  // If the key already exists, its value is updated and promoted to Most
  // Recently Used (MRU).
  // If the key is new and inserting it exceeds capacity, the unpinned Least
  // Recently Used (LRU) item is automatically evicted. Returns the evicted
  // key-value pair if an eviction occurred, or std::nullopt.
  std::optional<std::pair<Key, Value>> Put(Key key, Value value) {
    auto it = map_.find(key);
    if (it != map_.end()) {
      // Update existing item and promote to MRU
      it->second->value = std::move(value);
      if (it->second->pin_count > 0) {
        pinned_list_.splice(pinned_list_.begin(), pinned_list_, it->second);
      } else {
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
      }
      return std::nullopt;
    }

    std::optional<std::pair<Key, Value>> evicted = std::nullopt;
    if (map_.size() >= capacity_) {
      evicted = Evict();
      if (!evicted.has_value()) return std::nullopt;
    }

    // Insert new item at the front of the lru_list_ (MRU)
    lru_list_.push_front(CacheNode{std::move(key), std::move(value), 0});
    map_[lru_list_.front().key] = lru_list_.begin();
    return evicted;
  }

  // Retrieves a pointer to the stored value for the given key and promotes
  // the item to Most Recently Used (MRU). Returns nullptr if absent.
  Value* Get(const Key& key) {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return nullptr;
    }
    // Promote to MRU
    if (it->second->pin_count > 0) {
      pinned_list_.splice(pinned_list_.begin(), pinned_list_, it->second);
    } else {
      lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
    }
    return &(it->second->value);
  }

  // Retrieves a pointer to the stored value for the given key without
  // promoting it to MRU (preserves exact LRU order). Returns nullptr if
  // absent.
  Value* Peek(const Key& key) const {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return nullptr;
    }
    return &(it->second->value);
  }

  // Exposes a mutable pointer via Peek without perturbing LRU order.
  Value* PeekMutable(const Key& key) {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return nullptr;
    }
    return &(it->second->value);
  }

  // Increments the reference pin count for the given key.
  // Pinned items are protected entirely against automated LRU eviction.
  // Returns true if the key exists and was successfully pinned.
  bool Pin(const Key& key) {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return false;
    }
    if (it->second->pin_count == 0) {
      // Move from lru_list_ to pinned_list_
      pinned_list_.splice(pinned_list_.begin(), lru_list_, it->second);
    }
    it->second->pin_count++;
    return true;
  }

  // Decrements the reference pin count for the given key.
  // When pin count reaches 0, the item is eligible for normal LRU eviction.
  // Returns true if the key exists and was successfully unpinned.
  bool Unpin(const Key& key) {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return false;
    }
    if (it->second->pin_count > 0) {
      it->second->pin_count--;
      if (it->second->pin_count == 0) {
        // Move from pinned_list_ to lru_list_ (promote to MRU)
        lru_list_.splice(lru_list_.begin(), pinned_list_, it->second);
      }
    }
    return true;
  }

  // Returns the pin count for the given key, or 0 if absent.
  int GetPinCount(const Key& key) const {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return 0;
    }
    return it->second->pin_count;
  }

  // Erases the given key from the cache entirely.
  // Returns true if the element was successfully erased.
  bool Erase(const Key& key) {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return false;
    }
    if (it->second->pin_count > 0) {
      pinned_list_.erase(it->second);
    } else {
      lru_list_.erase(it->second);
    }
    map_.erase(it);
    return true;
  }

  // Evicts and returns the least recently used, unpinned key-value pair.
  // Returns std::nullopt if all items are pinned or empty.
  std::optional<std::pair<Key, Value>> Evict() {
    if (lru_list_.empty()) {
      return std::nullopt;
    }
    auto it = std::prev(lru_list_.end());
    std::pair<Key, Value> evicted{std::move(it->key), std::move(it->value)};
    map_.erase(evicted.first);
    lru_list_.erase(it);
    return evicted;
  }

 private:
  struct CacheNode {
    Key key;
    Value value;
    int pin_count = 0;
  };

  size_t capacity_;
  std::list<CacheNode> lru_list_;
  std::list<CacheNode> pinned_list_;
  absl::flat_hash_map<Key, typename std::list<CacheNode>::iterator> map_;
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_LRU_CACHE_H_
