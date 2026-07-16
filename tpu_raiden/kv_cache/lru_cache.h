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
#include <vector>

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
enum class NodeLocation { kLru, kPinned, kCandidate };

template <typename Key, typename Value>
class LRUCache {
 private:
  struct CacheNode;

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
  size_t active_size() const { return lru_list_.size() + pinned_list_.size(); }

  // Returns true if the cache is completely empty.
  bool empty() const { return map_.empty(); }

  // Returns true if the key is present in the cache (and not a candidate).
  bool Contains(const Key& key) const {
    auto it = map_.find(key);
    return it != map_.end() && it->second->location != NodeLocation::kCandidate;
  }

  // Clears all elements from the cache.
  void Clear() {
    map_.clear();
    lru_list_.clear();
    pinned_list_.clear();
    evict_candidate_list_.clear();
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
      Promote(it->second);
      return std::nullopt;
    }

    std::optional<std::pair<Key, Value>> evicted = std::nullopt;
    if (active_size() >= capacity_) {
      evicted = Evict();
      if (!evicted.has_value()) return std::nullopt;
    }

    // Insert new item at the front of the lru_list_ (MRU)
    lru_list_.emplace_front(std::move(key), std::move(value), 0,
                            NodeLocation::kLru);
    map_.emplace_hint(it, lru_list_.front().key, lru_list_.begin());
    return evicted;
  }

  // Inserts a key-value pair at the back of the lru_list_ (least recently
  // used position). If the key already exists, its value is updated and moved
  // to the back of lru_list_.
  void PutBack(Key key, Value value) {
    auto it = map_.find(key);
    if (it != map_.end()) {
      it->second->value = std::move(value);
      PromoteBack(it->second);
      return;
    }

    if (active_size() >= capacity_) {
      Evict();
    }

    lru_list_.emplace_back(std::move(key), std::move(value), 0,
                           NodeLocation::kLru);
    map_.emplace_hint(it, lru_list_.back().key, std::prev(lru_list_.end()));
  }

  // Retrieves a pointer to the stored value for the given key and promotes
  // the item to Most Recently Used (MRU). Returns nullptr if absent.
  Value* Get(const Key& key) {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return nullptr;
    }
    Promote(it->second);
    return &(it->second->value);
  }

  // Retrieves a pointer to the stored value for the given key without
  // promoting it to MRU (preserves exact LRU order). Returns nullptr if
  // absent or candidate.
  Value* Peek(const Key& key) const {
    auto it = map_.find(key);
    if (it == map_.end() || it->second->location == NodeLocation::kCandidate) {
      return nullptr;
    }
    return &(it->second->value);
  }

  // Exposes a mutable pointer via Peek without perturbing LRU order.
  // Returns nullptr if absent or candidate.
  Value* PeekMutable(const Key& key) {
    auto it = map_.find(key);
    if (it == map_.end() || it->second->location == NodeLocation::kCandidate) {
      return nullptr;
    }
    return &(it->second->value);
  }

  // Retrieves a pointer to the stored value even if it is a candidate.
  const Value* PeekIncludingCandidates(const Key& key) const {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return nullptr;
    }
    return &(it->second->value);
  }

  // Exposes a mutable pointer even if it is a candidate.
  Value* PeekMutableIncludingCandidates(const Key& key) {
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
      if (it->second->location == NodeLocation::kLru) {
        pinned_list_.splice(pinned_list_.begin(), lru_list_, it->second);
      } else if (it->second->location == NodeLocation::kCandidate) {
        pinned_list_.splice(pinned_list_.begin(), evict_candidate_list_,
                            it->second);
      }
      it->second->location = NodeLocation::kPinned;
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
        it->second->location = NodeLocation::kLru;
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

  // Exposes read-only access to the internal map for iteration.
  const absl::flat_hash_map<Key, typename std::list<CacheNode>::iterator>& map()
      const {
    return map_;
  }

  // Returns up to `count` oldest evictable (unpinned) keys in LRU order (oldest
  // first).
  std::vector<Key> GetEvictableKeys(size_t count) const {
    std::vector<Key> keys;
    keys.reserve(
        std::min(count, evict_candidate_list_.size() + lru_list_.size()));
    // 1. Scan evict_candidate_list_ first (oldest first, i.e. front to back)
    for (auto it = evict_candidate_list_.begin();
         it != evict_candidate_list_.end() && keys.size() < count; ++it) {
      keys.push_back(it->key);
    }
    // 2. Scan lru_list_ (oldest first, i.e. back to front)
    if (keys.size() < count) {
      for (auto it = lru_list_.rbegin();
           it != lru_list_.rend() && keys.size() < count; ++it) {
        keys.push_back(it->key);
      }
    }
    return keys;
  }

  // Erases the given key from the cache entirely.
  // Returns true if the element was successfully erased.
  bool Erase(const Key& key) {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return false;
    }
    if (it->second->location == NodeLocation::kPinned) {
      pinned_list_.erase(it->second);
    } else if (it->second->location == NodeLocation::kLru) {
      lru_list_.erase(it->second);
    } else if (it->second->location == NodeLocation::kCandidate) {
      evict_candidate_list_.erase(it->second);
    }
    map_.erase(it);
    return true;
  }

  // Evicts the least recently used, unpinned key-value pair to candidate list.
  // Returns the evicted pair if successful, or std::nullopt.
  std::optional<std::pair<Key, Value>> Evict() {
    if (lru_list_.empty()) {
      return std::nullopt;
    }
    auto it = std::prev(lru_list_.end());
    it->location = NodeLocation::kCandidate;
    evict_candidate_list_.splice(evict_candidate_list_.end(), lru_list_, it);
    return std::make_pair(it->key, it->value);
  }

  // Restores the last evicted candidate back to the LRU list (at the LRU
  // position). Returns true if a candidate was successfully restored.
  bool RestoreLastCandidate() {
    if (evict_candidate_list_.empty()) {
      return false;
    }
    auto it = std::prev(evict_candidate_list_.end());
    it->location = NodeLocation::kLru;
    lru_list_.splice(lru_list_.end(), evict_candidate_list_, it);
    return true;
  }

  std::vector<Key> GetEvictCandidateKeys() const {
    std::vector<Key> keys;
    keys.reserve(evict_candidate_list_.size());
    for (const auto& node : evict_candidate_list_) {
      keys.push_back(node.key);
    }
    return keys;
  }

  // Moves the key from the candidate list back to the active LRU list (as MRU)
  // if it was in the candidate list.
  void MarkAsActive(const Key& key) {
    auto it = map_.find(key);
    if (it == map_.end()) return;
    if (it->second->location == NodeLocation::kCandidate) {
      lru_list_.splice(lru_list_.begin(), evict_candidate_list_, it->second);
      it->second->location = NodeLocation::kLru;
    }
  }

 private:
  struct CacheNode {
    Key key;
    Value value;
    int pin_count = 0;
    NodeLocation location = NodeLocation::kLru;

    CacheNode(Key k, Value v, int p, NodeLocation loc = NodeLocation::kLru)
        : key(std::move(k)), value(std::move(v)), pin_count(p), location(loc) {}
  };

  void Promote(typename std::list<CacheNode>::iterator node_it) {
    if (node_it->pin_count > 0) {
      pinned_list_.splice(pinned_list_.begin(), pinned_list_, node_it);
      node_it->location = NodeLocation::kPinned;
    } else {
      if (node_it->location == NodeLocation::kLru) {
        lru_list_.splice(lru_list_.begin(), lru_list_, node_it);
      } else if (node_it->location == NodeLocation::kCandidate) {
        lru_list_.splice(lru_list_.begin(), evict_candidate_list_, node_it);
        node_it->location = NodeLocation::kLru;
      }
    }
  }

  void PromoteBack(typename std::list<CacheNode>::iterator node_it) {
    if (node_it->pin_count > 0) {
      pinned_list_.splice(pinned_list_.end(), pinned_list_, node_it);
      node_it->location = NodeLocation::kPinned;
    } else {
      if (node_it->location == NodeLocation::kLru) {
        lru_list_.splice(lru_list_.end(), lru_list_, node_it);
      } else if (node_it->location == NodeLocation::kCandidate) {
        lru_list_.splice(lru_list_.end(), evict_candidate_list_, node_it);
        node_it->location = NodeLocation::kLru;
      }
    }
  }

  size_t capacity_;
  std::list<CacheNode> lru_list_;
  std::list<CacheNode> pinned_list_;
  std::list<CacheNode> evict_candidate_list_;
  absl::flat_hash_map<Key, typename std::list<CacheNode>::iterator> map_;
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_LRU_CACHE_H_
