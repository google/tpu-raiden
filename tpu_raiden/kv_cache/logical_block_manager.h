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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_LOGICAL_BLOCK_MANAGER_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_LOGICAL_BLOCK_MANAGER_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace tpu_raiden {
namespace kv_cache {

// Manages logical blocks for KV cache utilized in TPU inference.
// Tracks block allocation, locking, and implements an LRU eviction policy
// to automatically free unlocked blocks when resources are exhausted.
//
// This class is not thread-safe and requires external synchronization for
// concurrent access.
class LogicalBlockManager {
 public:


  // Constructs a LogicalBlockManager managing a total of `num_blocks` blocks.
  // Block IDs are indexed from 0 to num_blocks - 1.
  explicit LogicalBlockManager(int num_blocks);

  // Allocates `num_blocks_to_allocate` blocks for `entity_id`.
  // Optionally locks the allocated blocks immediately.
  // Returns the assigned block IDs on success, or an error status if
  // insufficient blocks are available (even after attempting LRU eviction of
  // unlocked blocks).
  absl::StatusOr<std::vector<int>> Allocate(int num_blocks_to_allocate,
                                            bool lock = false);

  // Unlocks the specified blocks, making them eligible for LRU eviction if
  // future allocation requests require blocks.
  absl::Status Unlock(absl::Span<const int> block_ids);

  // Updates the access time (logical counter) for the specified block, marking
  // it as the most recently used.
  absl::Status AccessBlock(int block_id);

  // State inspection methods.
  bool IsAllocated(int block_id) const;
  bool IsLocked(int block_id) const;


  int total_blocks() const { return total_blocks_; }
  int num_free_blocks() const;
  int num_allocated_blocks() const;
  int num_locked_blocks() const;

 private:
  struct BlockState {
    bool is_allocated = false;
    bool is_locked = false;

    uint64_t last_access_counter = 0;
  };

  int total_blocks_;
  uint64_t access_counter_ = 0;
  std::vector<BlockState> blocks_;
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_LOGICAL_BLOCK_MANAGER_H_
