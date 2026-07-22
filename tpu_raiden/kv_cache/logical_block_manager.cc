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

#include "tpu_raiden/kv_cache/logical_block_manager.h"

#include <algorithm>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"

namespace tpu_raiden {
namespace kv_cache {

LogicalBlockManager::LogicalBlockManager(int num_blocks)
    : total_blocks_(num_blocks), blocks_(num_blocks) {}

absl::StatusOr<std::vector<int>> LogicalBlockManager::Allocate(
    int num_blocks_to_allocate, bool lock) {
  if (num_blocks_to_allocate <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Requested blocks must be > 0, got ", num_blocks_to_allocate));
  }
  if (num_blocks_to_allocate > total_blocks_) {
    return absl::InvalidArgumentError(
        absl::StrCat("Requested blocks (", num_blocks_to_allocate,
                     ") exceed total blocks (", total_blocks_, ")"));
  }

  // Find currently free blocks and unlocked allocated blocks (evictable).
  std::vector<int> free_block_ids;
  std::vector<int> evictable_block_ids;

  for (int i = 0; i < total_blocks_; ++i) {
    if (!blocks_[i].is_allocated) {
      free_block_ids.push_back(i);
    } else if (!blocks_[i].is_locked) {
      evictable_block_ids.push_back(i);
    }
  }

  if (free_block_ids.size() + evictable_block_ids.size() <
      num_blocks_to_allocate) {
    return absl::ResourceExhaustedError(absl::StrCat(
        "Insufficient available blocks. Requested: ", num_blocks_to_allocate,
        ", Free: ", free_block_ids.size(),
        ", Evictable: ", evictable_block_ids.size()));
  }

  // Determine which blocks to allocate.
  std::vector<int> allocated_block_ids;
  allocated_block_ids.reserve(num_blocks_to_allocate);

  // First, use free blocks.
  int free_to_use =
      std::min(static_cast<int>(free_block_ids.size()), num_blocks_to_allocate);
  for (int i = 0; i < free_to_use; ++i) {
    allocated_block_ids.push_back(free_block_ids[i]);
  }

  // If we still need more blocks, evict from evictable_block_ids based on LRU
  // policy. LRU policy evicts blocks with the smallest last_access_counter
  // first.
  int needed_blocks = num_blocks_to_allocate - free_to_use;
  if (needed_blocks > 0) {
    // Sort evictable_block_ids such that the least recently accessed comes
    // first.
    std::sort(evictable_block_ids.begin(), evictable_block_ids.end(),
              [this](int a, int b) {
                if (blocks_[a].last_access_counter ==
                    blocks_[b].last_access_counter) {
                  return a < b;
                }
                return blocks_[a].last_access_counter <
                       blocks_[b].last_access_counter;
              });

    for (int i = 0; i < needed_blocks; ++i) {
      int evict_id = evictable_block_ids[i];
      // Clear previous allocation state before reuse.
      blocks_[evict_id].is_allocated = false;
      allocated_block_ids.push_back(evict_id);
    }
  }

  // Increment global counter for new allocation access timestamp.
  ++access_counter_;

  // Assign the selected blocks to the requested entity.
  for (int block_id : allocated_block_ids) {
    blocks_[block_id].is_allocated = true;
    blocks_[block_id].is_locked = lock;
    blocks_[block_id].last_access_counter = access_counter_;
  }

  return allocated_block_ids;
}

absl::Status LogicalBlockManager::RestoreAllocated(
    absl::Span<const int> block_ids) {
  // Validate the entire batch before mutating any state.
  std::vector<bool> seen(total_blocks_, false);
  for (int block_id : block_ids) {
    if (block_id < 0 || block_id >= total_blocks_) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid block ID: ", block_id));
    }
    if (blocks_[block_id].is_allocated) {
      return absl::FailedPreconditionError(
          absl::StrCat("Cannot restore already allocated block ID: ", block_id));
    }
    if (seen[block_id]) {
      return absl::InvalidArgumentError(
          absl::StrCat("Duplicate block ID: ", block_id));
    }
    seen[block_id] = true;
  }

  ++access_counter_;
  for (int block_id : block_ids) {
    blocks_[block_id].is_allocated = true;
    blocks_[block_id].is_locked = true;
    blocks_[block_id].last_access_counter = access_counter_;
  }
  return absl::OkStatus();
}

absl::Status LogicalBlockManager::Unlock(absl::Span<const int> block_ids) {
  // First validate all blocks.
  for (int block_id : block_ids) {
    if (block_id < 0 || block_id >= total_blocks_) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid block ID: ", block_id));
    }
    if (!blocks_[block_id].is_allocated) {
      return absl::InvalidArgumentError(
          absl::StrCat("Cannot unlock unallocated block ID: ", block_id));
    }
  }

  // Perform state update.
  for (int block_id : block_ids) {
    blocks_[block_id].is_locked = false;
  }
  return absl::OkStatus();
}

absl::Status LogicalBlockManager::AccessBlock(int block_id) {
  if (block_id < 0 || block_id >= total_blocks_) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid block ID: ", block_id));
  }
  if (!blocks_[block_id].is_allocated) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cannot access unallocated block ID: ", block_id));
  }

  ++access_counter_;
  blocks_[block_id].last_access_counter = access_counter_;
  return absl::OkStatus();
}

bool LogicalBlockManager::IsAllocated(int block_id) const {
  if (block_id < 0 || block_id >= total_blocks_) return false;
  return blocks_[block_id].is_allocated;
}

bool LogicalBlockManager::IsLocked(int block_id) const {
  if (block_id < 0 || block_id >= total_blocks_) return false;
  return blocks_[block_id].is_locked;
}



int LogicalBlockManager::num_free_blocks() const {
  int count = 0;
  for (const auto& block : blocks_) {
    if (!block.is_allocated) ++count;
  }
  return count;
}

int LogicalBlockManager::num_allocated_blocks() const {
  int count = 0;
  for (const auto& block : blocks_) {
    if (block.is_allocated) ++count;
  }
  return count;
}

int LogicalBlockManager::num_locked_blocks() const {
  int count = 0;
  for (const auto& block : blocks_) {
    if (block.is_locked) ++count;
  }
  return count;
}

}  // namespace kv_cache
}  // namespace tpu_raiden
