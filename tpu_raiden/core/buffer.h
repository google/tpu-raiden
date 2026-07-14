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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_BUFFER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_BUFFER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"

namespace tpu_raiden {

// Represents a continuous memory region on a worker.
struct BufferShard {
  uint64_t handle;
  int64_t offset;
  int64_t size;
};

// Abstracts a sharded buffer composed of multiple memory shards.
// Either the buffer handle (within a shard) or the buffer index can be used
// to locate the buffer in the worker or controller.
class Buffer {
 public:
  Buffer(int index, std::vector<BufferShard> shards,
         std::optional<std::string> remote_address = std::nullopt)
      : index_(index),
        shards_(std::move(shards)),
        remote_address_(std::move(remote_address)) {}

  // Returns the index of this buffer. The index carries application-specific
  // semantics, such as a layer index or a block index.
  int index() const { return index_; }

  // Returns the shards composing this buffer.
  absl::Span<const BufferShard> shards() const { return shards_; }

  // Returns the remote controller address if the buffer resides on workers
  // attached to a remote controller, otherwise std::nullopt.
  std::optional<std::string> RemoteAddress() const { return remote_address_; }

 private:
  int index_;
  std::vector<BufferShard> shards_;
  std::optional<std::string> remote_address_;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_BUFFER_H_
