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

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_raiden/proto/worker_service.pb.h"

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
  Buffer() : index_(-1), memory_type_(rpc::MEMORY_TYPE_UNSPECIFIED) {}

  Buffer(int index, std::vector<BufferShard> shards,
         std::optional<std::string> remote_address = std::nullopt,
         rpc::MemoryType memory_type = rpc::MEMORY_TYPE_UNSPECIFIED)
      : index_(index),
        shards_(std::move(shards)),
        remote_address_(std::move(remote_address)),
        memory_type_(memory_type) {}

  // Returns true if shards are empty and index is negative.
  bool empty() const { return shards_.empty() && index_ < 0; }

  // Returns the index of this buffer. The index carries application-specific
  // semantics, such as a layer index or a block index.
  int index() const { return index_; }

  // Returns the shards composing this buffer.
  absl::Span<const BufferShard> shards() const { return shards_; }

  // Returns the memory type of this buffer.
  rpc::MemoryType memory_type() const { return memory_type_; }
  void set_memory_type(rpc::MemoryType memory_type) {
    memory_type_ = memory_type;
  }

  // Returns the remote controller address if the buffer resides on workers
  // attached to a remote controller, otherwise std::nullopt.
  std::optional<std::string> remote_address() const { return remote_address_; }
  void set_remote_address(std::optional<std::string> remote_address) {
    remote_address_ = std::move(remote_address);
  }

  // Converts this Buffer into a proto::BufferProto.
  proto::BufferProto ToProto() const {
    proto::BufferProto proto;
    if (index_ >= 0) {
      proto.set_index(index_);
    }
    for (const auto& shard : shards_) {
      proto.add_buffer_handles()->set_handle(shard.handle);
    }
    proto.set_memory_type(memory_type_);
    if (remote_address_.has_value()) {
      proto.set_remote_address(*remote_address_);
    }
    return proto;
  }

  // Converts a proto::BufferProto to a Buffer.
  static Buffer FromProto(
      const proto::BufferProto& proto,
      std::optional<std::string> remote_address = std::nullopt) {
    int index = proto.has_index() ? proto.index() : -1;
    std::vector<BufferShard> shards;
    shards.reserve(proto.buffer_handles_size());
    for (const auto& handle_proto : proto.buffer_handles()) {
      shards.push_back(BufferShard{
          .handle = handle_proto.handle(),
          .offset = 0,
          .size = 0,
      });
    }
    rpc::MemoryType memory_type = proto.memory_type();
    std::optional<std::string> addr = remote_address;
    if (!addr.has_value() && !proto.remote_address().empty()) {
      addr = proto.remote_address();
    }
    return Buffer(index, std::move(shards), std::move(addr), memory_type);
  }

 private:
  int index_;
  std::vector<BufferShard> shards_;
  std::optional<std::string> remote_address_;
  rpc::MemoryType memory_type_ = rpc::MEMORY_TYPE_UNSPECIFIED;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_BUFFER_H_
