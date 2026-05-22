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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_RAIDEN_MANAGER_BASE_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_RAIDEN_MANAGER_BASE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace tpu_raiden {

class RaidenManagerBase {
 public:
  RaidenManagerBase(size_t num_layers, size_t num_shards,
                    size_t slice_byte_size, int block_size = 1,
                    std::optional<int> local_port = std::nullopt,
                    int parallelism = 1);

  virtual ~RaidenManagerBase();

  // Direct C++ H2H network write (Push)
  absl::StatusOr<std::vector<int>> H2hWriteDirect(
      const std::string& peer, const std::vector<int>& src_block_ids,
      int64_t entity_id = 0);

  // Direct C++ H2H network read (Pull)
  absl::StatusOr<std::vector<int>> H2hReadDirect(
      const std::string& peer, const std::vector<int>& src_block_ids,
      int64_t entity_id = 0);

  std::optional<int> local_port() const;

  const uint8_t* GetHostPointer(size_t layer_idx, size_t shard_idx) const;

  void SetExternalHostPointers(const std::vector<const uint8_t*>& host_ptrs,
                               const std::vector<size_t>& host_sizes);

  size_t num_layers() const { return num_layers_; }
  size_t num_shards() const { return num_shards_; }
  size_t slice_byte_size() const { return slice_byte_size_; }
  int block_size() const { return block_size_; }

 protected:
  struct ShardBufferInfoBase {
    const uint8_t* host_ptr = nullptr;
    size_t host_size = 0;
    size_t device_size = 0;
    std::unique_ptr<uint8_t[], void (*)(void*)> owned_host_buffer = {
        nullptr, [](void*) {}};
  };

  struct LayerInfoBase {
    std::vector<ShardBufferInfoBase> shards;
  };

  size_t num_layers_ = 0;
  size_t num_shards_ = 0;
  size_t slice_byte_size_ = 0;
  int block_size_ = 1;
  int parallelism_ = 1;
  size_t shard_factor_ = 1;
  int64_t major_dim_size_ = 0;

  struct BlockTransportServer;
  std::unique_ptr<BlockTransportServer> server_;

  std::vector<LayerInfoBase> layers_;

  // Virtual hooks letting subclasses declare their own dynamic allocators if
  // needed E2E!
  virtual absl::StatusOr<std::vector<int>> AllocateBlocks(size_t num_blocks,
                                                          int64_t entity_id) {
    return absl::UnimplementedError("Block allocator is not available");
  }

  virtual int GetRemoteReadBlockId(int base_remote_id, int chunk_k) {
    return base_remote_id + chunk_k;
  }

  virtual absl::Status OnDataReceived() { return absl::OkStatus(); }

  void H2hWriteWorker(int stream_idx, const std::string& peer,
                      size_t blocks_per_stream,
                      const std::vector<int>& src_block_ids,
                      std::vector<int>& allocated_ids,
                      std::vector<absl::Status>& statuses);

  void H2hReadWorker(int stream_idx, const std::string& peer,
                     size_t blocks_per_stream, size_t remote_blocks_per_stream,
                     int base_remote_id, const std::vector<int>& allocated_ids,
                     std::vector<absl::Status>& statuses);
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_RAIDEN_MANAGER_BASE_H_
