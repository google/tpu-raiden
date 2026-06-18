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

#ifndef THIRD_PARTY_TPU_RAIDEN_TRANSPORT_BLOCK_TRANSPORT_H_
#define THIRD_PARTY_TPU_RAIDEN_TRANSPORT_BLOCK_TRANSPORT_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tpu_raiden/transport/raw_buffer_transport.h"

namespace tpu_raiden {
namespace transport {

enum class MajorOrder : uint8_t {
  kLayerMajor = 0,
  kBlockMajor = 1,
};

using BlockReceivedCallback =
    std::function<absl::Status(size_t layer_idx, size_t shard_idx,
                               int block_id, size_t size_bytes)>;

// Delegate interface for BlockTransport inheriting raw memory primitives.
class BlockTransportDelegate : public RawBufferTransportDelegate {
 public:
  ~BlockTransportDelegate() override = default;

  virtual absl::StatusOr<std::vector<int>> AllocateBlocks(
      size_t num_blocks, uint64_t uuid = 0) = 0;

  virtual absl::Status OnBlocksReceived(const std::vector<int>& block_ids,
                                        uint64_t uuid = 0) {
    return OnDataReceived();
  }

  virtual absl::Status OnSingleBlockReceived(int block_id, size_t size_bytes) {
    return OnDataReceived();
  }

  virtual absl::Status WaitForBlockRead(size_t layer_idx, size_t shard_idx,
                                        int block_id) {
    return absl::OkStatus();
  }

  virtual uint8_t* GetBlockHostPointer(size_t layer_idx, size_t shard_idx,
                                       int block_id) {
    return GetHostPointer(layer_idx, shard_idx) + block_id * bytes_per_block();
  }

  virtual size_t bytes_per_block() const { return slice_byte_size(); }

  virtual int GetRemoteReadBlockId(int base_remote_id, int chunk_k) = 0;

  virtual size_t num_layers() const = 0;
  virtual size_t num_shards() const = 0;
  virtual size_t slice_byte_size() const = 0;
  virtual size_t shard_factor() const = 0;
};

// High-speed Key-Value block transport engine extending RawBufferTransport.
class BlockTransport : public RawBufferTransport {
 public:
  using BlockPacketHeader = RawBufferTransport::PacketHeader;

  BlockTransport(BlockTransportDelegate* delegate, int local_port,
                 bool enable_conn_pool = true);
  ~BlockTransport() override;

  // Standard Scatter-Gather Push (op = 1 / op = 6)
  absl::StatusOr<std::vector<int>> Push(
      const std::string& peer, const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids = {}, int parallelism = 1,
      MajorOrder major_order = MajorOrder::kLayerMajor, uint64_t uuid = 0);

  // Synchronous Scatter-Gather Pull (op = 2)
  absl::StatusOr<std::vector<int>> Pull(
      const std::string& peer, const std::vector<int>& src_block_ids,
      const std::vector<int>& local_block_ids = {},
      const std::vector<uint8_t*>& explicit_dst_ptrs = {}, int parallelism = 1,
      MajorOrder major_order = MajorOrder::kLayerMajor,
      BlockReceivedCallback on_block_received = {});

  // Write a single block of data directly from a host pointer to a remote block ID.
  absl::Status WriteBlockDirect(const std::string& peer, int remote_block_id,
                                const uint8_t* data_ptr, size_t size_bytes);

 protected:
  absl::Status HandleCustomRequest(int client_fd,
                                   const PacketHeader& header) override;

 private:
  void H2hWriteWorker(int stream_idx, const std::string& peer,
                      size_t block_offset, size_t block_count,
                      const std::vector<int>& src_block_ids,
                      const std::vector<int>& dst_block_ids,
                      std::vector<int>& allocated_ids,
                      std::vector<absl::Status>& statuses,
                      MajorOrder major_order, uint64_t uuid = 0);

  void H2hReadWorker(int stream_idx, const std::string& peer,
                     size_t local_block_offset, size_t local_block_count,
                     size_t remote_block_offset, size_t remote_block_count,
                     const std::vector<int>& src_block_ids,
                     const std::vector<int>& allocated_ids,
                     const std::vector<uint8_t*>& explicit_dst_ptrs,
                     std::vector<absl::Status>& statuses,
                     MajorOrder major_order,
                     BlockReceivedCallback on_block_received);

  BlockTransportDelegate* block_delegate_;
};

}  // namespace transport
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TRANSPORT_BLOCK_TRANSPORT_H_
