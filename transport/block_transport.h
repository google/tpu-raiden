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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"

namespace tpu_raiden {
namespace transport {

// Delegate interface for BlockTransport to access layers/shards host memory
// and allocate blocks on the receiver dynamically.
class BlockTransportDelegate {
 public:
  virtual ~BlockTransportDelegate() = default;

  virtual absl::StatusOr<std::vector<int>> AllocateBlocks(
      size_t num_blocks, int64_t entity_id) = 0;

  virtual absl::Status OnDataReceived() = 0;

  virtual uint8_t* GetHostPointer(size_t layer_idx, size_t shard_idx) = 0;

  virtual size_t GetHostSize(size_t layer_idx, size_t shard_idx) = 0;

  virtual int GetRemoteReadBlockId(int base_remote_id, int chunk_k) = 0;

  virtual size_t num_layers() const = 0;
  virtual size_t num_shards() const = 0;
  virtual size_t slice_byte_size() const = 0;
  virtual int block_size() const = 0;
  virtual size_t shard_factor() const = 0;
};

// Standalone block-based POSIX TCP socket transport engine for TPU Raiden.
class BlockTransport {
 public:
  // Binary packet header layout for H2H transfers.
  struct alignas(8) BlockPacketHeader {
    uint8_t op;  // 1 = Push, 2 = Pull, 3 = Byte-Range Pull
    uint32_t remote_block_id;
    uint32_t local_block_id;
    uint32_t num_blocks;
  };

  BlockTransport(BlockTransportDelegate* delegate, int local_port);
  ~BlockTransport();

  // Push block data to remote peer (H2H Write)
  absl::StatusOr<std::vector<int>> Push(const std::string& peer,
                                        const std::vector<int>& src_block_ids,
                                        int parallelism = 1);

  // Pull block data from remote peer (H2H Read)
  absl::StatusOr<std::vector<int>> Pull(const std::string& peer,
                                        const std::vector<int>& src_block_ids,
                                        int parallelism = 1);

  // Pull byte-range weights chunk directly (Resharding Pull)
  absl::Status PullWeightsChunk(const std::string& source, size_t src_shard_idx,
                                size_t src_offset_bytes, size_t dst_shard_idx,
                                size_t dst_offset_bytes, size_t size_bytes);

  int local_port() const { return local_port_; }

 private:
  absl::StatusOr<int> ConnectToPeer(const std::string& peer);

  absl::Status ProcessSingleRequest(int client_fd);
  void ConnectionWorker(int client_fd);
  void ListenerLoop();

  void H2hWriteWorker(int stream_idx, const std::string& peer,
                      size_t blocks_per_stream,
                      const std::vector<int>& src_block_ids,
                      std::vector<int>& allocated_ids,
                      std::vector<absl::Status>& statuses);

  void H2hReadWorker(int stream_idx, const std::string& peer,
                     size_t blocks_per_stream, size_t remote_blocks_per_stream,
                     int base_remote_id, const std::vector<int>& allocated_ids,
                     std::vector<absl::Status>& statuses);

  BlockTransportDelegate* delegate_;
  int local_port_;
  int server_fd_ = -1;
  std::atomic<bool> stopping_{false};

  absl::Mutex mu_;
  std::vector<int> active_client_fds_ ABSL_GUARDED_BY(mu_);

  std::thread listener_thread_;
  std::vector<std::thread> worker_threads_;
};

}  // namespace transport
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TRANSPORT_BLOCK_TRANSPORT_H_
