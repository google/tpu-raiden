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
#include <functional>
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

enum class MajorOrder : uint8_t {
  kLayerMajor = 0,
  kBlockMajor = 1,
};

using BlockReceivedCallback =
    std::function<absl::Status(size_t layer_idx, size_t shard_idx,
                               int block_id, size_t size_bytes)>;

// Delegate interface for BlockTransport to access layers/shards host memory
// and allocate blocks on the receiver dynamically.
class BlockTransportDelegate {
 public:
  virtual ~BlockTransportDelegate() = default;

  virtual absl::StatusOr<std::vector<int>> AllocateBlocks(
      size_t num_blocks, int64_t entity_id) = 0;

  virtual absl::Status OnDataReceived() = 0;

  virtual absl::Status OnSingleBlockReceived(int block_id, size_t size_bytes) {
    return OnDataReceived();
  }

  virtual absl::Status WaitForBlockRead(size_t layer_idx, size_t shard_idx,
                                        int block_id) {
    return absl::OkStatus();
  }

  virtual uint8_t* GetHostPointer(size_t layer_idx, size_t shard_idx) = 0;

  virtual size_t GetHostSize(size_t layer_idx, size_t shard_idx) = 0;

  virtual uint8_t* GetBlockHostPointer(size_t layer_idx, size_t shard_idx,
                                       int block_id) {
    return GetHostPointer(layer_idx, shard_idx) + block_id * bytes_per_block();
  }

  virtual size_t bytes_per_block() const {
    return block_size() * slice_byte_size();
  }

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
    uint8_t major_order;  // See MajorOrder. Ignored by legacy ops.
    uint16_t reserved = 0;
    uint32_t remote_block_id;
    uint32_t local_block_id;
    uint32_t num_blocks;
  };

  BlockTransport(BlockTransportDelegate* delegate, int local_port);
  ~BlockTransport();

  // Push block data to remote peer (H2H Write)
  absl::StatusOr<std::vector<int>> Push(const std::string& peer,
                                        const std::vector<int>& src_block_ids,
                                        int parallelism = 1,
                                        MajorOrder major_order =
                                            MajorOrder::kLayerMajor);

  // Pull block data from remote peer (H2H Read)
  absl::StatusOr<std::vector<int>> Pull(
      const std::string& peer, const std::vector<int>& src_block_ids,
      const std::vector<int>& local_block_ids = {},
      const std::vector<uint8_t*>& explicit_dst_ptrs = {}, int parallelism = 1,
      MajorOrder major_order = MajorOrder::kLayerMajor,
      BlockReceivedCallback on_block_received = {});

  // Pull byte-range weights chunk directly (Resharding Pull)
  absl::Status PullWeightsChunk(const std::string& source, size_t src_shard_idx,
                                size_t src_offset_bytes, size_t dst_shard_idx,
                                size_t dst_offset_bytes, size_t size_bytes);

  // Write a single block of data directly from a host pointer to a remote
  // block ID (Direct Push).
  absl::Status WriteBlockDirect(const std::string& peer, int remote_block_id,
                                const uint8_t* data_ptr, size_t size_bytes);

  int local_port() const { return local_port_; }

 private:
  absl::StatusOr<int> ConnectToPeer(const std::string& peer);

  // Persistent connection pool (consumer side). Reuse warm, congestion-window-
  // ramped TCP connections across pulls instead of opening a fresh slow-start
  // connection every transfer. (H2H bandwidth Exp-3 / RC1.) Gated by env
  // RAIDEN_CONN_POOL (default on; "0" disables).
  absl::StatusOr<int> AcquireConnection(const std::string& peer);
  void ReleaseConnection(const std::string& peer, int fd);
  void ClosePooledConnections();

  absl::Status ProcessSingleRequest(int client_fd);
  void ConnectionWorker(int client_fd);
  void ListenerLoop();

  void H2hWriteWorker(int stream_idx, const std::string& peer,
                      size_t block_offset, size_t block_count,
                      const std::vector<int>& src_block_ids,
                      std::vector<int>& allocated_ids,
                      std::vector<absl::Status>& statuses,
                      MajorOrder major_order);

  void H2hReadWorker(int stream_idx, const std::string& peer,
                     size_t local_block_offset, size_t local_block_count,
                     size_t remote_block_offset, size_t remote_block_count,
                     const std::vector<int>& src_block_ids,
                     const std::vector<int>& allocated_ids,
                     const std::vector<uint8_t*>& explicit_dst_ptrs,
                     std::vector<absl::Status>& statuses,
                     MajorOrder major_order,
                     BlockReceivedCallback on_block_received);

  BlockTransportDelegate* delegate_;
  int local_port_;
  int server_fd_ = -1;
  std::atomic<bool> stopping_{false};

  absl::Mutex mu_;
  std::vector<int> active_client_fds_ ABSL_GUARDED_BY(mu_);

  absl::Mutex pool_mu_;
  absl::flat_hash_map<std::string, std::vector<int>> conn_pool_
      ABSL_GUARDED_BY(pool_mu_);
  bool pooling_enabled_ = true;

  std::thread listener_thread_;
  std::vector<std::thread> worker_threads_;
};

}  // namespace transport
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TRANSPORT_BLOCK_TRANSPORT_H_
