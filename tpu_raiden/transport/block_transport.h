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
#include <deque>
#include <functional>
#include <future>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tpu_raiden/transport/raw_buffer_transport.h"

namespace tpu_raiden {
namespace transport {

enum class MajorOrder : uint8_t {
  kLayerMajor = 0,
  kBlockMajor = 1,
};

using BlockReceivedCallback = std::function<absl::Status(
    size_t layer_idx, size_t shard_idx, int block_id, size_t size_bytes)>;

// Represents a contiguous span of memory.
struct BlockChunk {
  uint8_t* ptr;
  size_t size;
};

// Delegate interface for BlockTransport inheriting raw memory primitives.
class BlockTransportDelegate : public RawBufferTransportDelegate {
 public:
  ~BlockTransportDelegate() override = default;

  virtual bool use_block_chunks(uint64_t uuid) const { return false; }

  // Returns the active node ID (rank) of the worker.
  virtual int64_t node_id() const { return -1; }

  // Returns the list of contiguous chunks that constitute a block
  // for a specific transaction (identified by uuid).
  // Optionally accepts the sender_node_id to distinguish senders in many-to-one
  // transfers.
  virtual std::vector<BlockChunk> GetBlockChunks(size_t layer_idx,
                                                 size_t shard_idx, int block_id,
                                                 uint64_t uuid,
                                                 int64_t sender_node_id = -1,
                                                 absl::string_view peer = "") {
    // Default implementation: block is contiguous and of uniform size.
    return {{GetBlockHostPointer(layer_idx, shard_idx, block_id),
             bytes_per_block()}};
  }

  virtual absl::StatusOr<std::vector<int>> AllocateBlocks(
      size_t num_blocks, uint64_t uuid = 0) = 0;

  virtual absl::Status OnLayerReceived(size_t layer_idx, uint64_t uuid) {
    return absl::OkStatus();
  }

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
                 bool enable_conn_pool = true,
                 const std::vector<std::string>& local_ips = {},
                 int parallelism = 1);
  ~BlockTransport() override;

  // Standard Scatter-Gather Push (op = 1 / op = 6)
  absl::StatusOr<std::vector<int>> Push(
      const std::vector<std::string>& peers,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids = {}, int parallelism = 1,
      MajorOrder major_order = MajorOrder::kLayerMajor, uint64_t uuid = 0,
      int layer_idx = -1);

  // Asynchronous Scatter-Gather Push
  void Push(const std::vector<std::string>& peers,
            const std::vector<int>& src_block_ids,
            const std::vector<int>& dst_block_ids, int parallelism,
            MajorOrder major_order, uint64_t uuid, int layer_idx,
            std::function<void(absl::StatusOr<std::vector<int>>)> on_complete);

  // Synchronous Scatter-Gather Pull (op = 2)
  absl::StatusOr<std::vector<int>> Pull(
      const std::vector<std::string>& peers,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& local_block_ids = {},
      const std::vector<uint8_t*>& explicit_dst_ptrs = {}, int parallelism = 1,
      MajorOrder major_order = MajorOrder::kLayerMajor,
      BlockReceivedCallback on_block_received = {}, uint64_t uuid = 0);

  // Backward-compatible overloads
  absl::StatusOr<std::vector<int>> Push(
      absl::string_view peer, const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids = {}, int parallelism = 1,
      MajorOrder major_order = MajorOrder::kLayerMajor, uint64_t uuid = 0,
      int layer_idx = -1) {
    return Push(std::vector<std::string>{std::string(peer)}, src_block_ids,
                dst_block_ids, parallelism, major_order, uuid, layer_idx);
  }

  absl::StatusOr<std::vector<int>> Pull(
      absl::string_view peer, const std::vector<int>& src_block_ids,
      const std::vector<int>& local_block_ids = {},
      const std::vector<uint8_t*>& explicit_dst_ptrs = {}, int parallelism = 1,
      MajorOrder major_order = MajorOrder::kLayerMajor,
      BlockReceivedCallback on_block_received = {}, uint64_t uuid = 0) {
    return Pull(std::vector<std::string>{std::string(peer)}, src_block_ids,
                local_block_ids, explicit_dst_ptrs, parallelism, major_order,
                on_block_received, uuid);
  }

  // Write a single block of data directly from a host pointer to a remote block
  // ID.
  absl::Status WriteBlockDirect(absl::string_view peer, int remote_block_id,
                                const uint8_t* data_ptr, size_t size_bytes);

 protected:
  absl::Status HandleCustomRequest(int client_fd,
                                   const PacketHeader& header) override;

 private:
  struct WriteTask {
    uint64_t uuid;
    int layer_idx;
    int stream_idx;
    std::string peer;
    std::function<void()> run;
  };

  struct PeerQueue {
    std::deque<std::unique_ptr<WriteTask>> tasks;
    int active_streams = 0;
  };

  void SocketWorkerLoop();
  std::unique_ptr<WriteTask> SelectNextTask()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(scheduler_mu_);

  void H2hWriteWorker(int stream_idx, absl::string_view peer,
                      absl::string_view local_ip, size_t block_offset,
                      size_t block_count, const std::vector<int>& src_block_ids,
                      const std::vector<int>& dst_block_ids,
                      std::vector<int>& allocated_ids,
                      std::vector<absl::Status>& statuses,
                      MajorOrder major_order, uint64_t uuid = 0,
                      int layer_idx = -1, int parallelism = 1);

  void H2hReadWorker(int stream_idx, absl::string_view peer,
                     absl::string_view local_ip, size_t local_block_offset,
                     size_t local_block_count, size_t remote_block_offset,
                     size_t remote_block_count,
                     const std::vector<int>& src_block_ids,
                     const std::vector<int>& allocated_ids,
                     const std::vector<uint8_t*>& explicit_dst_ptrs,
                     std::vector<absl::Status>& statuses,
                     MajorOrder major_order,
                     BlockReceivedCallback on_block_received,
                     uint64_t uuid = 0);

  BlockTransportDelegate* block_delegate_;

  struct LayerProgress {
    size_t completed_chunks = 0;
    bool on_layer_received_called = false;
  };
  absl::Mutex progress_mu_;
  absl::flat_hash_map<std::pair<uint64_t, int>, LayerProgress> layer_progress_
      ABSL_GUARDED_BY(progress_mu_);

  absl::Mutex scheduler_mu_;
  absl::CondVar scheduler_cv_;
  absl::flat_hash_map<std::string, PeerQueue> peer_queues_
      ABSL_GUARDED_BY(scheduler_mu_);
  std::vector<std::string> active_peers_ ABSL_GUARDED_BY(scheduler_mu_);
  size_t rr_index_ ABSL_GUARDED_BY(scheduler_mu_) = 0;
  int parallelism_ = 1;
  std::atomic<bool> scheduler_stopping_{false};
  std::vector<std::thread> socket_workers_;
};

}  // namespace transport
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TRANSPORT_BLOCK_TRANSPORT_H_
