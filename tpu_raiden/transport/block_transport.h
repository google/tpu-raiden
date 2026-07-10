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

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_raiden/transport/raw_buffer_transport.h"

namespace tpu_raiden {
namespace transport {

enum class MajorOrder : uint8_t {
  kLayerMajor = 0,
  kBlockMajor = 1,
};

using BlockReceivedCallback = std::function<absl::Status(
    size_t layer_idx, size_t shard_idx, int block_id, size_t size_bytes)>;

// Represents a contiguous span of memory within a block transport operation.
struct BlockChunk {
  // Pointer to the local host memory address of this chunk.
  uint8_t* ptr;
  // Length of this memory chunk in bytes.
  size_t size;
};

enum class BlockChunkRegionValidationMode {
  kDisabled = 0,
  kWarn = 1,
  kFail = 2,
};

// Delegate interface for BlockTransport inheriting raw memory primitives.
class BlockTransportDelegate : public RawBufferTransportDelegate {
 public:
  ~BlockTransportDelegate() override = default;

  virtual bool use_block_chunks(uint64_t uuid) const { return false; }

  // Optional second-pass validation for chunked transfers. The transport always
  // validates chunks against the flat host buffer first. Delegates with
  // per-layer interior-layout knowledge can additionally reject or warn on
  // chunks that target padding or otherwise non-live bytes.
  virtual BlockChunkRegionValidationMode block_chunk_region_validation_mode()
      const {
    return BlockChunkRegionValidationMode::kDisabled;
  }

  virtual absl::Status ValidateBlockChunksInRegions(
      size_t layer_idx, size_t shard_idx,
      const std::vector<BlockChunk>& chunks) {
    return absl::OkStatus();
  }

  // Returns the active node ID (rank) of the worker.
  virtual int64_t node_id() const { return -1; }

  // Returns the list of contiguous chunks that constitute a block range
  // for a specific transaction (identified by uuid).
  // Optionally accepts the sender_node_id to distinguish senders in many-to-one
  // transfers.
  // If `src_block_id` is provided (not -1), it is used to resolve correct chunk
  // offsets when multiple source blocks merge into a single destination block
  // (heterogeneous block sizes).
  virtual std::vector<BlockChunk> GetBlockChunks(
      size_t layer_idx, size_t shard_idx, absl::Span<const int64_t> block_ids,
      size_t total_bytes, uint64_t uuid, int64_t sender_node_id = -1,
      absl::string_view peer = "", int64_t src_block_id = -1) {
    // Default implementation: blocks are contiguous and of uniform size.
    std::vector<BlockChunk> result;
    size_t accumulated_bytes = 0;
    for (int64_t block_id : block_ids) {
      if (accumulated_bytes >= total_bytes) break;
      size_t size =
          std::min(bytes_per_block(), total_bytes - accumulated_bytes);
      result.push_back(
          {GetBlockHostPointer(layer_idx, shard_idx, block_id), size});
      accumulated_bytes += size;
    }
    return result;
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

  using HostBlockReadyCallback = std::function<void(absl::Status)>;
  virtual void RegisterBlockReadinessCallback(size_t layer_idx,
                                              size_t shard_idx, int block_id,
                                              uint64_t uuid,
                                              HostBlockReadyCallback cb) {
    cb(absl::OkStatus());
  }

  virtual void ScheduleAsyncTask(std::function<void()> task) {
    std::thread(std::move(task)).detach();
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

  void Push(absl::string_view peer, const std::vector<int>& src_block_ids,
            const std::vector<int>& dst_block_ids, int parallelism,
            MajorOrder major_order, uint64_t uuid, int layer_idx,
            std::function<void(absl::StatusOr<std::vector<int>>)> on_complete) {
    Push(std::vector<std::string>{std::string(peer)}, src_block_ids,
         dst_block_ids, parallelism, major_order, uuid, layer_idx,
         std::move(on_complete));
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

  absl::Status HandleIncomingPush(int client_fd, const PacketHeader& header);
  absl::Status HandleIncomingPull(int client_fd, const PacketHeader& header);

  struct SendStreamState {
    int client_fd;
    uint64_t uuid;
    int remote_id;
    size_t count_or_size;
    MajorOrder major_order;
    size_t current_step = 0;
    size_t total_steps = 0;
  };

  void TriggerNextSendStep(std::shared_ptr<SendStreamState> state);
  void ResolveStepCoordinates(const std::shared_ptr<SendStreamState>& state,
                              size_t* layer, size_t* shard, size_t* block_idx);
  uint32_t GetChunksTotalSize(const std::vector<BlockChunk>& chunks);

  absl::Mutex active_sends_mu_;
  absl::flat_hash_map<uint64_t, std::shared_ptr<SendStreamState>> active_sends_
      ABSL_GUARDED_BY(active_sends_mu_);

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
