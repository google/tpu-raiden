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
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_raiden/transport/lib/raw_buffer_transport.h"

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

// Receiver-side expectation contract for one plan-declared (pool-keyed)
// transfer. The transport keeps a single progress map for every push; a
// registered plan supplies this spec and switches the expectation source for
// its uuid from the wire header's per-push parallelism to the plan's global
// per-pool push count. expected_pools retires all progress records for the
// uuid once the final declared pool completes.
struct PoolPushProgressSpec {
  size_t expected_pushes = 0;
  size_t expected_pools = 0;
};

// Delegate interface for BlockTransport inheriting raw memory primitives.
class BlockTransportDelegate : public lib::RawBufferTransportDelegate {
 public:
  ~BlockTransportDelegate() override = default;

  virtual bool use_block_chunks(uint64_t uuid) const { return false; }

  // The transport address space is historically one block array per manager
  // layer. Explicit pool tables widen that address space to one block array
  // per pool without changing the wire's integer index.
  virtual size_t num_block_arrays() const { return num_layers(); }

  // Geometry and host bounds for one wire-addressable block array. Legacy
  // delegates inherit the uniform layer behavior; pool-aware delegates may
  // map an array to an interior range of a different backing storage.
  virtual size_t block_bytes(size_t block_array_idx) const {
    return bytes_per_block();
  }
  virtual uint8_t* GetBlockArrayHostPointer(size_t block_array_idx,
                                            size_t shard_idx) {
    return GetHostPointer(block_array_idx, shard_idx);
  }
  virtual size_t GetBlockArrayHostSize(size_t block_array_idx,
                                       size_t shard_idx) {
    return GetHostSize(block_array_idx, shard_idx);
  }

  // Optional second-pass validation for chunked transfers. The transport always
  // validates chunks against the addressed block array first. Delegates with
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
  // (heterogeneous block sizes). If `dst_block_id` is provided (not -1), the
  // sender resolution is restricted to that destination block. This is needed
  // when one source block fans out to multiple blocks on the same peer.
  virtual std::vector<BlockChunk> GetBlockChunks(
      size_t layer_idx, size_t shard_idx, absl::Span<const int64_t> block_ids,
      size_t total_bytes, uint64_t uuid, int64_t sender_node_id = -1,
      absl::string_view peer = "", int64_t src_block_id = -1,
      int64_t dst_block_id = -1) {
    // Default implementation: blocks are contiguous and of uniform size.
    std::vector<BlockChunk> result;
    size_t accumulated_bytes = 0;
    for (int64_t block_id : block_ids) {
      if (accumulated_bytes >= total_bytes) break;
      size_t size =
          std::min(block_bytes(layer_idx), total_bytes - accumulated_bytes);
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

  // Resolves the receive-progress expectation source for one wire-addressed
  // block array. A registered pool plan returns its contract (plan-declared
  // mode); nullopt selects the legacy header-declared mode; an error rejects
  // the push before any payload is read (e.g. a pool outside the plan's
  // transfer set).
  virtual absl::StatusOr<std::optional<PoolPushProgressSpec>>
  GetPoolPushProgressSpec(size_t pool_idx, uint64_t uuid) const {
    return std::nullopt;
  }

  // Pool-keyed counterpart to OnLayerReceived, fired when a declared pool
  // reaches its plan-declared push count. An explicit-pool index is not
  // necessarily a constructor layer/storage index, so the default is a no-op.
  virtual absl::Status OnPoolReceived(size_t pool_idx, uint64_t uuid) {
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
    uint8_t* base = GetBlockArrayHostPointer(layer_idx, shard_idx);
    if (base == nullptr || block_id < 0) {
      return nullptr;
    }
    return base + static_cast<size_t>(block_id) * block_bytes(layer_idx);
  }

  virtual size_t bytes_per_block() const { return slice_byte_size(); }

  virtual int GetRemoteReadBlockId(int base_remote_id, int chunk_k) = 0;

  virtual size_t num_layers() const = 0;
  virtual size_t num_shards() const = 0;
  virtual size_t slice_byte_size() const = 0;
  virtual size_t shard_factor() const = 0;
};

// High-speed Key-Value block transport engine extending RawBufferTransport.
class BlockTransport : public lib::RawBufferTransport {
 public:
  BlockTransport(BlockTransportDelegate* delegate, int local_port,
                 const std::vector<std::string>& local_ips = {},
                 int parallelism = 1);
  ~BlockTransport() override;

  // Asynchronous Scatter-Gather Push
  void AsyncPush(
      const std::vector<std::string>& peers,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids, int parallelism,
      MajorOrder major_order, uint64_t uuid, int layer_idx,
      std::function<void(absl::StatusOr<std::vector<int>>)> on_complete);

  // Synchronous Scatter-Gather Push (op = 1 / op = 6)
  absl::StatusOr<std::vector<int>> SyncPush(
      const std::vector<std::string>& peers,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids = {}, int parallelism = 1,
      MajorOrder major_order = MajorOrder::kLayerMajor, uint64_t uuid = 0,
      int layer_idx = -1);

  // Synchronous Scatter-Gather Pull (op = 2)
  // When explicit_dst_ptrs is supplied it contains one base pointer per
  // (block array, shard), in block-array-major order.
  absl::StatusOr<std::vector<int>> SyncPull(
      const std::vector<std::string>& peers,
      const std::vector<int>& src_block_ids,
      const std::vector<int>& local_block_ids = {},
      const std::vector<uint8_t*>& explicit_dst_ptrs = {}, int parallelism = 1,
      MajorOrder major_order = MajorOrder::kLayerMajor,
      BlockReceivedCallback on_block_received = {}, uint64_t uuid = 0);

  // Drops receive-progress counters belonging to a finished, failed, or
  // timed-out plan so the UUID can be safely reused.
  void ForgetPushProgress(uint64_t uuid);

 private:
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
