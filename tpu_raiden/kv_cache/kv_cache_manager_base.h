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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_MANAGER_BASE_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_MANAGER_BASE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/future.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/semaphore.h"
#include "xla/stream_executor/stream.h"
#include "tpu_raiden/core/host_memory_allocator.h"
#include "tpu_raiden/core/numa_thread_pool.h"
#include "tpu_raiden/core/raiden_manager_base.h"
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/kv_cache/logical_block_manager.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"
#include "tpu_raiden/transport/block_transport.h"

namespace tpu_raiden {

struct BlockMetadata {
  int block_id;
  void* data_ptr;
  std::string address;
  xla::PjRtClient* pjrt_client = nullptr;
};

using RecvCallback =
    std::function<absl::Status(int block_id, size_t size_bytes)>;

namespace kv_cache {

struct KVCacheCopySpec {
  std::vector<int64_t> src_offsets;
  std::vector<int64_t> dst_offsets;
  std::vector<int64_t> sizes;
};

struct KVCacheHostSpan {
  uint8_t* ptr = nullptr;
  size_t nbytes = 0;
  int64_t slot_idx = -1;
  int64_t base_major = 0;
  int64_t num_major = 0;
  size_t layer_idx = 0;
  size_t shard_idx = 0;
};

using ::tpu_raiden::HostBufferAllocation;
using ::tpu_raiden::HostBufferAllocator;

class KVCacheManagerBase : public tpu_raiden::RaidenManagerBase {
 public:
  // Core C++ Constructor wrapping raw PJRT buffers directly (used by JAX and
  // PyTorch E2E)
  KVCacheManagerBase(
      const std::vector<std::vector<xla::PjRtBuffer*>>& layer_buffers,
      std::optional<int> local_port = std::nullopt,
      std::optional<int> host_blocks_to_allocate = std::nullopt,
      bool unsafe_skip_buffer_lock = false, int parallelism = 1,
      HostBufferAllocator host_allocator = nullptr,
      std::optional<std::string> bind_ip = std::nullopt);

  // Standard CPU-only Constructor for remote workers E2E
  KVCacheManagerBase(size_t num_layers, size_t num_shards,
                     size_t slice_byte_size,
                     std::optional<int> local_port = std::nullopt,
                     std::optional<int> host_blocks_to_allocate = std::nullopt,
                     int parallelism = 1,
                     HostBufferAllocator host_allocator = nullptr,
                     std::optional<std::string> bind_ip = std::nullopt);

  ~KVCacheManagerBase() override;

  void RegisterBlockReadinessCallback(
      size_t layer_idx, size_t shard_idx, int block_id, uint64_t uuid,
      transport::BlockTransportDelegate::HostBlockReadyCallback cb) override;

  // Async on-chip H2D offloads returning PJRT copy future E2E
  virtual absl::StatusOr<raiden::PjRtCopyFuture> H2d(
      const std::vector<int64_t>& src_offsets_major_dim = {},
      const std::vector<int64_t>& dst_offsets_major_dim = {},
      const std::vector<int64_t>& copy_sizes_major_dim = {},
      std::optional<int64_t> slot_idx = std::nullopt,
      std::optional<size_t> layer_idx = std::nullopt,
      std::optional<size_t> shard_idx = std::nullopt);

  // Async on-chip D2H offloads E2E
  virtual absl::StatusOr<raiden::PjRtCopyFuture> D2h(
      const std::vector<int64_t>& src_offsets_major_dim = {},
      const std::vector<int64_t>& dst_offsets_major_dim = {},
      const std::vector<int64_t>& copy_sizes_major_dim = {},
      std::optional<int64_t> slot_idx = std::nullopt,
      std::optional<size_t> layer_idx = std::nullopt,
      std::optional<size_t> shard_idx = std::nullopt);

  // Auto-allocating offloads E2E
  virtual absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
  D2hAutoAllocate(const std::vector<int64_t>& src_offsets_major_dim = {},
                  const std::vector<int64_t>& copy_sizes_major_dim = {});

  // Symmetrical H2H writes E2E
  virtual absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
  H2hWrite(std::string peer, const std::vector<int>& src_block_ids,
           const std::vector<int>& dst_block_ids = {}, uint64_t uuid = 0,
           int layer_idx = -1);

  virtual absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
  H2hRead(std::string peer, const std::vector<int>& src_block_ids);

  // Executes a distributed resharding push transfer based on precise
  // centralized Controller schedules.
  absl::Status PushKVCacheResharded(
      const tpu_raiden::rpc::StartTransferRequest& request);

  // Blocks until all pending asynchronous transfers/copies are complete.
  virtual absl::Status WaitForPendingWork() { return absl::OkStatus(); }

  virtual absl::StatusOr<raiden::PjRtCopyFuture> H2hReadExplicit(
      std::string peer, const std::vector<int>& src_block_ids,
      const std::vector<int>& local_block_ids,
      const std::vector<uint8_t*>& explicit_dst_ptrs, int parallelism = 1,
      tpu_raiden::transport::MajorOrder major_order =
          tpu_raiden::transport::MajorOrder::kLayerMajor,
      tpu_raiden::transport::BlockReceivedCallback on_block_received = {});

  // Pure StreamExecutor H2D copy using raw C++ device pointers
  absl::Status H2dDirect(stream_executor::Stream* stream,
                         const std::vector<uint8_t*>& device_buffers,
                         const std::vector<int64_t>& src_offsets,
                         const std::vector<int64_t>& dst_offsets,
                         const std::vector<int64_t>& copy_sizes);

  // Pure StreamExecutor D2H copy using raw C++ device pointers
  absl::Status D2hDirect(stream_executor::Stream* stream,
                         const std::vector<uint8_t*>& device_buffers,
                         const std::vector<int64_t>& src_offsets,
                         const std::vector<int64_t>& dst_offsets,
                         const std::vector<int64_t>& copy_sizes);

  absl::StatusOr<raiden::PjRtCopyFuture> H2dDirect(
      const std::vector<int64_t>& src_offsets = {},
      const std::vector<int64_t>& dst_offsets = {},
      const std::vector<int64_t>& copy_sizes = {}, int64_t device_id = -1);

  absl::StatusOr<raiden::PjRtCopyFuture> D2hDirect(
      const std::vector<int64_t>& src_offsets = {},
      const std::vector<int64_t>& dst_offsets = {},
      const std::vector<int64_t>& copy_sizes = {}, int64_t device_id = -1);

  // Layer-wise copies using caller-owned host buffers. The raw copy operation
  // captures the supplied address when issued, so independent calls can overlap
  // across layers.
  // NOTE: These functions are temporary. Long-term, KVCacheManager should own
  // these host buffers to enable serving prefix cache lookups directly from
  // RAM.

  absl::Status ConfigureHostStagingSlots(int64_t num_slots,
                                         int64_t max_major_per_slot);

  absl::StatusOr<KVCacheHostSpan> HostSpan(size_t layer_idx, size_t shard_idx,
                                           int64_t slot_idx, int64_t num_major);

  void SetExternalHostBuffer(
      const std::vector<raiden::BufferHoldAndAlias>& buffer_holds);

  // Returns the internal LogicalBlockManager.
  LogicalBlockManager* host_block_manager() const {
    return host_block_manager_.get();
  }

  size_t bytes_per_block() const override;

  bool use_block_chunks(uint64_t uuid) const override;

  virtual absl::Status RegisterActivePlan(
      uint64_t uuid, const tpu_raiden::rpc::StartTransferRequest& request,
      bool is_sender);

  virtual absl::Status RegisterRecv(uint64_t uuid, const std::string& req_id,
                                    int64_t expected_block_count) {
    return absl::UnimplementedError(
        "RegisterRecv not implemented in base class");
  }

  std::vector<tpu_raiden::transport::BlockChunk> GetBlockChunks(
      size_t layer_idx, size_t shard_idx, int block_id, uint64_t uuid,
      int64_t sender_node_id = -1, absl::string_view peer = "") override;

  bool IsDramDestination(uint64_t uuid) const;

 protected:
  const PJRT_Api* c_api_ = nullptr;
  const PJRT_RawBuffer_Extension* extension_ = nullptr;
  size_t physical_size_ = 0;
  int64_t staging_num_slots_ = 0;
  int64_t staging_max_major_per_slot_ = 0;
  bool is_blocked_layout_ = false;

  std::unique_ptr<LogicalBlockManager> host_block_manager_;

  // Separate PJRT active holds matrix to protect subclass scoping E2E!
  std::vector<std::vector<raiden::BufferHoldAndAlias>> buffer_holds_;

  std::unique_ptr<NumaThreadPool> dma_pool_;
  std::unique_ptr<NumaThreadPool> push_pool_;
  std::unique_ptr<NumaThreadPool> pull_pool_;

  struct CopyWork {
    size_t layer_idx;
    size_t shard_idx;
  };

  absl::StatusOr<std::vector<raiden::PjRtCopyFuture>> DispatchH2dWork(
      const std::vector<CopyWork>& works, std::optional<int64_t> slot_idx,
      bool is_partial, const std::vector<int64_t>& src_offsets_major_dim,
      const std::vector<int64_t>& dst_offsets_major_dim,
      const std::vector<int64_t>& copy_sizes_major_dim);

  absl::StatusOr<std::vector<raiden::PjRtCopyFuture>> DispatchD2hWork(
      const std::vector<CopyWork>& works, std::optional<int64_t> slot_idx,
      bool is_partial, const std::vector<int64_t>& src_offsets,
      const std::vector<int64_t>& dst_offsets,
      const std::vector<int64_t>& copy_sizes);

  // Override parent AllocateBlocks using our dynamic block manager!
  absl::StatusOr<std::vector<int>> AllocateBlocks(size_t num_blocks,
                                                  uint64_t uuid = 0) override {
    return host_block_manager_->Allocate(num_blocks, /*lock=*/true);
  }

  absl::Status OnSingleBlockReceived(int block_id, size_t size_bytes) override;

  absl::StatusOr<std::vector<raiden::PjRtCopyFuture>> DispatchD2hChunks(
      const std::vector<int64_t>& src_offsets,
      const std::vector<int64_t>& dst_offsets,
      const std::vector<int64_t>& copy_sizes,
      std::optional<int64_t> slot_idx = std::nullopt,
      std::optional<size_t> layer_idx = std::nullopt,
      std::optional<size_t> shard_idx = std::nullopt, int64_t device_id = -1);

 private:
  HostBufferAllocator host_allocator_ = nullptr;
  std::unique_ptr<xla::Semaphore> semaphore_;

  absl::Mutex recv_mu_;
  absl::flat_hash_map<int, RecvCallback> recv_callbacks_
      ABSL_GUARDED_BY(recv_mu_);

  mutable absl::Mutex plans_mu_;
  struct RegisteredPlan {
    tpu_raiden::rpc::StartTransferRequest request;
    bool is_sender = false;
  };
  absl::flat_hash_map<uint64_t, RegisteredPlan> active_plans_
      ABSL_GUARDED_BY(plans_mu_);
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_MANAGER_BASE_H_
