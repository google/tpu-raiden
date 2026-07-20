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
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
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
#include "tpu_raiden/kv_cache/pool_layout.h"
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
      std::optional<std::string> bind_ip = std::nullopt,
      std::optional<size_t> logical_slice_byte_size = std::nullopt,
      std::vector<int64_t> logical_dimensions = {},
      std::optional<size_t> logical_physical_size = std::nullopt,
      std::optional<int> assigned_numa_node_override = std::nullopt);

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

  virtual absl::StatusOr<raiden::PjRtCopyFuture> H2dWrite(
      absl::string_view peer,
      const std::vector<int64_t>& src_offsets_major_dim = {},
      const std::vector<int64_t>& dst_offsets_major_dim = {},
      const std::vector<int64_t>& copy_sizes_major_dim = {}) {
    return absl::UnimplementedError("H2dWrite is not implemented");
  }

  virtual absl::StatusOr<raiden::PjRtCopyFuture> H2dRead(
      absl::string_view peer,
      const std::vector<int64_t>& src_offsets_major_dim = {},
      const std::vector<int64_t>& dst_offsets_major_dim = {},
      const std::vector<int64_t>& copy_sizes_major_dim = {}) {
    return absl::UnimplementedError("H2dRead is not implemented");
  }

  // Async on-chip D2H offloads E2E
  virtual absl::StatusOr<raiden::PjRtCopyFuture> D2h(
      const std::vector<int64_t>& src_offsets_major_dim = {},
      const std::vector<int64_t>& dst_offsets_major_dim = {},
      const std::vector<int64_t>& copy_sizes_major_dim = {},
      std::optional<int64_t> slot_idx = std::nullopt,
      std::optional<size_t> layer_idx = std::nullopt,
      std::optional<size_t> shard_idx = std::nullopt);

  virtual absl::StatusOr<raiden::PjRtCopyFuture> D2hWrite(
      absl::string_view peer,
      const std::vector<int64_t>& src_offsets_major_dim = {},
      const std::vector<int64_t>& dst_offsets_major_dim = {},
      const std::vector<int64_t>& copy_sizes_major_dim = {}) {
    return absl::UnimplementedError("D2hWrite is not implemented");
  }

  // Auto-allocating offloads E2E
  virtual absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
  D2hAutoAllocate(const std::vector<int64_t>& src_offsets_major_dim = {},
                  const std::vector<int64_t>& copy_sizes_major_dim = {});

  // Symmetrical H2H writes E2E
  virtual absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
  H2hWrite(const std::vector<std::string>& peers,
           const std::vector<int>& src_block_ids,
           const std::vector<int>& dst_block_ids = {}, uint64_t uuid = 0,
           int layer_idx = -1);

  virtual absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
  H2hWrite(std::string peer, const std::vector<int>& src_block_ids,
           const std::vector<int>& dst_block_ids = {}, uint64_t uuid = 0,
           int layer_idx = -1);

  virtual absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
  H2hRead(const std::vector<std::string>& peers,
          const std::vector<int>& src_block_ids);

  virtual absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
  H2hRead(std::string peer, const std::vector<int>& src_block_ids);

  // Executes a distributed resharding push transfer based on precise
  // centralized Controller schedules.
  absl::Status PushKVCacheResharded(
      const tpu_raiden::rpc::StartTransferRequest& request);

  // Pool-plan executor hooks used by KVCacheListener. Designed as the
  // successor of PushKVCacheResharded/RegisterActivePlan for pool-addressed
  // plans; managers without transfer support fail closed instead of falling
  // back to the legacy whole-layer reshard path.
  virtual absl::Status PoolReshardPush(
      const tpu_raiden::rpc::StartTransferRequest&, absl::Span<const int64_t>,
      int = 8) {
    return absl::UnimplementedError(
        "pool reshard push is not supported by this manager");
  }

  virtual absl::Status PoolReshardRegisterRecv(
      const tpu_raiden::rpc::StartTransferRequest&, absl::Span<const int64_t>) {
    return absl::UnimplementedError(
        "pool reshard receive is not supported by this manager");
  }

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

  // BlockTransport's wire index addresses pools after explicit admission and
  // constructor storages otherwise.
  size_t num_block_arrays() const override;
  size_t block_bytes(size_t block_array_idx) const override;
  uint8_t* GetBlockArrayHostPointer(size_t block_array_idx,
                                    size_t shard_idx) override;
  size_t GetBlockArrayHostSize(size_t block_array_idx,
                               size_t shard_idx) override;

  int64_t LayerBlockByteSize(size_t layer_idx) const;

  // Returns the host address of one legacy (layer, shard, block) block.
  // Shared by the framework bindings; block granularity is bytes_per_block().
  absl::StatusOr<uintptr_t> GetBlockHostPointerValue(size_t layer_idx,
                                                     size_t shard_idx,
                                                     int block_id);

  // Registers explicit block pools over the wrapped storages. Fails while
  // active plans are registered. Until this is called the manager exposes one
  // implicit pool per constructor storage (tag "opaque", base_offset 0,
  // block_stride LayerBlockByteSize) and all legacy paths are unchanged.
  absl::Status RegisterPools(std::vector<PoolSpec> pools);

  absl::StatusOr<PoolBlockRef> GetPoolBlockRef(size_t pool_idx,
                                               size_t shard_idx,
                                               int64_t block_id) const;

  const PoolSpec* pool(size_t pool_idx) const;

  size_t num_pools() const;

  bool has_explicit_pools() const;

  std::vector<size_t> PoolIndicesWithTag(absl::string_view tag) const;

  // Partial D2H of whole pool blocks: copies each block's
  // [base_offset + id*stride, +stride) byte range from the pool's device
  // storage to the same offsets in the host mirror. All shards unless
  // shard_idx is given.
  absl::StatusOr<raiden::PjRtCopyFuture> D2hPoolBlocks(
      size_t pool_idx, absl::Span<const int64_t> block_ids,
      std::optional<size_t> shard_idx = std::nullopt);

  // Inverse of D2hPoolBlocks: host mirror -> device storage.
  absl::StatusOr<raiden::PjRtCopyFuture> H2dPoolBlocks(
      size_t pool_idx, absl::Span<const int64_t> block_ids,
      std::optional<size_t> shard_idx = std::nullopt);

  bool use_block_chunks(uint64_t uuid) const override;

  absl::StatusOr<std::optional<tpu_raiden::transport::PoolPushProgressSpec>>
  GetPoolPushProgressSpec(size_t pool_idx, uint64_t uuid) const override;

  void SetBlockChunkRegionValidation(
      tpu_raiden::transport::BlockChunkRegionValidationMode mode);

  tpu_raiden::transport::BlockChunkRegionValidationMode
  block_chunk_region_validation_mode() const override;

  absl::Status ValidateBlockChunksInRegions(
      size_t layer_idx, size_t shard_idx,
      const std::vector<tpu_raiden::transport::BlockChunk>& chunks) override;

  virtual absl::Status RegisterActivePlan(
      uint64_t uuid, const tpu_raiden::rpc::StartTransferRequest& request,
      bool is_sender);

  virtual absl::Status UnregisterActivePlan(uint64_t uuid);

  virtual absl::Status RegisterRecv(uint64_t uuid, const std::string& req_id,
                                    int64_t expected_block_count) {
    return absl::UnimplementedError(
        "RegisterRecv not implemented in base class");
  }

  // Resolves the host memory pointers (BlockChunks) for the given block_ids.
  // If `src_block_id` is provided (not -1), it is used to filter the active
  // plan to resolve the correct chunk offset, which is necessary when multiple
  // source blocks merge into a single destination block (heterogeneous block
  // sizes). If `dst_block_id` is provided (not -1), sender-side resolution is
  // restricted to schedule entries targeting that destination block, which is
  // necessary when one source block fans out to multiple destination blocks
  // on the same peer.
  std::vector<tpu_raiden::transport::BlockChunk> GetBlockChunks(
      size_t layer_idx, size_t shard_idx, absl::Span<const int64_t> block_ids,
      size_t total_bytes, uint64_t uuid, int64_t sender_node_id = -1,
      absl::string_view peer = "", int64_t src_block_id = -1,
      int64_t dst_block_id = -1) override;

  // With explicit pools the wire index addresses a pool; otherwise the legacy
  // uniform layer addressing applies unchanged.
  uint8_t* GetBlockHostPointer(size_t layer_idx, size_t shard_idx,
                               int block_id) override;

  bool IsDramDestination(uint64_t uuid) const;

 protected:
  const PJRT_Api* c_api_ = nullptr;
  const PJRT_RawBuffer_Extension* extension_ = nullptr;
  size_t max_physical_size_ = 0;
  int64_t staging_num_slots_ = 0;
  int64_t staging_max_major_per_slot_ = 0;
  bool is_blocked_layout_ = false;

  std::unique_ptr<LogicalBlockManager> host_block_manager_;

  // Per-layer device buffer holds bundled with the layer's on-device size.
  // For uniform models every layer has the same physical_size; for hybrid
  // (HMA) models sizes may differ (e.g. mamba conv_state bf16 vs ssm f32).
  struct LayerDeviceInfo {
    std::vector<raiden::BufferHoldAndAlias> holds;
    // Total on-device bytes for this layer's buffer.  Set by the
    // device-backed constructor from PjRtBuffer.  DMA functions use
    // this for per-layer offset and copy-size calculations.
    size_t physical_size = 0;
  };
  std::vector<LayerDeviceInfo> buffer_holds_;
  // Pool table. Explicit after RegisterPools; otherwise lazily materialized
  // implicit pools (one per storage, tag "opaque"). pools_mu_ guards the lazy
  // build and replacement; hot transfer paths read the table without the lock
  // because RegisterPools is barred while plans are active.
  mutable absl::Mutex pools_mu_;
  mutable std::vector<PoolSpec> pools_;
  bool explicit_pools_ = false;
  tpu_raiden::transport::BlockChunkRegionValidationMode
      block_chunk_region_validation_mode_ =
          tpu_raiden::transport::BlockChunkRegionValidationMode::kDisabled;

  // Returns the per-block byte size for a given layer.  Uses the layer's
  // actual physical_size when available (device-backed path); falls back
  // to the uniform slice_byte_size_ (CPU-only / test path).
  int64_t layer_block_byte_size(size_t layer_idx) const {
    const auto& info = buffer_holds_[layer_idx];
    return info.physical_size > 0
               ? static_cast<int64_t>(info.physical_size) / major_dim_size_
               : slice_byte_size_;
  }

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
  // Lazily materializes one implicit pool per storage when RegisterPools was
  // never called (or the host buffers were replaced since).
  void EnsureImplicitPools() const;

  // Grows a storage's host mirror so pool D2H/H2D and block refs can address
  // the whole block array at storage offsets. Host staging is sized at the
  // uniform layer-0 slice by the constructors, which can under-cover
  // heterogeneous storages.
  absl::Status EnsureHostMirrorCovers(size_t storage_idx, int64_t needed_bytes);

  // Shared implementation of D2hPoolBlocks/H2dPoolBlocks.
  absl::StatusOr<raiden::PjRtCopyFuture> CopyPoolBlocks(
      size_t pool_idx, absl::Span<const int64_t> block_ids,
      std::optional<size_t> shard_idx, bool device_to_host);

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
