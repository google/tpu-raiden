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
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/stream_executor/stream.h"
#include "core/raiden_manager_base.h"
#include "core/raw_transfer_core.h"
#include "kv_cache/logical_block_manager.h"

namespace tpu_raiden {
namespace kv_cache {

struct KVCacheCopySpec {
  std::vector<int64_t> src_offsets;
  std::vector<int64_t> dst_offsets;
  std::vector<int64_t> sizes;
};

class KVCacheManagerBase : public tpu_raiden::RaidenManagerBase {
 public:
  // Core C++ Constructor wrapping raw PJRT buffers directly (used by JAX and
  // PyTorch E2E)
  KVCacheManagerBase(
      const std::vector<std::vector<xla::PjRtBuffer*>>& layer_buffers,
      int block_size = 1, std::optional<int> local_port = std::nullopt,
      std::optional<int> host_blocks_to_allocate = std::nullopt,
      std::optional<std::vector<const uint8_t*>> external_host_ptrs =
          std::nullopt,
      bool unsafe_skip_buffer_lock = false, int transport_parallelism = 1);

  // Standard CPU-only Constructor for remote workers E2E
  KVCacheManagerBase(size_t num_layers, size_t num_shards,
                     size_t slice_byte_size, int block_size = 1,
                     std::optional<int> local_port = std::nullopt,
                     std::optional<int> host_blocks_to_allocate = std::nullopt,
                     int transport_parallelism = 1);

  ~KVCacheManagerBase() override;

  // Async on-chip H2D offloads returning PJRT copy future E2E
  virtual absl::StatusOr<raiden::PjRtCopyFuture> H2d(
      const std::vector<int64_t>& src_offsets_major_dim = {},
      const std::vector<int64_t>& dst_offsets_major_dim = {},
      const std::vector<int64_t>& copy_sizes_major_dim = {});

  // Async on-chip D2H offloads E2E
  virtual absl::StatusOr<raiden::PjRtCopyFuture> D2h(
      const std::vector<int64_t>& src_offsets_major_dim = {},
      const std::vector<int64_t>& dst_offsets_major_dim = {},
      const std::vector<int64_t>& copy_sizes_major_dim = {});

  // Auto-allocating offloads E2E
  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
  D2hAutoAllocate(const std::vector<int64_t>& src_offsets_major_dim = {},
                  const std::vector<int64_t>& copy_sizes_major_dim = {},
                  int64_t entity_id = 0);

  // Symmetrical H2H writes E2E
  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hWrite(
      std::string peer, const std::vector<int>& src_block_ids,
      int64_t entity_id = 0);

  // Symmetrical H2H reads E2E
  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hRead(
      std::string peer, const std::vector<int>& src_block_ids,
      int64_t entity_id = 0);

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
  absl::StatusOr<raiden::PjRtCopyFuture> D2hTo(size_t layer_idx,
                                               void* dst_host_ptr,
                                               size_t dst_size,
                                               const KVCacheCopySpec& copy_spec,
                                               size_t shard_idx = 0);

  // NOTE: This function is temporary. See D2hTo() for details.
  absl::StatusOr<raiden::PjRtCopyFuture> H2dFrom(
      size_t layer_idx, const void* src_host_ptr, size_t src_size,
      const KVCacheCopySpec& copy_spec, size_t shard_idx = 0);

  void SetExternalHostBuffer(
      const std::vector<raiden::BufferHoldAndAlias>& buffer_holds);

 protected:
  const PJRT_Api* c_api_ = nullptr;
  const PJRT_RawBuffer_Extension* extension_ = nullptr;
  size_t physical_size_ = 0;

  std::unique_ptr<LogicalBlockManager> block_manager_;

  // LogicalBlockManager is not thread-safe, but AllocateBlocks() is called
  // concurrently by the transport's receiver ConnectionWorker threads (one per
  // inbound H2H connection: parallelism>1 and/or multiple in-flight requests),
  // while the orchestrator thread concurrently Unlock()s completed blocks. All
  // block_manager_ operations must hold this mutex. Without it, two concurrent
  // Allocate() calls can hand out the SAME free block, so two requests write
  // different KV into the same host block and one silently clobbers the other.
  absl::Mutex block_manager_mutex_;

  // Separate PJRT active holds matrix to protect subclass scoping E2E!
  std::vector<std::vector<raiden::BufferHoldAndAlias>> buffer_holds_;

  // Override parent AllocateBlocks using our dynamic block manager!
  absl::StatusOr<std::vector<int>> AllocateBlocks(size_t num_blocks,
                                                  int64_t entity_id) override {
    absl::MutexLock lock(&block_manager_mutex_);
    return block_manager_->Allocate(num_blocks, entity_id, /*lock=*/true);
  }

  absl::StatusOr<raiden::PjRtCopyFuture> DispatchD2hChunks(
      const std::vector<int64_t>& src_offsets,
      const std::vector<int64_t>& dst_offsets,
      const std::vector<int64_t>& copy_sizes, int64_t device_id = -1);
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_MANAGER_BASE_H_
