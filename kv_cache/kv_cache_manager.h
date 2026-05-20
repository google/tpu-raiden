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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_MANAGER_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_MANAGER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include <nanobind/nanobind.h>
#include "xla/stream_executor/stream.h"
#include "kv_cache/logical_block_manager.h"
#include "raiden_lib/raw_transfer/raw_transfer_impl.h"

namespace tpu_raiden {
namespace kv_cache {

// Manages fast chunked transfers from CPU host memory to TPU device memory
// for key-value cache layers. Pre-extracts layout metadata and raw buffer
// handles to avoid Python overheads during execution.
class KVCacheManager {
 public:
  // Constructs a KVCacheManager wrapping lists of host and device JAX arrays.
  // Both lists must have identical lengths. Pre-extracts raw buffer handles,
  // memory access holds, and physical slice dimensions.
  KVCacheManager(nanobind::list device_arrays, int block_size = 1,
                 std::optional<int> local_port = std::nullopt,
                 int host_blocks_to_allocate = 64,
                 bool unsafe_skip_buffer_lock = false, int parallelism = 1);

  // Overloaded constructor for pure-CPU Host allocations (bypassing JAX and
  // PJRT completely on remote workers)
  KVCacheManager(size_t num_layers, size_t num_shards, size_t slice_byte_size,
                 int block_size = 1,
                 std::optional<int> local_port = std::nullopt,
                 std::optional<int> host_blocks_to_allocate = std::nullopt,
                 int parallelism = 1);

  // Explicit destructor to resolve unique_ptr incomplete type compilation
  // errors
  ~KVCacheManager();

  // Asynchronously copies chunks from host arrays to device arrays.
  // Returns a handle (raiden::PjRtCopyFuture) to await completion.
  absl::StatusOr<raiden::PjRtCopyFuture> H2d(
      nanobind::list src_offsets_major_dim = nanobind::list(),
      nanobind::list dst_offsets_major_dim = nanobind::list(),
      nanobind::list copy_sizes_major_dim = nanobind::list());

  // Overloaded H2d accepting standard C++ vectors directly
  __attribute__((visibility("default"))) absl::StatusOr<raiden::PjRtCopyFuture>
  H2dDirect(const std::vector<int64_t>& src_offsets = {},
            const std::vector<int64_t>& dst_offsets = {},
            const std::vector<int64_t>& copy_sizes = {},
            int64_t device_id = -1);

  // Pure StreamExecutor H2D copy using raw C++ device pointers
  absl::Status H2dDirect(stream_executor::Stream* stream,
                         const std::vector<uint8_t*>& device_buffers,
                         const std::vector<int64_t>& src_offsets,
                         const std::vector<int64_t>& dst_offsets,
                         const std::vector<int64_t>& copy_sizes);

  // Asynchronously copies chunks from device arrays to host arrays.
  // Returns a handle (raiden::PjRtCopyFuture) to await completion.
  absl::StatusOr<raiden::PjRtCopyFuture> D2h(
      nanobind::list src_offsets_major_dim = nanobind::list(),
      nanobind::list dst_offsets_major_dim = nanobind::list(),
      nanobind::list copy_sizes_major_dim = nanobind::list());

  // Overloaded D2h accepting standard C++ vectors directly
  __attribute__((visibility("default"))) absl::StatusOr<raiden::PjRtCopyFuture>
  D2hDirect(const std::vector<int64_t>& src_offsets = {},
            const std::vector<int64_t>& dst_offsets = {},
            const std::vector<int64_t>& copy_sizes = {},
            int64_t device_id = -1);

  // Pure StreamExecutor D2H copy using raw C++ device pointers
  absl::Status D2hDirect(stream_executor::Stream* stream,
                         const std::vector<uint8_t*>& device_buffers,
                         const std::vector<int64_t>& src_offsets,
                         const std::vector<int64_t>& dst_offsets,
                         const std::vector<int64_t>& copy_sizes);

  // Dynamically allocates host memory blocks via LogicalBlockManager and copies
  // sub-region chunks from device arrays into the newly allocated blocks.
  // Returns a pair containing the vector of allocated block IDs and the future
  // handle to await completion.
  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
  D2hAutoAllocate(nanobind::list src_offsets_major_dim = nanobind::list(),
                  nanobind::list copy_sizes_major_dim = nanobind::list(),
                  int64_t entity_id = 0);

  // Direct C++ H2H network write (Push)
  absl::StatusOr<std::vector<int>> H2hWriteDirect(
      const std::string& peer, const std::vector<int32_t>& src_block_ids,
      int64_t entity_id = 0);

  // Direct C++ H2H network read (Pull)
  absl::StatusOr<std::vector<int>> H2hReadDirect(
      const std::string& peer, const std::vector<int32_t>& src_block_ids,
      int64_t entity_id = 0);

  // Pushes logical block chunks symmetrically across all layers over network
  // streams directly into corresponding layer block structures on a remote
  // KVCacheManager peer. Returns a pair containing the vector of successfully
  // allocated target remote block IDs and the future handle to await local push
  // completion.
  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hWrite(
      std::string peer, nanobind::list src_block_ids, int64_t entity_id = 0);

  // Pulls logical block chunks symmetrically across all layers from a remote
  // KVCacheManager peer over network streams directly into corresponding local
  // host memory block structures. Returns a pair containing the vector of
  // successfully allocated local block IDs and the future handle to await
  // network pull completion.
  absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>> H2hRead(
      std::string peer, nanobind::list src_block_ids, int64_t entity_id = 0);

  // Returns the local listening port of the internal transport server if
  // enabled.
  std::optional<int> local_port() const;

  // Returns the allocated host memory virtual pointer address for a given layer
  // and shard.
  const uint8_t* GetHostPointer(size_t layer_idx, size_t shard_idx) const;

 private:
  struct ShardBufferInfo : public raiden::BufferHoldAndAlias {
    const uint8_t* host_ptr = nullptr;
    size_t host_size = 0;
    size_t device_size = 0;
    std::unique_ptr<uint8_t[], void (*)(void*)> owned_host_buffer = {
        nullptr, [](void*) {}};
  };

  struct LayerInfo {
    std::vector<ShardBufferInfo> shards;
  };

  std::optional<nanobind::list> device_arrays_;

  size_t num_layers_ = 0;
  size_t num_shards_ = 0;
  bool is_common_buffer_ = false;

  const PJRT_Api* c_api_ = nullptr;
  const PJRT_RawBuffer_Extension* extension_ = nullptr;

  size_t physical_size_ = 0;
  size_t slice_byte_size_ = 0;

  int64_t major_dim_size_ = 0;
  size_t shard_factor_ = 1;
  int block_size_ = 1;
  int parallelism_ = 1;
  std::unique_ptr<LogicalBlockManager> block_manager_;

  struct BlockTransportServer;
  std::unique_ptr<BlockTransportServer> server_;

  std::vector<LayerInfo> layers_;

  absl::StatusOr<raiden::PjRtCopyFuture> DispatchD2hChunks(
      const std::vector<int64_t>& src_offsets,
      const std::vector<int64_t>& dst_offsets,
      const std::vector<int64_t>& copy_sizes, int64_t device_id = -1);

  void H2hWriteWorker(int stream_idx, const std::string& peer,
                      size_t blocks_per_stream,
                      const nanobind::list& src_block_ids,
                      std::vector<int>& allocated_ids,
                      std::vector<absl::Status>& statuses);

  void H2hReadWorker(int stream_idx, const std::string& peer,
                     size_t blocks_per_stream, size_t remote_blocks_per_stream,
                     int base_remote_id, const std::vector<int>& allocated_ids,
                     std::vector<absl::Status>& statuses);
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_MANAGER_H_
