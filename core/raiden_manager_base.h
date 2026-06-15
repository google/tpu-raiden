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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_RAIDEN_MANAGER_BASE_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_RAIDEN_MANAGER_BASE_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/semaphore.h"
#include "core/raw_transfer_core.h"
#include "transport/block_transport.h"

namespace tpu_raiden {

struct BlockMetadata {
  int block_id;
  void* data_ptr;
  std::string address;
  xla::PjRtClient* pjrt_client = nullptr;
};

using RecvCallback =
    std::function<absl::Status(int block_id, size_t size_bytes)>;
using BlockReadinessCallback =
    std::function<absl::Status(size_t layer_idx, size_t shard_idx,
                               int block_id)>;

class RaidenManagerBase : public tpu_raiden::transport::BlockTransportDelegate {
 public:
  RaidenManagerBase(size_t num_layers, size_t num_shards,
                    size_t slice_byte_size, int block_size = 1,
                    std::optional<int> local_port = std::nullopt,
                    int parallelism = 1, size_t max_staging_blocks = 4);

  xla::Future<> RemoteD2DBlockWrite(const BlockMetadata& src,
                                    const BlockMetadata& dst,
                                    size_t size_bytes);

  xla::Future<> RemoteD2DBlockReceive(int block_id,
                                      raiden::BufferHoldAndAlias hold,
                                      size_t size_bytes);

  ~RaidenManagerBase() override;

  // Direct C++ H2H network write (Push)
  absl::StatusOr<std::vector<int>> H2hWriteDirect(
      const std::string& peer, const std::vector<int>& src_block_ids,
      int64_t entity_id = 0);

  // Direct C++ H2H network read (Pull)
  absl::StatusOr<std::vector<int>> H2hReadDirect(
      const std::string& peer, const std::vector<int>& src_block_ids,
      int64_t entity_id = 0);

  absl::Status PullWeightsChunk(const std::string& source, size_t src_shard_idx,
                                size_t src_offset_bytes, size_t dst_shard_idx,
                                size_t dst_offset_bytes, size_t size_bytes);

  absl::Status PushWeightsChunk(const std::string& peer, size_t dst_shard_idx,
                                size_t dst_offset_bytes,
                                const uint8_t* data_ptr, size_t size_bytes);

  std::optional<int> local_port() const;

  uint8_t* GetHostPointer(size_t layer_idx, size_t shard_idx) override;
  size_t GetHostSize(size_t layer_idx, size_t shard_idx) override;

  const uint8_t* GetHostPointer(size_t layer_idx, size_t shard_idx) const;

  void SetExternalHostPointers(const std::vector<const uint8_t*>& host_ptrs,
                               const std::vector<size_t>& host_sizes);

  void SetBlockReadinessCallback(BlockReadinessCallback callback);

  // Delegate overrides E2E
  size_t num_layers() const override { return num_layers_; }
  size_t num_shards() const override { return num_shards_; }
  size_t slice_byte_size() const override { return slice_byte_size_; }
  int block_size() const override { return block_size_; }
  size_t bytes_per_block() const override;
  size_t shard_factor() const override { return shard_factor_; }
  absl::Status WaitForBlockRead(size_t layer_idx, size_t shard_idx,
                                int block_id) override;

 protected:
  struct ShardBufferInfoBase {
    const uint8_t* host_ptr = nullptr;
    size_t host_size = 0;
    size_t device_size = 0;
    std::unique_ptr<uint8_t[], void (*)(void*)> owned_host_buffer = {
        nullptr, [](void*) {}};
    std::shared_ptr<void> host_owner;
  };

  struct LayerInfoBase {
    std::vector<ShardBufferInfoBase> shards;
  };

  size_t num_layers_ = 0;
  size_t num_shards_ = 0;
  size_t slice_byte_size_ = 0;
  int block_size_ = 1;
  int parallelism_ = 1;
  size_t shard_factor_ = 1;
  int64_t major_dim_size_ = 0;

  std::unique_ptr<tpu_raiden::transport::BlockTransport> server_;

  std::vector<LayerInfoBase> layers_;

  // Delegate allocator overrides
  absl::StatusOr<std::vector<int>> AllocateBlocks(size_t num_blocks,
                                                  int64_t entity_id) override {
    return absl::UnimplementedError("Block allocator is not available");
  }

  int GetRemoteReadBlockId(int base_remote_id, int chunk_k) override {
    return base_remote_id + chunk_k;
  }

  absl::Status OnDataReceived() override { return absl::OkStatus(); }

  absl::Status OnSingleBlockReceived(int block_id, size_t size_bytes) override;

 private:
  xla::Future<> DoD2DTransfer(const BlockMetadata& src,
                              const BlockMetadata& dst, size_t size_bytes);

  std::unique_ptr<xla::Semaphore> semaphore_;

  absl::Mutex recv_mu_;
  absl::flat_hash_map<int, RecvCallback> recv_callbacks_
      ABSL_GUARDED_BY(recv_mu_);

  absl::Mutex block_readiness_mu_;
  BlockReadinessCallback block_readiness_callback_
      ABSL_GUARDED_BY(block_readiness_mu_);
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_RAIDEN_MANAGER_BASE_H_
