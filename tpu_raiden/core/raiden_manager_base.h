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
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/semaphore.h"
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/core/tpu_utils.h"
#include "tpu_raiden/transport/block_transport.h"

namespace tpu_raiden {

class RaidenManagerBase : public tpu_raiden::transport::BlockTransportDelegate {
 public:
  RaidenManagerBase(size_t num_layers, size_t num_shards,
                    size_t slice_byte_size,
                    std::optional<int> local_port = std::nullopt,
                    int parallelism = 1,
                    std::optional<std::string> bind_ip = std::nullopt);

  ~RaidenManagerBase() override;

  // Direct C++ H2H network write (Push)
  absl::StatusOr<std::vector<int>> H2hWriteDirect(
      absl::string_view peer, const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids = {}, uint64_t uuid = 0,
      int layer_idx = -1);

  void H2hWriteDirectAsync(
      absl::string_view peer, const std::vector<int>& src_block_ids,
      const std::vector<int>& dst_block_ids, uint64_t uuid, int layer_idx,
      std::function<void(absl::StatusOr<std::vector<int>>)> on_complete);

  // Direct C++ H2H network read (Pull)
  absl::StatusOr<std::vector<int>> H2hReadDirect(
      absl::string_view peer, const std::vector<int>& src_block_ids);

  absl::Status PullWeightsChunk(absl::string_view source, size_t src_shard_idx,
                                size_t src_offset_bytes, size_t dst_shard_idx,
                                size_t dst_offset_bytes, size_t size_bytes);

  absl::Status PushWeightsChunk(absl::string_view peer, size_t dst_shard_idx,
                                size_t dst_offset_bytes,
                                const uint8_t* data_ptr, size_t size_bytes);

  virtual std::optional<int> local_port() const;
  virtual std::string local_ip() const;
  virtual std::vector<std::string> local_ips() const;
  std::optional<int> assigned_numa_node() const { return assigned_numa_node_; }

  uint8_t* GetHostPointer(size_t layer_idx, size_t shard_idx) override;
  size_t GetHostSize(size_t layer_idx, size_t shard_idx) override;

  virtual const uint8_t* GetHostPointer(size_t layer_idx,
                                        size_t shard_idx) const;

  void SetExternalHostPointers(const std::vector<const uint8_t*>& host_ptrs,
                               const std::vector<size_t>& host_sizes);

  // Delegate overrides E2E
  size_t num_layers() const override { return num_layers_; }
  size_t num_shards() const override { return num_shards_; }
  size_t slice_byte_size() const override { return slice_byte_size_; }
  size_t bytes_per_block() const override;
  size_t shard_factor() const override { return shard_factor_; }
  absl::Status WaitForBlockRead(size_t layer_idx, size_t shard_idx,
                                int block_id) override {
    return absl::OkStatus();
  }

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
  int parallelism_ = 1;
  size_t shard_factor_ = 1;
  int64_t major_dim_size_ = 0;
  std::optional<int> assigned_numa_node_ = std::nullopt;
  int local_port_cfg_ = 0;
  std::optional<std::string> bind_ip_cfg_ = std::nullopt;
  std::vector<std::string> local_ips_;

  void InitTransportServer();
  void ResolveLocalIpsLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(server_init_mu_);
  virtual std::vector<HostNicAddress> GetHostNics() const;

  void DetectAndAssignNumaNode(
      const std::vector<std::vector<xla::PjRtBuffer*>>& layer_buffers);

  mutable absl::Mutex server_init_mu_;
  std::unique_ptr<tpu_raiden::transport::BlockTransport> server_
      ABSL_GUARDED_BY(server_init_mu_);

  std::vector<LayerInfoBase> layers_;

  // Delegate allocator overrides
  absl::StatusOr<std::vector<int>> AllocateBlocks(size_t num_blocks,
                                                  uint64_t uuid = 0) override {
    return absl::UnimplementedError("Block allocator is not available");
  }

  int GetRemoteReadBlockId(int base_remote_id, int chunk_k) override {
    return base_remote_id + chunk_k;
  }

  absl::Status OnDataReceived() override { return absl::OkStatus(); }
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_RAIDEN_MANAGER_BASE_H_
