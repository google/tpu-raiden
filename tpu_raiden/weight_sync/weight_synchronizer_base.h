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

#ifndef THIRD_PARTY_TPU_RAIDEN_WEIGHT_SYNC_WEIGHT_SYNCHRONIZER_BASE_H_
#define THIRD_PARTY_TPU_RAIDEN_WEIGHT_SYNC_WEIGHT_SYNCHRONIZER_BASE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/pjrt_client.h"
#include "tpu_raiden/core/raiden_manager_base.h"
#include "tpu_raiden/core/raw_transfer_core.h"

namespace tpu_raiden {
namespace rpc {
class StartTransferRequest;
}  // namespace rpc

namespace weight_sync {

class WeightSynchronizerListener;

class WeightSynchronizerBase : public tpu_raiden::RaidenManagerBase {
 public:
  // Symmetrical core constructor wrapping raw PJRT buffers directly E2E
  WeightSynchronizerBase(
      const std::vector<std::vector<xla::PjRtBuffer*>>& layer_buffers,
      std::optional<int> local_port = std::nullopt,
      std::optional<std::vector<const uint8_t*>> external_host_ptrs =
          std::nullopt,
      bool unsafe_skip_buffer_lock = false, int parallelism = 1,
      std::optional<int> listener_port = std::nullopt,
      std::optional<std::string> bind_ip = std::nullopt);

  // CPU-only constructor for remote workers and mock E2E testing
  WeightSynchronizerBase(
      size_t num_layers, size_t num_shards, size_t slice_byte_size,
      std::optional<int> local_port = std::nullopt,
      std::optional<int> host_blocks_to_allocate = std::nullopt,
      int parallelism = 1, std::optional<int> listener_port = std::nullopt,
      std::optional<std::string> bind_ip = std::nullopt);

  std::optional<int> listener_port() const;
  bool is_listener_active() const;

  ~WeightSynchronizerBase() override;

  // Trainer pushes current weights to all inference server peers E2E (D2H +
  // network push)
  absl::Status PushWeights(const std::vector<std::string>& peers);

  /**
   * Executes a distributed resharding push transfer based on precise
   * centralized Controller schedules.
   *
   * Automatically copies active local weight buffers from TPU device HBM to
   * Host staging memory (via D2H), iterates over all active local shards, and
   * pipelines non-contiguous byte chunks across persistent TCP connections to
   * target remote peer host buffers.
   *
   * @param request Demarshaled StartTransferRequest protobuf containing exact
   *                1D memory copy byte chunks, peer network coordinates, and
   *                offset schedules.
   * @return absl::OkStatus() upon complete, successfully ACK-handshaked
   * delivery to all remote peers.
   */
  absl::Status PushWeightsResharded(
      const tpu_raiden::rpc::StartTransferRequest& request);

  // Inference server pulls current weights from the source peer E2E (network
  // pull + H2D)
  absl::Status PullWeights(absl::string_view source);

  void SetExternalHostBuffer(
      const std::vector<raiden::BufferHoldAndAlias>& buffer_holds);

  const uint8_t* GetHostBufferPtr(size_t layer_idx, size_t shard_idx) const {
    if (layer_idx >= num_layers_ || shard_idx >= num_shards_) {
      return nullptr;
    }
    return layers_[layer_idx].shards[shard_idx].host_ptr;
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2d();
  absl::StatusOr<raiden::PjRtCopyFuture> D2h();
  absl::StatusOr<raiden::PjRtCopyFuture> H2dChunk(
      size_t shard_idx, size_t host_offset_bytes, size_t device_offset_bytes,
      size_t size_bytes);

 protected:
  std::unique_ptr<WeightSynchronizerListener> listener_;
  const PJRT_Api* c_api_ = nullptr;
  const PJRT_RawBuffer_Extension* extension_ = nullptr;
  size_t physical_size_ = 0;

  // Separate PJRT active holds matrix E2E!
  std::vector<std::vector<raiden::BufferHoldAndAlias>> buffer_holds_;

  // Override parent AllocateBlocks as simple identity indices since we sync
  // entire buffers E2E!
  absl::StatusOr<std::vector<int>> AllocateBlocks(
      size_t num_blocks, uint64_t uuid = 0) override {
    std::vector<int> ids(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) ids[i] = static_cast<int>(i);
    return ids;
  }

  int GetRemoteReadBlockId(int base_remote_id, int chunk_k) override {
    return base_remote_id + chunk_k;
  }

  absl::Status OnDataReceived() override;
};

}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_WEIGHT_SYNC_WEIGHT_SYNCHRONIZER_BASE_H_
