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
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/pjrt_client.h"
#include "core/raiden_manager_base.h"
#include "core/raw_transfer_core.h"

namespace tpu_raiden {
namespace weight_sync {

class WeightSynchronizerBase : public tpu_raiden::RaidenManagerBase {
 public:
  // Symmetrical core constructor wrapping raw PJRT buffers directly E2E
  WeightSynchronizerBase(
      const std::vector<std::vector<xla::PjRtBuffer*>>& layer_buffers,
      std::optional<int> local_port = std::nullopt,
      std::optional<std::vector<const uint8_t*>> external_host_ptrs =
          std::nullopt,
      bool unsafe_skip_buffer_lock = false, int parallelism = 1);

  // CPU-only constructor for remote workers and mock E2E testing
  WeightSynchronizerBase(
      size_t num_layers, size_t num_shards, size_t slice_byte_size,
      std::optional<int> local_port = std::nullopt,
      std::optional<int> host_blocks_to_allocate = std::nullopt,
      int parallelism = 1);

  ~WeightSynchronizerBase() override;

  // Trainer pushes current weights to all inference server peers E2E (D2H +
  // network push)
  absl::Status PushWeights(const std::vector<std::string>& peers);

  // Inference server pulls current weights from the source peer E2E (network
  // pull + H2D)
  absl::Status PullWeights(const std::string& source);

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
  const PJRT_Api* c_api_ = nullptr;
  const PJRT_RawBuffer_Extension* extension_ = nullptr;
  size_t physical_size_ = 0;

  // Separate PJRT active holds matrix E2E!
  std::vector<std::vector<raiden::BufferHoldAndAlias>> buffer_holds_;

  // Override parent AllocateBlocks as simple identity indices since we sync
  // entire buffers E2E!
  absl::StatusOr<std::vector<int>> AllocateBlocks(size_t num_blocks,
                                                  int64_t entity_id) override {
    std::vector<int> ids(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i) ids[i] = static_cast<int>(i);
    return ids;
  }

  int GetRemoteReadBlockId(int base_remote_id, int chunk_k) override {
    // Symmetrical identity read maps E2E
    return chunk_k;
  }

  absl::Status OnDataReceived() override;
};

}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_WEIGHT_SYNC_WEIGHT_SYNCHRONIZER_BASE_H_
