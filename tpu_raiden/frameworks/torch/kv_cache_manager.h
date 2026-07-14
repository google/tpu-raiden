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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_API_TORCH_KV_CACHE_MANAGER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_API_TORCH_KV_CACHE_MANAGER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "xla/pjrt/pjrt_client.h"
#include "tpu_raiden/core/kv_cache_manager_with_transfer.h"

namespace tpu_raiden {
namespace kv_cache {
class KVCacheListener;
}  // namespace kv_cache
}  // namespace tpu_raiden

namespace tpu_raiden {
namespace torch {

class KVCacheManager : public KVCacheManagerWithTransfer {
 public:
  // PyTorch sharded constructor E2E (cache-only by default)
  KVCacheManager(const std::vector<std::vector<at::Tensor>>& device_tensors,
                 std::optional<int> local_port = std::nullopt,
                 std::optional<int> host_blocks_to_allocate = std::nullopt,
                 bool unsafe_skip_buffer_lock = false, int parallelism = 1);

  // New transfer-enabled constructor (flat list of tensors, single shard per
  // layer)
  KVCacheManager(const std::vector<at::Tensor>& kv_caches, int64_t node_id,
                 int64_t local_control_port, int64_t max_blocks,
                 int64_t num_slots, double timeout_s,
                 bool unsafe_skip_buffer_lock, int parallelism = 4,
                 std::optional<int> listener_port = std::nullopt);

  // Metadata-only constructor for host-memory Stage 1 resharding tests.
  KVCacheManager(size_t num_layers, size_t num_shards, size_t slice_byte_size,
                 int64_t node_id, std::optional<int> local_port,
                 std::optional<int> host_blocks_to_allocate, int parallelism);

  ~KVCacheManager() override;

  // Transfer-specific methods merged from TransferEngine
  std::vector<int64_t> RegisterKvCache(
      const std::vector<at::Tensor>& kv_caches);

  void RegisterHostBuffers(int64_t node_id);

  const std::vector<at::Tensor>& kv_caches() const { return kv_caches_; }

  std::optional<int> listener_port() const;
  bool is_listener_active() const;

  std::string transfer_address() const;
  std::string listener_address() const;

  absl::Status PushRegisteredPlan(uint64_t uuid, const std::string& peer,
                                  const std::vector<int>& src_block_ids,
                                  const std::vector<int>& dst_block_ids,
                                  int layer_idx = -1, int parallelism = 1);

  absl::StatusOr<std::string> ReadBlockBytes(size_t layer_idx, int block_id,
                                             size_t shard_idx = 0);

  absl::Status WriteBlockBytes(size_t layer_idx, int block_id,
                               absl::string_view payload, size_t shard_idx = 0);

 private:
  // Buffers unpacked from a 2D tensor list, together with the owning
  // DeviceBufferRefs that must outlive their use (see UnpackTorchTensor).
  struct UnpackedLayers {
    std::vector<std::vector<xla::PjRtBuffer*>> buffers;
    std::vector<torch_tpu::DeviceBufferRef> refs;
    xla::PjRtClient* client = nullptr;
    std::vector<int64_t> logical_dimensions;
    size_t logical_slice_byte_size = 0;
    size_t logical_physical_size = 0;
    bool has_logical_metadata = false;
  };
  static UnpackedLayers UnpackLayers(
      const std::vector<std::vector<at::Tensor>>& device_tensors);

  // Delegated-to constructor for BOTH public ctors. Moves the keep-alive refs
  // into buffer_refs_ so the materialized device buffers survive for this
  // manager's lifetime. `kv_caches` is retained for the disagg path's
  // kv_caches() accessor (empty for the offload path).
  KVCacheManager(UnpackedLayers unpacked, std::optional<int> local_port,
                 std::optional<int> host_blocks_to_allocate,
                 bool unsafe_skip_buffer_lock, int parallelism, int64_t node_id,
                 int64_t local_control_port, int64_t max_blocks,
                 int64_t num_slots, double timeout_s,
                 std::vector<at::Tensor> kv_caches);

  std::vector<at::Tensor> kv_caches_;
  // Keep-alives for the materialized device buffers backing the manager.
  std::vector<torch_tpu::DeviceBufferRef> buffer_refs_;
  std::unique_ptr<tpu_raiden::kv_cache::KVCacheListener> listener_;
};

}  // namespace torch
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_API_TORCH_KV_CACHE_MANAGER_H_
