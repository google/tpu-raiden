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

#include "tpu_raiden/frameworks/jax/weight_synchronizer.h"

#include <memory>
#include <optional>
#include <utility>

#include "tpu_raiden/weight_sync/weight_synchronizer_base.h"

namespace tpu_raiden {
namespace jax {

#ifndef WITHOUT_PYTHON
#include <nanobind/nanobind.h>
#include "tpu_raiden/frameworks/jax/utils.h"

namespace {
UnpackedWeights UnpackAndMove(nanobind::list jax_arrays) {
  auto layer_buffers = tpu_raiden::jax::UnpackJaxArrays(jax_arrays);
  return {std::move(layer_buffers), std::move(jax_arrays)};
}
}  // namespace

WeightSynchronizer::WeightSynchronizer(nanobind::list jax_arrays,
                                       std::optional<int> local_port,
                                       int parallelism,
                                       bool unsafe_skip_buffer_lock,
                                       std::optional<int> listener_port,
                                       std::optional<std::string> bind_ip)
    : WeightSynchronizer(UnpackAndMove(std::move(jax_arrays)), local_port,
                         parallelism, unsafe_skip_buffer_lock, listener_port,
                         bind_ip) {}

WeightSynchronizer::WeightSynchronizer(UnpackedWeights&& weights,
                                       std::optional<int> local_port,
                                       int parallelism,
                                       bool unsafe_skip_buffer_lock,
                                       std::optional<int> listener_port,
                                       std::optional<std::string> bind_ip)
    : jax_arrays_(std::move(weights.jax_arrays)) {
  impl_ = std::make_unique<weight_sync::WeightSynchronizerBase>(
      weights.layer_buffers, local_port,
      /*external_host_ptrs=*/std::nullopt, unsafe_skip_buffer_lock, parallelism,
      listener_port, bind_ip);
}

#endif  // WITHOUT_PYTHON

WeightSynchronizer::~WeightSynchronizer() = default;

// Forwarding methods implementations
absl::Status WeightSynchronizer::PullWeights(absl::string_view source) {
  return impl_->PullWeights(source);
}
absl::StatusOr<raiden::PjRtCopyFuture> WeightSynchronizer::D2h() {
  return impl_->D2h();
}
absl::StatusOr<raiden::PjRtCopyFuture> WeightSynchronizer::H2d() {
  return impl_->H2d();
}
absl::StatusOr<raiden::PjRtCopyFuture> WeightSynchronizer::H2dChunk(
    size_t shard_idx, size_t host_offset_bytes, size_t device_offset_bytes,
    size_t size_bytes) {
  return impl_->H2dChunk(shard_idx, host_offset_bytes, device_offset_bytes,
                         size_bytes);
}
absl::Status WeightSynchronizer::PullWeightsChunk(
    absl::string_view source, size_t src_shard_idx, size_t src_offset_bytes,
    size_t dst_shard_idx, size_t dst_offset_bytes, size_t size_bytes) {
  return impl_->PullWeightsChunk(source, src_shard_idx, src_offset_bytes,
                                 dst_shard_idx, dst_offset_bytes, size_bytes);
}
const uint8_t* WeightSynchronizer::GetHostBufferPtr(size_t layer_idx,
                                                    size_t shard_idx) const {
  return impl_->GetHostBufferPtr(layer_idx, shard_idx);
}
std::optional<int> WeightSynchronizer::local_port() const {
  return impl_->local_port();
}
std::optional<int> WeightSynchronizer::listener_port() const {
  return impl_->listener_port();
}
bool WeightSynchronizer::is_listener_active() const {
  return impl_->is_listener_active();
}
size_t WeightSynchronizer::num_layers() const { return impl_->num_layers(); }
size_t WeightSynchronizer::num_shards() const { return impl_->num_shards(); }
size_t WeightSynchronizer::slice_byte_size() const {
  return impl_->slice_byte_size();
}

}  // namespace jax
}  // namespace tpu_raiden
