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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_FRAMEWORKS_JAX_KV_CACHE_MANAGER_FFI_INTERNAL_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_FRAMEWORKS_JAX_KV_CACHE_MANAGER_FFI_INTERNAL_H_

#include <cstdint>
#include <memory>

#include "absl/strings/string_view.h"
#include "xla/ffi/api/ffi.h"
#include "xla/stream_executor/stream.h"

namespace tpu_raiden {
namespace kv_cache {

extern std::unique_ptr<stream_executor::Stream> g_streams[32];

xla::ffi::Error TriggerRaidenInitImpl(
    xla::ffi::AnyBuffer x, xla::ffi::AnyBuffer shard_idx_buf,
    int64_t slice_byte_size, int32_t local_port, int32_t parallelism,
    int32_t host_blocks_to_allocate, int32_t num_layers,
    absl::string_view local_ip, xla::ffi::Result<xla::ffi::AnyBuffer> out);

xla::ffi::Error TriggerRaidenH2dImpl(xla::ffi::AnyBuffer src_offsets,
                                     xla::ffi::AnyBuffer dst_offsets,
                                     xla::ffi::AnyBuffer copy_sizes,
                                     xla::ffi::AnyBuffer shard_idx_buf,
                                     xla::ffi::AnyBuffer cache_slice_buf,
                                     int32_t layer_idx,
                                     xla::ffi::Result<xla::ffi::AnyBuffer> out);

xla::ffi::Error TriggerRaidenD2hImpl(xla::ffi::AnyBuffer src_offsets,
                                     xla::ffi::AnyBuffer dst_offsets,
                                     xla::ffi::AnyBuffer copy_sizes,
                                     xla::ffi::AnyBuffer shard_idx_buf,
                                     xla::ffi::AnyBuffer cache_slice_buf,
                                     int32_t layer_idx,
                                     xla::ffi::Result<xla::ffi::AnyBuffer> out);

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_FRAMEWORKS_JAX_KV_CACHE_MANAGER_FFI_INTERNAL_H_
