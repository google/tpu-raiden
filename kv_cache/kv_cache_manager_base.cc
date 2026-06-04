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

#include "kv_cache/kv_cache_manager_base.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "core/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "core/raw_transfer_core.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

absl::Status ValidateCopySpecStatus(const KVCacheCopySpec& copy_spec) {
  const bool present = !copy_spec.src_offsets.empty() ||
                       !copy_spec.dst_offsets.empty() ||
                       !copy_spec.sizes.empty();
  if (present &&
      (copy_spec.src_offsets.size() != copy_spec.dst_offsets.size() ||
       copy_spec.src_offsets.size() != copy_spec.sizes.size())) {
    return absl::InvalidArgumentError(
        "src_offsets, dst_offsets, and sizes must have the same length");
  }
  for (size_t i = 0; i < copy_spec.src_offsets.size(); ++i) {
    if (copy_spec.src_offsets[i] < 0 || copy_spec.dst_offsets[i] < 0 ||
        copy_spec.sizes[i] < 0) {
      return absl::InvalidArgumentError(
          "copy offsets and sizes must be non-negative");
    }
  }
  return absl::OkStatus();
}

bool IsPartialCopy(const KVCacheCopySpec& copy_spec, int64_t major_dim_size) {
  if (copy_spec.src_offsets.empty()) return false;
  if (major_dim_size <= 0) return true;
  for (size_t i = 0; i < copy_spec.src_offsets.size(); ++i) {
    if (copy_spec.src_offsets[i] != 0 || copy_spec.dst_offsets[i] != 0 ||
        copy_spec.sizes[i] != major_dim_size) {
      return true;
    }
  }
  return false;
}

}  // namespace

KVCacheManagerBase::KVCacheManagerBase(
    const std::vector<std::vector<xla::PjRtBuffer*>>& layer_buffers,
    int block_size, std::optional<int> local_port,
    std::optional<int> host_blocks_to_allocate,
    std::optional<std::vector<const uint8_t*>> external_host_ptrs,
    bool unsafe_skip_buffer_lock, int parallelism,
    HostBufferAllocator host_allocator)
    : RaidenManagerBase(layer_buffers.size(),
                        layer_buffers.empty() ? 0 : layer_buffers[0].size(),
                        layer_buffers.empty() ? 0
                                              : raiden::GetMajorSliceByteSize(
                                                    layer_buffers[0][0]),
                        block_size, local_port, parallelism) {
  if (num_layers_ == 0 || num_shards_ == 0) {
    return;
  }

  xla::PjRtBuffer* first_buffer = layer_buffers[0][0];
  const xla::Shape& shape = first_buffer->on_device_shape();

  is_blocked_layout_ = (shape.dimensions().size() == 5);

  physical_size_ = first_buffer->GetOnDeviceSizeInBytes().value();
  extension_ = raiden::GetRawBufferExtension(first_buffer, &c_api_);

  int total_blocks = 0;
  if (!shape.dimensions().empty()) {
    major_dim_size_ = shape.dimensions(0);
    total_blocks =
        is_blocked_layout_ ? major_dim_size_ : (major_dim_size_ / block_size_);
    block_manager_ = std::make_unique<LogicalBlockManager>(total_blocks);
  }

  int num_host_blocks = host_blocks_to_allocate.value_or(64);
  if (external_host_ptrs.has_value()) {
    num_host_blocks = total_blocks;
  }

  size_t shard_idx = 0;
  layers_.reserve(num_layers_);
  buffer_holds_.reserve(num_layers_);

  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    const auto& dst_buffers = layer_buffers[layer_idx];
    if (dst_buffers.size() != num_shards_) {
      throw std::runtime_error("Number of shards mismatch across layers");
    }

    LayerInfoBase layer_info;
    layer_info.shards.reserve(num_shards_);
    std::vector<raiden::BufferHoldAndAlias> hold_info;
    hold_info.reserve(num_shards_);

    for (size_t i = 0; i < num_shards_; ++i) {
      xla::PjRtBuffer* dst_buffer = dst_buffers[i];
      ShardBufferInfoBase shard_info;

      shard_info.device_size = dst_buffer->GetOnDeviceSizeInBytes().value();
      if (shard_info.device_size < physical_size_) {
        throw std::runtime_error(
            "Device buffer shard size smaller than physical size");
      }

      size_t alloc_size = num_host_blocks * bytes_per_block();
      if (external_host_ptrs.has_value()) {
        if (shard_idx < external_host_ptrs->size()) {
          shard_info.host_ptr = (*external_host_ptrs)[shard_idx];
        } else {
          throw std::invalid_argument("External host pointers size mismatch");
        }
        shard_info.host_size = alloc_size;
        shard_idx++;
      } else if (host_allocator) {
        auto status_or_allocation = host_allocator(alloc_size);
        if (!status_or_allocation.ok()) {
          throw std::runtime_error(absl::StrCat(
              "Host allocator failed for size: ", alloc_size,
              ", error: ", status_or_allocation.status().ToString()));
        }
        HostBufferAllocation allocation = std::move(status_or_allocation).value();
        if (alloc_size > 0 && allocation.ptr == nullptr) {
          throw std::runtime_error(absl::StrCat(
              "Host allocator returned null buffer for size: ", alloc_size));
        }
        if (allocation.size < alloc_size) {
          throw std::runtime_error(absl::StrCat(
              "Host allocator returned undersized buffer. Requested: ",
              alloc_size, ", allocated: ", allocation.size));
        }
        shard_info.host_ptr = allocation.ptr;
        shard_info.host_size = allocation.size;
        shard_info.host_owner = std::move(allocation.owner);
      } else {
        void* ptr = nullptr;
        if (alloc_size > 0) {
          if (posix_memalign(&ptr, 64, alloc_size) != 0) {
            throw std::runtime_error(absl::StrCat(
                "Failed to allocate host buffer of size: ", alloc_size));
          }
          std::memset(ptr, 0, alloc_size);
        }
        shard_info.owned_host_buffer =
            std::unique_ptr<uint8_t[], void (*)(void*)>(
                static_cast<uint8_t*>(ptr), [](void* p) { free(p); });
        shard_info.host_ptr = shard_info.owned_host_buffer.get();
        shard_info.host_size = alloc_size;
      }

      auto status_or_hold = raiden::BufferHoldAndAlias::Acquire(
          dst_buffer, c_api_, extension_, unsafe_skip_buffer_lock);
      if (!status_or_hold.ok()) {
        throw std::runtime_error(
            std::string("Failed to acquire PJRT hold: ") +
            std::string(status_or_hold.status().message()));
      }
      hold_info.push_back(std::move(status_or_hold.value()));
      layer_info.shards.push_back(std::move(shard_info));
    }
    layers_.push_back(std::move(layer_info));
    buffer_holds_.push_back(std::move(hold_info));
  }
}

KVCacheManagerBase::KVCacheManagerBase(
    size_t num_layers, size_t num_shards, size_t slice_byte_size,
    int block_size, std::optional<int> local_port,
    std::optional<int> host_blocks_to_allocate, int parallelism,
    HostBufferAllocator host_allocator)
    : RaidenManagerBase(num_layers, num_shards, slice_byte_size, block_size,
                        local_port, parallelism) {
  int total_blocks = host_blocks_to_allocate.value_or(0);
  block_manager_ = std::make_unique<LogicalBlockManager>(total_blocks);

  layers_.reserve(num_layers_);
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    LayerInfoBase layer_info;
    layer_info.shards.reserve(num_shards_);

    for (size_t i = 0; i < num_shards_; ++i) {
      ShardBufferInfoBase shard_info;

      int num_host_blocks = host_blocks_to_allocate.value_or(0);
      size_t alloc_size = num_host_blocks * bytes_per_block();
      if (host_allocator) {
        auto status_or_allocation = host_allocator(alloc_size);
        if (!status_or_allocation.ok()) {
          throw std::runtime_error(absl::StrCat(
              "Host allocator failed for size: ", alloc_size,
              ", error: ", status_or_allocation.status().ToString()));
        }
        HostBufferAllocation allocation = std::move(status_or_allocation).value();
        if (alloc_size > 0 && allocation.ptr == nullptr) {
          throw std::runtime_error(absl::StrCat(
              "Host allocator returned null buffer for size: ", alloc_size));
        }
        if (allocation.size < alloc_size) {
          throw std::runtime_error(absl::StrCat(
              "Host allocator returned undersized buffer. Requested: ",
              alloc_size, ", allocated: ", allocation.size));
        }
        shard_info.host_ptr = allocation.ptr;
        shard_info.host_size = allocation.size;
        shard_info.host_owner = std::move(allocation.owner);
      } else {
        void* ptr = nullptr;
        if (alloc_size > 0) {
          if (posix_memalign(&ptr, 64, alloc_size) != 0) {
            throw std::runtime_error(absl::StrCat(
                "Failed to allocate host buffer of size: ", alloc_size));
          }
          std::memset(ptr, 0, alloc_size);
        }
        shard_info.owned_host_buffer =
            std::unique_ptr<uint8_t[], void (*)(void*)>(
                static_cast<uint8_t*>(ptr), [](void* p) { free(p); });
        shard_info.host_ptr = shard_info.owned_host_buffer.get();
        shard_info.host_size = alloc_size;
      }

      layer_info.shards.push_back(std::move(shard_info));
    }
    layers_.push_back(std::move(layer_info));
  }
}

KVCacheManagerBase::~KVCacheManagerBase() {
  buffer_holds_.clear();
  layers_.clear();
  block_manager_.reset();
}

absl::StatusOr<raiden::PjRtCopyFuture> KVCacheManagerBase::H2d(
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  bool is_partial = !src_offsets_major_dim.empty();
  if (is_partial) {
    if (src_offsets_major_dim.size() != dst_offsets_major_dim.size() ||
        src_offsets_major_dim.size() != copy_sizes_major_dim.size()) {
      return absl::InvalidArgumentError(
          "Lengths of offset and size lists must match");
    }
  }

  std::vector<xla::Future<raiden::BufferHolder>> shard_futures_to_join;
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    const auto& layer_info = layers_[layer_idx];
    const auto& layer_holds = buffer_holds_[layer_idx];
    for (size_t i = 0; i < num_shards_; ++i) {
      const auto& shard_info = layer_info.shards[i];
      const auto& shard_hold = layer_holds[i];

      std::vector<xla::Future<>> shard_futures;
      if (!is_partial) {
        xla::Future<> future = shard_hold.CopyRawHostToDevice(
            shard_info.host_ptr, 0, physical_size_);
        shard_futures.push_back(std::move(future));
      } else {
        for (size_t j = 0; j < src_offsets_major_dim.size(); ++j) {
          int64_t src_major_dim_offset = src_offsets_major_dim[j];
          int64_t dst_major_dim_offset = dst_offsets_major_dim[j];
          int64_t major_dim_size = copy_sizes_major_dim[j];

          int64_t src_offset = src_major_dim_offset * slice_byte_size_;
          int64_t dst_offset = dst_major_dim_offset * slice_byte_size_;
          int64_t size_to_copy = major_dim_size * slice_byte_size_;

          if (src_offset + size_to_copy > shard_info.host_size) {
            return absl::InvalidArgumentError(
                "Copy range exceeds source host buffer size");
          }
          if (dst_offset + size_to_copy > shard_info.device_size) {
            return absl::InvalidArgumentError(
                "Copy range exceeds destination device buffer size");
          }

          const uint8_t* src_ptr = shard_info.host_ptr + src_offset;
          xla::Future<> future =
              shard_hold.CopyRawHostToDevice(src_ptr, dst_offset, size_to_copy);
          shard_futures.push_back(std::move(future));
        }
      }
      shard_futures_to_join.push_back(raiden::CreateBufferFuture(
          std::move(shard_futures), shard_hold.c_hold, shard_hold.common_hold));
    }
  }
  return xla::JoinFutures(absl::MakeSpan(shard_futures_to_join));
}

absl::StatusOr<raiden::PjRtCopyFuture> KVCacheManagerBase::DispatchD2hChunks(
    const std::vector<int64_t>& src_offsets,
    const std::vector<int64_t>& dst_offsets,
    const std::vector<int64_t>& copy_sizes, int64_t device_id) {
  bool is_partial = !src_offsets.empty();
  std::vector<xla::Future<raiden::BufferHolder>> shard_futures_to_join;

  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    const auto& layer_info = layers_[layer_idx];
    const auto& layer_holds = buffer_holds_[layer_idx];
    for (size_t i = 0; i < num_shards_; ++i) {
      if (device_id >= 0 && static_cast<int64_t>(i) != device_id) {
        continue;
      }
      const auto& shard_info = layer_info.shards[i];
      const auto& shard_hold = layer_holds[i];
      uint8_t* dst_host_ptr = const_cast<uint8_t*>(shard_info.host_ptr);

      std::vector<xla::Future<>> shard_futures;
      if (!is_partial) {
        if (dst_host_ptr == nullptr) {
          return absl::FailedPreconditionError(
              "Destination host pointer is null");
        }
        if (physical_size_ > shard_info.host_size) {
          return absl::OutOfRangeError(
              "Copy range exceeds destination host buffer size");
        }
        xla::Future<> future =
            shard_hold.CopyRawDeviceToHost(dst_host_ptr, 0, physical_size_);
        shard_futures.push_back(std::move(future));
      } else {
        shard_futures.reserve(src_offsets.size());
        for (size_t j = 0; j < src_offsets.size(); ++j) {
          int64_t src_offset = src_offsets[j] * slice_byte_size_;
          int64_t dst_offset = dst_offsets[j] * slice_byte_size_;
          int64_t size_to_copy = copy_sizes[j] * slice_byte_size_;

          if (src_offset + size_to_copy > shard_info.device_size) {
            return absl::InvalidArgumentError(
                "Copy range exceeds source device buffer size");
          }
          if (dst_offset + size_to_copy > shard_info.host_size) {
            return absl::InvalidArgumentError(
                "Copy range exceeds destination host buffer size");
          }

          uint8_t* dst_ptr = dst_host_ptr + dst_offset;
          xla::Future<> future =
              shard_hold.CopyRawDeviceToHost(dst_ptr, src_offset, size_to_copy);
          shard_futures.push_back(std::move(future));
        }
      }
      shard_futures_to_join.push_back(raiden::CreateBufferFuture(
          std::move(shard_futures), shard_hold.c_hold, shard_hold.common_hold));
    }
  }
  return xla::JoinFutures(absl::MakeSpan(shard_futures_to_join));
}

absl::StatusOr<raiden::PjRtCopyFuture> KVCacheManagerBase::D2h(
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  size_t num_chunks = src_offsets_major_dim.size();
  if (num_chunks != dst_offsets_major_dim.size() ||
      num_chunks != copy_sizes_major_dim.size()) {
    return absl::InvalidArgumentError(
        "Lengths of offset and size lists must match");
  }
  return DispatchD2hChunks(src_offsets_major_dim, dst_offsets_major_dim,
                           copy_sizes_major_dim);
}

absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
KVCacheManagerBase::D2hAutoAllocate(
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim, int64_t entity_id) {
  size_t num_chunks = src_offsets_major_dim.size();
  if (num_chunks != copy_sizes_major_dim.size()) {
    return absl::InvalidArgumentError(
        "Lengths of offset and size lists must match");
  }

  int total_blocks_to_allocate = 0;
  std::vector<int> blocks_per_chunk;
  blocks_per_chunk.reserve(num_chunks);

  for (size_t j = 0; j < num_chunks; ++j) {
    int64_t copy_size = copy_sizes_major_dim[j];
    if (copy_size % block_size_ != 0) {
      return absl::InvalidArgumentError(
          "Copy size must be a multiple of block size");
    }
    int needed = copy_size / block_size_;
    total_blocks_to_allocate += needed;
    blocks_per_chunk.push_back(needed);
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<int> allocated_block_ids,
                        AllocateBlocks(total_blocks_to_allocate, entity_id));

  std::vector<int64_t> flat_src_offsets;
  std::vector<int64_t> flat_dst_offsets;
  std::vector<int64_t> flat_copy_sizes;
  flat_src_offsets.reserve(total_blocks_to_allocate);
  flat_dst_offsets.reserve(total_blocks_to_allocate);
  flat_copy_sizes.reserve(total_blocks_to_allocate);

  size_t block_id_idx = 0;
  for (size_t j = 0; j < num_chunks; ++j) {
    int64_t src_major_dim_offset = src_offsets_major_dim[j];
    int needed = blocks_per_chunk[j];

    for (int k = 0; k < needed; ++k) {
      int assigned_block_id = allocated_block_ids[block_id_idx++];
      flat_src_offsets.push_back(src_major_dim_offset + k * block_size_);
      flat_dst_offsets.push_back(assigned_block_id * block_size_);
      flat_copy_sizes.push_back(block_size_);
    }
  }

  ABSL_ASSIGN_OR_RETURN(
      raiden::PjRtCopyFuture future,
      DispatchD2hChunks(flat_src_offsets, flat_dst_offsets, flat_copy_sizes));
  return std::make_pair(allocated_block_ids, std::move(future));
}

absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
KVCacheManagerBase::H2hWrite(std::string peer,
                             const std::vector<int>& src_block_ids,
                             int64_t entity_id) {
  ABSL_ASSIGN_OR_RETURN(std::vector<int> allocated_ids,
                        H2hWriteDirect(peer, src_block_ids, entity_id));
  return std::make_pair(
      allocated_ids,
      raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{}));
}

absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
KVCacheManagerBase::H2hRead(std::string peer,
                            const std::vector<int>& src_block_ids,
                            int64_t entity_id) {
  ABSL_ASSIGN_OR_RETURN(std::vector<int> allocated_ids,
                        H2hReadDirect(peer, src_block_ids, entity_id));
  return std::make_pair(
      allocated_ids,
      raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{}));
}

absl::StatusOr<raiden::PjRtCopyFuture> KVCacheManagerBase::H2hReadExplicit(
    std::string peer, const std::vector<int>& src_block_ids,
    const std::vector<int>& local_block_ids,
    const std::vector<uint8_t*>& explicit_dst_ptrs) {
  if (!server_) {
    return absl::FailedPreconditionError("Transport server is not running");
  }
  ABSL_ASSIGN_OR_RETURN(std::vector<int> allocated_ids,
                        server_->Pull(peer, src_block_ids, local_block_ids,
                                      explicit_dst_ptrs, parallelism_));
  return raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{});
}

void KVCacheManagerBase::SetExternalHostBuffer(
    const std::vector<raiden::BufferHoldAndAlias>& buffer_holds) {
  size_t idx = 0;
  for (size_t l = 0; l < num_layers_; ++l) {
    for (size_t sh = 0; sh < num_shards_; ++sh) {
      if (idx < buffer_holds.size()) {
        auto u_ptr_or = buffer_holds[idx].buffer->client()->UnsafeBufferPointer(
            buffer_holds[idx].buffer);
        if (u_ptr_or.ok()) {
          layers_[l].shards[sh].host_ptr =
              reinterpret_cast<uint8_t*>(u_ptr_or.value());
          layers_[l].shards[sh].host_size =
              buffer_holds[idx].buffer->GetOnDeviceSizeInBytes().value();
        }
        idx++;
      }
    }
  }
}

absl::Status KVCacheManagerBase::H2dDirect(
    stream_executor::Stream* stream,
    const std::vector<uint8_t*>& device_buffers,
    const std::vector<int64_t>& src_offsets,
    const std::vector<int64_t>& dst_offsets,
    const std::vector<int64_t>& copy_sizes) {
  if (src_offsets.size() != dst_offsets.size() ||
      src_offsets.size() != copy_sizes.size()) {
    return absl::InvalidArgumentError(
        "Lengths of offset and size vectors must match");
  }
  if (device_buffers.size() != num_layers_) {
    return absl::InvalidArgumentError(
        "Number of device buffers must match layer count");
  }

  int64_t block_byte_size = block_size_ * slice_byte_size_;
  int64_t num_chunks = src_offsets.size();

  for (size_t l = 0; l < num_layers_; ++l) {
    const auto& layer_info = layers_[l];
    const auto& shard_info = layer_info.shards[0];
    const uint8_t* h_base = shard_info.host_ptr;
    uint8_t* d_base = device_buffers[l];

    for (int64_t i = 0; i < num_chunks; ++i) {
      int64_t copy_size = copy_sizes[i];
      if (copy_size == 0) continue;

      int64_t s_offset = src_offsets[i] * block_byte_size;
      int64_t d_offset = dst_offsets[i] * block_byte_size;
      size_t size_bytes = copy_size * block_byte_size;

      const uint8_t* src_ptr = h_base + s_offset;
      uint8_t* dst_ptr = d_base + d_offset;

      stream_executor::DeviceAddressBase device_addr(dst_ptr, size_bytes);
      ABSL_RETURN_IF_ERROR(stream->Memcpy(&device_addr, src_ptr, size_bytes));
    }
  }
  return absl::OkStatus();
}

absl::Status KVCacheManagerBase::D2hDirect(
    stream_executor::Stream* stream,
    const std::vector<uint8_t*>& device_buffers,
    const std::vector<int64_t>& src_offsets,
    const std::vector<int64_t>& dst_offsets,
    const std::vector<int64_t>& copy_sizes) {
  if (src_offsets.size() != dst_offsets.size() ||
      src_offsets.size() != copy_sizes.size()) {
    return absl::InvalidArgumentError(
        "Lengths of offset and size vectors must match");
  }
  if (device_buffers.size() != num_layers_) {
    return absl::InvalidArgumentError(
        "Number of device buffers must match layer count");
  }

  int64_t block_byte_size = block_size_ * slice_byte_size_;
  int64_t num_chunks = src_offsets.size();

  for (size_t l = 0; l < num_layers_; ++l) {
    const auto& layer_info = layers_[l];
    const auto& shard_info = layer_info.shards[0];
    uint8_t* h_base = const_cast<uint8_t*>(shard_info.host_ptr);
    const uint8_t* d_base = device_buffers[l];

    for (int64_t i = 0; i < num_chunks; ++i) {
      int64_t copy_size = copy_sizes[i];
      if (copy_size == 0) continue;

      int64_t s_offset = src_offsets[i] * block_byte_size;
      int64_t d_offset = dst_offsets[i] * block_byte_size;
      size_t size_bytes = copy_size * block_byte_size;

      const uint8_t* src_ptr = d_base + s_offset;
      uint8_t* dst_ptr = h_base + d_offset;

      stream_executor::DeviceAddressBase src_addr(const_cast<uint8_t*>(src_ptr),
                                                  size_bytes);
      ABSL_RETURN_IF_ERROR(stream->Memcpy(dst_ptr, src_addr, size_bytes));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<raiden::PjRtCopyFuture> KVCacheManagerBase::H2dDirect(
    const std::vector<int64_t>& src_offsets,
    const std::vector<int64_t>& dst_offsets,
    const std::vector<int64_t>& copy_sizes, int64_t device_id) {
  bool is_partial = !src_offsets.empty();
  if (is_partial) {
    if (src_offsets.size() != dst_offsets.size() ||
        src_offsets.size() != copy_sizes.size()) {
      return absl::InvalidArgumentError(
          "Lengths of offset and size vectors must match");
    }
  }

  std::vector<xla::Future<raiden::BufferHolder>> shard_futures_to_join;
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    const auto& layer_info = layers_[layer_idx];
    const auto& layer_holds = buffer_holds_[layer_idx];
    for (size_t i = 0; i < num_shards_; ++i) {
      if (device_id >= 0 && static_cast<int64_t>(i) != device_id) {
        continue;
      }
      const auto& shard_info = layer_info.shards[i];
      const auto& shard_hold = layer_holds[i];

      std::vector<xla::Future<>> shard_futures;
      if (!is_partial) {
        if (shard_info.host_ptr == nullptr) {
          return absl::FailedPreconditionError("Source host pointer is null");
        }
        if (physical_size_ > shard_info.host_size) {
          return absl::OutOfRangeError(
              "Copy range exceeds source host buffer size");
        }
        xla::Future<> future = shard_hold.CopyRawHostToDevice(
            shard_info.host_ptr, 0, physical_size_);
        shard_futures.push_back(std::move(future));
      } else {
        for (size_t j = 0; j < src_offsets.size(); ++j) {
          int64_t src_major_dim_offset = src_offsets[j];
          int64_t dst_major_dim_offset = dst_offsets[j];
          int64_t major_dim_size = copy_sizes[j];

          int64_t src_offset = src_major_dim_offset * slice_byte_size_;
          int64_t dst_offset = dst_major_dim_offset * slice_byte_size_;
          int64_t size_to_copy = major_dim_size * slice_byte_size_;

          if (src_offset + size_to_copy > shard_info.host_size) {
            return absl::InvalidArgumentError(
                "Copy range exceeds source host buffer size");
          }
          if (dst_offset + size_to_copy > shard_info.device_size) {
            return absl::InvalidArgumentError(
                "Copy range exceeds destination device buffer size");
          }

          const uint8_t* src_ptr = shard_info.host_ptr + src_offset;
          xla::Future<> future =
              shard_hold.CopyRawHostToDevice(src_ptr, dst_offset, size_to_copy);
          shard_futures.push_back(std::move(future));
        }
      }
      shard_futures_to_join.push_back(raiden::CreateBufferFuture(
          std::move(shard_futures), shard_hold.c_hold, shard_hold.common_hold));
    }
  }
  return xla::JoinFutures(absl::MakeSpan(shard_futures_to_join));
}

absl::StatusOr<raiden::PjRtCopyFuture> KVCacheManagerBase::D2hDirect(
    const std::vector<int64_t>& src_offsets,
    const std::vector<int64_t>& dst_offsets,
    const std::vector<int64_t>& copy_sizes, int64_t device_id) {
  return DispatchD2hChunks(src_offsets, dst_offsets, copy_sizes, device_id);
}

absl::StatusOr<raiden::PjRtCopyFuture> KVCacheManagerBase::D2hTo(
    size_t layer_idx, void* dst_host_ptr, size_t dst_size,
    const KVCacheCopySpec& copy_spec, size_t shard_idx) {
  // TODO: Long-term, KVCacheManager should own host buffers for prefix cache.
  ABSL_RETURN_IF_ERROR(ValidateCopySpecStatus(copy_spec));
  if (layer_idx >= layers_.size() ||
      shard_idx >= layers_[layer_idx].shards.size()) {
    return absl::OutOfRangeError("D2H layer or shard index out of range");
  }
  if (layer_idx >= buffer_holds_.size() ||
      shard_idx >= buffer_holds_[layer_idx].size()) {
    return absl::FailedPreconditionError(
        "D2H requires a device-backed KVCacheManagerBase");
  }
  if (dst_size > 0 && dst_host_ptr == nullptr) {
    return absl::InvalidArgumentError("Destination host pointer is null");
  }

  const auto& shard_info = layers_[layer_idx].shards[shard_idx];
  const auto& shard_hold = buffer_holds_[layer_idx][shard_idx];
  const bool is_partial = IsPartialCopy(copy_spec, major_dim_size_);
  std::vector<xla::Future<>> futures;
  if (!is_partial) {
    if (dst_size < physical_size_) {
      return absl::InvalidArgumentError("Destination host buffer is too small");
    }
    futures.push_back(
        shard_hold.CopyRawDeviceToHost(dst_host_ptr, 0, physical_size_));
  } else {
    futures.reserve(copy_spec.src_offsets.size());
    uint8_t* dst = static_cast<uint8_t*>(dst_host_ptr);
    for (size_t i = 0; i < copy_spec.src_offsets.size(); ++i) {
      const int64_t src_offset = copy_spec.src_offsets[i] * slice_byte_size_;
      const int64_t dst_offset = copy_spec.dst_offsets[i] * slice_byte_size_;
      const int64_t size_to_copy = copy_spec.sizes[i] * slice_byte_size_;
      if (src_offset + size_to_copy >
          static_cast<int64_t>(shard_info.device_size)) {
        return absl::InvalidArgumentError(
            "Copy range exceeds source device buffer size");
      }
      if (dst_offset + size_to_copy > static_cast<int64_t>(dst_size)) {
        return absl::InvalidArgumentError(
            "Copy range exceeds destination host buffer size");
      }
      futures.push_back(shard_hold.CopyRawDeviceToHost(
          dst + dst_offset, src_offset, size_to_copy));
    }
  }

  std::vector<xla::Future<raiden::BufferHolder>> shard_futures_to_join = {
      raiden::CreateBufferFuture(std::move(futures), shard_hold.c_hold,
                                 shard_hold.common_hold)};
  return xla::JoinFutures(absl::MakeSpan(shard_futures_to_join));
}

absl::StatusOr<raiden::PjRtCopyFuture> KVCacheManagerBase::H2dFrom(
    size_t layer_idx, const void* src_host_ptr, size_t src_size,
    const KVCacheCopySpec& copy_spec, size_t shard_idx) {
  // TODO: Long-term, KVCacheManager should own host buffers for prefix cache.
  ABSL_RETURN_IF_ERROR(ValidateCopySpecStatus(copy_spec));
  if (layer_idx >= layers_.size() ||
      shard_idx >= layers_[layer_idx].shards.size()) {
    return absl::OutOfRangeError("H2D layer or shard index out of range");
  }
  if (layer_idx >= buffer_holds_.size() ||
      shard_idx >= buffer_holds_[layer_idx].size()) {
    return absl::FailedPreconditionError(
        "H2D requires a device-backed KVCacheManagerBase");
  }
  if (src_size > 0 && src_host_ptr == nullptr) {
    return absl::InvalidArgumentError("Source host pointer is null");
  }

  const auto& shard_info = layers_[layer_idx].shards[shard_idx];
  const auto& shard_hold = buffer_holds_[layer_idx][shard_idx];
  const bool is_partial = IsPartialCopy(copy_spec, major_dim_size_);
  std::vector<xla::Future<>> futures;
  if (!is_partial) {
    if (src_size < physical_size_) {
      return absl::InvalidArgumentError("Source host buffer is too small");
    }
    futures.push_back(
        shard_hold.CopyRawHostToDevice(src_host_ptr, 0, physical_size_));
  } else {
    futures.reserve(copy_spec.src_offsets.size());
    const uint8_t* src = static_cast<const uint8_t*>(src_host_ptr);
    for (size_t i = 0; i < copy_spec.src_offsets.size(); ++i) {
      const int64_t src_offset = copy_spec.src_offsets[i] * slice_byte_size_;
      const int64_t dst_offset = copy_spec.dst_offsets[i] * slice_byte_size_;
      const int64_t size_to_copy = copy_spec.sizes[i] * slice_byte_size_;
      if (src_offset + size_to_copy > static_cast<int64_t>(src_size)) {
        return absl::InvalidArgumentError(
            "Copy range exceeds source host buffer size");
      }
      if (dst_offset + size_to_copy >
          static_cast<int64_t>(shard_info.device_size)) {
        return absl::InvalidArgumentError(
            "Copy range exceeds destination device buffer size");
      }
      futures.push_back(shard_hold.CopyRawHostToDevice(
          src + src_offset, dst_offset, size_to_copy));
    }
  }

  std::vector<xla::Future<raiden::BufferHolder>> shard_futures_to_join = {
      raiden::CreateBufferFuture(std::move(futures), shard_hold.c_hold,
                                 shard_hold.common_hold)};
  return xla::JoinFutures(absl::MakeSpan(shard_futures_to_join));
}

absl::Status KVCacheManagerBase::ConfigureHostStagingSlots(
    int64_t num_slots, int64_t max_major_per_slot) {
  if (num_slots <= 0) {
    return absl::InvalidArgumentError("num_slots must be positive");
  }
  if (max_major_per_slot <= 0) {
    return absl::InvalidArgumentError("max_major_per_slot must be positive");
  }
  const size_t required_size =
      static_cast<size_t>(num_slots) *
      static_cast<size_t>(max_major_per_slot) * slice_byte_size_;
  for (size_t l = 0; l < layers_.size(); ++l) {
    const auto& layer_info = layers_[l];
    for (size_t s = 0; s < layer_info.shards.size(); ++s) {
      const auto& shard_info = layer_info.shards[s];
      if (shard_info.host_size < required_size) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Host staging buffer is too small for requested slots at layer ", l,
            ", shard ", s, ". host_size: ", shard_info.host_size,
            ", required_size: ", required_size));
      }
    }
  }
  staging_num_slots_ = num_slots;
  staging_max_major_per_slot_ = max_major_per_slot;
  return absl::OkStatus();
}

absl::StatusOr<KVCacheHostSpan> KVCacheManagerBase::HostSpan(
    size_t layer_idx, size_t shard_idx, int64_t slot_idx, int64_t num_major) {
  if (staging_num_slots_ <= 0 || staging_max_major_per_slot_ <= 0) {
    return absl::FailedPreconditionError(
        "Host staging slots have not been configured");
  }
  if (slot_idx < 0 || slot_idx >= staging_num_slots_) {
    return absl::OutOfRangeError("slot_idx out of range");
  }
  if (num_major < 0 || num_major > staging_max_major_per_slot_) {
    return absl::OutOfRangeError("num_major out of range");
  }
  if (layer_idx >= layers_.size() ||
      shard_idx >= layers_[layer_idx].shards.size()) {
    return absl::OutOfRangeError("HostSpan layer or shard index out of range");
  }
  const int64_t base_major = slot_idx * staging_max_major_per_slot_;
  const size_t byte_offset =
      static_cast<size_t>(base_major) * slice_byte_size_;
  const size_t nbytes = static_cast<size_t>(num_major) * slice_byte_size_;
  const auto& shard_info = layers_[layer_idx].shards[shard_idx];
  if (byte_offset + nbytes > shard_info.host_size) {
    return absl::OutOfRangeError("HostSpan exceeds host staging buffer");
  }
  return KVCacheHostSpan{
      .ptr = const_cast<uint8_t*>(shard_info.host_ptr) + byte_offset,
      .nbytes = nbytes,
      .slot_idx = slot_idx,
      .base_major = base_major,
      .num_major = num_major,
      .layer_idx = layer_idx,
      .shard_idx = shard_idx};
}

absl::StatusOr<raiden::PjRtCopyFuture> KVCacheManagerBase::D2hToHostSlot(
    size_t layer_idx, int64_t slot_idx, int64_t num_major,
    const KVCacheCopySpec& copy_spec, size_t shard_idx) {
  ABSL_ASSIGN_OR_RETURN(KVCacheHostSpan span,
                        HostSpan(layer_idx, shard_idx, slot_idx, num_major));
  return D2hTo(layer_idx, span.ptr, span.nbytes, copy_spec, shard_idx);
}

absl::StatusOr<raiden::PjRtCopyFuture> KVCacheManagerBase::H2dFromHostSlot(
    size_t layer_idx, int64_t slot_idx, int64_t num_major,
    const KVCacheCopySpec& copy_spec, size_t shard_idx) {
  ABSL_ASSIGN_OR_RETURN(KVCacheHostSpan span,
                        HostSpan(layer_idx, shard_idx, slot_idx, num_major));
  return H2dFrom(layer_idx, span.ptr, span.nbytes, copy_spec, shard_idx);
}

size_t KVCacheManagerBase::bytes_per_block() const {
  if (is_blocked_layout_) {
    return slice_byte_size_;
  }
  return block_size_ * slice_byte_size_;
}

}  // namespace kv_cache
}  // namespace tpu_raiden
