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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>  // NOLINT(build/c++11)
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "core/numa_thread_pool.h"
#include "core/raiden_manager_base.h"
#include "core/raw_transfer_core.h"
#include "core/status_macros.h"
#include "core/tpu_utils.h"
#include "kv_cache/logical_block_manager.h"
#include "transport/block_transport.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

absl::Status ValidateOffsetsAndSizes(const std::vector<int64_t>& src_offsets,
                                     const std::vector<int64_t>& dst_offsets,
                                     const std::vector<int64_t>& sizes) {
  const bool present =
      !src_offsets.empty() || !dst_offsets.empty() || !sizes.empty();
  if (present && (src_offsets.size() != dst_offsets.size() ||
                  src_offsets.size() != sizes.size())) {
    return absl::InvalidArgumentError(
        "src_offsets, dst_offsets, and sizes must have the same length");
  }
  for (size_t i = 0; i < src_offsets.size(); ++i) {
    if (src_offsets[i] < 0 || dst_offsets[i] < 0 || sizes[i] < 0) {
      return absl::InvalidArgumentError(
          "copy offsets and sizes must be non-negative");
    }
  }
  return absl::OkStatus();
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
  LOG(ERROR) << "KVCacheManagerBase: on_device_shape: " << shape.ToString();

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
        auto status_or_allocation =
            host_allocator(alloc_size, dst_buffer->device());
        if (!status_or_allocation.ok()) {
          throw std::runtime_error(absl::StrCat(
              "Host allocator failed for size: ", alloc_size,
              ", error: ", status_or_allocation.status().ToString()));
        }
        HostBufferAllocation allocation =
            std::move(status_or_allocation).value();
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
        VLOG(1) << "KVCacheManagerBase: allocated host buffer for layer "
                << layer_idx << ", shard " << i << " at "
                << (void*)shard_info.host_ptr
                << ", size " << shard_info.host_size;
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

  // Initialize NUMA thread pool
  std::vector<int> unique_numa_nodes;
  for (const auto& layer : layer_buffers) {
    for (xla::PjRtBuffer* buf : layer) {
      if (buf && buf->device()) {
        int node = GetPjRtDeviceNumaNode(buf->device());
        if (node >= 0) {
          bool found = false;
          for (int n : unique_numa_nodes) {
            if (n == node) {
              found = true;
              break;
            }
          }
          if (!found) {
            unique_numa_nodes.push_back(node);
          }
        }
      }
    }
  }
  size_t pool_size = std::max<size_t>(
      {1, static_cast<size_t>(parallelism), unique_numa_nodes.size()});
  dma_pool_ = std::make_unique<NumaThreadPool>(pool_size);
  push_pool_ = std::make_unique<NumaThreadPool>(pool_size);
  pull_pool_ = std::make_unique<NumaThreadPool>(pool_size);
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
        auto status_or_allocation = host_allocator(alloc_size, nullptr);
        if (!status_or_allocation.ok()) {
          throw std::runtime_error(absl::StrCat(
              "Host allocator failed for size: ", alloc_size,
              ", error: ", status_or_allocation.status().ToString()));
        }
        HostBufferAllocation allocation =
            std::move(status_or_allocation).value();
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
  size_t pool_size = std::max<size_t>(1, parallelism);
  dma_pool_ = std::make_unique<NumaThreadPool>(pool_size);
  push_pool_ = std::make_unique<NumaThreadPool>(pool_size);
  pull_pool_ = std::make_unique<NumaThreadPool>(pool_size);
}

KVCacheManagerBase::~KVCacheManagerBase() {
  buffer_holds_.clear();
  layers_.clear();
  block_manager_.reset();
}

absl::StatusOr<std::vector<xla::Future<raiden::BufferHolder>>>
KVCacheManagerBase::H2d(const std::vector<int64_t>& src_offsets_major_dim,
                        const std::vector<int64_t>& dst_offsets_major_dim,
                        const std::vector<int64_t>& copy_sizes_major_dim,
                        std::optional<int64_t> slot_idx,
                        std::optional<size_t> target_layer_idx,
                        std::optional<size_t> target_shard_idx) {
  VLOG(1) << "KVCacheManagerBase::H2d called. Thread: "
          << std::this_thread::get_id() << ", slot_idx: "
          << (slot_idx.has_value() ? std::to_string(*slot_idx) : "none")
          << ", target_layer: "
          << (target_layer_idx.has_value() ? std::to_string(*target_layer_idx)
                                           : "all")
          << ", target_shard: "
          << (target_shard_idx.has_value() ? std::to_string(*target_shard_idx)
                                           : "all");

  if (buffer_holds_.empty()) {
    return absl::FailedPreconditionError(
        "H2d requires a device-backed KVCacheManagerBase");
  }
  TF_RETURN_IF_ERROR(ValidateOffsetsAndSizes(
      src_offsets_major_dim, dst_offsets_major_dim, copy_sizes_major_dim));
  if (target_layer_idx.has_value() && *target_layer_idx >= num_layers_) {
    return absl::OutOfRangeError("layer or shard index out of range");
  }
  if (target_shard_idx.has_value() && *target_shard_idx >= num_shards_) {
    return absl::OutOfRangeError("layer or shard index out of range");
  }
  bool is_partial = !src_offsets_major_dim.empty();

  std::vector<xla::Future<raiden::BufferHolder>> logical_futures(num_layers_ *
                                                                 num_shards_);

  // Group work by NUMA node
  std::map<int, std::vector<CopyWork>> grouped_work;
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    if (target_layer_idx.has_value() && *target_layer_idx != layer_idx) {
      continue;
    }
    for (size_t shard_idx = 0; shard_idx < num_shards_; ++shard_idx) {
      if (target_shard_idx.has_value() && *target_shard_idx != shard_idx) {
        continue;
      }
      int node = -1;
      if (layer_idx < buffer_holds_.size() &&
          shard_idx < buffer_holds_[layer_idx].size()) {
        auto* buf = buffer_holds_[layer_idx][shard_idx].buffer;
        if (buf && buf->device()) {
          node = GetPjRtDeviceNumaNode(buf->device());
        }
      }
      grouped_work[node].push_back({layer_idx, shard_idx});
    }
  }

  size_t total_works = 0;
  for (const auto& [node, works] : grouped_work) {
    total_works += works.size();
  }

  VLOG(1) << "H2d: grouped work into " << grouped_work.size()
          << " NUMA groups. Total works: " << total_works;

  struct PendingFuture {
    std::future<absl::StatusOr<std::vector<xla::Future<raiden::BufferHolder>>>>
        future;
    std::vector<CopyWork> works;
  };
  std::vector<PendingFuture> pending_futures;

  if (total_works <= 1) {
    // OPTIMIZATION: Single work item. Execute inline to avoid pool overhead.
    VLOG(1) << "H2d: Executing inline (single work). Thread: "
            << std::this_thread::get_id();
    for (const auto& [node, works] : grouped_work) {
      VLOG(1) << "H2d: Executing inline dispatch for NUMA node " << node
              << ", works count: " << works.size();
      auto status_or_local_futures =
          DispatchH2dWork(works, slot_idx, is_partial, src_offsets_major_dim,
                          dst_offsets_major_dim, copy_sizes_major_dim);
      if (!status_or_local_futures.ok()) {
        VLOG(1) << "H2d: Inline dispatch failed: "
                << status_or_local_futures.status().ToString();
        return status_or_local_futures.status();
      }
      auto local_futures = std::move(status_or_local_futures).value();
      for (size_t i = 0; i < works.size(); ++i) {
        const auto& work = works[i];
        logical_futures[work.layer_idx * num_shards_ + work.shard_idx] =
            std::move(local_futures[i]);
      }
    }
  } else {
    // Safe to parallelize via the dedicated dma_pool_.
    VLOG(1) << "H2d: Parallelizing dispatches on dma_pool_. Thread: "
            << std::this_thread::get_id();
    for (const auto& [node, works] : grouped_work) {
      VLOG(1) << "H2d: Scheduling dispatch for NUMA node " << node
              << ", works count: " << works.size();
      auto future = dma_pool_->Schedule(
          node >= 0 ? std::optional<int>(node) : std::nullopt,
          [this, works, is_partial, src_offsets_major_dim,
           dst_offsets_major_dim, copy_sizes_major_dim, slot_idx]() {
            return DispatchH2dWork(works, slot_idx, is_partial,
                                   src_offsets_major_dim, dst_offsets_major_dim,
                                   copy_sizes_major_dim);
          });
      pending_futures.push_back({std::move(future), works});
    }

    VLOG(1) << "H2d: Awaiting scheduled dispatches...";
    for (auto& pf : pending_futures) {
      auto status_or_local_futures = pf.future.get();
      if (!status_or_local_futures.ok()) {
        VLOG(1) << "H2d: Scheduled dispatch failed: "
                << status_or_local_futures.status().ToString();
        return status_or_local_futures.status();
      }
      VLOG(1) << "H2d: Scheduled dispatch completed successfully.";
      auto local_futures = std::move(status_or_local_futures).value();
      for (size_t i = 0; i < pf.works.size(); ++i) {
        const auto& work = pf.works[i];
        logical_futures[work.layer_idx * num_shards_ + work.shard_idx] =
            std::move(local_futures[i]);
      }
    }
  }

  VLOG(1) << "KVCacheManagerBase::H2d completed. Returning logical futures.";
  return logical_futures;
}

absl::StatusOr<std::vector<xla::Future<raiden::BufferHolder>>>
KVCacheManagerBase::DispatchD2hChunks(const std::vector<int64_t>& src_offsets,
                                      const std::vector<int64_t>& dst_offsets,
                                      const std::vector<int64_t>& copy_sizes,
                                      std::optional<int64_t> slot_idx,
                                      std::optional<size_t> target_layer_idx,
                                      std::optional<size_t> target_shard_idx,
                                      int64_t device_id) {
  VLOG(1) << "KVCacheManagerBase::DispatchD2hChunks called. Thread: "
          << std::this_thread::get_id() << ", slot_idx: "
          << (slot_idx.has_value() ? std::to_string(*slot_idx) : "none")
          << ", target_layer: "
          << (target_layer_idx.has_value() ? std::to_string(*target_layer_idx)
                                           : "all")
          << ", target_shard: "
          << (target_shard_idx.has_value() ? std::to_string(*target_shard_idx)
                                           : "all")
          << ", device_id: " << device_id;

  if (buffer_holds_.empty()) {
    return absl::FailedPreconditionError(
        "D2h requires a device-backed KVCacheManagerBase");
  }
  TF_RETURN_IF_ERROR(
      ValidateOffsetsAndSizes(src_offsets, dst_offsets, copy_sizes));
  if (target_layer_idx.has_value() && *target_layer_idx >= num_layers_) {
    return absl::OutOfRangeError("layer or shard index out of range");
  }
  if (target_shard_idx.has_value() && *target_shard_idx >= num_shards_) {
    return absl::OutOfRangeError("layer or shard index out of range");
  }
  bool is_partial = !src_offsets.empty();

  std::vector<xla::Future<raiden::BufferHolder>> logical_futures(num_layers_ *
                                                                 num_shards_);

  std::map<int, std::vector<CopyWork>> grouped_work;
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    if (target_layer_idx.has_value() && *target_layer_idx != layer_idx) {
      continue;
    }
    for (size_t shard_idx = 0; shard_idx < num_shards_; ++shard_idx) {
      if (target_shard_idx.has_value() && *target_shard_idx != shard_idx) {
        continue;
      }
      if (device_id >= 0 && static_cast<int64_t>(shard_idx) != device_id) {
        continue;
      }
      int node = -1;
      if (layer_idx < buffer_holds_.size() &&
          shard_idx < buffer_holds_[layer_idx].size()) {
        auto* buf = buffer_holds_[layer_idx][shard_idx].buffer;
        if (buf && buf->device()) {
          node = GetPjRtDeviceNumaNode(buf->device());
        }
      }
      grouped_work[node].push_back({layer_idx, shard_idx});
    }
  }

  size_t total_works = 0;
  for (const auto& [node, works] : grouped_work) {
    total_works += works.size();
  }

  VLOG(1) << "DispatchD2hChunks: grouped work into " << grouped_work.size()
          << " NUMA groups. Total works: " << total_works;

  struct PendingFuture {
    std::future<absl::StatusOr<std::vector<xla::Future<raiden::BufferHolder>>>>
        future;
    std::vector<CopyWork> works;
  };
  std::vector<PendingFuture> pending_futures;

  if (total_works <= 1) {
    // OPTIMIZATION: Single work item. Execute inline to avoid pool overhead.
    VLOG(1) << "DispatchD2hChunks: Executing inline (single work). Thread: "
            << std::this_thread::get_id();
    for (const auto& [node, works] : grouped_work) {
      VLOG(1) << "DispatchD2hChunks: Executing inline dispatch for NUMA node "
              << node << ", works count: " << works.size();
      auto status_or_local_futures = DispatchD2hWork(
          works, slot_idx, is_partial, src_offsets, dst_offsets, copy_sizes);
      if (!status_or_local_futures.ok()) {
        VLOG(1) << "DispatchD2hChunks: Inline dispatch failed: "
                << status_or_local_futures.status().ToString();
        return status_or_local_futures.status();
      }
      auto local_futures = std::move(status_or_local_futures).value();
      for (size_t i = 0; i < works.size(); ++i) {
        const auto& work = works[i];
        logical_futures[work.layer_idx * num_shards_ + work.shard_idx] =
            std::move(local_futures[i]);
      }
    }
  } else {
    // Safe to parallelize via the dedicated dma_pool_.
    VLOG(1)
        << "DispatchD2hChunks: Parallelizing dispatches on dma_pool_. Thread: "
        << std::this_thread::get_id();
    for (const auto& [node, works] : grouped_work) {
      VLOG(1) << "DispatchD2hChunks: Scheduling dispatch for NUMA node " << node
              << ", works count: " << works.size();
      auto future = dma_pool_->Schedule(
          node >= 0 ? std::optional<int>(node) : std::nullopt,
          [this, works, is_partial, src_offsets, dst_offsets, copy_sizes,
           slot_idx]() {
            return DispatchD2hWork(works, slot_idx, is_partial, src_offsets,
                                   dst_offsets, copy_sizes);
          });
      pending_futures.push_back({std::move(future), works});
    }

    VLOG(1) << "DispatchD2hChunks: Awaiting scheduled dispatches...";
    for (auto& pf : pending_futures) {
      auto status_or_local_futures = pf.future.get();
      if (!status_or_local_futures.ok()) {
        VLOG(1) << "DispatchD2hChunks: Scheduled dispatch failed: "
                << status_or_local_futures.status().ToString();
        return status_or_local_futures.status();
      }
      VLOG(1)
          << "DispatchD2hChunks: Scheduled dispatch completed successfully.";
      auto local_futures = std::move(status_or_local_futures).value();
      for (size_t i = 0; i < pf.works.size(); ++i) {
        const auto& work = pf.works[i];
        logical_futures[work.layer_idx * num_shards_ + work.shard_idx] =
            std::move(local_futures[i]);
      }
    }
  }

  VLOG(1) << "KVCacheManagerBase::DispatchD2hChunks completed. Returning "
             "logical futures.";
  return logical_futures;
}

absl::StatusOr<std::vector<xla::Future<raiden::BufferHolder>>>
KVCacheManagerBase::D2h(const std::vector<int64_t>& src_offsets_major_dim,
                        const std::vector<int64_t>& dst_offsets_major_dim,
                        const std::vector<int64_t>& copy_sizes_major_dim,
                        std::optional<int64_t> slot_idx,
                        std::optional<size_t> layer_idx,
                        std::optional<size_t> shard_idx) {
  return DispatchD2hChunks(src_offsets_major_dim, dst_offsets_major_dim,
                           copy_sizes_major_dim, slot_idx, layer_idx,
                           shard_idx);
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

  ASSIGN_OR_RETURN(std::vector<int> allocated_block_ids,
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

  ASSIGN_OR_RETURN(
      auto futures,
      DispatchD2hChunks(flat_src_offsets, flat_dst_offsets, flat_copy_sizes));
  auto future = xla::JoinFutures(absl::MakeSpan(futures));
  return std::make_pair(allocated_block_ids, std::move(future));
}

absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
KVCacheManagerBase::H2hWrite(std::string peer,
                             const std::vector<int>& src_block_ids,
                             int64_t entity_id) {
  ASSIGN_OR_RETURN(std::vector<int> allocated_ids,
                   H2hWriteDirect(peer, src_block_ids, entity_id));
  return std::make_pair(
      allocated_ids,
      raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{}));
}

absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
KVCacheManagerBase::H2hRead(std::string peer,
                            const std::vector<int>& src_block_ids,
                            int64_t entity_id) {
  ASSIGN_OR_RETURN(std::vector<int> allocated_ids,
                   H2hReadDirect(peer, src_block_ids, entity_id));
  return std::make_pair(
      allocated_ids,
      raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{}));
}

absl::StatusOr<raiden::PjRtCopyFuture> KVCacheManagerBase::H2hReadExplicit(
    std::string peer, const std::vector<int>& src_block_ids,
    const std::vector<int>& local_block_ids,
    const std::vector<uint8_t*>& explicit_dst_ptrs, int parallelism,
    tpu_raiden::transport::MajorOrder major_order,
    tpu_raiden::transport::BlockReceivedCallback on_block_received) {
  if (!server_) {
    return absl::FailedPreconditionError("Transport server is not running");
  }
  ASSIGN_OR_RETURN(
      std::vector<int> allocated_ids,
      server_->Pull(peer, src_block_ids, local_block_ids, explicit_dst_ptrs,
                    parallelism, major_order, on_block_received));
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
      TF_RETURN_IF_ERROR(stream->Memcpy(&device_addr, src_ptr, size_bytes));
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
      TF_RETURN_IF_ERROR(stream->Memcpy(dst_ptr, src_addr, size_bytes));
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
  ASSIGN_OR_RETURN(
      auto futures,
      DispatchD2hChunks(src_offsets, dst_offsets, copy_sizes,
                        /*slot_idx=*/std::nullopt, /*layer_idx=*/std::nullopt,
                        /*shard_idx=*/std::nullopt, device_id));
  return xla::JoinFutures(absl::MakeSpan(futures));
}

absl::Status KVCacheManagerBase::ConfigureHostStagingSlots(
    int64_t num_slots, int64_t max_major_per_slot) {
  if (num_slots <= 0) {
    return absl::InvalidArgumentError("num_slots must be positive");
  }
  if (max_major_per_slot <= 0) {
    return absl::InvalidArgumentError("max_major_per_slot must be positive");
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
  const size_t byte_offset = static_cast<size_t>(base_major) * slice_byte_size_;
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

size_t KVCacheManagerBase::bytes_per_block() const {
  if (is_blocked_layout_) {
    return slice_byte_size_;
  }
  return block_size_ * slice_byte_size_;
}

absl::StatusOr<std::vector<xla::Future<raiden::BufferHolder>>>
KVCacheManagerBase::DispatchH2dWork(
    const std::vector<CopyWork>& works, std::optional<int64_t> slot_idx,
    bool is_partial, const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  VLOG(1) << "KVCacheManagerBase::DispatchH2dWork started. Works count: "
          << works.size() << ", Thread: " << std::this_thread::get_id();
  std::vector<xla::Future<raiden::BufferHolder>> local_futures;
  for (const auto& work : works) {
    const auto& layer_info = layers_[work.layer_idx];
    const auto& shard_hold = buffer_holds_[work.layer_idx][work.shard_idx];
    const auto& shard_info = layer_info.shards[work.shard_idx];

    const uint8_t* base_host_ptr = nullptr;
    size_t host_size = 0;
    if (slot_idx.has_value()) {
      TF_ASSIGN_OR_RETURN(KVCacheHostSpan span,
                          HostSpan(work.layer_idx, work.shard_idx, *slot_idx,
                                   staging_max_major_per_slot_));
      base_host_ptr = span.ptr;
      host_size = span.nbytes;
    } else {
      base_host_ptr = shard_info.host_ptr;
      host_size = shard_info.host_size;
    }
    VLOG(1) << "DispatchH2dWork: Layer: " << work.layer_idx
            << ", Shard: " << work.shard_idx
            << ", base_host_ptr: " << (void*)base_host_ptr
            << ", host_size: " << host_size;

    std::vector<xla::Future<>> shard_futures;
    if (!is_partial) {
      if (base_host_ptr == nullptr) {
        return absl::FailedPreconditionError("Source host pointer is null");
      }
      if (physical_size_ > host_size) {
        return absl::InvalidArgumentError("Source host buffer is too small");
      }
      VLOG(1) << "DispatchH2dWork: calling CopyRawHostToDevice (Full). Layer: "
              << work.layer_idx << ", Shard: " << work.shard_idx
              << ", Size: " << physical_size_
              << ", Thread: " << std::this_thread::get_id();
      xla::Future<> future =
          shard_hold.CopyRawHostToDevice(base_host_ptr, 0, physical_size_);
      shard_futures.push_back(std::move(future));
    } else {
      for (size_t j = 0; j < src_offsets_major_dim.size(); ++j) {
        int64_t src_major_dim_offset = src_offsets_major_dim[j];
        int64_t dst_major_dim_offset = dst_offsets_major_dim[j];
        int64_t major_dim_size = copy_sizes_major_dim[j];

        int64_t src_offset = src_major_dim_offset * slice_byte_size_;
        int64_t dst_offset = dst_major_dim_offset * slice_byte_size_;
        int64_t size_to_copy = major_dim_size * slice_byte_size_;

        if (src_offset + size_to_copy > static_cast<int64_t>(host_size)) {
          return absl::InvalidArgumentError(
              "Copy range exceeds source host buffer size");
        }
        if (dst_offset + size_to_copy > shard_info.device_size) {
          return absl::InvalidArgumentError(
              "Copy range exceeds destination device buffer size");
        }

        const uint8_t* src_ptr = base_host_ptr + src_offset;
        VLOG(1)
            << "DispatchH2dWork: calling CopyRawHostToDevice (Partial). Layer: "
            << work.layer_idx << ", Shard: " << work.shard_idx
            << ", SrcOffset: " << src_offset << ", DstOffset: " << dst_offset
            << ", Size: " << size_to_copy
            << ", Thread: " << std::this_thread::get_id();
        xla::Future<> future =
            shard_hold.CopyRawHostToDevice(src_ptr, dst_offset, size_to_copy);
        shard_futures.push_back(std::move(future));
      }
    }
    local_futures.push_back(raiden::CreateBufferFuture(
        std::move(shard_futures), shard_hold.c_hold, shard_hold.common_hold));
  }
  VLOG(1) << "KVCacheManagerBase::DispatchH2dWork completed. Dispatched "
          << local_futures.size()
          << " buffer futures. Thread: " << std::this_thread::get_id();
  return local_futures;
}

absl::StatusOr<std::vector<xla::Future<raiden::BufferHolder>>>
KVCacheManagerBase::DispatchD2hWork(const std::vector<CopyWork>& works,
                                    std::optional<int64_t> slot_idx,
                                    bool is_partial,
                                    const std::vector<int64_t>& src_offsets,
                                    const std::vector<int64_t>& dst_offsets,
                                    const std::vector<int64_t>& copy_sizes) {
  VLOG(1) << "KVCacheManagerBase::DispatchD2hWork started. Works count: "
          << works.size() << ", Thread: " << std::this_thread::get_id();
  std::vector<xla::Future<raiden::BufferHolder>> local_futures;
  for (const auto& work : works) {
    const auto& layer_info = layers_[work.layer_idx];
    const auto& shard_hold = buffer_holds_[work.layer_idx][work.shard_idx];
    const auto& shard_info = layer_info.shards[work.shard_idx];

    uint8_t* dst_host_ptr = nullptr;
    size_t host_size = 0;
    if (slot_idx.has_value()) {
      TF_ASSIGN_OR_RETURN(KVCacheHostSpan span,
                          HostSpan(work.layer_idx, work.shard_idx, *slot_idx,
                                   staging_max_major_per_slot_));
      dst_host_ptr = span.ptr;
      host_size = span.nbytes;
    } else {
      dst_host_ptr = const_cast<uint8_t*>(shard_info.host_ptr);
      host_size = shard_info.host_size;
    }
    VLOG(1) << "DispatchD2hWork: Layer: " << work.layer_idx
            << ", Shard: " << work.shard_idx
            << ", dst_host_ptr: " << (void*)dst_host_ptr
            << ", host_size: " << host_size;

    std::vector<xla::Future<>> shard_futures;
    if (!is_partial) {
      if (dst_host_ptr == nullptr) {
        return absl::FailedPreconditionError(
            "Destination host pointer is null");
      }
      if (physical_size_ > host_size) {
        return absl::OutOfRangeError(
            "Copy range exceeds destination host buffer size");
      }
      VLOG(1) << "DispatchD2hWork: calling CopyRawDeviceToHost (Full). Layer: "
              << work.layer_idx << ", Shard: " << work.shard_idx
              << ", Size: " << physical_size_
              << ", Thread: " << std::this_thread::get_id();
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
        if (dst_offset + size_to_copy > static_cast<int64_t>(host_size)) {
          return absl::InvalidArgumentError(
              "Copy range exceeds destination host buffer size");
        }

        uint8_t* dst_ptr = dst_host_ptr + dst_offset;
        VLOG(1)
            << "DispatchD2hWork: calling CopyRawDeviceToHost (Partial). Layer: "
            << work.layer_idx << ", Shard: " << work.shard_idx
            << ", SrcOffset: " << src_offset << ", DstOffset: " << dst_offset
            << ", Size: " << size_to_copy
            << ", Thread: " << std::this_thread::get_id();
        xla::Future<> future =
            shard_hold.CopyRawDeviceToHost(dst_ptr, src_offset, size_to_copy);
        shard_futures.push_back(std::move(future));
      }
    }
    local_futures.push_back(raiden::CreateBufferFuture(
        std::move(shard_futures), shard_hold.c_hold, shard_hold.common_hold));
  }
  VLOG(1) << "KVCacheManagerBase::DispatchD2hWork completed. Dispatched "
          << local_futures.size()
          << " buffer futures. Thread: " << std::this_thread::get_id();
  return local_futures;
}

}  // namespace kv_cache
}  // namespace tpu_raiden
