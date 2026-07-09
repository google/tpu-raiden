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

#include "tpu_raiden/kv_cache/kv_cache_manager_base.h"

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
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/semaphore.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_raiden/core/numa_thread_pool.h"
#include "tpu_raiden/core/raiden_manager_base.h"
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/core/status_macros.h"
#include "tpu_raiden/core/tpu_utils.h"
#include "tpu_raiden/kv_cache/logical_block_manager.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"
#include "tpu_raiden/transport/block_transport.h"

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

// Coalesce runs of adjacent copies into one, so a run of N consecutive
// 1-block copies becomes one N-block copy.
void CoalesceMajorDimCopies(const std::vector<int64_t>& src_offsets,
                            const std::vector<int64_t>& dst_offsets,
                            const std::vector<int64_t>& sizes,
                            std::vector<int64_t>& out_src,
                            std::vector<int64_t>& out_dst,
                            std::vector<int64_t>& out_sizes) {
  out_src.clear();
  out_dst.clear();
  out_sizes.clear();
  for (size_t i = 0; i < src_offsets.size(); ++i) {
    if (!out_src.empty() &&
        src_offsets[i] == out_src.back() + out_sizes.back() &&
        dst_offsets[i] == out_dst.back() + out_sizes.back()) {
      out_sizes.back() += sizes[i];
    } else {
      out_src.push_back(src_offsets[i]);
      out_dst.push_back(dst_offsets[i]);
      out_sizes.push_back(sizes[i]);
    }
  }
}

}  // namespace

KVCacheManagerBase::KVCacheManagerBase(
    const std::vector<std::vector<xla::PjRtBuffer*>>& layer_buffers,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    bool unsafe_skip_buffer_lock, int parallelism,
    HostBufferAllocator host_allocator, std::optional<std::string> bind_ip)
    : RaidenManagerBase(layer_buffers.size(),
                        layer_buffers.empty() ? 0 : layer_buffers[0].size(),
                        layer_buffers.empty() ? 0
                                              : raiden::GetMajorSliceByteSize(
                                                    layer_buffers[0][0]),
                        local_port, parallelism, bind_ip),
      host_allocator_(host_allocator) {
  if (num_layers_ == 0 || num_shards_ == 0) {
    return;
  }

  DetectAndAssignNumaNode(layer_buffers);

  xla::PjRtBuffer* first_buffer = layer_buffers[0][0];
  const xla::Shape& shape = first_buffer->on_device_shape();
  LOG(ERROR) << "KVCacheManagerBase: on_device_shape: " << shape.ToString();

  is_blocked_layout_ = (shape.dimensions().size() == 5);

  // max_physical_size_ will be set to the max across all layers below.
  max_physical_size_ = 0;

  extension_ = raiden::GetRawBufferExtension(first_buffer, &c_api_);

  int num_host_blocks = host_blocks_to_allocate.value_or(64);
  host_block_manager_ = std::make_unique<LogicalBlockManager>(num_host_blocks);
  if (!shape.dimensions().empty()) {
    major_dim_size_ = shape.dimensions(0);
  }
  semaphore_ = std::make_unique<xla::Semaphore>(std::max<int>(4, parallelism));

  layers_.reserve(num_layers_);
  buffer_holds_.reserve(num_layers_);

  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    const auto& dst_buffers = layer_buffers[layer_idx];
    if (dst_buffers.size() != num_shards_) {
      throw std::runtime_error("Number of shards mismatch across layers");
    }

    LayerInfoBase layer_info;
    LayerDeviceInfo device_info;
    // Store the per-layer on-device buffer size.  For uniform models
    // every layer has the same value; for hybrid (HMA) models they
    // may differ (e.g. mamba conv_state bf16 vs ssm_state f32).
    device_info.physical_size =
        layer_buffers[layer_idx][0]->GetOnDeviceSizeInBytes().value();
    max_physical_size_ =
        std::max(max_physical_size_, device_info.physical_size);
    VLOG(1) << "KVCacheManagerBase: layer " << layer_idx << " on_device_shape: "
            << layer_buffers[layer_idx][0]->on_device_shape().ToString()
            << " size: " << device_info.physical_size;
    layer_info.shards.reserve(num_shards_);
    device_info.holds.reserve(num_shards_);

    for (size_t i = 0; i < num_shards_; ++i) {
      xla::PjRtBuffer* dst_buffer = dst_buffers[i];
      ShardBufferInfoBase shard_info;

      shard_info.device_size = dst_buffer->GetOnDeviceSizeInBytes().value();
      if (shard_info.device_size < device_info.physical_size) {
        throw std::runtime_error(
            "Device buffer shard size smaller than physical size");
      }

      // Allocate host buffer using the max slice size (bytes_per_block)
      // so the buffer is large enough for any layer.
      size_t alloc_size = num_host_blocks * bytes_per_block();
      if (host_allocator) {
        const xla::PjRtDevice* target_dev = dst_buffer->device();
        auto status_or_allocation = host_allocator(alloc_size, target_dev);
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
                << (void*)shard_info.host_ptr << ", size "
                << shard_info.host_size;
      }

      auto status_or_hold = raiden::BufferHoldAndAlias::Acquire(
          dst_buffer, c_api_, extension_, unsafe_skip_buffer_lock);
      if (!status_or_hold.ok()) {
        throw std::runtime_error(
            std::string("Failed to acquire PJRT hold: ") +
            std::string(status_or_hold.status().message()));
      }
      device_info.holds.push_back(std::move(status_or_hold.value()));
      layer_info.shards.push_back(std::move(shard_info));
    }
    layers_.push_back(std::move(layer_info));
    buffer_holds_.push_back(std::move(device_info));
  }

  constexpr size_t kPoolSize = 4;
  dma_pool_ = std::make_unique<NumaThreadPool>(kPoolSize);
  push_pool_ = std::make_unique<NumaThreadPool>(kPoolSize);
  pull_pool_ = std::make_unique<NumaThreadPool>(kPoolSize);
}

KVCacheManagerBase::KVCacheManagerBase(
    size_t num_layers, size_t num_shards, size_t slice_byte_size,
    std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
    int parallelism, HostBufferAllocator host_allocator,
    std::optional<std::string> bind_ip)
    : RaidenManagerBase(num_layers, num_shards, slice_byte_size, local_port,
                        parallelism, bind_ip),
      host_allocator_(host_allocator) {
  int total_blocks = host_blocks_to_allocate.value_or(0);
  host_block_manager_ = std::make_unique<LogicalBlockManager>(total_blocks);
  semaphore_ = std::make_unique<xla::Semaphore>(std::max<int>(4, parallelism));

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
  constexpr size_t kPoolSize = 4;
  dma_pool_ = std::make_unique<NumaThreadPool>(kPoolSize);
  push_pool_ = std::make_unique<NumaThreadPool>(kPoolSize);
  pull_pool_ = std::make_unique<NumaThreadPool>(kPoolSize);
  InitTransportServer();
}

KVCacheManagerBase::~KVCacheManagerBase() {
  buffer_holds_.clear();
  layers_.clear();
  host_block_manager_.reset();
}

absl::StatusOr<raiden::PjRtCopyFuture> KVCacheManagerBase::H2d(
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim,
    std::optional<int64_t> slot_idx, std::optional<size_t> target_layer_idx,
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
  std::vector<int64_t> src_c, dst_c, sizes_c;
  CoalesceMajorDimCopies(src_offsets_major_dim, dst_offsets_major_dim,
                         copy_sizes_major_dim, src_c, dst_c, sizes_c);
  bool is_partial = !src_c.empty();

  std::vector<raiden::PjRtCopyFuture> logical_futures(num_layers_ *
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
          shard_idx < buffer_holds_[layer_idx].holds.size()) {
        auto* buf = buffer_holds_[layer_idx].holds[shard_idx].buffer;
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
    std::future<absl::StatusOr<std::vector<raiden::PjRtCopyFuture>>> future;
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
          DispatchH2dWork(works, slot_idx, is_partial, src_c, dst_c, sizes_c);
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
    for (const auto& [node, works_binding] : grouped_work) {
      auto works = works_binding;
      VLOG(1) << "H2d: Scheduling dispatch for NUMA node " << node
              << ", works count: " << works.size();
      auto future = dma_pool_->Schedule(
          node >= 0 ? std::make_optional(node) : std::nullopt,
          [this, works, is_partial, src_c, dst_c, sizes_c, slot_idx]() {
            return DispatchH2dWork(works, slot_idx, is_partial, src_c, dst_c,
                                   sizes_c);
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
  return raiden::JoinPjRtCopyFutures(absl::MakeSpan(logical_futures));
}

absl::StatusOr<std::vector<raiden::PjRtCopyFuture>>
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
  std::vector<int64_t> src_c, dst_c, sizes_c;
  CoalesceMajorDimCopies(src_offsets, dst_offsets, copy_sizes, src_c, dst_c,
                         sizes_c);
  bool is_partial = !src_c.empty();

  std::vector<raiden::PjRtCopyFuture> logical_futures(num_layers_ *
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
          shard_idx < buffer_holds_[layer_idx].holds.size()) {
        auto* buf = buffer_holds_[layer_idx].holds[shard_idx].buffer;
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
    std::future<absl::StatusOr<std::vector<raiden::PjRtCopyFuture>>> future;
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
      auto status_or_local_futures =
          DispatchD2hWork(works, slot_idx, is_partial, src_c, dst_c, sizes_c);
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
    for (const auto& [node, works_binding] : grouped_work) {
      auto works = works_binding;
      VLOG(1) << "DispatchD2hChunks: Scheduling dispatch for NUMA node " << node
              << ", works count: " << works.size();
      auto future = dma_pool_->Schedule(
          node >= 0 ? std::make_optional(node) : std::nullopt,
          [this, works, is_partial, src_c, dst_c, sizes_c, slot_idx]() {
            return DispatchD2hWork(works, slot_idx, is_partial, src_c, dst_c,
                                   sizes_c);
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

absl::StatusOr<raiden::PjRtCopyFuture> KVCacheManagerBase::D2h(
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim,
    std::optional<int64_t> slot_idx, std::optional<size_t> layer_idx,
    std::optional<size_t> shard_idx) {
  ASSIGN_OR_RETURN(
      auto logical_futures,
      DispatchD2hChunks(src_offsets_major_dim, dst_offsets_major_dim,
                        copy_sizes_major_dim, slot_idx, layer_idx, shard_idx));
  return raiden::JoinPjRtCopyFutures(absl::MakeSpan(logical_futures));
}

absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
KVCacheManagerBase::D2hAutoAllocate(
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
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
    int needed = copy_size;
    total_blocks_to_allocate += needed;
    blocks_per_chunk.push_back(needed);
  }

  ASSIGN_OR_RETURN(std::vector<int> allocated_block_ids,
                   AllocateBlocks(total_blocks_to_allocate));

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
      flat_src_offsets.push_back(src_major_dim_offset + k);
      flat_dst_offsets.push_back(assigned_block_id);
      flat_copy_sizes.push_back(1);
    }
  }

  ASSIGN_OR_RETURN(
      auto futures,
      DispatchD2hChunks(flat_src_offsets, flat_dst_offsets, flat_copy_sizes));
  return std::make_pair(allocated_block_ids,
                        raiden::JoinPjRtCopyFutures(absl::MakeSpan(futures)));
}

absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
KVCacheManagerBase::H2hWrite(const std::vector<std::string>& peers,
                             const std::vector<int>& src_block_ids,
                             const std::vector<int>& dst_block_ids,
                             uint64_t uuid, int layer_idx) {
  ASSIGN_OR_RETURN(
      std::vector<int> allocated_ids,
      H2hWriteDirect(peers, src_block_ids, dst_block_ids, uuid, layer_idx));
  return std::make_pair(
      allocated_ids,
      raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{}));
}

absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
KVCacheManagerBase::H2hWrite(std::string peer,
                             const std::vector<int>& src_block_ids,
                             const std::vector<int>& dst_block_ids,
                             uint64_t uuid, int layer_idx) {
  return H2hWrite(std::vector<std::string>{std::move(peer)}, src_block_ids,
                  dst_block_ids, uuid, layer_idx);
}

absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
KVCacheManagerBase::H2hRead(const std::vector<std::string>& peers,
                            const std::vector<int>& src_block_ids) {
  ASSIGN_OR_RETURN(std::vector<int> allocated_ids,
                   H2hReadDirect(peers, src_block_ids));
  return std::make_pair(
      allocated_ids,
      raiden::PjRtCopyFuture(std::vector<raiden::BufferHolder>{}));
}

absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
KVCacheManagerBase::H2hRead(std::string peer,
                            const std::vector<int>& src_block_ids) {
  return H2hRead(std::vector<std::string>{std::move(peer)}, src_block_ids);
}

absl::StatusOr<raiden::PjRtCopyFuture> KVCacheManagerBase::H2hReadExplicit(
    std::string peer, const std::vector<int>& src_block_ids,
    const std::vector<int>& local_block_ids,
    const std::vector<uint8_t*>& explicit_dst_ptrs, int parallelism,
    tpu_raiden::transport::MajorOrder major_order,
    tpu_raiden::transport::BlockReceivedCallback on_block_received) {
  absl::MutexLock lock(server_init_mu_);
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

  int64_t num_chunks = src_offsets.size();

  for (size_t l = 0; l < num_layers_; ++l) {
    int64_t block_byte_size = layer_block_byte_size(l);
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

  int64_t num_chunks = src_offsets.size();

  for (size_t l = 0; l < num_layers_; ++l) {
    int64_t block_byte_size = layer_block_byte_size(l);
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

  std::vector<raiden::PjRtCopyFuture> shard_futures_to_join;
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    const auto& layer_info = layers_[layer_idx];
    const auto& layer_holds = buffer_holds_[layer_idx].holds;
    for (size_t i = 0; i < num_shards_; ++i) {
      if (device_id >= 0 && static_cast<int64_t>(i) != device_id) {
        continue;
      }
      const auto& shard_info = layer_info.shards[i];
      const auto& shard_hold = layer_holds[i];

      std::vector<raiden::H2dCopy> copies;
      size_t layer_phys_size = buffer_holds_[layer_idx].physical_size > 0
                                   ? buffer_holds_[layer_idx].physical_size
                                   : max_physical_size_;
      int64_t layer_block_size = layer_block_byte_size(layer_idx);
      if (!is_partial) {
        if (shard_info.host_ptr == nullptr) {
          return absl::FailedPreconditionError("Source host pointer is null");
        }
        if (layer_phys_size > shard_info.host_size) {
          return absl::OutOfRangeError(
              "Copy range exceeds source host buffer size");
        }

        copies.push_back(
            {shard_info.host_ptr, 0, static_cast<int64_t>(layer_phys_size)});
      } else {
        for (size_t j = 0; j < src_offsets.size(); ++j) {
          int64_t src_major_dim_offset = src_offsets[j];
          int64_t dst_major_dim_offset = dst_offsets[j];
          int64_t major_dim_size = copy_sizes[j];

          int64_t src_offset = src_major_dim_offset * layer_block_size;
          int64_t dst_offset = dst_major_dim_offset * layer_block_size;
          int64_t size_to_copy = major_dim_size * layer_block_size;

          if (src_offset + size_to_copy >
              static_cast<int64_t>(shard_info.host_size)) {
            return absl::InvalidArgumentError(
                "Copy range exceeds source host buffer size");
          }
          if (dst_offset + size_to_copy >
              static_cast<int64_t>(shard_info.device_size)) {
            return absl::InvalidArgumentError(
                "Copy range exceeds destination device buffer size");
          }

          copies.push_back(
              {shard_info.host_ptr + src_offset, dst_offset, size_to_copy});
        }
      }
      TF_ASSIGN_OR_RETURN(raiden::PjRtCopyFuture cf,
                          raiden::IssueH2dShard(shard_hold, copies));
      shard_futures_to_join.push_back(std::move(cf));
    }
  }
  return raiden::JoinPjRtCopyFutures(absl::MakeSpan(shard_futures_to_join));
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
  return raiden::JoinPjRtCopyFutures(absl::MakeSpan(futures));
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
  int64_t layer_block_size = layer_block_byte_size(layer_idx);
  const int64_t base_major = slot_idx * staging_max_major_per_slot_;
  const size_t byte_offset = static_cast<size_t>(base_major) * layer_block_size;
  const size_t nbytes = static_cast<size_t>(num_major) * layer_block_size;
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

size_t KVCacheManagerBase::bytes_per_block() const { return slice_byte_size_; }

bool KVCacheManagerBase::use_block_chunks(uint64_t uuid) const {
  absl::MutexLock l(plans_mu_);
  auto it = active_plans_.find(uuid);
  if (it == active_plans_.end()) {
    return false;
  }
  return it->second.request.use_block_chunks();
}

absl::StatusOr<std::vector<raiden::PjRtCopyFuture>>
KVCacheManagerBase::DispatchH2dWork(
    const std::vector<CopyWork>& works, std::optional<int64_t> slot_idx,
    bool is_partial, const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  VLOG(1) << "KVCacheManagerBase::DispatchH2dWork started. Works count: "
          << works.size() << ", Thread: " << std::this_thread::get_id();
  std::vector<raiden::PjRtCopyFuture> local_futures;
  for (const auto& work : works) {
    const auto& layer_info = layers_[work.layer_idx];
    const auto& shard_hold =
        buffer_holds_[work.layer_idx].holds[work.shard_idx];
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

    size_t layer_phys_size = buffer_holds_[work.layer_idx].physical_size > 0
                                 ? buffer_holds_[work.layer_idx].physical_size
                                 : max_physical_size_;
    int64_t layer_block_size = layer_block_byte_size(work.layer_idx);
    std::vector<raiden::H2dCopy> copies;
    if (!is_partial) {
      if (base_host_ptr == nullptr) {
        return absl::FailedPreconditionError("Source host pointer is null");
      }
      if (layer_phys_size > host_size) {
        return absl::InvalidArgumentError("Source host buffer is too small");
      }
      VLOG(1) << "DispatchH2dWork: calling CopyRawHostToDevice (Full). Layer: "
              << work.layer_idx << ", Shard: " << work.shard_idx
              << ", Size: " << layer_phys_size
              << ", Thread: " << std::this_thread::get_id();

      copies.push_back(
          {base_host_ptr, 0, static_cast<int64_t>(layer_phys_size)});
    } else {
      for (size_t j = 0; j < src_offsets_major_dim.size(); ++j) {
        int64_t src_major_dim_offset = src_offsets_major_dim[j];
        int64_t dst_major_dim_offset = dst_offsets_major_dim[j];
        int64_t major_dim_size = copy_sizes_major_dim[j];

        int64_t src_offset = src_major_dim_offset * layer_block_size;
        int64_t dst_offset = dst_major_dim_offset * layer_block_size;
        int64_t size_to_copy = major_dim_size * layer_block_size;

        if (src_offset + size_to_copy > static_cast<int64_t>(host_size)) {
          return absl::InvalidArgumentError(
              "Copy range exceeds source host buffer size");
        }
        if (dst_offset + size_to_copy > shard_info.device_size) {
          return absl::InvalidArgumentError(
              "Copy range exceeds destination device buffer size");
        }
        VLOG(1)
            << "DispatchH2dWork: calling CopyRawHostToDevice (Partial). Layer: "
            << work.layer_idx << ", Shard: " << work.shard_idx
            << ", SrcOffset: " << src_offset << ", DstOffset: " << dst_offset
            << ", Size: " << size_to_copy
            << ", Thread: " << std::this_thread::get_id();

        copies.push_back(
            {base_host_ptr + src_offset, dst_offset, size_to_copy});
      }
    }
    TF_ASSIGN_OR_RETURN(raiden::PjRtCopyFuture cf,
                        raiden::IssueH2dShard(shard_hold, copies));
    local_futures.push_back(std::move(cf));
  }
  VLOG(1) << "KVCacheManagerBase::DispatchH2dWork completed. Dispatched "
          << local_futures.size()
          << " buffer futures. Thread: " << std::this_thread::get_id();
  return local_futures;
}

absl::StatusOr<std::vector<raiden::PjRtCopyFuture>>
KVCacheManagerBase::DispatchD2hWork(const std::vector<CopyWork>& works,
                                    std::optional<int64_t> slot_idx,
                                    bool is_partial,
                                    const std::vector<int64_t>& src_offsets,
                                    const std::vector<int64_t>& dst_offsets,
                                    const std::vector<int64_t>& copy_sizes) {
  VLOG(1) << "KVCacheManagerBase::DispatchD2hWork started. Works count: "
          << works.size() << ", Thread: " << std::this_thread::get_id();
  std::vector<raiden::PjRtCopyFuture> local_futures;
  for (const auto& work : works) {
    const auto& layer_info = layers_[work.layer_idx];
    const auto& shard_hold =
        buffer_holds_[work.layer_idx].holds[work.shard_idx];
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

    size_t layer_phys_size = buffer_holds_[work.layer_idx].physical_size > 0
                                 ? buffer_holds_[work.layer_idx].physical_size
                                 : max_physical_size_;
    int64_t layer_block_size = layer_block_byte_size(work.layer_idx);
    std::vector<raiden::D2hCopy> copies;
    if (!is_partial) {
      if (dst_host_ptr == nullptr) {
        return absl::FailedPreconditionError(
            "Destination host pointer is null");
      }
      if (layer_phys_size > host_size) {
        return absl::OutOfRangeError(
            "Copy range exceeds destination host buffer size");
      }
      VLOG(1) << "DispatchD2hWork: calling CopyRawDeviceToHost (Full). Layer: "
              << work.layer_idx << ", Shard: " << work.shard_idx
              << ", Size: " << layer_phys_size
              << ", Thread: " << std::this_thread::get_id();

      copies.push_back(
          {dst_host_ptr, 0, static_cast<int64_t>(layer_phys_size)});
    } else {
      copies.reserve(src_offsets.size());
      for (size_t j = 0; j < src_offsets.size(); ++j) {
        int64_t src_offset = src_offsets[j] * layer_block_size;
        int64_t dst_offset = dst_offsets[j] * layer_block_size;
        int64_t size_to_copy = copy_sizes[j] * layer_block_size;

        if (src_offset + size_to_copy > shard_info.device_size) {
          return absl::InvalidArgumentError(
              "Copy range exceeds source device buffer size");
        }
        if (dst_offset + size_to_copy > static_cast<int64_t>(host_size)) {
          return absl::InvalidArgumentError(
              "Copy range exceeds destination host buffer size");
        }
        VLOG(1)
            << "DispatchD2hWork: calling CopyRawDeviceToHost (Partial). Layer: "
            << work.layer_idx << ", Shard: " << work.shard_idx
            << ", SrcOffset: " << src_offset << ", DstOffset: " << dst_offset
            << ", Size: " << size_to_copy
            << ", Thread: " << std::this_thread::get_id();

        copies.push_back({dst_host_ptr + dst_offset, src_offset, size_to_copy});
      }
    }
    TF_ASSIGN_OR_RETURN(raiden::PjRtCopyFuture cf,
                        raiden::IssueD2hShard(shard_hold, copies));
    local_futures.push_back(std::move(cf));
  }
  VLOG(1) << "KVCacheManagerBase::DispatchD2hWork completed. Dispatched "
          << local_futures.size()
          << " buffer futures. Thread: " << std::this_thread::get_id();
  return local_futures;
}

absl::Status KVCacheManagerBase::OnSingleBlockReceived(int block_id,
                                                       size_t size_bytes) {
  RecvCallback cb;
  {
    absl::MutexLock l(recv_mu_);
    auto it = recv_callbacks_.find(block_id);
    if (it != recv_callbacks_.end()) {
      cb = std::move(it->second);
      recv_callbacks_.erase(it);
    }
  }
  if (cb) {
    return cb(block_id, size_bytes);
  }
  return absl::OkStatus();
}

void KVCacheManagerBase::RegisterBlockReadinessCallback(
    size_t layer_idx, size_t shard_idx, int block_id, uint64_t uuid,
    transport::BlockTransportDelegate::HostBlockReadyCallback cb) {
  cb(absl::OkStatus());
}

absl::Status KVCacheManagerBase::PushKVCacheResharded(
    const tpu_raiden::rpc::StartTransferRequest& request) {
  // 1. Register the active plan so GetBlockChunks can use it
  TF_RETURN_IF_ERROR(
      RegisterActivePlan(request.uuid(), request, /*is_sender=*/true));

  int numa = assigned_numa_node().value_or(-1);
  for (size_t l = 0; l < num_layers_; ++l) {
    VLOG(1) << "StartPushInternal (D2H start) layer " << l
            << ": uuid=" << request.uuid() << ", numa=" << numa;
  }

  // 2. D2H to copy from device to host.
  ASSIGN_OR_RETURN(raiden::PjRtCopyFuture d2h_future, D2h());

  // 3. Group entries by dst_peer and collect unique block IDs
  std::map<std::string, std::vector<std::pair<int, int>>> peer_transfers;
  for (const auto& [shard_idx, schedule] : request.shard_push_schedules()) {
    for (const auto& entry : schedule.entries()) {
      peer_transfers[entry.dst_peer()].push_back(
          {entry.src_block_id(), entry.dst_block_id()});
    }
  }

  d2h_future.OnReady([this, request, peer_transfers, numa](auto status_or) {
    if (!status_or.ok()) {
      LOG(ERROR) << "D2H copy failed for resharded push uuid " << request.uuid()
                 << ": " << status_or.status().ToString();
      return;
    }

    for (size_t l = 0; l < num_layers_; ++l) {
      VLOG(1) << "StartPushInternal (H2H start layer " << l
              << "): uuid=" << request.uuid() << ", numa=" << numa;
    }

    transport::BlockTransport* transport_server = nullptr;
    {
      absl::MutexLock lock(server_init_mu_);
      transport_server = server_.get();
    }
    if (!transport_server) {
      LOG(ERROR)
          << "Transport server is not running during resharded push for uuid "
          << request.uuid();
      return;
    }

    for (const auto& [peer, transfers] : peer_transfers) {
      std::vector<int> src_block_ids;
      std::vector<int> dst_block_ids;
      std::set<std::pair<int, int>> seen;
      for (const auto& p : transfers) {
        if (seen.insert(p).second) {
          src_block_ids.push_back(p.first);
          dst_block_ids.push_back(p.second);
        }
      }

      if (src_block_ids.empty()) continue;

      transport_server->Push(
          peer, src_block_ids, dst_block_ids, /*parallelism=*/1,
          transport::MajorOrder::kLayerMajor, request.uuid(),
          /*layer_idx=*/-1, [uuid = request.uuid(), peer](auto push_res) {
            if (!push_res.ok()) {
              LOG(ERROR) << "Resharded push to " << peer << " failed for uuid "
                         << uuid << ": " << push_res.status().ToString();
            } else {
              VLOG(1) << "Resharded push to " << peer << " completed for uuid "
                      << uuid;
            }
          });
    }
  });

  return absl::OkStatus();
}

absl::Status KVCacheManagerBase::RegisterActivePlan(
    uint64_t uuid, const tpu_raiden::rpc::StartTransferRequest& request,
    bool is_sender) {
  absl::MutexLock l(plans_mu_);
  if (auto [it, inserted] =
          active_plans_.try_emplace(uuid, RegisteredPlan{request, is_sender});
      !inserted) {
    return absl::AlreadyExistsError(
        absl::StrCat("Plan with UUID ", uuid, " is already registered!"));
  }
  VLOG(1) << "RegisterActivePlan: Registered plan for UUID " << uuid
          << ", is_sender: " << is_sender << ", shard_push_schedules size: "
          << request.shard_push_schedules().size();
  return absl::OkStatus();
}

bool KVCacheManagerBase::IsDramDestination(uint64_t uuid) const {
  absl::MutexLock l(plans_mu_);
  auto it = active_plans_.find(uuid);
  if (it != active_plans_.end()) {
    return it->second.request.dst_mem_type() ==
           tpu_raiden::rpc::MEMORY_TYPE_DRAM;
  }
  return false;
}

std::vector<tpu_raiden::transport::BlockChunk>
KVCacheManagerBase::GetBlockChunks(size_t layer_idx, size_t shard_idx,
                                   absl::Span<const int64_t> block_ids,
                                   size_t total_bytes, uint64_t uuid,
                                   int64_t sender_node_id,
                                   absl::string_view peer) {
  RegisteredPlan plan;
  bool has_plan = false;
  {
    absl::MutexLock l(plans_mu_);
    auto it = active_plans_.find(uuid);
    if (it != active_plans_.end()) {
      plan = it->second;
      has_plan = true;
    }
  }

  if (!has_plan || uuid == 0) {
    std::vector<tpu_raiden::transport::BlockChunk> chunks;
    size_t accumulated_bytes = 0;
    for (int64_t block_id : block_ids) {
      if (accumulated_bytes >= total_bytes) break;
      size_t size =
          std::min(bytes_per_block(), total_bytes - accumulated_bytes);
      chunks.push_back(
          {GetBlockHostPointer(layer_idx, shard_idx, block_id), size});
      accumulated_bytes += size;
    }
    return chunks;
  }

  const auto& request = plan.request;
  const auto& schedules = request.shard_push_schedules();
  auto schedule_it = schedules.find(static_cast<int32_t>(shard_idx));

  bool is_sender = plan.is_sender;

  std::vector<tpu_raiden::transport::BlockChunk> chunks;
  size_t block_size_bytes = bytes_per_block();
  uint8_t* base_host_ptr = GetHostPointer(layer_idx, shard_idx);
  size_t accumulated_bytes = 0;

  for (int64_t block_id : block_ids) {
    if (accumulated_bytes >= total_bytes) break;
    size_t block_start_byte = static_cast<size_t>(block_id) * block_size_bytes;
    std::vector<tpu_raiden::transport::BlockChunk> block_resolved_chunks;

    if (is_sender) {
      if (schedule_it != schedules.end()) {
        const auto& schedule = schedule_it->second;
        for (const auto& entry : schedule.entries()) {
          if (!peer.empty() && entry.dst_peer() != peer) {
            continue;
          }
          if (static_cast<size_t>(entry.src_block_id()) == block_id) {
            size_t src_base_offset = entry.src_offset_bytes();
            size_t size = entry.size_bytes();
            size_t src_stride = entry.src_stride_bytes();
            int count = entry.count();
            if (count <= 0) count = 1;

            for (int c = 0; c < count; ++c) {
              size_t src_offset = src_base_offset + c * src_stride;
              block_resolved_chunks.push_back(
                  {.ptr = base_host_ptr + block_start_byte + src_offset,
                   .size = size});
            }
          }
        }
      }
    } else {
      int found_src_shard = -1;
      if (sender_node_id != -1) {
        found_src_shard = static_cast<int>(sender_node_id);
      } else {
        for (const auto& [src_shard, src_schedule] : schedules) {
          for (const auto& entry : src_schedule.entries()) {
            if (static_cast<size_t>(entry.dst_shard_idx()) == shard_idx &&
                static_cast<size_t>(entry.dst_block_id()) == block_id) {
              found_src_shard = src_shard;
              break;
            }
          }
          if (found_src_shard != -1) break;
        }
      }

      if (found_src_shard != -1) {
        auto schedule_found_it = schedules.find(found_src_shard);
        if (schedule_found_it != schedules.end()) {
          const auto& schedule = schedule_found_it->second;
          for (const auto& entry : schedule.entries()) {
            if (static_cast<size_t>(entry.dst_shard_idx()) == shard_idx &&
                static_cast<size_t>(entry.dst_block_id()) == block_id) {
              if (!peer.empty() && entry.dst_peer() != peer) {
                continue;
              }
              size_t dst_base_offset = entry.dst_offset_bytes();
              size_t size = entry.size_bytes();
              size_t dst_stride = entry.dst_stride_bytes();
              int count = entry.count();
              if (count <= 0) count = 1;

              for (int c = 0; c < count; ++c) {
                size_t dst_offset = dst_base_offset + c * dst_stride;
                block_resolved_chunks.push_back(
                    {.ptr = base_host_ptr + block_start_byte + dst_offset,
                     .size = size});
              }
            }
          }
        }
      }
    }

    for (const auto& chunk : block_resolved_chunks) {
      if (accumulated_bytes >= total_bytes) break;
      size_t size_to_add =
          std::min(chunk.size, total_bytes - accumulated_bytes);
      if (size_to_add > 0) {
        chunks.push_back({.ptr = chunk.ptr, .size = size_to_add});
        accumulated_bytes += size_to_add;
      }
    }
  }

  if (!chunks.empty()) {
    return chunks;
  }
  return {};
}

}  // namespace kv_cache
}  // namespace tpu_raiden
