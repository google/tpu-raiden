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

#include "kv_cache/kv_cache_store.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include "xla/pjrt/status_casters.h"

namespace tpu_raiden {
namespace kv_cache {

KVCacheStore::KVCacheStore(int block_size, int capacity)
    : block_size_(block_size), capacity_(capacity) {}

absl::StatusOr<std::pair<std::vector<bool>, raiden::PjRtCopyFuture>>
KVCacheStore::LookupAndFetch(const std::vector<uint64_t>& block_hashes,
                             nanobind::list device_arrays,
                             const std::vector<int>& dst_offsets_major_dim,
                             const std::vector<int>& copy_sizes_major_dim) {
  size_t num_chunks = block_hashes.size();
  if (dst_offsets_major_dim.size() != num_chunks ||
      copy_sizes_major_dim.size() != num_chunks) {
    return absl::InvalidArgumentError("Lengths of lists must match");
  }

  std::vector<bool> hits(num_chunks, false);
  raiden::PjRtCopyFuture acc({});

  for (size_t i = 0; i < num_chunks; ++i) {
    uint64_t hash = block_hashes[i];
    std::shared_ptr<std::vector<std::vector<uint8_t>>> host_buffers;
    std::vector<int> host_block_ids;
    std::shared_ptr<raiden::PjRtCopyFuture> insert_future;
    {
      absl::MutexLock lock(&mutex_);
      auto it = cache_map_.find(hash);
      if (it != cache_map_.end()) {
        hits[i] = true;
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
        host_buffers = it->second->host_buffers;
        host_block_ids = it->second->internal_block_ids;
        insert_future = it->second->insert_future;
      }
    }

    if (hits[i]) {
      if (insert_future) {
        insert_future->Await();
      }
      int needed = copy_sizes_major_dim[i] / block_size_;
      if (host_block_ids.size() < needed) {
        return absl::InternalError("Cached entry does not have enough blocks");
      }

      int dst_major_dim_offset = dst_offsets_major_dim[i];
      std::vector<int64_t> hit_src_offsets;
      std::vector<int64_t> hit_dst_offsets;
      std::vector<int64_t> hit_sizes;
      hit_src_offsets.reserve(needed);
      hit_dst_offsets.reserve(needed);
      hit_sizes.reserve(needed);

      for (int k = 0; k < needed; ++k) {
        hit_src_offsets.push_back(host_block_ids[k] * block_size_);
        hit_dst_offsets.push_back(dst_major_dim_offset + k * block_size_);
        hit_sizes.push_back(block_size_);
      }

      std::vector<const uint8_t*> host_ptrs;
      host_ptrs.reserve(host_buffers->size());
      for (const auto& buf : *host_buffers) {
        host_ptrs.push_back(buf.data());
      }
      KVCacheManager manager(device_arrays, block_size_, std::nullopt,
                             std::nullopt, host_ptrs);
      auto fut_or = manager.H2d(hit_src_offsets, hit_dst_offsets, hit_sizes);
      if (!fut_or.ok()) {
        return fut_or.status();
      }
      acc.Append(std::move(fut_or.value()));
    } else {
      break;
    }
  }

  return std::pair<std::vector<bool>, raiden::PjRtCopyFuture>{std::move(hits),
                                                              std::move(acc)};
}

absl::Status KVCacheStore::Insert(
    const std::vector<uint64_t>& block_hashes, nanobind::list device_arrays,
    const std::vector<int>& src_offsets_major_dim,
    const std::vector<int>& copy_sizes_major_dim) {
  size_t num_chunks = block_hashes.size();
  if (src_offsets_major_dim.size() != num_chunks ||
      copy_sizes_major_dim.size() != num_chunks) {
    return absl::InvalidArgumentError("Lengths of lists must match");
  }

  KVCacheManager manager(device_arrays, block_size_, std::nullopt, 0);
  size_t num_layers = manager.num_layers();
  size_t num_shards = manager.num_shards();
  size_t slice_byte_size = manager.slice_byte_size();

  // Convert src_offsets and copy_sizes to int64_t vectors for D2hAutoAllocate
  std::vector<int64_t> src_offsets_64(src_offsets_major_dim.begin(),
                                      src_offsets_major_dim.end());
  std::vector<int64_t> copy_sizes_64(copy_sizes_major_dim.begin(),
                                     copy_sizes_major_dim.end());

  // Calculate total memory blocks and space needed for the entire insert batch
  int total_needed_blocks = 0;
  for (int copy_size : copy_sizes_major_dim) {
    total_needed_blocks += copy_size / block_size_;
  }

  size_t shard_alloc_size = total_needed_blocks * block_size_ * slice_byte_size;
  auto host_buffers = std::make_shared<std::vector<std::vector<uint8_t>>>(
      num_layers * num_shards);
  std::vector<const uint8_t*> host_ptrs;
  std::vector<size_t> host_sizes;
  for (size_t j = 0; j < num_layers * num_shards; ++j) {
    (*host_buffers)[j].resize(shard_alloc_size);
    host_ptrs.push_back((*host_buffers)[j].data());
    host_sizes.push_back(shard_alloc_size);
  }

  manager.SetExternalHostPointers(host_ptrs, host_sizes);
  auto fut_or = manager.D2hAutoAllocate(src_offsets_64, copy_sizes_64);
  if (!fut_or.ok()) {
    return fut_or.status();
  }

  auto [allocated_block_ids, future] = fut_or.value();
  auto insert_future =
      std::make_shared<raiden::PjRtCopyFuture>(std::move(future));

  size_t block_idx = 0;
  for (size_t i = 0; i < num_chunks; ++i) {
    uint64_t hash = block_hashes[i];
    int needed = copy_sizes_major_dim[i] / block_size_;
    std::vector<int> chunk_block_ids(
        allocated_block_ids.begin() + block_idx,
        allocated_block_ids.begin() + block_idx + needed);
    block_idx += needed;

    {
      absl::MutexLock lock(&mutex_);
      auto map_it = cache_map_.find(hash);
      if (map_it != cache_map_.end()) {
        lru_list_.erase(map_it->second);
        cache_map_.erase(map_it);
      }
      while (lru_list_.size() >= capacity_) {
        auto back = lru_list_.back();
        cache_map_.erase(back.block_hash);
        lru_list_.pop_back();
      }

      lru_list_.push_front(
          {hash, chunk_block_ids, host_buffers, insert_future});
      cache_map_[hash] = lru_list_.begin();
    }
  }

  return absl::OkStatus();
}

NB_MODULE(kv_cache_store, m) {
  nanobind::class_<raiden::PjRtCopyFuture>(m, "PjRtCopyFuture")
      .def("Await",
           [](raiden::PjRtCopyFuture& future) {
             nanobind::gil_scoped_release release;
             future.Await();
           })
      .def("IsReady", &raiden::PjRtCopyFuture::IsReady);

  nanobind::class_<tpu_raiden::kv_cache::KVCacheStore>(m, "KVCacheStore")
      .def(nanobind::init<int, int>(), nanobind::arg("block_size"),
           nanobind::arg("capacity"))
      .def(
          "lookup_and_fetch",
          [](tpu_raiden::kv_cache::KVCacheStore& self,
             const std::vector<uint64_t>& block_hashes,
             nanobind::list device_arrays, const std::vector<int>& dst_offsets,
             const std::vector<int>& copy_sizes) {
            return xla::ValueOrThrow(self.LookupAndFetch(
                block_hashes, device_arrays, dst_offsets, copy_sizes));
          },
          nanobind::arg("block_hashes"), nanobind::arg("device_arrays"),
          nanobind::arg("dst_offsets_major_dim"),
          nanobind::arg("copy_sizes_major_dim"))
      .def(
          "insert",
          [](tpu_raiden::kv_cache::KVCacheStore& self,
             const std::vector<uint64_t>& block_hashes,
             nanobind::list device_arrays, const std::vector<int>& src_offsets,
             const std::vector<int>& copy_sizes) {
            xla::ThrowIfError(self.Insert(block_hashes, device_arrays,
                                          src_offsets, copy_sizes));
          },
          nanobind::arg("block_hashes"), nanobind::arg("device_arrays"),
          nanobind::arg("src_offsets_major_dim"),
          nanobind::arg("copy_sizes_major_dim"));
}

}  // namespace kv_cache
}  // namespace tpu_raiden
