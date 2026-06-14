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

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ATen/ops/from_blob.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/raw_transfer_core.h"
#include "tpu_raiden/frameworks/torch/kv_cache_manager.h"
#include "tpu_raiden/frameworks/torch/torch_nanobind_utils.h"
#include "tpu_raiden/frameworks/torch/weight_synchronizer.h"

namespace nb = nanobind;

using ::tpu_raiden::torch::KVCacheManager;
using ::tpu_raiden::torch::WeightSynchronizer;

NB_MODULE(_tpu_raiden_torch, m) {
  // =========================================================================
  // 1. Bind PjRtCopyFuture (shared between both)
  // =========================================================================
  nb::class_<::raiden::PjRtCopyFuture>(m, "PjRtCopyFuture")
      .def("Await",
           [](::raiden::PjRtCopyFuture& future) {
             nb::gil_scoped_release release;
             absl::Status status = future.Await().status();
             if (!status.ok()) {
               throw std::runtime_error(std::string("Async copy failed: ") +
                                        std::string(status.message()));
             }
           })
      .def("IsReady", &::raiden::PjRtCopyFuture::IsReady);

  // =========================================================================
  // 2. Bind KVCacheManager
  // =========================================================================
  nb::class_<KVCacheManager>(m, "KVCacheManager")
      .def(nb::init<const std::vector<std::vector<at::Tensor>>&, int,
                    std::optional<int>, std::optional<int>,
                    std::optional<std::vector<uintptr_t>>, bool, int>(),
           nb::arg("device_tensors"), nb::arg("block_size") = 1,
           nb::arg("local_port") = nb::none(),
           nb::arg("host_blocks_to_allocate") = nb::none(),
           nb::arg("external_host_ptrs") = nb::none(),
           nb::arg("unsafe_skip_buffer_lock") = false,
           nb::arg("parallelism") = 1)
      .def(nb::init<const std::vector<at::Tensor>&, int64_t, int64_t, int64_t,
                    int64_t, double, bool>(),
           nb::arg("kv_caches"), nb::arg("tp_rank"),
           nb::arg("local_control_port"), nb::arg("max_blocks"),
           nb::arg("num_slots"), nb::arg("timeout_s") = 120.0,
           nb::arg("unsafe_skip_buffer_lock") = true)
      .def(
          "H2d",
          [](KVCacheManager& self,
             const std::vector<int64_t>& src_offsets_major_dim,
             const std::vector<int64_t>& dst_offsets_major_dim,
             const std::vector<int64_t>& copy_sizes_major_dim) {
            auto status_or =
                self.H2d(src_offsets_major_dim, dst_offsets_major_dim,
                         copy_sizes_major_dim);
            if (!status_or.ok()) {
              throw std::runtime_error(
                  "KVCacheManager H2d failed: " +
                  std::string(status_or.status().message()));
            }
            return status_or.value();
          },
          nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
          nb::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
          nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{})
      .def(
          "D2h",
          [](KVCacheManager& self,
             const std::vector<int64_t>& src_offsets_major_dim,
             const std::vector<int64_t>& dst_offsets_major_dim,
             const std::vector<int64_t>& copy_sizes_major_dim) {
            auto status_or =
                self.D2h(src_offsets_major_dim, dst_offsets_major_dim,
                         copy_sizes_major_dim);
            if (!status_or.ok()) {
              throw std::runtime_error(
                  "KVCacheManager D2h failed: " +
                  std::string(status_or.status().message()));
            }
            return status_or.value();
          },
          nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
          nb::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
          nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{})
      .def(
          "D2hAutoAllocate",
          [](KVCacheManager& self,
             const std::vector<int64_t>& src_offsets_major_dim,
             const std::vector<int64_t>& copy_sizes_major_dim,
             int64_t entity_id) {
            auto status_or = self.D2hAutoAllocate(
                src_offsets_major_dim, copy_sizes_major_dim, entity_id);
            if (!status_or.ok()) {
              throw std::runtime_error(
                  "KVCacheManager D2hAutoAllocate failed: " +
                  std::string(status_or.status().message()));
            }
            return status_or.value();
          },
          nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
          nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{},
          nb::arg("entity_id") = 0)
      .def(
          "H2hWrite",
          [](KVCacheManager& self, std::string peer,
             const std::vector<int>& src_block_ids, int64_t entity_id) {
            auto status_or =
                self.H2hWrite(std::move(peer), src_block_ids, entity_id);
            if (!status_or.ok()) {
              throw std::runtime_error(
                  "KVCacheManager H2hWrite failed: " +
                  std::string(status_or.status().message()));
            }
            return status_or.value();
          },
          nb::arg("peer"), nb::arg("src_block_ids"), nb::arg("entity_id") = 0,
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "H2hRead",
          [](KVCacheManager& self, std::string peer,
             const std::vector<int>& src_block_ids, int64_t entity_id) {
            auto status_or =
                self.H2hRead(std::move(peer), src_block_ids, entity_id);
            if (!status_or.ok()) {
              throw std::runtime_error(
                  "KVCacheManager H2hRead failed: " +
                  std::string(status_or.status().message()));
            }
            return status_or.value();
          },
          nb::arg("peer"), nb::arg("src_block_ids"), nb::arg("entity_id") = 0,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("local_port", &KVCacheManager::local_port)
      .def_prop_ro("num_layers", &KVCacheManager::num_layers)
      .def_prop_ro("num_shards", &KVCacheManager::num_shards)
      .def_prop_ro("block_size", &KVCacheManager::block_size)
      .def_prop_ro("slice_byte_size", &KVCacheManager::slice_byte_size)
      .def_prop_ro("local_control_port", &KVCacheManager::local_control_port)
      .def("notify_for_read", &KVCacheManager::NotifyForRead, nb::arg("req_id"),
           nb::arg("uuid"), nb::arg("block_ids"))
      .def("start_read", &KVCacheManager::StartRead, nb::arg("req_id"),
           nb::arg("uuid"), nb::arg("remote_endpoint"),
           nb::arg("remote_block_ids"), nb::arg("local_block_ids"),
           nb::arg("parallelism") = 1)
      .def("complete_read", [](KVCacheManager& self) {
        auto [done_sending, done_recving, failed_recving] =
            self.CompleteReadRaw();
        return nb::make_tuple(done_sending, done_recving, failed_recving);
      });

  // =========================================================================
  // 3. Bind WeightSynchronizer
  // =========================================================================
  nb::class_<WeightSynchronizer>(m, "WeightSynchronizer")
      .def(nb::init<const std::vector<std::vector<at::Tensor>>&,
                    std::optional<int>, int>(),
           nb::arg("device_tensors"), nb::arg("local_port") = nb::none(),
           nb::arg("parallelism") = 1)
      .def(
          "PushWeights",
          [](WeightSynchronizer& self, const std::vector<std::string>& peers) {
            absl::Status s = self.PushWeights(peers);
            if (!s.ok()) {
              throw std::runtime_error(
                  "WeightSynchronizer PushWeights failed: " +
                  std::string(s.message()));
            }
          },
          nb::arg("peers"), nb::call_guard<nb::gil_scoped_release>())
      .def(
          "PullWeights",
          [](WeightSynchronizer& self, const std::string& source) {
            absl::Status s = self.PullWeights(source);
            if (!s.ok()) {
              throw std::runtime_error(
                  "WeightSynchronizer PullWeights failed: " +
                  std::string(s.message()));
            }
          },
          nb::arg("source"), nb::call_guard<nb::gil_scoped_release>())
      .def(
          "D2h",
          [](WeightSynchronizer& self) {
            auto status_or_future = self.D2h();
            if (!status_or_future.ok()) {
              throw std::runtime_error(
                  "WeightSynchronizer D2H failed: " +
                  std::string(status_or_future.status().message()));
            }
            absl::Status status = status_or_future.value().Await().status();
            if (!status.ok()) {
              throw std::runtime_error("WeightSynchronizer D2H copy failed: " +
                                       std::string(status.message()));
            }
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "H2dChunk",
          [](WeightSynchronizer& self, size_t shard_idx,
             size_t host_offset_bytes, size_t device_offset_bytes,
             size_t size_bytes) {
            auto status_or_future = self.H2dChunk(
                shard_idx, host_offset_bytes, device_offset_bytes, size_bytes);
            if (!status_or_future.ok()) {
              throw std::runtime_error(
                  "WeightSynchronizer H2dChunk failed: " +
                  std::string(status_or_future.status().message()));
            }
            absl::Status status = status_or_future.value().Await().status();
            if (!status.ok()) {
              throw std::runtime_error(
                  "WeightSynchronizer H2dChunk copy failed: " +
                  std::string(status.message()));
            }
          },
          nb::arg("shard_idx"), nb::arg("host_offset_bytes"),
          nb::arg("device_offset_bytes"), nb::arg("size_bytes"),
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "PullWeightsChunk",
          [](WeightSynchronizer& self, const std::string& source,
             size_t src_shard_idx, size_t src_offset_bytes,
             size_t dst_shard_idx, size_t dst_offset_bytes, size_t size_bytes) {
            absl::Status s = self.PullWeightsChunk(
                source, src_shard_idx, src_offset_bytes, dst_shard_idx,
                dst_offset_bytes, size_bytes);
            if (!s.ok()) {
              throw std::runtime_error(
                  "WeightSynchronizer PullWeightsChunk failed: " +
                  std::string(s.message()));
            }
          },
          nb::arg("source"), nb::arg("src_shard_idx"),
          nb::arg("src_offset_bytes"), nb::arg("dst_shard_idx"),
          nb::arg("dst_offset_bytes"), nb::arg("size_bytes"),
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "get_host_buffer",
          [](WeightSynchronizer& self, size_t layer_idx,
             size_t shard_idx) -> at::Tensor {
            const uint8_t* ptr = self.GetHostBufferPtr(layer_idx, shard_idx);
            if (!ptr) {
              throw std::runtime_error("Invalid layer or shard index");
            }
            size_t size = self.slice_byte_size() + 256 * 1024;
            return at::from_blob(const_cast<uint8_t*>(ptr),
                                 {static_cast<int64_t>(size)}, at::kByte);
          },
          nb::arg("layer_idx") = 0, nb::arg("shard_idx") = 0)
      .def_prop_ro("local_port", &WeightSynchronizer::local_port)
      .def_prop_ro("num_layers", &WeightSynchronizer::num_layers)
      .def_prop_ro("num_shards", &WeightSynchronizer::num_shards)
      .def_prop_ro("slice_byte_size", &WeightSynchronizer::slice_byte_size);
}
