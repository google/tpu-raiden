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

#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"
#include "absl/status/statusor.h"
#include "api/torch/kv_cache_manager.h"
#include "core/raw_transfer_core.h"
#include "frameworks/torch/torch_nanobind_utils.h"
#include "torch/extension.h"

namespace nb = nanobind;

using ::tpu_raiden::torch::KVCacheManager;

// Nanobind FFI bindings module definition for PyTorch E2E
NB_MODULE(_kv_cache_manager, m) {
  nb::class_<::raiden::PjRtCopyFuture>(m, "PjRtCopyFuture")
      .def("Await", &::raiden::PjRtCopyFuture::Await,
           nb::call_guard<nb::gil_scoped_release>())
      .def("IsReady", &::raiden::PjRtCopyFuture::IsReady);

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
      .def_prop_ro("slice_byte_size", &KVCacheManager::slice_byte_size);
}
