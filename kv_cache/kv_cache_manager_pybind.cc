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

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "kv_cache/kv_cache_manager.h"

namespace nb = nanobind;

namespace {

template <typename T>
T ValueOrThrow(absl::StatusOr<T> value_or) {
  if (!value_or.ok()) {
    throw std::runtime_error(std::string(value_or.status().message()));
  }
  return std::move(value_or).value();
}

}  // namespace

NB_MODULE(kv_cache_manager, m) {
  nb::class_<tpu_raiden::kv_cache::KVCacheTransferFuture>(m, "PjRtCopyFuture")
      .def("Await",
           [](tpu_raiden::kv_cache::KVCacheTransferFuture& future) {
             nb::gil_scoped_release release;
             future.Await();
           })
      .def("IsReady",
           &tpu_raiden::kv_cache::KVCacheTransferFuture::IsReady);

  nb::class_<tpu_raiden::kv_cache::KVCacheManager>(m, "KVCacheManager")
      .def(
          "__init__",
          [](tpu_raiden::kv_cache::KVCacheManager* self, nb::list device_arrays,
             int block_size, std::optional<int> local_port,
             std::optional<int> host_blocks_to_allocate,
             std::optional<std::vector<uintptr_t>> external_host_ptrs,
             bool unsafe_skip_buffer_lock, int parallelism) {
            std::optional<std::vector<const uint8_t*>> cpp_ptrs = std::nullopt;
            if (external_host_ptrs.has_value()) {
              std::vector<const uint8_t*> ptrs;
              ptrs.reserve(external_host_ptrs->size());
              for (uintptr_t p : *external_host_ptrs) {
                ptrs.push_back(reinterpret_cast<const uint8_t*>(p));
              }
              cpp_ptrs = std::move(ptrs);
            }
            new (self) tpu_raiden::kv_cache::KVCacheManager(
                device_arrays, block_size, local_port, host_blocks_to_allocate,
                cpp_ptrs, unsafe_skip_buffer_lock, parallelism);
          },
          nb::arg("device_arrays"), nb::arg("block_size") = 1,
          nb::arg("local_port") = nb::none(),
          nb::arg("host_blocks_to_allocate") = nb::none(),
          nb::arg("external_host_ptrs") = nb::none(),
          nb::arg("unsafe_skip_buffer_lock") = false,
          nb::arg("parallelism") = 1)
      .def(
          "h2d",
          [](tpu_raiden::kv_cache::KVCacheManager& self,
             const std::vector<int64_t>& src_offsets,
             const std::vector<int64_t>& dst_offsets,
             const std::vector<int64_t>& copy_sizes) {
            return ValueOrThrow(self.H2d(src_offsets, dst_offsets, copy_sizes));
          },
          nb::arg("src_offsets_major_dim") = std::vector<int64_t>(),
          nb::arg("dst_offsets_major_dim") = std::vector<int64_t>(),
          nb::arg("copy_sizes_major_dim") = std::vector<int64_t>())
      .def(
          "d2h",
          [](tpu_raiden::kv_cache::KVCacheManager& self,
             const std::vector<int64_t>& src_offsets,
             const std::vector<int64_t>& dst_offsets,
             const std::vector<int64_t>& copy_sizes) {
            return ValueOrThrow(self.D2h(src_offsets, dst_offsets, copy_sizes));
          },
          nb::arg("src_offsets_major_dim") = std::vector<int64_t>(),
          nb::arg("dst_offsets_major_dim") = std::vector<int64_t>(),
          nb::arg("copy_sizes_major_dim") = std::vector<int64_t>())
      .def(
          "d2h_auto_allocate",
          [](tpu_raiden::kv_cache::KVCacheManager& self,
             const std::vector<int64_t>& src_offsets,
             const std::vector<int64_t>& copy_sizes, int64_t entity_id) {
            return ValueOrThrow(
                self.D2hAutoAllocate(src_offsets, copy_sizes, entity_id));
          },
          nb::arg("src_offsets_major_dim") = std::vector<int64_t>(),
          nb::arg("copy_sizes_major_dim") = std::vector<int64_t>(),
          nb::arg("entity_id") = 0)
      .def(
          "h2h_write",
          [](tpu_raiden::kv_cache::KVCacheManager& self, std::string peer,
             const std::vector<int>& src_block_ids, int64_t entity_id) {
            return ValueOrThrow(
                self.H2hWrite(peer, src_block_ids, entity_id));
          },
          nb::arg("peer"), nb::arg("src_block_ids"), nb::arg("entity_id") = 0)
      .def(
          "h2h_read",
          [](tpu_raiden::kv_cache::KVCacheManager& self, std::string peer,
             const std::vector<int>& src_block_ids, int64_t entity_id) {
            return ValueOrThrow(self.H2hRead(peer, src_block_ids, entity_id));
          },
          nb::arg("peer"), nb::arg("src_block_ids"), nb::arg("entity_id") = 0)
      .def("local_port", &tpu_raiden::kv_cache::KVCacheManager::local_port);
}
