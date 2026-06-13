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

#include "absl/status/statusor.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>  // IWYU pragma: keep
#include <nanobind/stl/pair.h>  // IWYU pragma: keep
#include <nanobind/stl/string.h>  // IWYU pragma: keep
#include <nanobind/stl/vector.h>  // IWYU pragma: keep
#include "core/raw_transfer_core.h"
#include "frameworks/jax/kv_cache_manager.h"
#include "frameworks/jax/nb_statusor.h"  // IWYU pragma: keep

namespace nb = nanobind;

// Define a wrapper struct to avoid exposing PjRtCopyFuture directly,
// which would force a dependency on the _raw_transfer module.
struct KVCacheManagerFuture {
  raiden::PjRtCopyFuture future;

  void Await() {
    nb::gil_scoped_release release;
    absl::Status status = future.Await().status();
    if (!status.ok()) {
      throw std::runtime_error(std::string("Async copy failed: ") +
                               std::string(status.message()));
    }
  }

  bool IsReady() const { return future.IsReady(); }
};

NB_MODULE(_kv_cache_manager, m) {
  // Removed the import of _raw_transfer to sever the dependency.
  // nb::module_::import_("frameworks.jax._raw_transfer");

  // Bind the new Future class as RaidenFuture to maintain duck-typing
  // compatibility.
  nb::class_<KVCacheManagerFuture>(m, "RaidenFuture")
      .def("Await", &KVCacheManagerFuture::Await)
      .def("wait", &KVCacheManagerFuture::Await)
      .def("IsReady", &KVCacheManagerFuture::IsReady)
      .def("is_ready", &KVCacheManagerFuture::IsReady);

  nb::class_<tpu_raiden::kv_cache::jax::KVCacheManager>(m, "KVCacheManager")
      .def(nb::init<nb::list, int, std::optional<int>, std::optional<int>,
                    std::optional<std::vector<uintptr_t>>, bool, int>(),
           nb::arg("device_arrays"), nb::arg("block_size") = 1,
           nb::arg("local_port") = nb::none(),
           nb::arg("host_blocks_to_allocate") = nb::none(),
           nb::arg("external_host_ptrs") = nb::none(),
           nb::arg("unsafe_skip_buffer_lock") = false,
           nb::arg("parallelism") = 1)
      .def(nb::init<nanobind::list, int64_t, int64_t, int64_t, int64_t, double,
                    bool>(),
           nb::arg("kv_caches"), nb::arg("tp_rank") = 0,
           nb::arg("local_control_port"), nb::arg("max_blocks"),
           nb::arg("num_slots"), nb::arg("timeout_s") = 120.0,
           nb::arg("unsafe_skip_buffer_lock") = true)

      // Use lambdas to wrap the returned raiden::PjRtCopyFuture into
      // KVCacheManagerFuture
      .def(
          "h2d",
          [](tpu_raiden::kv_cache::jax::KVCacheManager& self,
             const std::vector<int64_t>& src_offsets,
             const std::vector<int64_t>& dst_offsets,
             const std::vector<int64_t>& copy_sizes)
              -> absl::StatusOr<KVCacheManagerFuture> {
            auto res = self.H2d(src_offsets, dst_offsets, copy_sizes);
            if (!res.ok()) return res.status();
            return KVCacheManagerFuture{std::move(res.value())};
          },
          nb::arg("src_offsets_major_dim") = nb::list(),
          nb::arg("dst_offsets_major_dim") = nb::list(),
          nb::arg("copy_sizes_major_dim") = nb::list())

      .def(
          "d2h",
          [](tpu_raiden::kv_cache::jax::KVCacheManager& self,
             const std::vector<int64_t>& src_offsets,
             const std::vector<int64_t>& dst_offsets,
             const std::vector<int64_t>& copy_sizes)
              -> absl::StatusOr<KVCacheManagerFuture> {
            auto res = self.D2h(src_offsets, dst_offsets, copy_sizes);
            if (!res.ok()) return res.status();
            return KVCacheManagerFuture{std::move(res.value())};
          },
          nb::arg("src_offsets_major_dim") = nb::list(),
          nb::arg("dst_offsets_major_dim") = nb::list(),
          nb::arg("copy_sizes_major_dim") = nb::list())

      .def(
          "d2h_auto_allocate",
          [](tpu_raiden::kv_cache::jax::KVCacheManager& self,
             const std::vector<int64_t>& src_offsets,
             const std::vector<int64_t>& copy_sizes, int64_t entity_id)
              -> absl::StatusOr<
                  std::pair<std::vector<int>, KVCacheManagerFuture>> {
            auto res = self.D2hAutoAllocate(src_offsets, copy_sizes, entity_id);
            if (!res.ok()) return res.status();
            return std::make_pair(
                res.value().first,
                KVCacheManagerFuture{std::move(res.value().second)});
          },
          nb::arg("src_offsets_major_dim") = nb::list(),
          nb::arg("copy_sizes_major_dim") = nb::list(),
          nb::arg("entity_id") = 0)

      .def(
          "h2h_write",
          [](tpu_raiden::kv_cache::jax::KVCacheManager& self, std::string peer,
             const std::vector<int>& src_block_ids, int64_t entity_id)
              -> absl::StatusOr<
                  std::pair<std::vector<int>, KVCacheManagerFuture>> {
            auto res = self.H2hWrite(peer, src_block_ids, entity_id);
            if (!res.ok()) return res.status();
            return std::make_pair(
                res.value().first,
                KVCacheManagerFuture{std::move(res.value().second)});
          },
          nb::arg("peer"), nb::arg("src_block_ids"), nb::arg("entity_id") = 0)

      .def(
          "h2h_read",
          [](tpu_raiden::kv_cache::jax::KVCacheManager& self, std::string peer,
             const std::vector<int>& src_block_ids, int64_t entity_id)
              -> absl::StatusOr<
                  std::pair<std::vector<int>, KVCacheManagerFuture>> {
            auto res = self.H2hRead(peer, src_block_ids, entity_id);
            if (!res.ok()) return res.status();
            return std::make_pair(
                res.value().first,
                KVCacheManagerFuture{std::move(res.value().second)});
          },
          nb::arg("peer"), nb::arg("src_block_ids"), nb::arg("entity_id") = 0)

      .def("local_port", &tpu_raiden::kv_cache::KVCacheManagerBase::local_port)
      .def("get_host_pointer",
           static_cast<const uint8_t* (
               tpu_raiden::kv_cache::KVCacheManagerBase::*)(size_t, size_t)
                           const>(
               &tpu_raiden::kv_cache::KVCacheManagerBase::GetHostPointer),
           nb::arg("layer_idx"), nb::arg("shard_idx"))
      .def_prop_ro("num_layers",
                   &tpu_raiden::kv_cache::KVCacheManagerBase::num_layers)
      .def_prop_ro("num_shards",
                   &tpu_raiden::kv_cache::KVCacheManagerBase::num_shards)
      .def_prop_ro("slice_byte_size",
                   &tpu_raiden::kv_cache::KVCacheManagerBase::slice_byte_size)
      .def_prop_ro(
          "local_control_port",
          &tpu_raiden::kv_cache::jax::KVCacheManager::local_control_port)
      .def("notify_for_read",
           &tpu_raiden::kv_cache::jax::KVCacheManager::NotifyForRead,
           nb::arg("req_id"), nb::arg("uuid"), nb::arg("block_ids"))
      .def("start_read", &tpu_raiden::kv_cache::jax::KVCacheManager::StartRead,
           nb::arg("req_id"), nb::arg("uuid"), nb::arg("remote_endpoint"),
           nb::arg("remote_block_ids"), nb::arg("local_block_ids"),
           nb::arg("parallelism") = 1)
      .def("complete_read",
           [](tpu_raiden::kv_cache::jax::KVCacheManager& self) {
             auto [done_sending, done_recving, failed_recving] =
                 self.CompleteReadRaw();
             return nb::make_tuple(done_sending, done_recving, failed_recving);
           });
}
