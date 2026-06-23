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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>  // IWYU pragma: keep
#include <nanobind/stl/pair.h>  // IWYU pragma: keep
#include <nanobind/stl/shared_ptr.h>  // IWYU pragma: keep
#include <nanobind/stl/string.h>  // IWYU pragma: keep
#include <nanobind/stl/string_view.h>  // IWYU pragma: keep
#include <nanobind/stl/vector.h>  // IWYU pragma: keep
#include "tpu_raiden/core/raiden_future.h"
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/frameworks/jax/kv_cache_manager.h"
#include "tpu_raiden/frameworks/jax/nb_statusor.h"  // IWYU pragma: keep
#include "tpu_raiden/frameworks/jax/raw_transfer_internal.h"
#include "tpu_raiden/frameworks/jax/weight_synchronizer.h"
#include "tpu_raiden/kv_cache/kv_cache_store.h"

namespace nb = nanobind;

using ::tpu_raiden::jax::WeightSynchronizer;

namespace tpu_raiden {
namespace kv_cache {
namespace {
class KVCacheStoreWrapper {
 public:
  explicit KVCacheStoreWrapper(size_t lru_capacity) {
    controller_ = std::make_unique<KVCacheStore>(lru_capacity);
  }
  KVCacheStore* operator->() { return controller_.get(); }
  KVCacheStore& operator*() { return *controller_; }

 private:
  std::unique_ptr<KVCacheStore> controller_;
};
}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden

NB_MODULE(_tpu_raiden_jax, m) {
  // =========================================================================
  // 1. Bind KVCacheManager
  // =========================================================================

  // Bind the new Future class as RaidenFuture to maintain duck-typing
  // compatibility.
  nb::class_<tpu_raiden::RaidenFuture>(m, "RaidenFuture")
      .def("Await",
           [](tpu_raiden::RaidenFuture& self) {
             nb::gil_scoped_release release;
             absl::Status status = self.Await();
             if (!status.ok()) {
               throw std::runtime_error("Async copy failed: " +
                                        std::string(status.message()));
             }
           })
      .def("wait",
           [](tpu_raiden::RaidenFuture& self) {
             nb::gil_scoped_release release;
             absl::Status status = self.Await();
             if (!status.ok()) {
               throw std::runtime_error("Async copy failed: " +
                                        std::string(status.message()));
             }
           })
      .def("IsReady", &tpu_raiden::RaidenFuture::IsReady)
      .def("is_ready", &tpu_raiden::RaidenFuture::IsReady);

  nb::class_<tpu_raiden::kv_cache::jax::KVCacheManager>(m, "KVCacheManager")
      .def(nb::init<nb::list, std::optional<int>, std::optional<int>, bool,
                    int>(),
           nb::arg("device_arrays"),
           nb::arg("local_port") = nb::none(),
           nb::arg("host_blocks_to_allocate") = nb::none(),
           nb::arg("unsafe_skip_buffer_lock") = false,
           nb::arg("parallelism") = 1)
      .def(nb::init<nanobind::list, int64_t, int64_t, int64_t, int64_t, double,
                    bool>(),
           nb::arg("kv_caches"), nb::arg("node_id") = 0,
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
              -> absl::StatusOr<tpu_raiden::RaidenFuture> {
            auto res = self.H2d(src_offsets, dst_offsets, copy_sizes);
            if (!res.ok()) return res.status();
            return tpu_raiden::RaidenFuture{std::move(res.value())};
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
              -> absl::StatusOr<tpu_raiden::RaidenFuture> {
            auto res = self.D2h(src_offsets, dst_offsets, copy_sizes);
            if (!res.ok()) return res.status();
            return tpu_raiden::RaidenFuture{std::move(res.value())};
          },
          nb::arg("src_offsets_major_dim") = nb::list(),
          nb::arg("dst_offsets_major_dim") = nb::list(),
          nb::arg("copy_sizes_major_dim") = nb::list())

      .def(
          "d2h_auto_allocate",
          [](tpu_raiden::kv_cache::jax::KVCacheManager& self,
             const std::vector<int64_t>& src_offsets,
             const std::vector<int64_t>& copy_sizes)
              -> absl::StatusOr<
                  std::pair<std::vector<int>, tpu_raiden::RaidenFuture>> {
            auto res = self.D2hAutoAllocate(src_offsets, copy_sizes);
            if (!res.ok()) return res.status();
            return std::make_pair(
                res.value().first,
                tpu_raiden::RaidenFuture{std::move(res.value().second)});
          },
          nb::arg("src_offsets_major_dim") = nb::list(),
          nb::arg("copy_sizes_major_dim") = nb::list())

      .def(
          "h2h_write",
          [](tpu_raiden::kv_cache::jax::KVCacheManager& self, std::string peer,
             const std::vector<int>& src_block_ids)
              -> absl::StatusOr<
                  std::pair<std::vector<int>, tpu_raiden::RaidenFuture>> {
            auto res = self.H2hWrite(peer, src_block_ids);
            if (!res.ok()) return res.status();
            return std::make_pair(
                res.value().first,
                tpu_raiden::RaidenFuture{std::move(res.value().second)});
          },
          nb::arg("peer"), nb::arg("src_block_ids"))

      .def(
          "h2h_read",
          [](tpu_raiden::kv_cache::jax::KVCacheManager& self, std::string peer,
             const std::vector<int>& src_block_ids)
              -> absl::StatusOr<
                  std::pair<std::vector<int>, tpu_raiden::RaidenFuture>> {
            auto res = self.H2hRead(peer, src_block_ids);
            if (!res.ok()) return res.status();
            return std::make_pair(
                res.value().first,
                tpu_raiden::RaidenFuture{std::move(res.value().second)});
          },
          nb::arg("peer"), nb::arg("src_block_ids"))

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
           nb::arg("parallelism") = 1,
           nb::arg("local_host_block_ids") = nb::none())
      .def("complete_read",
           [](tpu_raiden::kv_cache::jax::KVCacheManager& self) {
             auto [done_sending, done_recving, failed_recving] =
                 self.CompleteReadRaw();
             return nb::make_tuple(done_sending, done_recving, failed_recving);
           });

  // =========================================================================
  // 2. Bind WeightSynchronizer
  // =========================================================================
  nb::class_<WeightSynchronizer>(m, "WeightSynchronizer")
      .def(nb::init<nb::list, std::optional<int>, int, bool,
                    std::optional<int>>(),
           nb::arg("jax_arrays"), nb::arg("local_port") = nb::none(),
           nb::arg("parallelism") = 1,
           nb::arg("unsafe_skip_buffer_lock") = false,
           nb::arg("control_port") = nb::none())
      .def(
          "PullWeights",
          [](WeightSynchronizer& self, absl::string_view source) {
            absl::Status s = self.PullWeights(source);
            if (!s.ok()) {
              throw std::runtime_error("Weight sync PullWeights failed: " +
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
                  "Weight sync D2H failed: " +
                  std::string(status_or_future.status().message()));
            }
            absl::Status status = status_or_future.value().Await();
            if (!status.ok()) {
              throw std::runtime_error("Weight sync D2H copy failed: " +
                                       std::string(status.message()));
            }
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "H2d",
          [](WeightSynchronizer& self) {
            auto status_or_future = self.H2d();
            if (!status_or_future.ok()) {
              throw std::runtime_error(
                  "Weight sync H2D failed: " +
                  std::string(status_or_future.status().message()));
            }
            absl::Status status = status_or_future.value().Await();
            if (!status.ok()) {
              throw std::runtime_error("Weight sync H2D copy failed: " +
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
                shard_idx, host_offset_bytes, device_offset_bytes, size_bytes
            );
            if (!status_or_future.ok()) {
              throw std::runtime_error(
                  "Weight sync H2DChunk failed: " +
                  std::string(status_or_future.status().message()));
            }
            absl::Status status = status_or_future.value().Await();
            if (!status.ok()) {
              throw std::runtime_error("Weight sync H2DChunk copy failed: " +
                                       std::string(status.message()));
            }
          },
          nb::arg("shard_idx"), nb::arg("host_offset_bytes"),
          nb::arg("device_offset_bytes"), nb::arg("size_bytes"),
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "PullWeightsChunk",
          [](WeightSynchronizer& self, absl::string_view source,
             size_t src_shard_idx, size_t src_offset_bytes,
             size_t dst_shard_idx, size_t dst_offset_bytes, size_t size_bytes) {
            absl::Status s = self.PullWeightsChunk(
                source, src_shard_idx, src_offset_bytes, dst_shard_idx,
                dst_offset_bytes, size_bytes
            );
            if (!s.ok()) {
              throw std::runtime_error(
                  "Weight sync PullWeightsChunk failed: " +
                  std::string(s.message())
              );
            }
          },
          nb::arg("source"), nb::arg("src_shard_idx"),
          nb::arg("src_offset_bytes"), nb::arg("dst_shard_idx"),
          nb::arg("dst_offset_bytes"), nb::arg("size_bytes"),
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "get_host_buffer",
          [](WeightSynchronizer& self, size_t layer_idx, size_t shard_idx) {
            const uint8_t* ptr = self.GetHostBufferPtr(layer_idx, shard_idx);
            if (!ptr) {
              throw std::runtime_error("Invalid layer or shard index");
            }
            size_t size = self.slice_byte_size() + 256 * 1024;
            size_t shape[1] = {size};
            return nb::ndarray<uint8_t, nb::numpy, nb::c_contig>(
                const_cast<uint8_t*>(ptr), 1, shape,
                nb::handle() /* view only, no ownership copy */
            );
          },
          nb::arg("layer_idx") = 0, nb::arg("shard_idx") = 0)
      .def_prop_ro("local_port", &WeightSynchronizer::local_port)
      .def_prop_ro("control_port", &WeightSynchronizer::control_port)
      .def_prop_ro("is_control_service_active",
                   &WeightSynchronizer::is_control_service_active)
      .def_prop_ro("num_layers", &WeightSynchronizer::num_layers)
      .def_prop_ro("num_shards", &WeightSynchronizer::num_shards)
      .def_prop_ro("slice_byte_size", &WeightSynchronizer::slice_byte_size);

  // =========================================================================
  // 3. Bind KVCacheStore
  // =========================================================================
  nb::class_<tpu_raiden::kv_cache::RaidenId>(m, "RaidenId")
      .def(nb::init<std::string, std::string, std::string, int>(),
           nb::arg("job_name"), nb::arg("job_replica_id") = "",
           nb::arg("data_name"), nb::arg("data_replica_idx") = 0)
      .def_rw("job_name", &tpu_raiden::kv_cache::RaidenId::job_name)
      .def_rw("job_replica_id", &tpu_raiden::kv_cache::RaidenId::job_replica_id)
      .def_rw("data_name", &tpu_raiden::kv_cache::RaidenId::data_name)
      .def_rw("data_replica_idx",
              &tpu_raiden::kv_cache::RaidenId::data_replica_idx);

  nb::class_<tpu_raiden::kv_cache::KVCacheStoreWrapper>(m, "KVCacheStore")
      .def(nb::init<size_t>(), nb::arg("capacity"))
      .def(
          "lookup",
          [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self,
             const std::vector<uint64_t>& block_hashes) {
            auto res = self->Lookup(block_hashes);
            if (!res.ok()) {
              throw std::runtime_error("KVCacheStore lookup failed: " +
                                       std::string(res.status().message()));
            }
            return res.value();
          },
          nb::arg("block_hashes"),
          "Checks the LRU directory for cached block hashes. Returns a list of "
          "all matched replica pairs prior to the first miss.")
      .def(
          "insert",
          [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self,
             const std::vector<uint64_t>& block_hashes,
             const std::vector<std::vector<tpu_raiden::kv_cache::RaidenId>>&
                 slices,
             bool on_host) {
            return self->Insert(block_hashes, slices, on_host);
          },
          nb::arg("block_hashes"), nb::arg("slices"), nb::arg("on_host"))
      .def(
          "delete",
          [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self,
             const std::vector<uint64_t>& block_hashes,
             const std::vector<std::vector<tpu_raiden::kv_cache::RaidenId>>&
                 slices) { self->Delete(block_hashes, slices); },
          nb::arg("block_hashes"), nb::arg("slices"))
      .def("capacity",
           [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self) {
             return self->capacity();
           })
      .def(
          "pin",
          [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self,
             const std::vector<uint64_t>& block_hashes) {
            return self->Pin(block_hashes);
          },
          nb::arg("block_hashes"))
      .def(
          "release",
          [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self,
             const std::vector<uint64_t>& block_hashes) {
            self->Release(block_hashes);
          },
          nb::arg("block_hashes"));
}
