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

#include "ATen/ops/from_blob.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/shared_ptr.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/string_view.h"
#include "nanobind/stl/vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tpu_raiden/core/raiden_future.h"
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/frameworks/torch/kv_cache_manager.h"
#include "tpu_raiden/frameworks/torch/torch_nanobind_utils.h"
#include "tpu_raiden/frameworks/torch/weight_synchronizer.h"
#include "tpu_raiden/kv_cache/kv_cache_store.h"

namespace nb = nanobind;

using ::tpu_raiden::torch::KVCacheManager;
using ::tpu_raiden::torch::WeightSynchronizer;

namespace tpu_raiden {
namespace kv_cache {
namespace {

std::vector<std::string> ToStdStringVector(
    const std::vector<nb::bytes>& bytes_vec) {
  std::vector<std::string> str_vec;
  str_vec.reserve(bytes_vec.size());
  for (const auto& b : bytes_vec) {
    str_vec.push_back(std::string(b.c_str(), b.size()));
  }
  return str_vec;
}

class KVCacheStoreWrapper {
 public:
  explicit KVCacheStoreWrapper(size_t lru_capacity,
                               std::string global_registry_address = "",
                               RaidenId raiden_id = {}) {
    controller_ = std::make_unique<KVCacheStore>(
        lru_capacity, global_registry_address, std::move(raiden_id));
  }
  KVCacheStore* operator->() { return controller_.get(); }
  KVCacheStore& operator*() { return *controller_; }

 private:
  std::unique_ptr<KVCacheStore> controller_;
};
}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden

using ::tpu_raiden::kv_cache::ToStdStringVector;

NB_MODULE(_tpu_raiden_torch, m) {
  // =========================================================================
  // 1. Bind RaidenFuture
  // =========================================================================
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
      .def("is_ready", &tpu_raiden::RaidenFuture::IsReady)
      // Non-blocking success check; valid once is_ready() is true. True if the
      // transfer is still pending or completed successfully, False if a ready
      // event carries an error. Lets pollers distinguish success from failure
      // without calling the blocking Await() (which can deadlock the live
      // model).
      .def("ok",
           [](tpu_raiden::RaidenFuture& self) { return self.PollError().ok(); })
      // Non-blocking error message ("" when ok); pair with ok() for logging.
      .def("error_message", [](tpu_raiden::RaidenFuture& self) {
        absl::Status status = self.PollError();
        return status.ok() ? std::string() : std::string(status.message());
      });

  // =========================================================================
  // 2. Bind KVCacheManager
  // =========================================================================
  nb::class_<KVCacheManager>(m, "KVCacheManager")
      .def(nb::init<const std::vector<std::vector<at::Tensor>>&,
                    std::optional<int>, std::optional<int>, bool, int>(),
           nb::arg("device_tensors"), nb::arg("local_port") = nb::none(),
           nb::arg("host_blocks_to_allocate") = nb::none(),
           nb::arg("unsafe_skip_buffer_lock") = false,
           nb::arg("parallelism") = 1)
      .def(nb::init<const std::vector<at::Tensor>&, int64_t, int64_t, int64_t,
                    int64_t, double, bool, int, std::optional<int>>(),
           nb::arg("kv_caches"), nb::arg("node_id"),
           nb::arg("local_control_port"), nb::arg("max_blocks"),
           nb::arg("num_slots"), nb::arg("timeout_s") = 120.0,
           nb::arg("unsafe_skip_buffer_lock") = true,
           nb::arg("parallelism") = 4, nb::arg("listener_port") = nb::none())
      .def(
          "RegisterRecv",
          [](KVCacheManager& self, uint64_t uuid, const std::string& req_id,
             int expected_block_count) {
            absl::Status status =
                self.RegisterRecv(uuid, req_id, expected_block_count);
            if (!status.ok()) {
              throw std::runtime_error("KVCacheManager RegisterRecv failed: " +
                                       std::string(status.message()));
            }
          },
          nb::arg("uuid"), nb::arg("req_id"), nb::arg("expected_block_count"))
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
            return tpu_raiden::RaidenFuture{std::move(status_or.value())};
          },
          nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
          nb::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
          nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{},
          nb::call_guard<nb::gil_scoped_release>())
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
            return tpu_raiden::RaidenFuture{std::move(status_or.value())};
          },
          nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
          nb::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
          nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{},
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "D2hAutoAllocate",
          [](KVCacheManager& self,
             const std::vector<int64_t>& src_offsets_major_dim,
             const std::vector<int64_t>& copy_sizes_major_dim) {
            auto status_or = self.D2hAutoAllocate(src_offsets_major_dim,
                                                  copy_sizes_major_dim);
            if (!status_or.ok()) {
              throw std::runtime_error(
                  "KVCacheManager D2hAutoAllocate failed: " +
                  std::string(status_or.status().message()));
            }
            return std::make_pair(
                status_or.value().first,
                tpu_raiden::RaidenFuture{std::move(status_or.value().second)});
          },
          nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
          nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{})
      .def(
          "H2hWrite",
          [](KVCacheManager& self, std::string peer,
             const std::vector<int>& src_block_ids) {
            auto status_or = self.H2hWrite(std::move(peer), src_block_ids);
            if (!status_or.ok()) {
              throw std::runtime_error(
                  "KVCacheManager H2hWrite failed: " +
                  std::string(status_or.status().message()));
            }
            return std::make_pair(
                status_or.value().first,
                tpu_raiden::RaidenFuture{std::move(status_or.value().second)});
          },
          nb::arg("peer"), nb::arg("src_block_ids"),
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "H2hRead",
          [](KVCacheManager& self, std::string peer,
             const std::vector<int>& src_block_ids) {
            auto status_or = self.H2hRead(std::move(peer), src_block_ids);
            if (!status_or.ok()) {
              throw std::runtime_error(
                  "KVCacheManager H2hRead failed: " +
                  std::string(status_or.status().message()));
            }
            return std::make_pair(
                status_or.value().first,
                tpu_raiden::RaidenFuture{std::move(status_or.value().second)});
          },
          nb::arg("peer"), nb::arg("src_block_ids"),
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("local_port", &KVCacheManager::local_port)
      .def_prop_ro("num_layers", &KVCacheManager::num_layers)
      .def_prop_ro("num_shards", &KVCacheManager::num_shards)
      .def_prop_ro("slice_byte_size", &KVCacheManager::slice_byte_size)
      .def_prop_ro("local_control_port", &KVCacheManager::local_control_port)
      .def("get_local_endpoints",
           [](const KVCacheManager& self) {
             auto eps = self.get_local_endpoints();
             nb::list py_eps;
             for (const auto& ep : eps) {
               nb::dict d;
               d["endpoint"] = ep.endpoint;
               d["shards"] = ep.shards;
               py_eps.append(d);
             }
             return py_eps;
           })
      .def_prop_ro("listener_port", &KVCacheManager::listener_port)
      .def_prop_ro("is_listener_active", &KVCacheManager::is_listener_active)
      .def_prop_ro("transfer_address", &KVCacheManager::transfer_address)
      .def_prop_ro("listener_address", &KVCacheManager::listener_address)

      .def("notify_for_read", &KVCacheManager::NotifyForRead, nb::arg("req_id"),
           nb::arg("uuid"), nb::arg("block_ids"))
      .def(
          "start_read",
          [](KVCacheManager& self, const std::string& req_id, uint64_t uuid,
             nb::object remote_endpoint,
             const std::vector<int64_t>& remote_block_ids,
             const std::vector<int64_t>& local_block_ids, int parallelism,
             std::optional<std::vector<int64_t>> local_host_block_ids) {
            if (nb::isinstance<nb::str>(remote_endpoint)) {
              self.StartRead(req_id, uuid,
                             nb::cast<std::string>(remote_endpoint),
                             remote_block_ids, local_block_ids, parallelism,
                             local_host_block_ids);
            } else if (nb::isinstance<nb::list>(remote_endpoint)) {
              std::vector<tpu_raiden::EndpointDescriptor> descriptors;
              nb::list ep_list = nb::cast<nb::list>(remote_endpoint);
              for (size_t i = 0; i < ep_list.size(); ++i) {
                nb::dict d = nb::cast<nb::dict>(ep_list[i]);
                tpu_raiden::EndpointDescriptor desc;
                desc.endpoint = nb::cast<std::string>(d["endpoint"]);
                desc.shards = nb::cast<std::vector<int64_t>>(d["shards"]);
                descriptors.push_back(std::move(desc));
              }
              self.StartRead(req_id, uuid, descriptors, remote_block_ids,
                             local_block_ids, parallelism,
                             local_host_block_ids);
            } else {
              throw std::runtime_error(
                  "remote_endpoint must be str or list of dicts");
            }
          },
          nb::arg("req_id"), nb::arg("uuid"), nb::arg("remote_endpoint"),
          nb::arg("remote_block_ids"), nb::arg("local_block_ids"),
          nb::arg("parallelism") = 1,
          nb::arg("local_host_block_ids") = nb::none())
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
                    std::optional<int>, int, std::optional<int>,
                    std::optional<std::string>>(),
           nb::arg("device_tensors"), nb::arg("local_port") = nb::none(),
           nb::arg("parallelism") = 1, nb::arg("listener_port") = nb::none(),
           nb::arg("bind_ip") = nb::none())
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
          [](WeightSynchronizer& self, absl::string_view source) {
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
            absl::Status status = status_or_future.value().Await();
            if (!status.ok()) {
              throw std::runtime_error("WeightSynchronizer D2H copy failed: " +
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
                  "WeightSynchronizer H2D failed: " +
                  std::string(status_or_future.status().message()));
            }
            absl::Status status = status_or_future.value().Await();
            if (!status.ok()) {
              throw std::runtime_error("WeightSynchronizer H2D copy failed: " +
                                       std::string(status.message()));
            }
          },
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
      .def_prop_ro("listener_port", &WeightSynchronizer::listener_port)
      .def_prop_ro("is_listener_active",
                   &WeightSynchronizer::is_listener_active)
      .def_prop_ro("num_layers", &WeightSynchronizer::num_layers)
      .def_prop_ro("num_shards", &WeightSynchronizer::num_shards)
      .def_prop_ro("slice_byte_size", &WeightSynchronizer::slice_byte_size);

  // =========================================================================
  // 4. Bind KVCacheStore
  // =========================================================================
  nb::class_<tpu_raiden::kv_cache::RaidenId>(m, "RaidenId")
      .def(nb::init<std::string, std::string, std::string, int>(),
           nb::arg("job_name") = "", nb::arg("job_replica_id") = "",
           nb::arg("data_name") = "", nb::arg("data_replica_idx") = 0)
      .def_rw("job_name", &tpu_raiden::kv_cache::RaidenId::job_name)
      .def_rw("job_replica_id", &tpu_raiden::kv_cache::RaidenId::job_replica_id)
      .def_rw("data_name", &tpu_raiden::kv_cache::RaidenId::data_name)
      .def_rw("data_replica_idx",
              &tpu_raiden::kv_cache::RaidenId::data_replica_idx);

  nb::enum_<tpu_raiden::kv_cache::BlockStatus>(m, "BlockStatus")
      .value("INIT", tpu_raiden::kv_cache::BlockStatus::INIT)
      .value("REMOTE", tpu_raiden::kv_cache::BlockStatus::REMOTE)
      .value("HOST", tpu_raiden::kv_cache::BlockStatus::HOST)
      .value("HBM", tpu_raiden::kv_cache::BlockStatus::HBM);

  nb::class_<tpu_raiden::kv_cache::RaidenBlockID>(m, "RaidenBlockID")
      .def(nb::init<tpu_raiden::kv_cache::RaidenId, int,
                    tpu_raiden::kv_cache::BlockStatus>(),
           nb::arg("raiden_id") = tpu_raiden::kv_cache::RaidenId(),
           nb::arg("host_block_id") = -1,
           nb::arg("status") = tpu_raiden::kv_cache::BlockStatus::INIT)
      .def_rw("raiden_id", &tpu_raiden::kv_cache::RaidenBlockID::raiden_id)
      .def_rw("host_block_id",
              &tpu_raiden::kv_cache::RaidenBlockID::host_block_id)
      .def_rw("status", &tpu_raiden::kv_cache::RaidenBlockID::status);

  nb::class_<tpu_raiden::kv_cache::KVCacheStoreWrapper>(m, "KVCacheStore")
      .def(nb::init<size_t, std::string, tpu_raiden::kv_cache::RaidenId>(),
           nb::arg("capacity"), nb::arg("global_registry_address") = "",
           nb::arg("raiden_id") = tpu_raiden::kv_cache::RaidenId())
      .def_prop_ro(
          "raiden_id",
          [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self) {
            return (*self).raiden_id();
          },
          "Returns the RaidenId associated with this store.")
      .def(
          "lookup",
          [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self,
             const std::vector<nb::bytes>& block_hashes, bool enable_global) {
            auto hashes = ToStdStringVector(block_hashes);
            auto res = self->Lookup(hashes, enable_global);
            if (!res.ok()) {
              throw std::runtime_error("KVCacheStore lookup failed: " +
                                       std::string(res.status().message()));
            }
            std::vector<std::pair<
                nb::bytes, std::vector<tpu_raiden::kv_cache::RaidenBlockID>>>
                py_res;
            py_res.reserve(res.value().size());
            for (const auto& pair : res.value()) {
              py_res.push_back(std::make_pair(
                  nb::bytes(pair.first.data(), pair.first.size()),
                  pair.second));
            }
            return py_res;
          },
          nb::arg("block_hashes"), nb::arg("enable_global") = false,
          "Checks the LRU directory for cached block hashes. Returns a list of "
          "all matched replica pairs prior to the first miss.")
      .def(
          "insert",
          [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self,
             const std::vector<nb::bytes>& block_hashes,
             const std::vector<
                 std::vector<tpu_raiden::kv_cache::RaidenBlockID>>& slices,
             bool on_host) {
            auto hashes = ToStdStringVector(block_hashes);
            auto res = self->Insert(hashes, slices, on_host);
            std::vector<std::pair<
                nb::bytes, std::vector<tpu_raiden::kv_cache::RaidenBlockID>>>
                py_evicted;
            py_evicted.reserve(res.second.size());
            for (const auto& pair : res.second) {
              py_evicted.push_back(std::make_pair(
                  nb::bytes(pair.first.data(), pair.first.size()),
                  pair.second));
            }
            return std::make_pair(res.first, py_evicted);
          },
          nb::arg("block_hashes"), nb::arg("slices"), nb::arg("on_host"))
      .def(
          "insert_and_pin",
          [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self,
             const std::vector<nb::bytes>& block_hashes,
             const std::vector<
                 std::vector<tpu_raiden::kv_cache::RaidenBlockID>>& slices,
             bool on_host) {
            auto hashes = ToStdStringVector(block_hashes);
            auto res = self->InsertAndPin(hashes, slices, on_host);
            std::vector<std::pair<
                nb::bytes, std::vector<tpu_raiden::kv_cache::RaidenBlockID>>>
                py_evicted;
            py_evicted.reserve(res.second.size());
            for (const auto& pair : res.second) {
              py_evicted.push_back(std::make_pair(
                  nb::bytes(pair.first.data(), pair.first.size()),
                  pair.second));
            }
            return std::make_pair(res.first, py_evicted);
          },
          nb::arg("block_hashes"), nb::arg("slices"), nb::arg("on_host"))
      .def(
          "release_and_delete",
          [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self,
             const std::vector<nb::bytes>& block_hashes,
             const std::vector<std::pair<
                 nb::bytes, std::vector<tpu_raiden::kv_cache::RaidenBlockID>>>&
                 pending_evict_entries) {
            auto hashes = ToStdStringVector(block_hashes);
            std::vector<std::pair<
                std::string, std::vector<tpu_raiden::kv_cache::RaidenBlockID>>>
                evicted;
            evicted.reserve(pending_evict_entries.size());
            for (const auto& pair : pending_evict_entries) {
              evicted.push_back(std::make_pair(
                  std::string(pair.first.c_str(), pair.first.size()),
                  pair.second));
            }
            auto res = self->ReleaseAndDelete(hashes, evicted);
            std::vector<std::pair<
                nb::bytes, std::vector<tpu_raiden::kv_cache::RaidenBlockID>>>
                py_rem_evicted;
            py_rem_evicted.reserve(res.second.size());
            for (const auto& pair : res.second) {
              py_rem_evicted.push_back(std::make_pair(
                  nb::bytes(pair.first.data(), pair.first.size()),
                  pair.second));
            }
            return std::make_pair(res.first, py_rem_evicted);
          },
          nb::arg("block_hashes"),
          nb::arg("pending_evict_entries") = std::vector<std::pair<
              nb::bytes, std::vector<tpu_raiden::kv_cache::RaidenBlockID>>>())
      .def(
          "delete",
          [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self,
             const std::vector<nb::bytes>& block_hashes,
             const std::vector<
                 std::vector<tpu_raiden::kv_cache::RaidenBlockID>>& slices) {
            auto hashes = ToStdStringVector(block_hashes);
            self->Delete(hashes, slices);
          },
          nb::arg("block_hashes"), nb::arg("slices"))
      .def("capacity",
           [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self) {
             return self->capacity();
           })
      .def(
          "pin",
          [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self,
             const std::vector<nb::bytes>& block_hashes) {
            auto hashes = ToStdStringVector(block_hashes);
            return self->Pin(hashes);
          },
          nb::arg("block_hashes"))
      .def(
          "release",
          [](tpu_raiden::kv_cache::KVCacheStoreWrapper& self,
             const std::vector<nb::bytes>& block_hashes) {
            auto hashes = ToStdStringVector(block_hashes);
            self->Release(hashes);
          },
          nb::arg("block_hashes"));
}
