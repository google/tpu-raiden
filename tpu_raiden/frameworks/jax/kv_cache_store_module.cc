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

/* Copyright 2026 The TPU Raiden Authors. All Rights Reserved.
==============================================================================*/

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "xla/pjrt/status_casters.h"
#include "tpu_raiden/kv_cache/kv_cache_store.h"

namespace nb = nanobind;

namespace tpu_raiden {
namespace kv_cache {
namespace jax {
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
}  // namespace jax
}  // namespace kv_cache
}  // namespace tpu_raiden

NB_MODULE(kv_cache_store, m) {
  nb::class_<tpu_raiden::kv_cache::RaidenId>(m, "RaidenId")
      .def(nb::init<std::string, std::string, std::string, int>(),
           nb::arg("job_name"), nb::arg("job_replica_id") = "",
           nb::arg("data_name"), nb::arg("data_replica_idx") = 0)
      .def_rw("job_name", &tpu_raiden::kv_cache::RaidenId::job_name)
      .def_rw("job_replica_id", &tpu_raiden::kv_cache::RaidenId::job_replica_id)
      .def_rw("data_name", &tpu_raiden::kv_cache::RaidenId::data_name)
      .def_rw("data_replica_idx",
              &tpu_raiden::kv_cache::RaidenId::data_replica_idx);

  nb::class_<tpu_raiden::kv_cache::jax::KVCacheStoreWrapper>(m, "KVCacheStore")
      .def(nb::init<size_t>(), nb::arg("capacity"))
      .def(
          "lookup",
          [](tpu_raiden::kv_cache::jax::KVCacheStoreWrapper& self,
             const std::vector<uint64_t>& block_hashes) {
            return xla::ValueOrThrow(self->Lookup(block_hashes));
          },
          nb::arg("block_hashes"),
          "Checks the LRU directory for cached block hashes. Returns a list of "
          "all matched replica pairs prior to the first miss.")
      .def(
          "insert",
          [](tpu_raiden::kv_cache::jax::KVCacheStoreWrapper& self,
             const std::vector<uint64_t>& block_hashes,
             const std::vector<std::vector<tpu_raiden::kv_cache::RaidenId>>&
                 slices,
             bool on_host) {
            return self->Insert(block_hashes, slices, on_host);
          },
          nb::arg("block_hashes"), nb::arg("slices"), nb::arg("on_host"))
      .def(
          "delete",
          [](tpu_raiden::kv_cache::jax::KVCacheStoreWrapper& self,
             const std::vector<uint64_t>& block_hashes,
             const std::vector<std::vector<tpu_raiden::kv_cache::RaidenId>>&
                 slices) { self->Delete(block_hashes, slices); },
          nb::arg("block_hashes"), nb::arg("slices"))
      .def("capacity",
           [](tpu_raiden::kv_cache::jax::KVCacheStoreWrapper& self) {
             return self->capacity();
           })
      .def(
          "pin",
          [](tpu_raiden::kv_cache::jax::KVCacheStoreWrapper& self,
             const std::vector<uint64_t>& block_hashes) {
            return self->Pin(block_hashes);
          },
          nb::arg("block_hashes"))
      .def(
          "release",
          [](tpu_raiden::kv_cache::jax::KVCacheStoreWrapper& self,
             const std::vector<uint64_t>& block_hashes) {
            self->Release(block_hashes);
          },
          nb::arg("block_hashes"));
}
