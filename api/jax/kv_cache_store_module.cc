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

#include <optional>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include "xla/pjrt/status_casters.h"
#include "api/jax/kv_cache_manager.h"
#include "kv_cache/kv_cache_store.h"

namespace nb = nanobind;

NB_MODULE(kv_cache_store, m) {
  nb::module_::import_(
      "google3.third_party.tpu_raiden.raiden_lib.raw_transfer.jax.raw_"
      "transfer");
  nb::class_<tpu_raiden::kv_cache::KVCacheStore>(m, "KVCacheStore")
      .def(nb::init<int, int>(), nb::arg("block_size"), nb::arg("capacity"))
      .def(
          "lookup_and_fetch",
          [](tpu_raiden::kv_cache::KVCacheStore& self,
             const std::vector<uint64_t>& block_hashes, nb::list device_arrays,
             const std::vector<int>& dst_offsets,
             const std::vector<int>& copy_sizes) {
            tpu_raiden::kv_cache::jax::KVCacheManager jax_manager(
                device_arrays, self.block_size(), std::nullopt, std::nullopt);
            return xla::ValueOrThrow(self.LookupAndFetch(
                block_hashes, jax_manager, dst_offsets, copy_sizes));
          },
          nb::arg("block_hashes"), nb::arg("device_arrays"),
          nb::arg("dst_offsets_major_dim"), nb::arg("copy_sizes_major_dim"))
      .def(
          "insert",
          [](tpu_raiden::kv_cache::KVCacheStore& self,
             const std::vector<uint64_t>& block_hashes, nb::list device_arrays,
             const std::vector<int>& src_offsets,
             const std::vector<int>& copy_sizes) {
            tpu_raiden::kv_cache::jax::KVCacheManager jax_manager(
                device_arrays, self.block_size(), std::nullopt, std::nullopt);
            xla::ThrowIfError(self.Insert(block_hashes, jax_manager,
                                          src_offsets, copy_sizes));
          },
          nb::arg("block_hashes"), nb::arg("device_arrays"),
          nb::arg("src_offsets_major_dim"), nb::arg("copy_sizes_major_dim"));
}
