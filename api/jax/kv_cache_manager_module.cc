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

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "api/jax/nb_status.h"
#include "api/jax/kv_cache_manager.h"
#include "raiden_lib/raw_transfer/raw_transfer_core.h"

namespace nb = nanobind;

NB_MODULE(_kv_cache_manager, m) {
  // Bind JAX PjRtCopyFuture dynamically E2E!
  nb::class_<raiden::PjRtCopyFuture>(m, "PjRtCopyFuture")
      .def("Await", [](raiden::PjRtCopyFuture& self) { self.Await(); })
      .def("IsReady", &raiden::PjRtCopyFuture::IsReady);

  nb::class_<tpu_raiden::kv_cache::jax::KVCacheManager>(m, "KVCacheManager")
      .def(nb::init<nb::list, int, std::optional<int>, std::optional<int>,
                    std::optional<std::vector<uintptr_t>>, bool, int>(),
           nb::arg("device_arrays"), nb::arg("block_size") = 1,
           nb::arg("local_port") = nb::none(),
           nb::arg("host_blocks_to_allocate") = nb::none(),
           nb::arg("external_host_ptrs") = nb::none(),
           nb::arg("unsafe_skip_buffer_lock") = false,
           nb::arg("parallelism") = 1)
      .def("h2d", &tpu_raiden::kv_cache::KVCacheManagerBase::H2d,
           nb::arg("src_offsets_major_dim") = nb::list(),
           nb::arg("dst_offsets_major_dim") = nb::list(),
           nb::arg("copy_sizes_major_dim") = nb::list())
      .def("d2h", &tpu_raiden::kv_cache::KVCacheManagerBase::D2h,
           nb::arg("src_offsets_major_dim") = nb::list(),
           nb::arg("dst_offsets_major_dim") = nb::list(),
           nb::arg("copy_sizes_major_dim") = nb::list())
      .def("d2h_auto_allocate",
           &tpu_raiden::kv_cache::KVCacheManagerBase::D2hAutoAllocate,
           nb::arg("src_offsets_major_dim") = nb::list(),
           nb::arg("copy_sizes_major_dim") = nb::list(),
           nb::arg("entity_id") = 0)
      .def("h2h_write", &tpu_raiden::kv_cache::KVCacheManagerBase::H2hWrite,
           nb::arg("peer"), nb::arg("src_block_ids"), nb::arg("entity_id") = 0)
      .def("h2h_read", &tpu_raiden::kv_cache::KVCacheManagerBase::H2hRead,
           nb::arg("peer"), nb::arg("src_block_ids"), nb::arg("entity_id") = 0)
      .def("local_port", &tpu_raiden::kv_cache::KVCacheManagerBase::local_port)
      .def("get_host_pointer",
           &tpu_raiden::kv_cache::KVCacheManagerBase::GetHostPointer,
           nb::arg("layer_idx"), nb::arg("shard_idx"))
      .def_prop_ro("num_layers",
                   &tpu_raiden::kv_cache::KVCacheManagerBase::num_layers)
      .def_prop_ro("num_shards",
                   &tpu_raiden::kv_cache::KVCacheManagerBase::num_shards)
      .def_prop_ro("slice_byte_size",
                   &tpu_raiden::kv_cache::KVCacheManagerBase::slice_byte_size);
}
