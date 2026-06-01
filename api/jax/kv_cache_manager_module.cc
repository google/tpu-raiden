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
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "api/jax/disagg_kv_cache_manager.h"
#include "api/jax/kv_cache_manager.h"
#include "api/jax/nb_statusor.h"
#include "core/raw_transfer_core.h"

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
                   &tpu_raiden::kv_cache::KVCacheManagerBase::slice_byte_size);

  // Bind DisaggTransferRequest
  nb::enum_<tpu_raiden::kv_cache::DisaggTransferRequest::Type>(
      m, "DisaggTransferRequestType")
      .value("PREFILL_D2H",
             tpu_raiden::kv_cache::DisaggTransferRequest::Type::kPrefillD2H)
      .value("DECODE_H2D",
             tpu_raiden::kv_cache::DisaggTransferRequest::Type::kDecodeH2D)
      .value("H2H_WRITE",
             tpu_raiden::kv_cache::DisaggTransferRequest::Type::kH2HWrite)
      .value("H2H_READ",
             tpu_raiden::kv_cache::DisaggTransferRequest::Type::kH2HRead);

  nb::class_<tpu_raiden::kv_cache::DisaggTransferRequest>(
      m, "DisaggTransferRequest")
      .def(nb::init<>())
      .def_rw("uuid", &tpu_raiden::kv_cache::DisaggTransferRequest::uuid)
      .def_rw("req_id", &tpu_raiden::kv_cache::DisaggTransferRequest::req_id)
      .def_rw("type", &tpu_raiden::kv_cache::DisaggTransferRequest::type)
      .def_rw("pull_mode",
              &tpu_raiden::kv_cache::DisaggTransferRequest::pull_mode)
      .def_rw("src_offsets",
              &tpu_raiden::kv_cache::DisaggTransferRequest::src_offsets)
      .def_rw("dst_offsets",
              &tpu_raiden::kv_cache::DisaggTransferRequest::dst_offsets)
      .def_rw("sizes", &tpu_raiden::kv_cache::DisaggTransferRequest::sizes)
      .def_rw("peer", &tpu_raiden::kv_cache::DisaggTransferRequest::peer)
      .def_rw("block_ids",
              &tpu_raiden::kv_cache::DisaggTransferRequest::block_ids)
      .def_rw("entity_id",
              &tpu_raiden::kv_cache::DisaggTransferRequest::entity_id)
      .def_rw("callback",
              &tpu_raiden::kv_cache::DisaggTransferRequest::callback);

  // Bind JaxDisaggKVCacheManager
  nb::class_<tpu_raiden::kv_cache::jax::DisaggKVCacheManager>(
      m, "DisaggKVCacheManager")
      .def(nb::init<nb::list, int, std::optional<int>, std::optional<int>,
                    std::optional<std::vector<uintptr_t>>, bool, int, int>(),
           nb::arg("device_arrays"), nb::arg("block_size") = 1,
           nb::arg("local_port") = nb::none(),
           nb::arg("host_blocks_to_allocate") = nb::none(),
           nb::arg("external_host_ptrs") = nb::none(),
           nb::arg("unsafe_skip_buffer_lock") = false,
           nb::arg("transport_parallelism") = 1,
           nb::arg("worker_parallelism") = 1)
      // Release the GIL across these blocking C++ calls: background threads
      // (orchestrator / H2H workers) acquire the GIL to invoke Python
      // completion callbacks, so holding the GIL here while Stop() joins them
      // -- or while submit contends on running_mutex_ -- can deadlock.
      .def("start", &tpu_raiden::kv_cache::DisaggKVCacheManagerBase::Start,
           nb::call_guard<nb::gil_scoped_release>())
      .def("stop", &tpu_raiden::kv_cache::DisaggKVCacheManagerBase::Stop,
           nb::call_guard<nb::gil_scoped_release>())
      .def("submit_request",
           &tpu_raiden::kv_cache::DisaggKVCacheManagerBase::SubmitRequest,
           nb::arg("request"), nb::call_guard<nb::gil_scoped_release>())
      .def("register_peer",
           &tpu_raiden::kv_cache::DisaggKVCacheManagerBase::RegisterPeer,
           nb::arg("name"), nb::arg("ip"), nb::arg("zmq_port"),
           nb::arg("transport_port"))
      .def("zmq_control_port",
           &tpu_raiden::kv_cache::DisaggKVCacheManagerBase::zmq_control_port)
      .def("local_port", &tpu_raiden::kv_cache::KVCacheManagerBase::local_port)
      .def_prop_ro("num_layers",
                   &tpu_raiden::kv_cache::KVCacheManagerBase::num_layers)
      .def_prop_ro("num_shards",
                   &tpu_raiden::kv_cache::KVCacheManagerBase::num_shards)
      .def_prop_ro("slice_byte_size",
                   &tpu_raiden::kv_cache::KVCacheManagerBase::slice_byte_size);
}
