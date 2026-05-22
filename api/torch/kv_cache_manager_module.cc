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
#include <vector>

#include "pybind11/gil.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "absl/status/statusor.h"
#include "api/torch/kv_cache_manager.h"
#include "raiden_lib/raw_transfer/raw_transfer_core.h"

namespace py = pybind11;

using ::tpu_raiden::torch::KVCacheManager;

// Pybind11 FFI bindings module definition for PyTorch E2E
PYBIND11_MODULE(_kv_cache_manager, m) {
  py::class_<KVCacheManager>(m, "KVCacheManager")
      .def(py::init<const std::vector<std::vector<at::Tensor>>&, int,
                    std::optional<int>, std::optional<int>, int>(),
           py::arg("device_tensors"), py::arg("block_size") = 1,
           py::arg("local_port") = py::none(),
           py::arg("host_blocks_to_allocate") = py::none(),
           py::arg("parallelism") = 1)
      .def(
          "H2d",
          [](KVCacheManager& self, int stream_idx, const std::string& peer,
             const std::vector<int>& src_block_ids,
             const std::vector<int>& dst_block_ids, int64_t entity_id) {
            absl::Status s = self.H2d(stream_idx, peer, src_block_ids,
                                      dst_block_ids, entity_id);
            if (!s.ok()) {
              throw std::runtime_error("KVCacheManager H2d failed: " +
                                       std::string(s.message()));
            }
          },
          py::arg("stream_idx"), py::arg("peer"), py::arg("src_block_ids"),
          py::arg("dst_block_ids"), py::arg("entity_id") = 0,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "D2h",
          [](KVCacheManager& self, int stream_idx, const std::string& peer,
             const std::vector<int>& src_block_ids,
             const std::vector<int>& dst_block_ids, int64_t entity_id) {
            absl::Status s = self.D2h(stream_idx, peer, src_block_ids,
                                      dst_block_ids, entity_id);
            if (!s.ok()) {
              throw std::runtime_error("KVCacheManager D2h failed: " +
                                       std::string(s.message()));
            }
          },
          py::arg("stream_idx"), py::arg("peer"), py::arg("src_block_ids"),
          py::arg("dst_block_ids"), py::arg("entity_id") = 0,
          py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("local_port", &KVCacheManager::local_port)
      .def_property_readonly("num_layers", &KVCacheManager::num_layers)
      .def_property_readonly("num_shards", &KVCacheManager::num_shards)
      .def_property_readonly("block_size", &KVCacheManager::block_size)
      .def_property_readonly("slice_byte_size",
                             &KVCacheManager::slice_byte_size);
}
