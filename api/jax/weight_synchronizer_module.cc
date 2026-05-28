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

#include "absl/status/statusor.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "api/jax/weight_synchronizer.h"
#include "core/raw_transfer_core.h"

namespace nb = nanobind;

using ::tpu_raiden::jax::WeightSynchronizer;

// Exported Nanobind python module definition E2E!
NB_MODULE(_weight_synchronizer, m) {
  nb::class_<WeightSynchronizer>(m, "WeightSynchronizer")
      .def(nb::init<const nb::list&, std::optional<int>, int, bool>(),
           nb::arg("jax_arrays"), nb::arg("local_port") = nb::none(),
           nb::arg("parallelism") = 1,
           nb::arg("unsafe_skip_buffer_lock") = false)
      .def(
          "PushWeights",
          [](WeightSynchronizer& self, const std::vector<std::string>& peers) {
            absl::Status s = self.PushWeights(peers);
            if (!s.ok()) {
              throw std::runtime_error("Weight sync PushWeights failed: " +
                                       std::string(s.message()));
            }
          },
          nb::arg("peers"), nb::call_guard<nb::gil_scoped_release>())
      .def(
          "PullWeights",
          [](WeightSynchronizer& self, const std::string& source) {
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
            status_or_future.value().Await();
          },
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "H2dChunk",
          [](WeightSynchronizer& self, size_t shard_idx, size_t host_offset_bytes,
             size_t device_offset_bytes, size_t size_bytes) {
            auto status_or_future = self.H2dChunk(
                shard_idx, host_offset_bytes, device_offset_bytes, size_bytes
            );
            if (!status_or_future.ok()) {
              throw std::runtime_error(
                  "Weight sync H2DChunk failed: " +
                  std::string(status_or_future.status().message())
              );
            }
            status_or_future.value().Await();
          },
          nb::arg("shard_idx"), nb::arg("host_offset_bytes"),
          nb::arg("device_offset_bytes"), nb::arg("size_bytes"),
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "PullWeightsChunk",
          [](WeightSynchronizer& self, const std::string& source,
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
      .def_prop_ro("num_layers", &WeightSynchronizer::num_layers)
      .def_prop_ro("num_shards", &WeightSynchronizer::num_shards)
      .def_prop_ro("slice_byte_size", &WeightSynchronizer::slice_byte_size);
}
