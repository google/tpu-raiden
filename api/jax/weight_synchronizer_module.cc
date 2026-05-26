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
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "api/jax/weight_synchronizer.h"
#include "raiden_lib/raw_transfer/raw_transfer_core.h"

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
      .def_prop_ro("local_port", &WeightSynchronizer::local_port)
      .def_prop_ro("num_layers", &WeightSynchronizer::num_layers)
      .def_prop_ro("num_shards", &WeightSynchronizer::num_shards)
      .def_prop_ro("slice_byte_size", &WeightSynchronizer::slice_byte_size);
}
