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
#include <utility>
#include <vector>

#include "pybind11/gil.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "absl/status/status.h"
#include "api/torch/weight_synchronizer.h"
#include "core/raw_transfer_core.h"
#include "torch/extension.h"

namespace py = pybind11;

using ::tpu_raiden::torch::WeightSynchronizer;

// Pybind11 FFI bindings module definition for PyTorch E2E
PYBIND11_MODULE(_weight_synchronizer, m) {
  py::class_<WeightSynchronizer>(m, "WeightSynchronizer")
      .def(py::init<const std::vector<std::vector<at::Tensor>>&,
                    std::optional<int>, int>(),
           py::arg("device_tensors"), py::arg("local_port") = py::none(),
           py::arg("parallelism") = 1)
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
          py::arg("peers"), py::call_guard<py::gil_scoped_release>())
      .def(
          "PullWeights",
          [](WeightSynchronizer& self, const std::string& source) {
            absl::Status s = self.PullWeights(source);
            if (!s.ok()) {
              throw std::runtime_error(
                  "WeightSynchronizer PullWeights failed: " +
                  std::string(s.message()));
            }
          },
          py::arg("source"), py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("local_port", &WeightSynchronizer::local_port)
      .def_property_readonly("num_layers", &WeightSynchronizer::num_layers)
      .def_property_readonly("num_shards", &WeightSynchronizer::num_shards)
      .def_property_readonly("slice_byte_size",
                             &WeightSynchronizer::slice_byte_size);
}
