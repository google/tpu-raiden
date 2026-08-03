# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "common",
    srcs = ["common.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "telemetry",
    srcs = ["telemetry.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/absl/logging",
        "//third_party/py/prometheus_client",
        "//tpu_raiden/telemetry:_raiden_telemetry",
    ],
)

py_test(
    name = "telemetry_test",
    srcs = ["telemetry_test.py"],
    deps = [
        ":telemetry",
        "//third_party/py/prometheus_client",
        "//tpu_raiden/telemetry:_raiden_telemetry",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)
