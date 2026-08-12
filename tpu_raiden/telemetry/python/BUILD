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

load("@nanobind_bazel//:build_defs.bzl", "nanobind_extension")
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "telemetry_binding",
    srcs = ["telemetry_binding.cc"],
    hdrs = ["telemetry_binding.h"],
    copts = [
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
    ],
    deps = [
        "//tpu_raiden/telemetry:metrics_3p_prometheus_exporter",
        "//tpu_raiden/telemetry:metrics_api",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@nanobind",
    ],
)

nanobind_extension(
    name = "_telemetry_binding_test_ext",
    testonly = True,
    srcs = ["telemetry_binding_test_ext.cc"],
    features = [
        "-use_header_modules",
    ],
    deps = [
        ":telemetry_binding",
    ],
)

py_test(
    name = "telemetry_binding_test",
    srcs = ["telemetry_binding_test.py"],
    deps = [
        ":_telemetry_binding_test_ext",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)
