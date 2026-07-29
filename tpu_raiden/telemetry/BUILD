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

# Copyright 2026 Google LLC
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

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("//third_party/pybind11/google3_utils:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# --- CL 1: Base C++ Telemetry API ---

cc_library(
    name = "metrics_api",
    srcs = ["metrics_api.cc"],
    hdrs = ["metrics_api.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "metrics_api_test",
    srcs = ["metrics_api_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":metrics_api",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

# --- CL 2: 3P Prometheus Exporter & Pybind Snapshot Exporter ---

cc_library(
    name = "metrics_3p_prometheus_exporter",
    srcs = ["prometheus_exporter.cc"],
    hdrs = ["prometheus_exporter.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":metrics_api",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "prometheus_exporter_test",
    srcs = ["prometheus_exporter_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":metrics_3p_prometheus_exporter",
        ":metrics_api",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

pybind_extension(
    name = "telemetry_pybind",
    srcs = ["telemetry_pybind.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":metrics_3p_prometheus_exporter",
        ":metrics_api",
    ],
)

# --- CL 4: 1P Google Streamz Exporter & Bazel select() Build Isolation ---

cc_library(
    name = "metrics_1p_exporter",
    srcs = ["streamz_exporter.cc"],
    hdrs = ["streamz_exporter.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":metrics_api",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "streamz_exporter_test",
    srcs = ["streamz_exporter_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":metrics_1p_exporter",
        ":metrics_api",
        "@com_google_googletest//:gtest_main",
    ],
)

config_setting(
    name = "enable_1p_telemetry_config",
    define_values = {"enable_1p_telemetry": "true"},
)

config_setting(
    name = "enable_1p_telemetry_setting",
    values = {"define": "enable_1p_telemetry=true"},
)

cc_library(
    name = "telemetry_backend_selector",
    deps = select({
        ":enable_1p_telemetry_config": [":metrics_1p_exporter"],
        ":enable_1p_telemetry_setting": [":metrics_1p_exporter"],
        "//conditions:default": [":metrics_3p_prometheus_exporter"],
    }),
)

alias(
    name = "default_telemetry_exporter",
    actual = ":telemetry_backend_selector",
)
