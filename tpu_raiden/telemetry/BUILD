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

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "metrics_api",
    srcs = ["metrics_api.cc"],
    hdrs = ["metrics_api.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "metrics_api_test",
    srcs = ["metrics_api_test.cc"],
    deps = [
        ":metrics_api",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "metrics_3p_prometheus_exporter",
    srcs = ["prometheus_exporter.cc"],
    hdrs = ["prometheus_exporter.h"],
    deps = [
        ":metrics_api",
        "//third_party/prometheus_cpp_client:prometheus_client_core",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "prometheus_exporter_test",
    srcs = ["prometheus_exporter_test.cc"],
    deps = [
        ":metrics_3p_prometheus_exporter",
        ":metrics_api",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
