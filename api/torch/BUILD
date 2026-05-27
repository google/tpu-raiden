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
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "kv_cache_manager_torch",
    srcs = [
        "kv_cache_manager.cc",
    ],
    hdrs = [
        "kv_cache_manager.h",
    ],
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//kv_cache:kv_cache_manager_base",
        "@torch_tpu//shims/torch:aten_headers",
        "@torch_tpu//shims/torch:torch_headers",
        "@torch_tpu//shims/torch/c10/core:base_headers",
        "@torch_tpu//shims/torch/torch/headeronly:torch_headeronly",
        "@torch_tpu_headers//:headers",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:chlo_builder",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:func_builder",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
        "@stablehlo//:stablehlo_ops",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

nanobind_extension(
    name = "_kv_cache_manager",
    srcs = [
        "kv_cache_manager_module.cc",
    ],
    copts = [
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_manager_torch",
        "//kv_cache:kv_cache_manager_base",
        "//raiden_lib/raw_transfer:raw_transfer_impl",
        "@pybind11//:pybind11",
        "@torch_tpu//shims/torch:torch_headers",
        "@com_google_absl//absl/status:statusor",
    ],
)

py_library(
    name = "kv_cache_manager_torch_py",
    srcs = ["kv_cache_manager.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_kv_cache_manager",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu:torch_tpu",
    ],
)

py_test(
    name = "kv_cache_manager_test",
    srcs = ["kv_cache_manager_test.py"],
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = [
        ":kv_cache_manager_torch_py",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu:torch_tpu",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@numpy//numpy",
    ],
)

cc_library(
    name = "weight_synchronizer_torch",
    srcs = [
        "weight_synchronizer.cc",
    ],
    hdrs = [
        "weight_synchronizer.h",
    ],
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@torch_tpu//shims/torch:aten_headers",
        "@torch_tpu//shims/torch:torch_headers",
        "@torch_tpu//shims/torch/c10/core:base_headers",
        "@torch_tpu//shims/torch/torch/headeronly:torch_headeronly",
        "@torch_tpu_headers//:headers",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:chlo_builder",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:func_builder",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
        "@stablehlo//:stablehlo_ops",
        "//weight_sync:weight_synchronizer_base",
        "@com_google_absl//absl/status",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

nanobind_extension(
    name = "_weight_synchronizer",
    srcs = [
        "weight_synchronizer_module.cc",
    ],
    copts = [
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":weight_synchronizer_torch",
        "//raiden_lib/raw_transfer:raw_transfer_impl",
        "@pybind11//:pybind11",
        "@torch_tpu//shims/torch:torch_headers",
        "//weight_sync:weight_synchronizer_base",
        "@com_google_absl//absl/status",
    ],
)

py_library(
    name = "weight_synchronizer_torch_py",
    srcs = ["weight_synchronizer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_weight_synchronizer",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu:torch_tpu",
    ],
)

py_test(
    name = "weight_synchronizer_test",
    srcs = ["weight_synchronizer_test.py"],
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = [
        ":weight_synchronizer_torch_py",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu:torch_tpu",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@numpy//numpy",
    ],
)
