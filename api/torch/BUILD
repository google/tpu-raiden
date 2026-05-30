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

load("@nanobind_abseil/google:nanobind.bzl", "nanobind_extension")
load("@nanobind_bazel//:build_defs.bzl", "nanobind_extension")
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "torch_tpu_stablehlo_headers",
    deps = [
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:chlo_builder",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:func_builder",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "kv_cache_manager_torch",
    srcs = ["kv_cache_manager.cc"],
    hdrs = ["kv_cache_manager.h"],
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
        ":torch_tpu_stablehlo_headers",
        "//core:xla_raw_transfer_headers",
        "//frameworks/torch:torch_utils",
        "//kv_cache:kv_cache_manager_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@torch_tpu//shims/torch:aten_headers",
        "@torch_tpu//shims/torch:torch_headers",
        "@torch_tpu//shims/torch/c10/core:base_headers",
        "@torch_tpu//shims/torch/torch/headeronly:torch_headeronly",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

nanobind_extension(
    name = "_kv_cache_manager",
    srcs = ["kv_cache_manager_module.cc"],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_manager_torch",
        "//core:raw_transfer_core",
        "//frameworks/torch:torch_nanobind_utils",
        "//kv_cache:kv_cache_manager_base",
        "@com_google_absl//absl/status:statusor",
        "@torch_tpu//shims/torch:torch_headers",
    ],
)

py_library(
    name = "kv_cache_manager_torch_py",
    srcs = ["kv_cache_manager.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_kv_cache_manager",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
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
        ":weight_synchronizer_torch_py",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
    ],
)

py_library(
    name = "weight_synchronizer_torch_py",
    srcs = ["weight_synchronizer.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//frameworks/torch:_weight_synchronizer",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
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
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
    ],
)

nanobind_extension(
    name = "_raiden_transfer_engine",
    srcs = ["raiden_transfer_engine.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_manager_torch",
        "//core:raw_transfer_core",
        "//frameworks/torch:torch_nanobind_utils",
        "//kv_cache:kv_cache_manager_base",
        "//transport:socket_transport",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@nanobind",
        "@torch_tpu//shims/torch:aten_headers",
        "@torch_tpu//shims/torch:torch_headers",
    ],
)

py_library(
    name = "raiden_transfer_engine",
    srcs = ["raiden_transfer_engine.py"],
    data = [
        "raiden_transfer_engine.pyi",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":_raiden_transfer_engine",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
        "@torch_tpu//torch_tpu:_loader",
    ],
)

py_test(
    name = "raiden_transfer_engine_test",
    srcs = ["raiden_transfer_engine_test.py"],
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = [
        ":raiden_transfer_engine",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
    ],
)
