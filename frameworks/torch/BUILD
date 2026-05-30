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

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "weight_synchronizer_torch",
    srcs = ["weight_synchronizer.cc"],
    hdrs = ["weight_synchronizer.h"],
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
        "//frameworks/torch:torch_utils",
        "//weight_sync:weight_synchronizer_base",
        "@com_google_absl//absl/status",
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:chlo_builder",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:func_builder",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
        "@stablehlo//:stablehlo_ops",
        "@torch_tpu//shims/torch:aten_headers",
        "@torch_tpu//shims/torch:torch_headers",
        "@torch_tpu//shims/torch/c10/core:base_headers",
        "@torch_tpu//shims/torch/torch/headeronly:torch_headeronly",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

nanobind_extension(
    name = "_weight_synchronizer",
    srcs = ["weight_synchronizer_module.cc"],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_nanobind_utils",
        ":weight_synchronizer_torch",
        "//core:raw_transfer_core",
        "//weight_sync:weight_synchronizer_base",
        "@com_google_absl//absl/status",
        "@torch_tpu//shims/torch:torch_headers",
    ],
)

cc_library(
    name = "torch_tpu_utils",
    srcs = ["torch_tpu_utils.cc"],
    hdrs = ["torch_tpu_utils.h"],
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@torch_tpu//shims/torch:aten_headers",
        "@torch_tpu//shims/torch:torch_headers",
        "@torch_tpu//shims/torch/c10/core:base_headers",
        "@torch_tpu//shims/torch/torch/headeronly:torch_headeronly",
        "@torch_tpu//torch_tpu/eager:device_buffer",
        "@torch_tpu//torch_tpu/eager:materialize",
        "@torch_tpu//torch_tpu/eager:tensor_to_buffer",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "torch_utils",
    srcs = ["torch_utils.cc"],
    hdrs = ["torch_utils.h"],
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    deps = [
        ":torch_tpu_utils",
        "@torch_tpu//shims/torch:aten_headers",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "torch_nanobind_utils",
    hdrs = ["torch_nanobind_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@nanobind",
        "@pybind11",
        "@torch_tpu//shims/torch:aten_headers",
        "@torch_tpu//shims/torch:torch_headers",
    ],
)
