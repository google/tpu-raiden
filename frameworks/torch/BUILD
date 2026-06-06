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
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("//core:header_only_cc_info.bzl", "header_only_cc_info")

package(default_visibility = ["//visibility:public"])

header_only_cc_info(
    name = "torch_tpu_tensor_to_buffer_headers",
    deps = [
        "@torch_tpu//torch_tpu/eager:tensor_to_buffer",
    ],
)

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
        "//core:xla_raw_transfer_headers",
        "//frameworks/torch:torch_utils",
        "//weight_sync:weight_synchronizer_base",
        "@com_google_absl//absl/status",
        "@torch_tpu//shims/torch:aten_headers",
        "@torch_tpu//shims/torch:torch_headers",
        "@torch_tpu//shims/torch/c10/core:base_headers",
        "@torch_tpu//shims/torch/torch/headeronly:torch_headeronly",
    ],
)

nanobind_extension(
    name = "_weight_synchronizer",
    srcs = ["weight_synchronizer_module.cc"],
    features = [
        "-use_header_modules",
    ],
    linkopts = ["-Wl,--allow-shlib-undefined"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_nanobind_utils",
        ":weight_synchronizer_torch",
        "//core:raw_transfer_core",
        "//weight_sync:weight_synchronizer_base",
        "@com_google_absl//absl/status",
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
        ":torch_tpu_tensor_to_buffer_headers",
        "//core:xla_raw_transfer_headers",
        "@com_google_absl//absl/status",
        "@torch_tpu//shims/torch:aten_headers",
        "@torch_tpu//shims/torch:torch_headers",
        "@torch_tpu//shims/torch/torch/headeronly:torch_headeronly",
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
        "//core:xla_raw_transfer_headers",
        "@torch_tpu//shims/torch:aten_headers",
    ],
)

cc_library(
    name = "torch_nanobind_utils",
    hdrs = ["torch_nanobind_utils.h"],
    copts = [
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@nanobind",
        "@pybind11",
        "@torch_tpu//shims/torch:aten_headers",
        "@torch_tpu//shims/torch:torch_headers",
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
        ":torch_tpu_utils",
        ":torch_utils",
        "//core:host_memory_allocator",
        "//core:utils",
        "//core:xla_raw_transfer_headers",
        "//kv_cache:kv_cache_manager_base",
        "@com_google_absl//absl/status:statusor",
        "@torch_tpu//shims/torch:aten_headers",
        "@torch_tpu//shims/torch:torch_headers",
    ],
)

nanobind_extension(
    name = "_kv_cache_manager",
    srcs = ["kv_cache_manager_module.cc"],
    features = [
        "-use_header_modules",
    ],
    linkopts = ["-Wl,--allow-shlib-undefined"],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_manager_torch",
        ":torch_nanobind_utils",
        "//core:raw_transfer_core",
        "//kv_cache:kv_cache_manager_base",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "transfer_engine_torch",
    srcs = ["transfer_engine.cc"],
    hdrs = ["transfer_engine.h"],
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
        ":kv_cache_manager_torch",
        "//core:transfer_engine_base",
        "//kv_cache:kv_cache_manager_base",
        "@torch_tpu//shims/torch:aten_headers",
    ],
    alwayslink = 1,
)

nanobind_extension(
    name = "_transfer_engine",
    srcs = ["transfer_engine_module.cc"],
    features = [
        "-use_header_modules",
    ],
    linkopts = ["-Wl,--allow-shlib-undefined"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_nanobind_utils",
        ":transfer_engine_torch",
        "//core:transfer_engine_base",
        "//kv_cache:kv_cache_manager_base",
        "@torch_tpu//shims/torch:aten_headers",
    ],
)

cc_library(
    name = "transfer_engine_torch_mock",
    srcs = ["transfer_engine.cc"],
    hdrs = ["transfer_engine.h"],
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    defines = ["RAIDEN_TORCH_USE_MOCKS"],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_manager_torch_mock",
        "//core:transfer_engine_base",
        "//kv_cache:kv_cache_manager_base",
        "@torch_tpu//shims/torch:aten_headers",
    ],
)

cc_library(
    name = "torch_raw_transfer_torch",
    srcs = ["torch_raw_transfer.cc"],
    hdrs = ["torch_raw_transfer.h"],
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_tpu_utils",
        "//core:host_memory_allocator",
        "//core:raw_transfer_core",
        "//core:utils",
        "//core:xla_raw_transfer_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@torch_tpu//shims/torch:aten_headers",
    ],
    alwayslink = 1,
)

nanobind_extension(
    name = "_torch_raw_transfer",
    srcs = ["torch_raw_transfer_module.cc"],
    features = ["-use_header_modules"],
    linkopts = ["-Wl,--allow-shlib-undefined"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_nanobind_utils",
        ":torch_raw_transfer_torch",
        "//core:host_memory_allocator",
        "//core:raw_transfer_core",
        "//core:utils",
        "//core:xla_raw_transfer_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@torch_tpu//shims/torch:aten_headers",
    ],
)

cc_library(
    name = "torch_raw_transfer_torch_mock",
    srcs = ["torch_raw_transfer.cc"],
    hdrs = ["torch_raw_transfer.h"],
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_tpu_utils_mock",
        "//core:host_memory_allocator",
        "//core:raw_transfer_core",
        "//core:utils",
        "//core:xla_raw_transfer_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@torch_tpu//shims/torch:aten_headers",
    ],
)

cc_library(
    name = "torch_tpu_utils_mock",
    srcs = ["torch_tpu_utils_mock.cc"],
    hdrs = [
        "torch_tpu_utils.h",
        "torch_tpu_utils_mock.h",
    ],
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@torch_tpu//shims/torch:aten_headers",
        "@torch_tpu//shims/torch:torch_headers",
        "@torch_tpu//shims/torch/torch/headeronly:torch_headeronly",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "torch_utils_mock",
    srcs = ["torch_utils.cc"],
    hdrs = ["torch_utils.h"],
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":torch_tpu_utils_mock",
        "//core:xla_raw_transfer_headers",
        "@torch_tpu//shims/torch:aten_headers",
    ],
)

cc_library(
    name = "kv_cache_manager_torch_mock",
    srcs = ["kv_cache_manager.cc"],
    hdrs = ["kv_cache_manager.h"],
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_tpu_utils_mock",
        ":torch_utils_mock",
        "//core:host_memory_allocator",
        "//core:utils",
        "//core:xla_raw_transfer_headers",
        "//kv_cache:kv_cache_manager_base",
        "@com_google_absl//absl/status:statusor",
        "@torch_tpu//shims/torch:aten_headers",
        "@torch_tpu//shims/torch:torch_headers",
    ],
)

cc_library(
    name = "weight_synchronizer_torch_mock",
    srcs = ["weight_synchronizer.cc"],
    hdrs = ["weight_synchronizer.h"],
    copts = [
        "-std=c++20",
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_utils_mock",
        "//core:xla_raw_transfer_headers",
        "//weight_sync:weight_synchronizer_base",
        "@com_google_absl//absl/status",
        "@torch_tpu//shims/torch:aten_headers",
        "@torch_tpu//shims/torch:torch_headers",
        "@torch_tpu//shims/torch/c10/core:base_headers",
        "@torch_tpu//shims/torch/torch/headeronly:torch_headeronly",
    ],
)

cc_test(
    name = "kv_cache_manager_test",
    srcs = ["kv_cache_manager_test.cc"],
    copts = [
        "-std=c++20",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":kv_cache_manager_torch_mock",
        ":torch_tpu_utils_mock",
        "@torch_tpu//shims/torch",
        "@torch_tpu//shims/torch:torch_headers",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_main",
    ],
)

cc_test(
    name = "transfer_engine_test",
    srcs = ["transfer_engine_test.cc"],
    copts = [
        "-std=c++20",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":torch_tpu_utils_mock",
        ":transfer_engine_torch_mock",
        "@torch_tpu//shims/torch",
        "@torch_tpu//shims/torch:torch_headers",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_main",
    ],
)

cc_test(
    name = "torch_raw_transfer_test",
    srcs = ["torch_raw_transfer_test.cc"],
    copts = [
        "-std=c++20",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":torch_raw_transfer_torch_mock",
        ":torch_tpu_utils_mock",
        "@torch_tpu//shims/torch",
        "@torch_tpu//shims/torch:torch_headers",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_main",
    ],
)
