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

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "kv_cache_manager_jax",
    srcs = [
        "kv_cache_manager.cc",
    ],
    hdrs = [
        "kv_cache_manager.h",
    ],
    copts = [
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//core:utils",
        "//kv_cache:kv_cache_manager_base",
        "@nanobind",
    ],
)

cc_library(
    name = "jax_utils",
    hdrs = [
        "jax_utils.h",
    ],
    copts = [
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@jax//jaxlib:py_client",
        "@nanobind",
        "@rules_python//python/cc:current_py_cc_headers",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
    ],
)

cc_library(
    name = "utils",
    hdrs = [
        "utils.h",
    ],
    copts = [
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":jax_utils",
        "@nanobind",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/python/ifrt",
    ],
)

cc_library(
    name = "nb_statusor",
    hdrs = ["nb_statusor.h"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@nanobind",
        "@rules_python//python/cc:current_py_cc_headers",
    ],
)

nanobind_extension(
    name = "_kv_cache_manager",
    srcs = [
        "kv_cache_manager_module.cc",
        "nb_statusor.h",
    ],
    copts = [
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_manager_jax",
        ":nb_statusor",
        "//core:raw_transfer_core",
        "//kv_cache:kv_cache_manager_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@nanobind",
        "@rules_python//python/cc:current_py_cc_headers",
    ],
)

nanobind_extension(
    name = "kv_cache_store",
    srcs = [
        "kv_cache_store_module.cc",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_manager_jax",
        "//kv_cache:kv_cache_store",
        "@nanobind",
        "@xla//xla/pjrt:status_casters",
    ],
)

cc_library(
    name = "kv_cache_manager_ffi",
    srcs = [
        "kv_cache_manager_ffi.cc",
    ],
    hdrs = [
        "kv_cache_manager_ffi.h",
    ],
    copts = [
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_manager_jax",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@xla//xla/ffi/api:ffi",
        "@xla//xla/stream_executor:platform",
        "@xla//xla/stream_executor:platform_manager",
        "@xla//xla/stream_executor:stream",
        "@xla//xla/stream_executor:stream_executor_h",
        "@xla//xla/stream_executor/tpu:tpu_executor",
    ],
    alwayslink = 1,
)

nanobind_extension(
    name = "_kv_cache_manager_ffi",
    srcs = [
        "kv_cache_manager_ffi_module.cc",
    ],
    copts = [
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_manager_ffi",
        ":kv_cache_manager_jax",
    ],
)

cc_library(
    name = "weight_synchronizer_ffi",
    srcs = [
        "weight_synchronizer_ffi.cc",
    ],
    hdrs = [
        "weight_synchronizer_ffi.h",
    ],
    copts = [
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":weight_synchronizer_jax",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@xla//xla/ffi/api:ffi",
        "@xla//xla/stream_executor:device_address",
        "@xla//xla/stream_executor:platform",
        "@xla//xla/stream_executor:platform_manager",
        "@xla//xla/stream_executor:stream",
        "@xla//xla/stream_executor:stream_executor_h",
    ],
    alwayslink = 1,
)

nanobind_extension(
    name = "_weight_synchronizer_ffi",
    srcs = [
        "weight_synchronizer_ffi_module.cc",
    ],
    copts = [
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":weight_synchronizer_ffi",
        ":weight_synchronizer_jax",
        "@nanobind",
    ],
)

cc_library(
    name = "weight_synchronizer_jax",
    srcs = [
        "weight_synchronizer.cc",
    ],
    hdrs = [
        "weight_synchronizer.h",
    ],
    copts = [
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":jax_utils",
        ":utils",
        "//weight_sync:weight_synchronizer_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@nanobind",
        "@xla//xla/pjrt:pjrt_client",
    ],
    alwayslink = 1,
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
        ":weight_synchronizer_jax",
        "//core:raw_transfer_core",
        "//weight_sync:weight_synchronizer_base",
        "@com_google_absl//absl/status:statusor",
        "@nanobind",
    ],
)

nanobind_extension(
    name = "_raw_transfer",
    srcs = [
        "raw_transfer.cc",
        "raw_transfer_internal.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":jax_utils",
        "//core:raw_transfer_impl",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_external",
    ],
)

cc_library(
    name = "kv_cache_manager_jax_mock",
    srcs = ["kv_cache_manager_mock.cc"],
    hdrs = ["kv_cache_manager.h"],
    copts = [
        "-fexceptions",
    ],
    defines = ["WITHOUT_PYTHON"],
    features = ["-use_header_modules"],
    deps = [
        "//kv_cache:kv_cache_manager_base",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "kv_cache_manager_ffi_mock",
    srcs = ["kv_cache_manager_ffi.cc"],
    hdrs = [
        "kv_cache_manager_ffi.h",
        "kv_cache_manager_ffi_internal.h",
    ],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        ":kv_cache_manager_jax_mock",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@xla//xla/ffi/api:ffi",
        "@xla//xla/stream_executor:platform",
        "@xla//xla/stream_executor:platform_manager",
        "@xla//xla/stream_executor:stream",
        "@xla//xla/stream_executor:stream_executor_h",
    ],
)

cc_test(
    name = "kv_cache_manager_ffi_test",
    srcs = ["kv_cache_manager_ffi_test.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        ":kv_cache_manager_ffi_mock",
        ":kv_cache_manager_jax_mock",
        "@xla//xla/ffi/api:c_api",
        "@xla//xla/ffi/api:ffi",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_external",
        "@xla//xla/pjrt/c_api_client:pjrt_c_api_client",
        "@xla//xla/stream_executor:platform_manager",
        "@xla//xla/stream_executor/host:host_platform",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_main",
    ],
)

cc_library(
    name = "weight_synchronizer_jax_mock",
    srcs = ["weight_synchronizer_mock.cc"],
    hdrs = ["weight_synchronizer.h"],
    copts = ["-fexceptions"],
    defines = ["WITHOUT_PYTHON"],
    features = ["-use_header_modules"],
    deps = [
        "//weight_sync:weight_synchronizer_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "weight_synchronizer_ffi_mock",
    srcs = ["weight_synchronizer_ffi.cc"],
    hdrs = [
        "weight_synchronizer_ffi.h",
        "weight_synchronizer_ffi_internal.h",
    ],
    copts = [
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":weight_synchronizer_jax_mock",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@xla//xla/ffi/api:ffi",
        "@xla//xla/stream_executor:device_address",
        "@xla//xla/stream_executor:platform",
        "@xla//xla/stream_executor:platform_manager",
        "@xla//xla/stream_executor:stream",
        "@xla//xla/stream_executor:stream_executor_h",
    ],
)

cc_test(
    name = "weight_synchronizer_ffi_test",
    srcs = ["weight_synchronizer_ffi_test.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        ":weight_synchronizer_ffi_mock",
        ":weight_synchronizer_jax_mock",
        "@com_google_absl//absl/log",
        "@xla//xla/ffi/api:c_api",
        "@xla//xla/ffi/api:ffi",
        "@xla//xla/stream_executor:platform_manager",
        "@xla//xla/stream_executor/host:host_platform",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_main",
    ],
)

cc_library(
    name = "raw_transfer_jax_mock",
    srcs = ["raw_transfer.cc"],
    hdrs = [
        "jax_utils.h",
        "mock_nanobind.h",
        "raw_transfer_internal.h",
    ],
    copts = [
        "-fexceptions",
        "-Wno-unused-variable",
    ],
    defines = ["WITHOUT_PYTHON"],
    features = ["-use_header_modules"],
    deps = [
        "//core:raw_transfer_core",
        "//core:raw_transfer_impl",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_external",
        "@xla//xla/pjrt/c_api_client:pjrt_c_api_client",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_test(
    name = "raw_transfer_test",
    srcs = ["raw_transfer_test.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        ":raw_transfer_jax_mock",
        "//core:raw_transfer_core",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@xla//xla/stream_executor/host:host_platform",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_main",
    ],
)
