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
load("@rules_python//python:defs.bzl", "py_library", "py_test")

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
        "-frtti",
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//tpu_raiden/core:host_memory_allocator",
        "//tpu_raiden/core:kv_cache_manager_with_transfer",
        "//tpu_raiden/core:kv_manager_holder",
        "//tpu_raiden/core:metrics_collector",
        "//tpu_raiden/core:raw_transfer_core",
        "//tpu_raiden/core:status_macros",
        "//tpu_raiden/core:tpu_utils",
        "//tpu_raiden/core:utils",
        "//tpu_raiden/core/controller:controller_client",
        "//tpu_raiden/core/controller:worker_service_server",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@nanobind",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/tsl/platform:logging",
    ],
)

cc_library(
    name = "jax_utils",
    hdrs = [
        "jax_utils.h",
    ],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":mock_nanobind",
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
        "-frtti",
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
    name = "_tpu_raiden_jax",
    srcs = [
        "nb_statusor.h",
        "tpu_raiden_jax_module.cc",
    ],
    copts = [
        "-fexceptions",
        "-frtti",
        "-march=x86-64-v2",
        "-fvisibility=hidden",
    ],
    features = [
        "-use_header_modules",
    ],
    linkopts = ["-Wl,--exclude-libs,ALL"],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_manager_jax",
        ":nb_statusor",
        ":raw_transfer_jax",
        ":weight_synchronizer_jax",
        "//tpu_raiden/core:raw_transfer_core",
        "//tpu_raiden/kv_cache:kv_cache_store",
        "//tpu_raiden/weight_sync:weight_synchronizer_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@nanobind",
        "@rules_python//python/cc:current_py_cc_headers",
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
        "//tpu_raiden/core:kv_cache_manager_with_transfer",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@xla//xla/ffi/api:ffi",
        "@xla//xla/stream_executor:platform",
        "@xla//xla/stream_executor:platform_manager",
        "@xla//xla/stream_executor:stream",
        "@xla//xla/stream_executor:stream_executor_h",
        "@xla//xla/tpu:tpu_library_init_fns",
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
        "-frtti",
        "-march=x86-64-v2",
        "-fvisibility=hidden",
    ],
    features = [
        "-use_header_modules",
    ],
    linkopts = ["-Wl,--exclude-libs,ALL"],
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
        "weight_synchronizer_ffi_internal.h",
    ],
    copts = [
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    tags = ["manual"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_raiden/weight_sync:weight_synchronizer_base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
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
        "-frtti",
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    tags = ["manual"],
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
        "-frtti",
    ],
    features = [
        "-use_header_modules",
    ],
    tags = ["manual"],
    visibility = ["//visibility:public"],
    deps = [
        ":jax_utils",
        ":utils",
        "//tpu_raiden/core:raw_transfer_core",
        "//tpu_raiden/weight_sync:weight_synchronizer_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@nanobind",
    ],
    alwayslink = 1,
)

# Removed _weight_synchronizer (consolidated into _tpu_raiden_jax)

cc_library(
    name = "mock_nanobind",
    hdrs = [
        "mock_nanobind.h",
    ],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = ["//visibility:private"],
)

nanobind_extension(
    name = "_raw_transfer",
    srcs = [
        "mock_nanobind.h",
        "raw_transfer.cc",
        "raw_transfer_internal.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
        "-frtti",
        "-fvisibility=hidden",
    ],
    features = ["-use_header_modules"],
    linkopts = ["-Wl,--exclude-libs,ALL"],
    visibility = ["//visibility:public"],
    deps = [
        ":jax_utils",
        "//tpu_raiden/core:raw_transfer_core",
        "//tpu_raiden/core:raw_transfer_impl",
        "//tpu_raiden/core:status_macros",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@nanobind",
        "@xla//xla/pjrt:status_casters",
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
        "//tpu_raiden/core:kv_cache_manager_with_transfer",
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
        "//tpu_raiden/core:kv_cache_manager_with_transfer",
        "@xla//xla/ffi/api:c_api",
        "@xla//xla/ffi/api:ffi",
        "@xla//xla/stream_executor:platform_manager",
        "@xla//xla/stream_executor/host:host_platform",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_main",
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
        "-frtti",
        "-march=x86-64-v2",
    ],
    features = ["-use_header_modules"],
    deps = [
        "//tpu_raiden/weight_sync:weight_synchronizer_base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
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
        "//tpu_raiden/weight_sync:weight_synchronizer_base",
        "@com_google_absl//absl/container:flat_hash_set",
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
    name = "raw_transfer_jax",
    srcs = ["raw_transfer.cc"],
    hdrs = [
        "jax_utils.h",
        "raw_transfer_internal.h",
    ],
    copts = [
        "-fexceptions",
        "-frtti",
        "-Wno-unused-variable",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":mock_nanobind",
        "//tpu_raiden/core:raw_transfer_core",
        "//tpu_raiden/core:raw_transfer_impl",
        "//tpu_raiden/core:status_macros",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@jax//jaxlib:py_client",
        "@nanobind",
        "@rules_python//python/cc:current_py_cc_headers",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_external",  # buildcleaner: keep
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/tsl/concurrency:ref_count",
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
        "//tpu_raiden/core:raw_transfer_core",
        "//tpu_raiden/core:raw_transfer_impl",
        "//tpu_raiden/core:status_macros",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_external",  # buildcleaner: keep
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/tsl/concurrency:ref_count",
    ],
)

cc_test(
    name = "raw_transfer_test",
    srcs = ["raw_transfer_test.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        ":raw_transfer_jax_mock",
        "//tpu_raiden/core:raw_transfer_core",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@xla//xla/stream_executor/host:host_platform",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_main",
    ],
)

py_test(
    name = "kv_cache_manager_test_cpu",
    timeout = "long",
    srcs = ["kv_cache_manager_test.py"],
    args = ["--device_type=cpu"],
    main = "kv_cache_manager_test.py",
    tags = [
        "cpu:4",
        "notap",
    ],
    deps = [
        ":_tpu_raiden_jax",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jax//jax",
        "@jax//jax/_src/lib",
        "@pypi//numpy",
    ],
)

py_library(
    name = "weight_synchronizer_ffi_py",
    srcs = ["weight_synchronizer_ffi.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_weight_synchronizer_ffi",
        "@jax//jax",
        "@jax//jax/experimental:compute_on",
        "@pypi//numpy",
    ],
)

py_library(
    name = "resharding_planner",
    srcs = ["resharding_planner.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_raiden/kv_cache:nd_slice_math",
        "@jax//jax",
    ],
)

py_test(
    name = "resharding_planner_test",
    srcs = ["resharding_planner_test.py"],
    deps = [
        ":resharding_planner",
        "@com_google_absl_py//absl/testing:absltest",
        "@jax//jax",
        "@pypi//numpy",
    ],
)

py_library(
    name = "jax_test_utils",
    srcs = ["utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl_py//absl/flags",
        "@jax//jax",
        "@pypi//numpy",
    ],
)

py_test(
    name = "raw_transfer_perf_gl",
    timeout = "long",
    srcs = ["raw_transfer_perf_test.py"],
    main = "raw_transfer_perf_test.py",
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = [
        ":_raw_transfer",
        ":jax_test_utils",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jax//jax",
        "@pypi//numpy",
    ],
)

py_test(
    name = "raw_transfer_perf_gf",
    timeout = "long",
    srcs = ["raw_transfer_perf_test.py"],
    main = "raw_transfer_perf_test.py",
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostfish:4",
        "requires-net:external",
    ],
    deps = [
        ":_raw_transfer",
        ":jax_test_utils",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jax//jax",
        "@pypi//numpy",
    ],
)

py_test(
    name = "raw_transfer_test_gl",
    timeout = "long",
    srcs = ["raw_transfer_test.py"],
    main = "raw_transfer_test.py",
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = [
        ":_raw_transfer",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jax//jax",
        "@pypi//numpy",
    ],
)

py_test(
    name = "raw_transfer_test_gf",
    timeout = "long",
    srcs = ["raw_transfer_test.py"],
    main = "raw_transfer_test.py",
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostfish:4",
        "requires-net:external",
    ],
    deps = [
        ":_raw_transfer",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jax//jax",
        "@pypi//numpy",
    ],
)

cc_library(
    name = "kv_cache_manager_core_lib",
    srcs = ["kv_cache_manager.cc"],
    hdrs = [
        "kv_cache_manager.h",
    ],
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    defines = ["WITHOUT_PYTHON"],
    features = ["-use_header_modules"],
    deps = [
        "//tpu_raiden/core:host_memory_allocator",
        "//tpu_raiden/core:kv_cache_manager_with_transfer",
        "//tpu_raiden/core:kv_manager_holder",
        "//tpu_raiden/core:metrics_collector",
        "//tpu_raiden/core:raw_transfer_core",
        "//tpu_raiden/core:status_macros",
        "//tpu_raiden/core:tpu_utils",
        "//tpu_raiden/core:utils",
        "//tpu_raiden/core/controller:controller_client",
        "//tpu_raiden/core/controller:worker_service_server",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/tsl/platform:logging",
    ],
)

cc_test(
    name = "kv_cache_manager_wrapper_test",
    srcs = ["kv_cache_manager_wrapper_test.cc"],
    copts = [
        "-DWITHOUT_PYTHON",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    tags = ["notap"],
    deps = [
        ":kv_cache_manager_core_lib",
        "//tpu_raiden/core:kv_cache_manager_with_transfer",
        "//tpu_raiden/core:raw_transfer_core",
        "//tpu_raiden/core/controller:controller_service",
        "//tpu_raiden/core/controller:raiden_controller",
        "//tpu_raiden/core/controller:test_util",
        "//tpu_raiden/rpc:raiden_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_main",
    ],
)
