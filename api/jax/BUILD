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
load("//third_party/bazel_rules/rules_python/python:py_binary.bzl", "py_binary")

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
        "//kv_cache:kv_cache_manager_base",
        "//raiden_lib/raw_transfer:raw_transfer_impl",
        "@nanobind",
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
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_manager_jax",
        "//kv_cache:kv_cache_manager_base",
        "//raiden_lib/raw_transfer:raw_transfer_impl",
        "@nanobind",
    ],
)

py_library(
    name = "kv_cache_manager_jax_py",
    srcs = ["kv_cache_manager.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_kv_cache_manager",
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
        "@xla//xla/ffi/api:c_api",
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

py_library(
    name = "kv_cache_manager_ffi_py",
    srcs = ["kv_cache_manager_ffi.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_kv_cache_manager_ffi",
        "@jax//jax",
        "@jax//jax/experimental:compute_on",
        "@numpy//numpy",
    ],
)

py_test(
    name = "kv_cache_manager_test_gl",
    timeout = "long",
    srcs = ["kv_cache_manager_test.py"],
    main = "kv_cache_manager_test.py",
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = [
        ":kv_cache_manager_jax_py",
        "//learning/brain/research/jax:tpu_support",
        "//third_party/py/absl/flags",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jax//jax",
        "@numpy//numpy",
    ],
)

py_test(
    name = "kv_cache_manager_test_gf",
    timeout = "long",
    srcs = ["kv_cache_manager_test.py"],
    main = "kv_cache_manager_test.py",
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostfish:4",
        "requires-net:external",
    ],
    deps = [
        ":kv_cache_manager_jax_py",
        "//learning/brain/research/jax:tpu_support",
        "//third_party/py/absl/flags",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jax//jax",
        "@numpy//numpy",
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
        ":kv_cache_manager_jax_py",
        "//third_party/py/absl/flags",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jax//jax",
        "@numpy//numpy",
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
        "//raiden_lib/raw_transfer:raw_transfer_impl",
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
        "//raiden_lib/raw_transfer:raw_transfer_impl",
        "//weight_sync:weight_synchronizer_base",
        "@com_google_absl//absl/status:statusor",
        "@nanobind",
    ],
)

py_library(
    name = "resharding_planner",
    srcs = ["resharding_planner.py"],
    visibility = ["//visibility:public"],
    deps = ["@jax//jax"],
)

py_library(
    name = "resharding_engine",
    srcs = ["resharding_engine.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":resharding_planner",
        ":weight_synchronizer_jax_py",
        "@jax//jax",
        "@numpy//numpy",
    ],
)

py_library(
    name = "resharding_test_lib",
    srcs = ["resharding_test_lib.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":resharding_planner",
        "@jax//jax",
        "@numpy//numpy",
    ],
)

py_library(
    name = "weight_synchronizer_jax_py",
    srcs = ["weight_synchronizer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_weight_synchronizer",
        ":resharding_planner",
    ],
)

py_test(
    name = "weight_synchronizer_test",
    srcs = ["weight_synchronizer_test.py"],
    deps = [
        ":weight_synchronizer_jax_py",
        "@com_google_absl_py//absl/testing:absltest",
        "@jax//jax",
        "@numpy//numpy",
    ],
)

py_test(
    name = "resharding_planner_test",
    srcs = ["resharding_planner_test.py"],
    deps = [
        ":resharding_planner",
        "@com_google_absl_py//absl/testing:absltest",
        "@jax//jax",
        "@numpy//numpy",
    ],
)

py_test(
    name = "resharding_test",
    srcs = ["resharding_test.py"],
    deps = [
        ":resharding_engine",
        ":resharding_test_lib",
        "@com_google_absl_py//absl/testing:absltest",
        "@jax//jax",
        "@numpy//numpy",
    ],
)

py_library(
    name = "mc_jax_ffi_test_main_lib",
    srcs = ["mc_jax_ffi_test_main.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_manager_ffi_py",
        "//learning/brain/research/jax:tpu_support",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "@jax//jax",
        "@numpy//numpy",
    ],
)

py_binary(
    name = "mc_jax_ffi_test_main",
    srcs = ["mc_jax_ffi_test_main.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":mc_jax_ffi_test_main_lib",
    ],
)

py_test(
    name = "mc_jax_ffi_test_main_test",
    srcs = ["mc_jax_ffi_test_main_test.py"],
    deps = [
        ":mc_jax_ffi_test_main_lib",
        "@com_google_absl_py//absl/testing:absltest",
        "@jax//jax",
    ],
)
