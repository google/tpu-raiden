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

py_library(
    name = "kv_cache_manager_torch_py",
    srcs = ["kv_cache_manager.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//frameworks/torch:_kv_cache_manager",
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

py_library(
    name = "transfer_engine",
    srcs = ["transfer_engine.py"],
    data = [
        "transfer_engine.pyi",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//frameworks/torch:_transfer_engine",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
        "@torch_tpu//torch_tpu:_loader",
    ],
)

py_test(
    name = "transfer_engine_test",
    srcs = ["transfer_engine_test.py"],
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = [
        ":transfer_engine",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
    ],
)

py_library(
    name = "torch_raw_transfer",
    srcs = ["torch_raw_transfer.py"],
    deps = [
        "//frameworks/torch:_torch_raw_transfer",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
    ],
)

py_test(
    name = "torch_raw_transfer_perf_test_gl",
    timeout = "long",
    srcs = ["torch_raw_transfer_perf_test.py"],
    main = "torch_raw_transfer_perf_test.py",
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = [
        ":torch_raw_transfer",
        "//pyglib/contrib/g3_multiprocessing",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
    ],
)

py_test(
    name = "torch_raw_transfer_perf_test_gf",
    timeout = "long",
    srcs = ["torch_raw_transfer_perf_test.py"],
    main = "torch_raw_transfer_perf_test.py",
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostfish:4",
        "requires-net:external",
    ],
    deps = [
        ":torch_raw_transfer",
        "//pyglib/contrib/g3_multiprocessing",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
    ],
)

test_suite(
    name = "torch_raw_transfer_perf_test",
    tests = [
        ":torch_raw_transfer_perf_test_gf",
        ":torch_raw_transfer_perf_test_gl",
    ],
)

py_test(
    name = "torch_raw_transfer_test_gl",
    timeout = "long",
    srcs = ["torch_raw_transfer_test.py"],
    main = "torch_raw_transfer_test.py",
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = [
        ":torch_raw_transfer",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
    ],
)

py_test(
    name = "torch_raw_transfer_test_gf",
    timeout = "long",
    srcs = ["torch_raw_transfer_test.py"],
    main = "torch_raw_transfer_test.py",
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostfish:4",
        "requires-net:external",
    ],
    deps = [
        ":torch_raw_transfer",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
    ],
)

test_suite(
    name = "torch_raw_transfer_test",
    tests = [
        ":torch_raw_transfer_test_gf",
        ":torch_raw_transfer_test_gl",
    ],
)
