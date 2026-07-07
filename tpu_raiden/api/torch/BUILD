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

load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "torch_tpu_common_loader",
    srcs = ["torch_tpu_common_loader.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@torch_tpu//torch_tpu",
        "@torch_tpu//torch_tpu:_loader",
    ],
)

py_library(
    name = "kv_cache_manager_torch_py",
    srcs = ["kv_cache_manager.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_tpu_common_loader",
        "//tpu_raiden/frameworks/torch:_tpu_raiden_torch",
    ],
)

py_library(
    name = "weight_synchronizer_torch_py",
    srcs = ["weight_synchronizer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_tpu_common_loader",
        "//tpu_raiden/frameworks/torch:_tpu_raiden_torch",
        "@torch_tpu//shims/torch:pytorch",
    ],
)

py_library(
    name = "kv_cache_store",
    srcs = ["kv_cache_store.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":torch_tpu_common_loader",
        "//tpu_raiden/frameworks/torch:_tpu_raiden_torch",
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

py_test(
    name = "kv_cache_manager_transfer_test",
    srcs = ["kv_cache_manager_transfer_test.py"],
    tags = [
        "cpu:8",
        "notap",
        "requires-accel-hbm",
        "requires-ghostlite:8",
        "requires-net:external",
    ],
    deps = [
        ":kv_cache_manager_torch_py",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@pypi//numpy",
        "@torch_tpu//shims/torch:pytorch",
        "@torch_tpu//torch_tpu",
    ],
)
