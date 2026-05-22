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
        "-fexceptions",
        "-march=x86-64-v2",
    ],
    features = [
        "-use_header_modules",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//kv_cache:kv_cache_manager_base",
        "//third_party/py/torch:aten_headers",
        "//third_party/py/torch:torch_headers",
        "//third_party/py/torch/c10/core:base_headers",
        "//third_party/py/torch/torch/headeronly:torch_headeronly",
        "//third_party/py/torch_tpu/eager:device_buffer",
        "//third_party/py/torch_tpu/eager:materialize",
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
        "//third_party/py/torch:torch_headers",
        "//third_party/pybind11",
        "@com_google_absl//absl/status:statusor",
    ],
)

py_library(
    name = "kv_cache_manager_torch_py",
    srcs = ["kv_cache_manager.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_kv_cache_manager",
        "//third_party/py/torch:pytorch",
        "//third_party/py/torch_tpu",
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
        "//third_party/py/torch:pytorch",
        "//third_party/py/torch_tpu",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@numpy//numpy",
    ],
)
