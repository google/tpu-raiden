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
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "jax_dma_kv_cache_benchmark",
    testonly = True,
    srcs = ["jax_dma_kv_cache_benchmark.py"],
    deps = [
        "//tpu_raiden/frameworks/jax:_tpu_raiden_jax",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/testing:absltest",       # VERIFY label
        "@com_google_absl_py//absl/testing:parameterized",  # VERIFY label
        "@jax//jax",
        "@pypi//numpy",
        "@pypi//libtpu",
    ],
)

load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "h2h_benchmark_runner",
    srcs = ["h2h_benchmark_runner.cc"],
    deps = [
        "//tpu_raiden/core:tpu_utils",
        "//tpu_raiden/kv_cache:kv_cache_manager_base",
        "//tpu_raiden/kv_cache:logical_block_manager",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

