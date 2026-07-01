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

load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

exports_files(["multi_host_perf_test_oss.py"])

py_binary(
    name = "multi_host_perf_test_oss",
    testonly = True,
    srcs = ["multi_host_perf_test_oss.py"],
    deps = [
        "//tpu_raiden/frameworks/jax:_tpu_raiden_jax",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "@jax//jax",
        "@pypi//numpy",
        "@pypi//libtpu",
    ],
)

py_binary(
    name = "h2h_perf_test_oss",
    testonly = True,
    srcs = ["h2h_perf_test_oss.py"],
    deps = [
        "//tpu_raiden/frameworks/jax:_tpu_raiden_jax",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "@jax//jax",
        "@pypi//numpy",
    ],
)
