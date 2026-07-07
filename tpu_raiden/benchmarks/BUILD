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


load("@rules_python//python:defs.bzl", "py_binary", "py_library")


py_library(
    name = "perf_core",
    testonly = True,
    srcs = ["perf_core.py"],
    deps = [
        "//tpu_raiden/frameworks/jax:_tpu_raiden_jax",
        "@jax//jax",
        "@pypi//numpy",
        "@pypi//libtpu",
    ],
)


py_binary(
    name = "h2d_d2h_perf_test",
    testonly = True,
    srcs = ["h2d_d2h_perf_test.py"],
    deps = [
        ":perf_core",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)

py_binary(
    name = "h2d_d2h_benchmark_gating",
    testonly = True,
    srcs = ["h2d_d2h_benchmark_gating.py"],
    data = ["gating_baselines.json"],
    deps = [
        ":perf_core",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
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
        "@pypi//libtpu",
    ],
)

# Same source as the binary above, exposed as a library so the H2H gate can reuse
# the PUSH transport (run_launcher / run_sender / run_receiver) in-process.
py_library(
    name = "h2h_perf_test_oss_lib",
    testonly = True,
    srcs = ["h2h_perf_test_oss.py"],
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
    name = "h2h_benchmark_gating",
    testonly = True,
    srcs = ["h2h_benchmark_gating.py"],
    data = ["h2h_gating_baselines.json"],
    deps = [
        ":h2h_perf_test_oss_lib",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
        "@pypi//numpy",
    ],
)

# C++ vs Python comparison as ONE BAP workload: both benchmark binaries are
# runfiles data deps, so `bazel run` on the TPU runner builds and ships them.
py_binary(
    name = "h2h_compare_bench",
    testonly = True,
    srcs = ["h2h_compare_bench.py"],
    data = [
        ":h2h_perf_test_oss",
        # main has the C++ runner at top-level examples/ (no path split like V7).
        "//examples/microbenchmarks:h2h_benchmark_runner",
    ],
    deps = [
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)
