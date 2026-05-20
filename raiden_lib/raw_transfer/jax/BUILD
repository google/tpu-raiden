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
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//perftools/accelerators/xprof/api/python:xprof_session",
        "@jax//jax",
        "@numpy//numpy",
    ],
)

nanobind_extension(
    name = "raw_transfer",
    srcs = [
        "raw_transfer.cc",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    pytype_srcs = ["raw_transfer.pyi"],
    deps = [
        "//raiden_lib/raw_transfer:raw_transfer_impl",
        "//third_party/tensorflow/compiler/xla:future",
        "//third_party/tensorflow/compiler/xla:shape_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@jax//jaxlib:py_client",
        "@nanobind",
        "@rules_python//python/cc:current_py_cc_headers",
        "@xla//xla/pjrt:common_pjrt_client",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_layout",
        "@xla//xla/pjrt:raw_buffer",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_external",
        "@xla//xla/pjrt/c_api_client:pjrt_c_api_client",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/tsl/concurrency:ref_count",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:statusor",
    ],
)

nanobind_extension(
    name = "raw_transfer_profiled",
    srcs = [
        "raw_transfer_profiled.cc",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    pytype_srcs = ["raw_transfer.pyi"],
    deps = [
        "//base:profiler",
        "//perftools/profiles/proto:builder",
        "//perftools/profiles/proto:encoder",
        "//raiden_lib/raw_transfer:raw_transfer_impl",
        "//third_party/tensorflow/compiler/xla:future",
        "//third_party/tensorflow/compiler/xla:shape_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@jax//jaxlib:py_client",
        "@nanobind",
        "@rules_python//python/cc:current_py_cc_headers",
        "@xla//xla/pjrt:common_pjrt_client",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_layout",
        "@xla//xla/pjrt:raw_buffer",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_external",
        "@xla//xla/pjrt/c_api_client:pjrt_c_api_client",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/tsl/concurrency:ref_count",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:statusor",
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
        ":raw_transfer",
        ":raw_transfer_profiled",
        ":utils",
        "//learning/brain/research/jax:tpu_support",
        "//perftools/accelerators/xprof/api/python:traceme",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jax//jax",
        "@numpy//numpy",
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
        ":raw_transfer",
        ":raw_transfer_profiled",
        ":utils",
        "//learning/brain/research/jax:tpu_support",
        "//perftools/accelerators/xprof/api/python:traceme",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jax//jax",
        "@numpy//numpy",
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
        ":raw_transfer",
        "//learning/brain/research/jax:tpu_support",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jax//jax",
        "@numpy//numpy",
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
        ":raw_transfer",
        "//learning/brain/research/jax:tpu_support",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jax//jax",
        "@numpy//numpy",
    ],
)
