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

load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//raiden_lib/raw_transfer:header_only_cc_info.bzl", "header_only_cc_info")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

header_only_cc_info(
    name = "xla_stablehlo_headers",
    deps = [
        "@stablehlo//:attr_type_builder_util",
        "@stablehlo//:chlo_builder",
        "@stablehlo//:mlir_builder",
        "@stablehlo//:stablehlo_builder",
        "@xla//xla:future",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:common_pjrt_client",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_layout",
        "@xla//xla/pjrt:raw_buffer",
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_external",
        "@xla//xla/pjrt/c_api_client:pjrt_c_api_client",
        "@xla//xla/tsl/concurrency:ref_count",
    ],
)

cc_binary(
    name = "_torch_raw_transfer.so",
    srcs = [
        "torch_raw_transfer.cc",
    ],
    copts = [
        "-std=c++20",
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    linkopts = [
        "-Wl,--unresolved-symbols=ignore-all",
    ],
    linkshared = True,
    deps = [
        "//raiden_lib/raw_transfer:raw_transfer_impl_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_torch_headers//:torch_headers",
        "@local_torch_tpu_headers//:headers",
        "@pybind11//:pybind11",
        ":xla_stablehlo_headers",
    ],
)

alias(
    name = "_torch_raw_transfer",
    actual = ":_torch_raw_transfer.so",
)

py_library(
    name = "torch_raw_transfer",
    srcs = ["torch_raw_transfer.py"],
    data = [
        ":_torch_raw_transfer.so",
        "torch_raw_transfer.pyi",
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
        "//third_party/py/torch:pytorch",
        "//third_party/py/torch_tpu",
        "//third_party/py/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//third_party/py/torch_tpu/_internal/utils:hardware",
        "//third_party/py/torch_tpu/tests/distributed:distributed_utils",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@numpy//numpy",
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
        "//third_party/py/torch:pytorch",
        "//third_party/py/torch_tpu",
        "//third_party/py/torch_tpu/_internal/distributed/launchers:singlehost_wrapper",
        "//third_party/py/torch_tpu/_internal/utils:hardware",
        "//third_party/py/torch_tpu/tests/distributed:distributed_utils",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@numpy//numpy",
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
        "//third_party/py/torch:pytorch",
        "//third_party/py/torch_tpu",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@numpy//numpy",
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
        "//third_party/py/torch:pytorch",
        "//third_party/py/torch_tpu",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@numpy//numpy",
    ],
)

test_suite(
    name = "torch_raw_transfer_test",
    tests = [
        ":torch_raw_transfer_test_gf",
        ":torch_raw_transfer_test_gl",
    ],
)
