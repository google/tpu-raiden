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

load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("@rules_python//python:proto.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "weight_synchronizer_service_proto",
    srcs = ["weight_synchronizer_service.proto"],
)

cc_proto_library(
    name = "weight_synchronizer_service_cc_proto",
    deps = [":weight_synchronizer_service_proto"],
)

py_proto_library(
    name = "weight_synchronizer_service_py_pb2",
    deps = [":weight_synchronizer_service_proto"],
)

cc_library(
    name = "weight_synchronizer_base",
    srcs = [
        "weight_synchronizer_base.cc",
        "weight_synchronizer_control_service.cc",
    ],
    hdrs = [
        "weight_synchronizer_base.h",
        "weight_synchronizer_control_service.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":weight_synchronizer_service_cc_proto",
        "//tpu_raiden/core:raiden_manager_base",
        "//tpu_raiden/core:raw_transfer_core",
        "//tpu_raiden/core:status_macros",
        "//tpu_raiden/core:tpu_utils",
        "//tpu_raiden/core:xla_raw_transfer_headers",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = 1,
)

cc_test(
    name = "weight_synchronizer_test",
    srcs = [
        "weight_synchronizer_test.cc",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    linkopts = ["-Wl,--allow-shlib-undefined"],
    deps = [
        ":weight_synchronizer_base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_external",
    ],
)

cc_test(
    name = "weight_synchronizer_control_service_test",
    srcs = ["weight_synchronizer_control_service_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":weight_synchronizer_base",
        ":weight_synchronizer_service_cc_proto",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
