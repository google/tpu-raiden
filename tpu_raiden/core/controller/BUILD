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

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "worker_service_client",
    srcs = ["worker_service_client.cc"],
    hdrs = ["worker_service_client.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_raiden/proto:worker_service_cc_grpc",
        "//tpu_raiden/proto:worker_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "worker_service_impl",
    srcs = ["worker_service_impl.cc"],
    hdrs = ["worker_service_impl.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_raiden/core:host_memory_allocator",
        "//tpu_raiden/proto:worker_service_cc_grpc",
        "//tpu_raiden/proto:worker_service_cc_proto",
        "//tpu_raiden/rpc:raiden_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "worker_service_server",
    srcs = ["worker_service_server.cc"],
    hdrs = ["worker_service_server.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":worker_service_impl",
        "//tpu_raiden/core:host_memory_allocator",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "raiden_controller",
    srcs = ["raiden_controller.cc"],
    hdrs = ["raiden_controller.h"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":worker_service_client",
        "//tpu_raiden/kv_cache:logical_block_manager",
        "//tpu_raiden/proto:worker_service_cc_proto",
        "//tpu_raiden/rpc:raiden_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "controller_client",
    srcs = ["controller_client.cc"],
    hdrs = ["controller_client.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_raiden/proto:controller_service_cc_grpc",
        "//tpu_raiden/proto:controller_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "controller_service",
    srcs = ["controller_service.cc"],
    hdrs = ["controller_service.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_raiden/proto:controller_service_cc_grpc",
        "//tpu_raiden/proto:controller_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "test_util",
    testonly = True,
    hdrs = ["test_util.h"],
    visibility = ["//:internal"],
    deps = [
        ":worker_service_client",
        ":worker_service_impl",
        "@com_github_grpc_grpc//:grpc++",
    ],
)

cc_test(
    name = "worker_service_test",
    srcs = ["worker_service_test.cc"],
    deps = [
        ":test_util",
        ":worker_service_client",
        ":worker_service_impl",
        "//tpu_raiden/proto:worker_service_cc_proto",
        "//tpu_raiden/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "raiden_controller_test",
    srcs = ["raiden_controller_test.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        ":raiden_controller",
        ":test_util",
        "//tpu_raiden/proto:worker_service_cc_proto",
        "//tpu_raiden/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "controller_service_test",
    srcs = ["controller_service_test.cc"],
    deps = [
        ":controller_client",
        ":controller_service",
        "//tpu_raiden/proto:controller_service_cc_grpc",
        "//tpu_raiden/proto:controller_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "worker_service_server_test",
    srcs = ["worker_service_server_test.cc"],
    deps = [
        ":worker_service_client",
        ":worker_service_server",
        "//tpu_raiden/proto:worker_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
