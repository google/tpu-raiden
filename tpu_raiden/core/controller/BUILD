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

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

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
        "@xla//xla/tsl/concurrency:future",
    ],
)

cc_library(
    name = "worker_service_impl",
    srcs = ["worker_service_impl.cc"],
    hdrs = ["worker_service_impl.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_raiden/core:host_memory_allocator",
        "//tpu_raiden/core:kv_manager_holder",
        "//tpu_raiden/core:raw_transfer_core",
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
        "//tpu_raiden/core:kv_manager_holder",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "orchestrator_service_client",
    srcs = ["orchestrator_service_client.cc"],
    hdrs = ["orchestrator_service_client.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_raiden/proto:orchestrator_service_cc_grpc",
        "//tpu_raiden/proto:orchestrator_service_cc_proto",
        "//tpu_raiden/rpc:raiden_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
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
        ":controller_server",
        ":orchestrator_service_client",
        ":worker_registry",
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
        "@xla//xla/tsl/concurrency:future",
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
    name = "worker_registry",
    srcs = ["worker_registry.cc"],
    hdrs = ["worker_registry.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":worker_service_client",
        "//tpu_raiden/proto:worker_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "controller_service",
    srcs = ["controller_service.cc"],
    hdrs = ["controller_service.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":worker_registry",
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
    name = "controller_server",
    srcs = ["controller_server.cc"],
    hdrs = ["controller_server.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":controller_service",
        ":worker_registry",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
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
        ":controller_client",
        ":controller_service",
        ":worker_service_client",
        ":worker_service_impl",
        "//tpu_raiden/core:raw_transfer_core",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "worker_service_test",
    srcs = ["worker_service_test.cc"],
    deps = [
        ":test_util",
        ":worker_service_client",
        ":worker_service_impl",
        "//tpu_raiden/core:kv_manager_holder",
        "//tpu_raiden/core:raw_transfer_core",
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
        ":controller_client",
        ":raiden_controller",
        ":raiden_orchestrator",
        ":test_util",
        "//tpu_raiden/core:kv_manager_holder",
        "//tpu_raiden/proto:controller_service_cc_proto",
        "//tpu_raiden/proto:worker_service_cc_proto",
        "//tpu_raiden/rpc:raiden_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
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
        ":test_util",
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

cc_test(
    name = "worker_registry_test",
    srcs = ["worker_registry_test.cc"],
    deps = [
        ":worker_registry",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "controller_server_test",
    srcs = ["controller_server_test.cc"],
    deps = [
        ":controller_client",
        ":controller_server",
        ":worker_registry",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "raiden_orchestrator",
    srcs = ["raiden_orchestrator.cc"],
    hdrs = ["raiden_orchestrator.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_raiden/proto:orchestrator_service_cc_grpc",
        "//tpu_raiden/proto:orchestrator_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_binary(
    name = "raiden_orchestrator_main",
    srcs = ["raiden_orchestrator_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":raiden_orchestrator",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "raiden_orchestrator_test",
    srcs = ["raiden_orchestrator_test.cc"],
    deps = [
        ":raiden_orchestrator",
        "//tpu_raiden/proto:orchestrator_service_cc_grpc",
        "//tpu_raiden/proto:orchestrator_service_cc_proto",
        "//tpu_raiden/rpc:raiden_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_googletest//:gtest_main",
    ],
)
