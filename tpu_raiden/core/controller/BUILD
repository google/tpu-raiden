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

cc_test(
    name = "worker_service_test",
    srcs = ["worker_service_test.cc"],
    deps = [
        ":worker_service_client",
        ":worker_service_impl",
        "//tpu_raiden/proto:worker_service_cc_proto",
        "//tpu_raiden/rpc:raiden_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
