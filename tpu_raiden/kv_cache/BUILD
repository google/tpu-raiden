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
load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "logical_block_manager",
    srcs = ["logical_block_manager.cc"],
    hdrs = ["logical_block_manager.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "logical_block_manager_test",
    srcs = ["logical_block_manager_test.cc"],
    deps = [
        ":logical_block_manager",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "pool_layout",
    srcs = ["pool_layout.cc"],
    hdrs = ["pool_layout.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        "//tpu_raiden/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "pool_layout_test",
    srcs = ["pool_layout_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":pool_layout",
        "//tpu_raiden/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "kv_cache_manager_base",
    srcs = [
        "kv_cache_manager_base.cc",
    ],
    hdrs = [
        "kv_cache_manager_base.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":logical_block_manager",
        ":pool_layout",
        "//tpu_raiden/core:host_memory_allocator",
        "//tpu_raiden/core:numa_thread_pool",
        "//tpu_raiden/core:raiden_manager_base",
        "//tpu_raiden/core:raw_transfer_core",
        "//tpu_raiden/core:raw_transfer_impl",
        "//tpu_raiden/core:status_macros",
        "//tpu_raiden/core:tpu_utils",
        "//tpu_raiden/core:xla_raw_transfer_headers",
        "//tpu_raiden/rpc:raiden_service_cc_proto",
        "//tpu_raiden/transport:block_transport",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@xla//xla/pjrt:semaphore",
        "@xla//xla/stream_executor:device_address",
    ],
    alwayslink = 1,
)

cc_library(
    name = "kv_cache_store_internal",
    srcs = [
        "kv_cache_store_internal.cc",
    ],
    hdrs = [
        "kv_cache_store_internal.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_manager_base",
        ":logical_block_manager",
        "//tpu_raiden/core:raw_transfer_core",
        "//tpu_raiden/core:status_macros",
        "//tpu_raiden/kv_cache/global_registry:global_registry_client_cc",
        "//tpu_raiden/transport:block_transport",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

# JAX test targets successfully relocated to api/jax/BUILD.

cc_test(
    name = "kv_cache_store_internal_test",
    srcs = ["kv_cache_store_internal_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    tags = [
        "no_oss",
        "requires-jellyfish",
    ],
    deps = [
        ":kv_cache_manager_base",
        ":kv_cache_store_internal",
        "//tpu_raiden/core:raw_transfer_core",
        "//tpu_raiden/core:tpu_pjrt_manager",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_main",
    ],
)

cc_library(
    name = "lru_cache",
    hdrs = ["lru_cache.h"],
    deps = ["@com_google_absl//absl/container:flat_hash_map"],
)

cc_test(
    name = "lru_cache_test",
    srcs = ["lru_cache_test.cc"],
    deps = [
        ":lru_cache",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "kv_cache_store",
    srcs = ["kv_cache_store.cc"],
    hdrs = ["kv_cache_store.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":lru_cache",
        "//tpu_raiden/core:raw_transfer_core",
        "//tpu_raiden/kv_cache/global_registry:global_registry_client_cc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "kv_cache_store_test",
    srcs = ["kv_cache_store_test.cc"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":kv_cache_store",
        ":lru_cache",
        "//tpu_raiden/kv_cache/global_registry:global_registry_client_cc",
        "//tpu_raiden/kv_cache/global_registry:global_registry_server_lib",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

py_library(
    name = "kv_cache_controller",
    srcs = ["kv_cache_controller.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_raiden/rpc:raiden_controller",
        "//tpu_raiden/rpc:raiden_service_py_pb2",
    ],
)

cc_library(
    name = "kv_cache_listener",
    srcs = ["kv_cache_listener.cc"],
    hdrs = ["kv_cache_listener.h"],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":kv_cache_manager_base",
        "//tpu_raiden/rpc:raiden_service_cc_proto",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)
