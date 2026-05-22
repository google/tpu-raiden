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
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

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
        "//core:raiden_manager_base",
        "//raiden_lib/raw_transfer:raw_transfer_impl",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/c_api_client:pjrt_c_api_client",
        "@xla//xla/stream_executor:stream",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
    alwayslink = 1,
)

nanobind_extension(
    name = "kv_cache_manager",
    srcs = [
        "kv_cache_manager_pybind.cc",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":kv_cache_manager_lib",
        "@nanobind",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/pjrt/c_api_client:pjrt_c_api_client",
        "@xla//xla/tsl/platform:errors",
    ],
)

nanobind_extension(
    name = "kv_cache_store",
    srcs = [
        "kv_cache_store.cc",
        "kv_cache_store.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":kv_cache_manager_lib",
        "//raiden_lib/raw_transfer:raw_transfer_impl",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@nanobind",
        "@xla//xla/pjrt:status_casters",
    ],
)

# JAX test targets successfully relocated to api/jax/BUILD.
