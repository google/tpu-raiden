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

load("@rules_cc//cc:defs.bzl", "cc_library")
load("//tools/build_defs/testing:bzl_library.bzl", "bzl_library")
load(":header_only_cc_info.bzl", "header_only_cc_info")

package(default_visibility = ["//visibility:public"])

header_only_cc_info(
    name = "tsl_platform_headers",
    deps = [
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "raiden_manager_base",
    srcs = [
        "raiden_manager_base.cc",
    ],
    hdrs = [
        "raiden_manager_base.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":tsl_platform_headers",
        "//transport:block_transport",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = 1,
)

header_only_cc_info(
    name = "xla_raw_transfer_headers",
    deps = [
        "@xla//xla:future",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:common_pjrt_client",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_layout",
        "@xla//xla/pjrt:raw_buffer",
        "@xla//xla/pjrt:status_casters",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_extension_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_external",
        "@xla//xla/pjrt/c_api_client:pjrt_c_api_client",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/stream_executor:stream",
        "@xla//xla/tsl/concurrency:ref_count",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "host_memory_allocator",
    srcs = [
        "host_memory_allocator.cc",
    ],
    hdrs = [
        "host_memory_allocator.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":xla_raw_transfer_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@xla//xla/pjrt:common_pjrt_client",
        "@xla//xla/pjrt:host_memory_allocator",
        "@xla//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "raw_transfer_core",
    hdrs = [
        "raw_transfer_core.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
        "-Wno-unused-variable",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":xla_raw_transfer_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "raw_transfer_impl",
    hdrs = [
        "raw_transfer_impl.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
        "-Wno-unused-variable",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":raw_transfer_core",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@xla//xla:future",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:raw_buffer",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_extension_hdrs",
        "@xla//xla/pjrt/c_api_client:pjrt_c_api_client",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "transfer_engine_base",
    srcs = [
        "transfer_engine_base.cc",
    ],
    hdrs = [
        "transfer_engine_base.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":raw_transfer_core",
        "//kv_cache:kv_cache_manager_base",
        "//transport:socket_transport",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

bzl_library(
    name = "header_only_cc_info_bzl",
    srcs = ["header_only_cc_info.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
    deps = ["//third_party/bazel_rules/rules_cc/cc/common"],
)
