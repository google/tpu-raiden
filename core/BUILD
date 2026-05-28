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

package(default_visibility = ["//visibility:public"])

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
        "//transport:block_transport",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@xla//xla:future",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:common_pjrt_client",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:raw_buffer",
        "@xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_extension_hdrs",
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_external",
        "@xla//xla/pjrt/c_api_client:pjrt_c_api_client",
        "@xla//xla/tsl/concurrency:ref_count",
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
