# Copyright 2026 Google LLC
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

load("@jax//jaxlib:pywrap.bzl", "nanobind_pywrap_extension", "pywrap_library", "pywrap_binaries")

nanobind_pywrap_extension(
    name = "raw_transfer",
    srcs = [
        "raw_transfer.cc",
        "raw_transfer_impl.h",
    ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    deps = [
        "@jax//jaxlib:py_client",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@nanobind",
        "@xla//xla:future",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_layout",
        "@xla//xla/pjrt/c:pjrt_c_api_raw_buffer_external",
        "@xla//xla/pjrt/c_api_client:pjrt_c_api_client",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla:shape_util",
        "@xla//xla/tsl/concurrency:ref_count",
        "@xla//xla/tsl/platform:statusor",
    ],
)

pywrap_library(
    name = "raw_transfer_pywrap",
    deps = [":raw_transfer"],
)

pywrap_binaries(
    name = "raw_transfer_binaries",
    dep = ":raw_transfer_pywrap",
)
