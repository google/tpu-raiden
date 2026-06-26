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

# #3 dual-XLA jax-fork: build a REPLACEMENT libjax_common.so that aggregates jaxlib's
# entire common-lib code (:_jax_common_split, which carries the single XLA) PLUS the
# raiden engine code. Because both sides depend on the same @xla cc_libraries, bazel
# links ONE copy of XLA -> no dual-XLA double-free. Installed over jaxlib's
# libjax_common.so, the existing jax extension stubs AND the engine stub both forward
# their Wrapped_PyInit_* into this one lib, sharing the single XLA. No LD_PRELOAD.

load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

# Combined common library: jaxlib's XLA + raiden engine, one XLA copy.
cc_binary(
    name = "libjax_common.so",
    linkshared = True,
    additional_linker_inputs = [":libjax_common.lds"],
    linkopts = [
        "-Wl,--version-script=$(location :libjax_common.lds)",
    ],
    deps = [
        "@pybind11//:pybind11",
        "@jax//jaxlib:_jax_common_split",
        "//tpu_raiden/frameworks/jax:engine_common",
    ],
)

# Engine extension stub: tiny .so that forwards PyInit__tpu_raiden_jax to the
# Wrapped_PyInit symbol inside the combined libjax_common.so.
cc_binary(
    name = "_tpu_raiden_jax.so",
    srcs = ["engine_stub.c"],
    linkshared = True,
    deps = [":libjax_common.so"],
)
