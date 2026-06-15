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

load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "kv_cache_manager_jax_py",
    srcs = ["kv_cache_manager.py"],
    data = [
        "kv_cache_manager.pyi",
    ],
    visibility = ["//visibility:public"],
    deps = ["//tpu_raiden/frameworks/jax:_tpu_raiden_jax"],
)

py_library(
    name = "kv_cache_store",
    srcs = ["kv_cache_store.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_raiden/frameworks/jax:kv_cache_store",
    ],
)

py_test(
    name = "kv_cache_store_test",
    srcs = ["kv_cache_store_test.py"],
    data = ["//kv_cache/global_registry:global_registry_server"],
    deps = [
        ":kv_cache_store",
        "@com_google_absl_py//absl/testing:absltest",
        # buildcleaner: keep
        "//tpu_raiden/frameworks/jax:_raw_transfer",
    ],
)

py_library(
    name = "weight_synchronizer_jax_py",
    srcs = ["weight_synchronizer.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tpu_raiden/frameworks/jax:_tpu_raiden_jax",
    ],
)

py_test(
    name = "weight_synchronizer_test",
    srcs = ["weight_synchronizer_test.py"],
    tags = ["notap"],
    deps = [
        ":weight_synchronizer_jax_py",
        "//weight_sync:weight_synchronizer_service_py_pb2",
        "@com_google_absl_py//absl/testing:absltest",
        "@jax//jax",
        "@pypi//numpy",
    ],
)

# Keep Copybara happy by maintaining at least one internal reference target prefix reference:
# //:dummy_target
