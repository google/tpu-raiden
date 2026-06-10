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
    visibility = ["//visibility:public"],
    deps = ["//frameworks/jax:_kv_cache_manager"],
)

py_library(
    name = "kv_cache_store",
    srcs = ["kv_cache_store.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//frameworks/jax:kv_cache_store",
    ],
)

py_test(
    name = "kv_cache_store_test",
    srcs = ["kv_cache_store_test.py"],
    data = ["//kv_cache/global_registry:global_registry_server"],
    deps = [
        ":kv_cache_manager_jax_py",
        ":kv_cache_store",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jax//jax",
        "@pypi//numpy",
        # buildcleaner: keep
        ":raw_transfer",
    ],
)

py_library(
    name = "kv_cache_manager_ffi_py",
    srcs = ["kv_cache_manager_ffi.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//frameworks/jax:_kv_cache_manager_ffi",
        "@jax//jax",
        "@jax//jax/experimental:compute_on",
        "@pypi//numpy",
    ],
)

py_library(
    name = "weight_synchronizer_ffi_py",
    srcs = ["weight_synchronizer_ffi.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//frameworks/jax:_weight_synchronizer_ffi",
        "@jax//jax",
        "@jax//jax/experimental:compute_on",
        "@pypi//numpy",
    ],
)

py_test(
    name = "kv_cache_manager_test_cpu",
    timeout = "long",
    srcs = ["kv_cache_manager_test.py"],
    args = ["--device_type=cpu"],
    main = "kv_cache_manager_test.py",
    tags = [
        "cpu:4",
        "notap",
    ],
    deps = [
        ":kv_cache_manager_jax_py",
        "@com_google_absl_py//absl/flags",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jax//jax",
        "@pypi//numpy",
    ],
)

py_library(
    name = "resharding_planner",
    srcs = ["resharding_planner.py"],
    visibility = ["//visibility:public"],
    deps = ["@jax//jax"],
)

py_library(
    name = "resharding_engine",
    srcs = ["resharding_engine.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":resharding_planner",
        ":weight_synchronizer_ffi_py",
        "//rpc:coordination_helper",
        "@jax//jax",
        "@pypi//numpy",
    ],
)

py_library(
    name = "resharding_test_lib",
    srcs = ["resharding_test_lib.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":resharding_planner",
        "@jax//jax",
        "@pypi//numpy",
    ],
)

py_library(
    name = "weight_synchronizer_jax_py",
    srcs = ["weight_synchronizer.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//frameworks/jax:_weight_synchronizer",
    ],
)

py_test(
    name = "weight_synchronizer_test",
    srcs = ["weight_synchronizer_test.py"],
    deps = [
        ":weight_synchronizer_jax_py",
        "@com_google_absl_py//absl/testing:absltest",
        "@jax//jax",
        "@pypi//numpy",
    ],
)

py_test(
    name = "resharding_planner_test",
    srcs = ["resharding_planner_test.py"],
    deps = [
        ":resharding_planner",
        "@com_google_absl_py//absl/testing:absltest",
        "@jax//jax",
        "@pypi//numpy",
    ],
)

py_test(
    name = "resharding_test",
    srcs = ["resharding_test.py"],
    deps = [
        ":resharding_engine",
        ":resharding_test_lib",
        "@com_google_absl_py//absl/testing:absltest",
        "@jax//jax",
        "@pypi//numpy",
    ],
)

py_library(
    name = "transfer_engine_jax_py",
    srcs = ["transfer_engine.py"],
    data = [
        "transfer_engine.pyi",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//frameworks/jax:_transfer_engine",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "@com_google_absl_py//absl/flags",
        "@jax//jax",
        "@pypi//numpy",
    ],
)

py_library(
    name = "raw_transfer",
    srcs = [
        "raw_transfer.py",
    ],
    deps = [
        "//frameworks/jax:_raw_transfer",
    ],
)

# Keep Copybara happy by maintaining at least one internal reference target prefix reference:
# //:dummy_target
