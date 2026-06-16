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

package(default_visibility = ["//visibility:public"])

pytype_strict_binary(
    name = "perf_test_runner",
    testonly = True,
    srcs = ["perf_test_runner.py"],
    deps = [
        "//net/grpc/python:use_insecure_channel_for_test",
        "//rpc:coordination_helper",
        "//third_party/py/absl:app",
        "//tpu_raiden/frameworks/jax:_tpu_raiden_jax",
        "@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio",
        "@com_google_absl_py//absl/flags",
        "@jax//jax",
        "@pypi//numpy",
    ],
)
