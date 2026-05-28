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

load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_grpc_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_python//python:defs.bzl", "py_library")
load("@rules_python//python:proto.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "coordination_proto",
    srcs = ["coordination.proto"],
    has_services = True,
)

py_proto_library(
    name = "coordination_py_pb2",
    has_services = True,
    deps = [":coordination_proto"],
)

py_grpc_library(
    name = "coordination_py_pb2_grpc",
    srcs = [":coordination_proto"],
    deps = [":coordination_py_pb2"],
)

py_library(
    name = "coordination_helper",
    testonly = True,
    srcs = ["coordination_helper.py"],
    deps = [
        ":coordination_py_pb2",
        ":coordination_py_pb2_grpc",
        "@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio",
    ],
)
