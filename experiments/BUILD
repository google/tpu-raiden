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

load("@rules_python//python:defs.bzl", "py_binary", "py_test")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "sources",
    srcs = glob(["**"]),
    visibility = ["//:internal"],
)

py_binary(
    name = "reshard_stage1_e2e",
    srcs = ["reshard_stage1_e2e.py"],
    main = "reshard_stage1_e2e.py",
    deps = [
        "//tpu_raiden/experimental:kv_cache_manager_host_py",
        "//tpu_raiden/rpc:raiden_service_py_pb2",
    ],
)

py_test(
    name = "reshard_stage1_local_primitive_test",
    srcs = ["reshard_stage1_local_primitive_test.py"],
    main = "reshard_stage1_local_primitive_test.py",
    tags = [
        "cpu:4",
        "notap",
    ],
    deps = [
        "//tpu_raiden/experimental:kv_cache_manager_host_py",
        "//tpu_raiden/rpc:raiden_service_py_pb2",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)
