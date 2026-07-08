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

# Peak-bandwidth roofline sweep. Depends only on the current checkout's
# perf_core, so BAP builds it directly (no historical-commit / partial-clone
# issues).
load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "roofline_sweep",
    testonly = True,
    srcs = ["roofline_sweep.py"],
    deps = ["//tpu_raiden/benchmarks:perf_core"],
)

py_binary(
    name = "dist_sweep",
    testonly = True,
    srcs = ["dist_sweep.py"],
    deps = ["//tpu_raiden/benchmarks:perf_core"],
)
