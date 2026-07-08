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

# Window sweep: measure EVERY commit in the last DAYS vs its parent on one
# machine. sweep_window ships measure.py / BUILD.inject as runfiles data and
# injects them into each checked-out commit; the per-commit build happens inside
# with an isolated --output_base.
load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "sweep_window",
    testonly = True,
    srcs = ["sweep_window.py"],
    data = [
        "measure.py",
        "BUILD.inject",
    ],
)
