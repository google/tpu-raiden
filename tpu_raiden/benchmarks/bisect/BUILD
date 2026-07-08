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

# Perf-regression localization workloads. Both ship measure.py / BUILD.inject /
# run_one.sh as runfiles data and inject them into each checked-out commit; the
# per-commit build happens inside with an isolated --output_base. py_binary (no
# rules_shell dependency).
load("@rules_python//python:defs.bzl", "py_binary")

# Path A: confirm ONE suspect commit (parent vs suspect), one click in the UI.
py_binary(
    name = "confirm_commit",
    testonly = True,
    srcs = ["confirm_commit.py"],
    data = [
        "measure.py",
        "BUILD.inject",
    ],
)

# Path B: automated full-range git-bisect.
py_binary(
    name = "run_bisect",
    testonly = True,
    srcs = ["run_bisect.py"],
    data = [
        "measure.py",
        "run_one.sh",
        "BUILD.inject",
    ],
)
