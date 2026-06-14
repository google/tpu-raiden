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

"""Single-source the wheel version from pyproject.toml.

Mirrors torch_tpu/bazel/wheel_version.bzl: reads the base `version` from the
[project] section of pyproject.toml and appends the WHEEL_VERSION_EXTRAS env var
(e.g. ".dev20260613120000") so CI can stamp dated dev builds.
"""

def _raiden_version_repo_impl(ctx):
    pyproject_path = ctx.path(ctx.attr.pyproject_toml)
    content = ctx.read(pyproject_path)
    version = None

    in_project_section = False
    for line in content.splitlines():
        line = line.strip()

        # Skip empty lines and comments.
        if not line or line.startswith("#"):
            continue

        # Track the current TOML table/section.
        if line.startswith("[") and line.endswith("]"):
            in_project_section = line == "[project]"
            continue

        # Only parse 'version' while inside the '[project]' section.
        if in_project_section and line.startswith("version"):
            parts = line.split("=", 1)
            if len(parts) == 2 and parts[0].strip() == "version":
                version = parts[1].strip().strip('"').strip("'")
                break

    if not version:
        fail("Version not found in pyproject.toml [project] section")

    suffix = ctx.os.environ.get("WHEEL_VERSION_EXTRAS", "")
    full_version = version + suffix

    ctx.file("BUILD.bazel", "")
    ctx.file("version.bzl", "WHEEL_VERSION = '{}'\n".format(full_version))

raiden_version_repo = repository_rule(
    implementation = _raiden_version_repo_impl,
    attrs = {
        "pyproject_toml": attr.label(mandatory = True, allow_single_file = True),
    },
    environ = ["WHEEL_VERSION_EXTRAS"],
)
