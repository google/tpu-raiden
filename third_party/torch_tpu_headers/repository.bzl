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

"""TODO: sixiang - Write module docstring."""

def _torch_tpu_headers_repository_impl(repository_ctx):
    workspace_file = str(repository_ctx.path(repository_ctx.attr.workspace_file))
    workspace_root = workspace_file[:workspace_file.rfind("/")]
    source = repository_ctx.path(workspace_root + "/" + repository_ctx.attr.path)
    result = repository_ctx.execute(
        [
            "bash",
            "-c",
            "cd \"$1\" && find . -type f -name '*.h' | sort",
            "find_torch_tpu_headers",
            str(source),
        ],
        timeout = 60,
    )
    if result.return_code != 0:
        fail("failed to enumerate TorchTPU headers: " + result.stderr)

    for line in result.stdout.splitlines():
        if not line:
            continue
        relative = line[2:] if line.startswith("./") else line
        repository_ctx.symlink(source.get_child(relative), relative)

    repository_ctx.file(
        "BUILD.bazel",
        """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "headers",
    hdrs = glob(["**/*.h"]),
    strip_include_prefix = ".",
    include_prefix = "torch_tpu",
)
""",
    )

torch_tpu_headers_repository = repository_rule(
    implementation = _torch_tpu_headers_repository_impl,
    attrs = {
        "path": attr.string(mandatory = True),
        "workspace_file": attr.label(
            allow_single_file = True,
            default = Label("//:MODULE.bazel"),
        ),
    },
    local = True,
)
