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

# Copyright 2026 The TPU Raiden Authors. All Rights Reserved.
# ==============================================================================

"""Repository rule to configure TPU Raiden dependencies."""

def _torch_tpu_configure_impl(repository_ctx):
    workspace_file = str(repository_ctx.path(repository_ctx.attr.workspace_file))
    workspace_root = workspace_file[:workspace_file.rfind("/")]

    enable_torch = repository_ctx.os.environ.get("WITH_TORCH", "1")
    if enable_torch == "0":
        _create_dummy_torch_tpu(repository_ctx)
        return

    sibling_torch_tpu = workspace_root + "/../torch_tpu"

    result = repository_ctx.execute(["test", "-d", sibling_torch_tpu])
    if result.return_code != 0:
        # Sibling checkout not found, use dummy
        _create_dummy_torch_tpu(repository_ctx)
        return

    # Sibling found, symlink its contents
    # We symlink top-level files/directories to make it look like the real repo
    # but we want to avoid symlinking WORKSPACE/MODULE.bazel if we want to override them,
    # though usually we just want to reuse them.
    # Actually, symlinking everything is fine if it already has BUILD files.

    # We need to find all top-level items in sibling_torch_tpu
    # and symlink them.
    # We can use find or ls.
    result = repository_ctx.execute([
        "bash",
        "-c",
        "cd \"$1\" && ls -A",
        "list_sibling_contents",
        sibling_torch_tpu,
    ])
    if result.return_code != 0:
        fail("failed to list sibling torch_tpu contents: " + result.stderr)

    for line in result.stdout.splitlines():
        if not line:
            continue

        # Avoid symlinking WORKSPACE or MODULE.bazel if we want to be safe,
        # but Bzlmod might need them if it is resolved as a module.
        # Actually, since we are defining this repo via extension, Bzlmod
        # treats it as a repo, not a module, so WORKSPACE/MODULE.bazel are ignored
        # in the symlinked directory (Bazel generated WORKSPACE is used).
        repository_ctx.symlink(sibling_torch_tpu + "/" + line, line)

def _create_dummy_torch_tpu(repository_ctx):
    repository_ctx.file("WORKSPACE", "")
    repository_ctx.file("BUILD.bazel", """
package(default_visibility = ["//visibility:public"])
cc_library(
    name = "torch_tpu",
)
""")
    repository_ctx.file("eager/BUILD.bazel", """
package(default_visibility = ["//visibility:public"])
cc_library(name = "device_buffer")
cc_library(name = "materialize")
cc_library(name = "tensor_to_buffer")
""")

torch_tpu_configure = repository_rule(
    implementation = _torch_tpu_configure_impl,
    attrs = {
        "workspace_file": attr.label(
            allow_single_file = True,
            default = Label("//:MODULE.bazel"),
        ),
    },
    environ = ["WITH_TORCH"],
    local = True,
)
