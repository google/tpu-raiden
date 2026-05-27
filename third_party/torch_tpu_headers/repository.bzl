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
