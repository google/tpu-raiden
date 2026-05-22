"""Repository rule for exposing headers from a local source tree."""

def _local_headers_repository_impl(repository_ctx):
    source_root = repository_ctx.path(repository_ctx.attr.path)
    include_root = repository_ctx.attr.include_root
    find_cmd = ("cd {source_root} && " +
                "find {include_root} -type f \\( -name '*.h' -o -name '*.hpp' \\)").format(
        source_root = source_root,
        include_root = include_root
    )
    result = repository_ctx.execute(["bash", "-c", find_cmd], quiet = True)
    if result.return_code != 0:
        fail("failed to list headers under {}: {}".format(source_root, result.stderr))

    dirs = {}
    for rel_path in result.stdout.splitlines():
        if "/" in rel_path:
            dirs[rel_path.rsplit("/", 1)[0]] = True

    for dirname in sorted(dirs):
        mkdir_result = repository_ctx.execute(["mkdir", "-p", dirname], quiet = True)
        if mkdir_result.return_code != 0:
            fail("failed to create {}: {}".format(dirname, mkdir_result.stderr))

    for rel_path in result.stdout.splitlines():
        repository_ctx.symlink("{}/{}".format(source_root, rel_path), rel_path)

    repository_ctx.file(
        "BUILD.bazel",
        """
cc_library(
    name = "headers",
    hdrs = glob([
        "{include_root}/**/*.h",
        "{include_root}/**/*.hpp",
    ]),
    includes = ["."],
    visibility = ["//visibility:public"],
)
""".format(include_root = include_root)
    )

local_headers_repository = repository_rule(
    implementation = _local_headers_repository_impl,
    attrs = {
        "include_root": attr.string(default = "."),
        "path": attr.string(mandatory = True),
    }
)
