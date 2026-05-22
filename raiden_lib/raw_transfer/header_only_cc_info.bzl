def _header_only_cc_info_impl(ctx):
    merged = cc_common.merge_cc_infos(
        cc_infos = [dep[CcInfo] for dep in ctx.attr.deps],
    )
    return [
        CcInfo(
            compilation_context = merged.compilation_context,
            linking_context = cc_common.create_linking_context(
                linker_inputs = depset(),
            ),
        ),
    ]

header_only_cc_info = rule(
    attrs = {
        "deps": attr.label_list(
            providers = [CcInfo],
        ),
    },
    implementation = _header_only_cc_info_impl,
)
