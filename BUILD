package(default_visibility = ["//visibility:public"])

exports_files([
    "LICENSE",
    "README.md",
])

config_setting(
    name = "jax_disabled",
    values = {"define": "with_jax=false"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "torch_disabled",
    values = {"define": "with_torch=false"},
    visibility = ["//visibility:public"],
)

py_binary(
    name = "bootstrap_python",
    srcs = ["bootstrap_python.py"],
)
