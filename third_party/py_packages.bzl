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

"""Repository rules and module extensions to import and map pre-compiled wheels (JAX, PyTorch, NumPy, XLA).

This file discovers installed Python packages in the environment (e.g., jaxlib, torch,
numpy) and exposes their C++ headers and shared libraries to Bazel. It also downloads
and patches XLA and JAX headers dynamically based on the installed package versions.
"""

# ==============================================================================
# 0. Helper to get the last non-empty line (Verifier patch)
# ==============================================================================
def _get_last_line(stdout):
    lines = stdout.split("\n")
    last_line = ""
    for line in lines:
        stripped = line.strip()
        if stripped:
            last_line = stripped
    return last_line

# ==============================================================================
# 1. Local Python Packages Repository Rule (Symlinks Wheels)
# ==============================================================================
def _local_python_packages_impl(repository_ctx):
    python_bin = repository_ctx.os.environ.get("PYTHON_BIN_PATH", "python3")

    # Discover jaxlib installation directory
    result = repository_ctx.execute([python_bin, "-c", "import jaxlib; import os; print(os.path.dirname(jaxlib.__file__))"])
    if result.return_code != 0:
        _generate_dummy_build(repository_ctx)
        return
    jaxlib_path = _get_last_line(result.stdout)

    # Discover JAX C++ extension filename (handles renaming in newer JAX versions)
    result = repository_ctx.execute([python_bin, "-c", """
import os
path = "{}"
if os.path.exists(os.path.join(path, "_pywrap_xla.so")):
    print("_pywrap_xla.so")
elif os.path.exists(os.path.join(path, "_jax.so")):
    print("_jax.so")
else:
    print("_pywrap_xla.so") # default fallback
""".format(jaxlib_path)])
    jax_ext = _get_last_line(result.stdout) if result.return_code == 0 else "_pywrap_xla.so"

    # Discover torch installation directory
    result = repository_ctx.execute([python_bin, "-c", "import torch; import os; print(os.path.dirname(torch.__file__))"])
    if result.return_code != 0:
        _generate_dummy_build(repository_ctx)
        return
    torch_path = _get_last_line(result.stdout)

    # Discover numpy include directory
    result = repository_ctx.execute([python_bin, "-c", "import numpy; print(numpy.get_include())"])
    if result.return_code != 0:
        _generate_dummy_build(repository_ctx)
        return
    numpy_include_path = _get_last_line(result.stdout)

    # Symlink the directories into the external repository
    repository_ctx.symlink(jaxlib_path, "jaxlib_src")
    repository_ctx.symlink(torch_path, "torch_src")
    repository_ctx.symlink(numpy_include_path, "numpy_include")

    # Recreate the torch/headeronly directory structure as file symlinks to bypass Bazel glob limitations (inserted by Verifier)
    repository_ctx.execute(["mkdir", "-p", "torch/headeronly"])

    # 1. First, copy the real torch/headeronly headers if they exist in the installation
    real_headeronly = torch_path + "/include/torch/headeronly"
    if repository_ctx.path(real_headeronly).exists:
        repository_ctx.execute([
            "bash",
            "-c",
            "cp -rs {real_headeronly}/* torch/headeronly/".format(real_headeronly = real_headeronly),
        ])

    # 2. Then, copy c10 headers with --no-clobber to fill in the missing ones (like Layout.h) without creating cycles
    result = repository_ctx.execute([
        "bash",
        "-c",
        "cp -rsn {torch_path}/include/c10/* torch/headeronly/".format(torch_path = torch_path),
    ])
    if result.return_code != 0:
        fail("Failed to create PyTorch compatibility symlinks: " + result.stderr)

    # Generate the real BUILD file exposing the headers and libraries
    build_content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "jaxlib_headers",
    hdrs = glob(["jaxlib_src/include/**/*.h"]),
    strip_include_prefix = "jaxlib_src/include",
)

cc_import(
    name = "pywrap_xla",
    shared_library = "jaxlib_src/%JAX_EXT%",
)

cc_library(
    name = "torch_headers",
    hdrs = glob([
        "torch_src/include/**/*.h",
        "torch_src/include/**/*.hpp",
        "torch/headeronly/**/*.h",
        "torch/headeronly/**/*.hpp",
    ]),
    includes = [
        "torch_src/include",
        "torch_src/include/torch/csrc/api/include",
        ".",
    ],
)

cc_import(
    name = "libtorch",
    shared_library = "torch_src/lib/libtorch.so",
)

cc_import(
    name = "libtorch_cpu",
    shared_library = "torch_src/lib/libtorch_cpu.so",
)

cc_import(
    name = "libtorch_python",
    shared_library = "torch_src/lib/libtorch_python.so",
)

cc_library(
    name = "numpy_headers",
    hdrs = glob(["numpy_include/**/*.h"]),
    strip_include_prefix = "numpy_include",
)

cc_library(
    name = "jax_py_client",
    deps = [
        ":pywrap_xla",
        "@jax_headers//:headers",
    ],
)

cc_library(
    name = "aten_headers",
    deps = [":torch_headers"],
)
""".replace("%JAX_EXT%", jax_ext)
    repository_ctx.file("BUILD", build_content)

def _generate_dummy_build(repository_ctx):
    # NOTE: We stub all targets (including those that are cc_import in the real build)
    # as cc_library. This is because cc_import requires a physical shared library (.so)
    # file to exist at loading time, which would force us to create dummy .so files.
    # cc_library is completely self-contained and perfectly satisfies the loader phase.
    dummy_content = """
package(default_visibility = ["//visibility:public"])
cc_library(name = "jaxlib_headers")
cc_library(name = "pywrap_xla")
cc_library(name = "torch_headers")
cc_library(name = "libtorch")
cc_library(name = "libtorch_cpu")
cc_library(name = "libtorch_python")
cc_library(name = "numpy_headers")
cc_library(name = "jax_py_client")
cc_library(name = "aten_headers")
"""
    repository_ctx.file("BUILD", dummy_content)

local_python_packages = repository_rule(
    implementation = _local_python_packages_impl,
    environ = ["PYTHON_BIN_PATH"],
)

# ==============================================================================
# 2. Dynamic XLA Headers Repository Rule (Resolves, Downloads & Patches XLA)
# ==============================================================================
def _xla_headers_impl(repository_ctx):
    python_bin = repository_ctx.os.environ.get("PYTHON_BIN_PATH", "python3")

    # A. Get JAX Git Hash from the installed wheel
    result = repository_ctx.execute([
        python_bin,
        "-c",
        "try:\n  import jaxlib.version as v\n  print(v._git_hash if v._git_hash else v.__version__)\nexcept:\n  print('ERROR')",
    ])
    output = _get_last_line(result.stdout)
    if result.return_code != 0 or output == "ERROR":
        # Bootstrap Fallback: If JAX is not installed yet, generate dummy headers target
        _generate_dummy_headers(repository_ctx)
        return

    is_commit = len(output) == 40  # Standard git commit hash length

    # B. Fetch JAX's MODULE.bazel from GitHub
    if is_commit:
        url = "https://raw.githubusercontent.com/google/jax/{}/MODULE.bazel".format(output)
    else:
        # Fallback to release tag
        url = "https://raw.githubusercontent.com/google/jax/jaxlib-v{}/MODULE.bazel".format(output)

    # Download the file, allowing failure to prevent crashing the Bazel loader
    download_res = repository_ctx.download(url = url, output = "JAX_MODULE.bazel", allow_fail = True)

    # C. Parse JAX's MODULE.bazel for the XLA commit pin
    xla_commit = None
    if download_res.success and repository_ctx.path("JAX_MODULE.bazel").exists:
        module_content = repository_ctx.read("JAX_MODULE.bazel")
        xla_commit = _extract_xla_commit(module_content)

    if not xla_commit:
        # Fallback to a safe default commit matching our JAX baseline if parsing/download fails.
        # NOTE: Update this baseline commit whenever upgrading the primary JAX version we test against.
        xla_commit = "a98e42b8b195e578a97025654f4e7b9b3fb5001a"

    # D. Download and extract XLA source at the exact commit
    xla_url = "https://github.com/openxla/xla/archive/{}.tar.gz".format(xla_commit)
    repository_ctx.download_and_extract(
        url = xla_url,
        stripPrefix = "xla-{}".format(xla_commit),
    )

    # Delete all BUILD/BUILD.bazel files in the extracted source to merge everything
    # into a single root package. This avoids package boundary issues with glob
    # and loading errors from XLA's own BUILD files (e.g. rules_license).
    repository_ctx.execute(["find", ".", "(", "-name", "BUILD", "-o", "-name", "BUILD.bazel", ")", "-delete"])

    # E. Apply local compiler compatibility patches
    # The patches reside in the main workspace and are passed via label references
    patches = [
        repository_ctx.path(Label("//:third_party/xla/future_sfinae.patch")),
        repository_ctx.path(Label("//:third_party/xla/attribute_map_static_assert.patch")),
        repository_ctx.path(Label("//:third_party/xla/attribute_map_cc_static_assert.patch")),
    ]
    for patch in patches:
        repository_ctx.patch(patch, strip = 1)

    # F. Generate the BUILD file exposing the headers
    build_content = """
package(default_visibility = ["//visibility:public"])

proto_library(
    name = "tsl_protos",
    srcs = glob(["xla/tsl/protobuf/*.proto"]),
    deps = [
        "@com_google_protobuf//:any_proto",
        "@com_google_protobuf//:wrappers_proto",
    ],
)

cc_proto_library(
    name = "tsl_cc_protos",
    deps = [":tsl_protos"],
)

proto_library(
    name = "xla_protos",
    srcs = glob(
        ["xla/**/*.proto"],
        exclude = ["xla/tsl/**/*.proto"],
    ),
    deps = [
        ":tsl_protos",
        "@com_google_protobuf//:any_proto",
        "@com_google_protobuf//:duration_proto",
        "@com_google_protobuf//:timestamp_proto",
        "@com_google_protobuf//:wrappers_proto",
    ],
)

cc_proto_library(
    name = "xla_cc_protos",
    deps = [":xla_protos"],
)

cc_library(
    name = "headers",
    hdrs = glob(["xla/**/*.h"]),
    includes = [
        ".",
        "xla",
        "third_party/tsl",
    ],
    deps = [
        ":tsl_cc_protos",
        ":xla_cc_protos",
        "@com_google_protobuf//:protobuf",
        "@eigen_archive//:eigen3",
        "@ml_dtypes_py//ml_dtypes:float8",
        "@ml_dtypes_py//ml_dtypes:intn",
        "@ml_dtypes_py//ml_dtypes:mxfloat",
        "@highwayhash",
        "@highwayhash//:arch_specific",
        "@highwayhash//:hh_types",
        "@farmhash_archive//:farmhash",
        "@llvm-project//mlir:IR",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/utility",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_github_grpc_grpc//:grpc++",
        "@local_py_packages//:numpy_headers",
    ],
)
"""
    repository_ctx.file("BUILD", build_content)

def _extract_xla_commit(content):
    # Robust Starlark string parsing to extract the XLA commit from JAX's git_override or archive_override
    if not content:
        return None

    # Find the "xla" override block
    xla_index = content.find('module_name = "xla"')
    if xla_index == -1:
        xla_index = content.find('name = "xla"')
    if xla_index == -1:
        return None

    # Search for the next "commit" or "strip_prefix" field in the block
    commit_index = content.find('commit = "', xla_index)
    if commit_index == -1:
        # Fallback to strip_prefix if they use archive_override
        commit_index = content.find('strip_prefix = "xla-', xla_index)
        if commit_index == -1:
            return None
        start = commit_index + len('strip_prefix = "xla-')
    else:
        start = commit_index + len('commit = "')

    end = content.find('"', start)
    if end == -1:
        return None
    return content[start:end]

def _generate_dummy_headers(repository_ctx):
    repository_ctx.file("BUILD", """
package(default_visibility = ["//visibility:public"])
cc_library(name = "headers")
""")

xla_headers = repository_rule(
    implementation = _xla_headers_impl,
    environ = ["PYTHON_BIN_PATH"],
)

def _generate_dummy_jax_headers(repository_ctx):
    repository_ctx.file("BUILD", """
package(default_visibility = ["//visibility:public"])
cc_library(name = "headers")
""")

def _jax_headers_impl(repository_ctx):
    python_bin = repository_ctx.os.environ.get("PYTHON_BIN_PATH", "python3")

    # A. Get JAX Git Hash from the installed wheel
    result = repository_ctx.execute([
        python_bin,
        "-c",
        "try:\n  import jaxlib.version as v\n  print(v._git_hash if hasattr(v, '_git_hash') and v._git_hash else v.__version__)\nexcept:\n  print('ERROR')",
    ])
    output = _get_last_line(result.stdout)
    if result.return_code != 0 or output == "ERROR":
        _generate_dummy_jax_headers(repository_ctx)
        return

    commit = output
    is_commit = len(commit) == 40

    # B. Download and extract JAX source
    if is_commit:
        url = "https://github.com/google/jax/archive/{}.tar.gz".format(commit)
        strip_prefix = "jax-{}".format(commit)
    else:
        url = "https://github.com/google/jax/archive/jaxlib-v{}.tar.gz".format(commit)
        strip_prefix = "jax-jaxlib-v{}".format(commit)

    repository_ctx.download_and_extract(
        url = url,
        stripPrefix = strip_prefix,
    )

    # Delete all BUILD/BUILD.bazel files in the extracted source to avoid loading errors from JAX's own BUILD files.
    repository_ctx.execute(["find", ".", "(", "-name", "BUILD", "-o", "-name", "BUILD.bazel", ")", "-delete"])

    build_content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "headers",
    hdrs = glob([
        "jaxlib/**/*.h",
        "jaxlib/**/*.hpp",
    ]),
    includes = ["."],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/log:check",
        "@nanobind",
        "@rules_python//python/cc:current_py_cc_headers",
        "@xla_headers//:headers",
    ],
)
"""
    repository_ctx.file("BUILD", build_content)

jax_headers = repository_rule(
    implementation = _jax_headers_impl,
    environ = ["PYTHON_BIN_PATH"],
)

# ==============================================================================
# 3. Module Extension (Exposes Both Repositories to Bzlmod)
# ==============================================================================
def local_py_packages_extension_impl(_module_ctx):
    local_python_packages(name = "local_py_packages")
    xla_headers(name = "xla_headers")
    jax_headers(name = "jax_headers")

local_py_packages_extension = module_extension(
    implementation = local_py_packages_extension_impl,
)
