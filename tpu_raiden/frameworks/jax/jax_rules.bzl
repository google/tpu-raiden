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

"""Build rules for TPU Raiden JAX framework integration.

This module provides wrappers around standard Bazel rules that dynamically
disable JAX targets when JAX support is disabled.
"""

load("@nanobind_bazel//:build_defs.bzl", nanobind_extension_rule = "nanobind_extension")
load("@rules_cc//cc:defs.bzl", cc_library_rule = "cc_library", cc_test_rule = "cc_test")
load("@rules_python//python:defs.bzl", py_binary_rule = "py_binary", py_library_rule = "py_library", py_test_rule = "py_test")

def raiden_jax_cc_library(name, srcs = [], hdrs = [], deps = [], **kwargs):
    cc_library_rule(
        name = name,
        srcs = select({
            "//:jax_disabled": [],
            "//conditions:default": srcs,
        }),
        hdrs = select({
            "//:jax_disabled": [],
            "//conditions:default": hdrs,
        }),
        deps = select({
            "//:jax_disabled": [],
            "//conditions:default": deps,
        }),
        **kwargs
    )

def raiden_jax_nanobind_extension(name, srcs = [], deps = [], **kwargs):
    nanobind_extension_rule(
        name = name,
        srcs = select({
            "//:jax_disabled": [],
            "//conditions:default": srcs,
        }),
        deps = select({
            "//:jax_disabled": [],
            "//conditions:default": deps,
        }),
        **kwargs
    )

def raiden_jax_cc_test(name, srcs = [], deps = [], **kwargs):
    dummy_main_generator = name + "_dummy_main_generator"
    native.genrule(
        name = dummy_main_generator,
        outs = [name + "_dummy_main.cc"],
        cmd = "echo 'int main() { return 0; }' > $@",
    )

    cc_test_rule(
        name = name,
        srcs = select({
            "//:jax_disabled": [":" + name + "_dummy_main.cc"],
            "//conditions:default": srcs,
        }),
        deps = select({
            "//:jax_disabled": [],
            "//conditions:default": deps,
        }),
        **kwargs
    )

def raiden_jax_py_library(name, srcs = [], deps = [], **kwargs):
    py_library_rule(
        name = name,
        srcs = select({
            "//:jax_disabled": [],
            "//conditions:default": srcs,
        }),
        deps = select({
            "//:jax_disabled": [],
            "//conditions:default": deps,
        }),
        **kwargs
    )

def raiden_jax_py_binary(name, srcs = [], deps = [], **kwargs):
    """Wrapper around py_binary that is disabled when JAX is disabled.

    Args:
        name: A unique name for this target.
        srcs: The list of source files.
        deps: The list of other targets that this library depends on.
        **kwargs: Additional arguments passed to the underlying py_binary.
    """
    dummy_main_generator = name + "_dummy_main_generator"
    dummy_main_file = name + "_dummy_main.py"
    native.genrule(
        name = dummy_main_generator,
        outs = [dummy_main_file],
        cmd = "echo '' > $@",
    )

    original_main = kwargs.pop("main", None)
    actual_main = original_main if original_main else (name + ".py")

    kwargs["main"] = select({
        "//:jax_disabled": ":" + dummy_main_file,
        "//conditions:default": actual_main,
    })

    py_binary_rule(
        name = name,
        srcs = select({
            "//:jax_disabled": [":" + dummy_main_file],
            "//conditions:default": srcs,
        }),
        deps = select({
            "//:jax_disabled": [],
            "//conditions:default": deps,
        }),
        **kwargs
    )

def raiden_jax_py_test(name, srcs = [], deps = [], **kwargs):
    """Wrapper around py_test that is disabled when JAX is disabled.

    Args:
        name: A unique name for this target.
        srcs: The list of source files.
        deps: The list of other targets that this library depends on.
        **kwargs: Additional arguments passed to the underlying py_test.
    """
    dummy_main_generator = name + "_dummy_main_generator"
    dummy_main_file = name + "_dummy_main.py"
    native.genrule(
        name = dummy_main_generator,
        outs = [dummy_main_file],
        cmd = "echo '' > $@",
    )

    original_main = kwargs.pop("main", None)
    actual_main = original_main if original_main else (name + ".py")

    kwargs["main"] = select({
        "//:jax_disabled": ":" + dummy_main_file,
        "//conditions:default": actual_main,
    })

    py_test_rule(
        name = name,
        srcs = select({
            "//:jax_disabled": [":" + dummy_main_file],
            "//conditions:default": srcs,
        }),
        deps = select({
            "//:jax_disabled": [],
            "//conditions:default": deps,
        }),
        **kwargs
    )

def raiden_jax_pytype_strict_binary(name, srcs = [], deps = [], **kwargs):
    """Wrapper around pytype_strict_binary that is disabled when JAX is disabled.

    Args:
        name: A unique name for this target.
        srcs: The list of source files.
        deps: The list of other targets that this library depends on.
        **kwargs: Additional arguments passed to the underlying pytype_strict_binary.
    """
    dummy_main_generator = name + "_dummy_main_generator"
    dummy_main_file = name + "_dummy_main.py"
    native.genrule(
        name = dummy_main_generator,
        outs = [dummy_main_file],
        cmd = "echo '' > $@",
    )

    original_main = kwargs.pop("main", None)
    actual_main = original_main if original_main else (name + ".py")

    kwargs["main"] = select({
        "//:jax_disabled": ":" + dummy_main_file,
        "//conditions:default": actual_main,
    })

    py_binary_rule(
        name = name,
        srcs = select({
            "//:jax_disabled": [":" + dummy_main_file],
            "//conditions:default": srcs,
        }),
        deps = select({
            "//:jax_disabled": [],
            "//conditions:default": deps,
        }),
        **kwargs
    )
