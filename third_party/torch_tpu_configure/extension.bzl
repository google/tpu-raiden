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

"""Module extension to import TPU Raiden dependencies."""

load("../torch_tpu_headers:repository.bzl", "torch_tpu_headers_repository")
load(":repository.bzl", "torch_tpu_configure")

def _raiden_imports_impl(_module_ctx):
    torch_tpu_configure(name = "torch_tpu")
    torch_tpu_headers_repository(
        name = "torch_tpu_headers",
        path = "../torch_tpu/torch_tpu",
    )

raiden_imports = module_extension(
    implementation = _raiden_imports_impl,
)
