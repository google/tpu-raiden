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

# third_party/tpu_raiden/kokoro/gcp_ubuntu/verify_imports.py
# Copyright 2026 Google LLC
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
"""Verify imports of all TPU Raiden C++ extensions on CPU without mocking."""

import importlib
import importlib.util
import os
import subprocess
import sys

# Resolve REPO_ROOT relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

JAX_EXTENSIONS = [
    ("_raw_transfer", []),
    ("_kv_cache_manager", ["KVCacheManager"]),
    ("_kv_cache_manager_ffi", []),
    ("_weight_synchronizer", []),
    ("_transfer_engine", []),
]

TORCH_EXTENSIONS = [
    ("_torch_raw_transfer", []),
    ("_kv_cache_manager", ["KVCacheManager"]),
    ("_weight_synchronizer", []),
    ("_transfer_engine", []),
]


def run_ldd(so_path):
  """Run ldd on the .so file and print output."""
  print(f"--- Running ldd on {so_path} ---")
  try:
    result = subprocess.run(
        ["ldd", so_path], capture_output=True, text=True, check=False
    )
    print(result.stdout)
    if result.stderr:
      print("ldd stderr:", result.stderr)
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"Failed to run ldd: {e}")


def verify_extension(path, module_name, attributes=None):
  """Load a C++ extension directly by path and verify its attributes."""
  print(f"Verifying extension '{module_name}' from path '{path}'...")
  if not os.path.exists(path):
    print(f"  ERROR: File not found at {path}")
    return False

  if attributes is None:
    attributes = []

  try:
    # Use a unique name in sys.modules to avoid collision between JAX and Torch
    unique_name = (
        f"verify_{os.path.basename(os.path.dirname(path))}_{module_name}"
    )

    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None:
      print(f"  ERROR: Failed to create spec for {module_name} from {path}")
      return False

    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    print(f"  Successfully imported '{module_name}' as '{unique_name}'")

    failed = False
    for attr in attributes:
      print(f"  Checking attribute '{attr}'...")
      if hasattr(mod, attr):
        print(f"    Found '{attr}'")
        # Force resolution of dynamic symbols
        _ = getattr(mod, attr)
      else:
        print(f"    ERROR: Attribute '{attr}' not found")
        failed = True

    return not failed

  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"  ERROR: Failed to load '{module_name}' from '{path}': {e}")
    if path.endswith(".so"):
      run_ldd(path)
    return False


def main():
  print("=== Starting 100% Mock-Free Path-Based Import Verification ===")
  print(f"Python version: {sys.version}")
  print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
  print(f"REPO_ROOT: {REPO_ROOT}")

  failed = False

  # 1. Verify JAX Extensions
  print("\n--- Verifying JAX Extensions ---")
  for ext_name, attrs in JAX_EXTENSIONS:
    so_path = os.path.join(REPO_ROOT, "frameworks", "jax", f"{ext_name}.so")
    if not verify_extension(so_path, ext_name, attrs):
      failed = True

  # 2. Verify Torch Extensions
  print("\n--- Verifying Torch Extensions ---")
  for ext_name, attrs in TORCH_EXTENSIONS:
    so_path = os.path.join(REPO_ROOT, "frameworks", "torch", f"{ext_name}.so")
    if not verify_extension(so_path, ext_name, attrs):
      failed = True

  # 3. Verify torch_tpu Python package
  print("\n--- Verifying torch_tpu Python package ---")
  try:
    # pylint: disable=g-import-not-at-top
    import torch_tpu  # pylint: disable=unused-import

    print("  Successfully imported 'torch_tpu'")
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"  ERROR: Failed to import 'torch_tpu': {e}")
    failed = True

  if failed:
    print("\n=== Import Verification FAILED! ===")
    sys.exit(1)
  else:
    print("\n=== Import Verification PASSED! ===")


if __name__ == "__main__":
  main()
