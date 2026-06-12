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

import argparse
import glob
import os
import subprocess
import sys
import venv


def main():
  parser = argparse.ArgumentParser(
      description="Setup virtualenv and install wheels."
  )
  parser.add_argument(
      "--venv_dir", required=True, help="Path to the virtualenv directory"
  )
  parser.add_argument(
      "--wheels_dir",
      required=True,
      help="Path to the directory containing wheels",
  )
  args = parser.parse_args()

  venv_dir = os.path.abspath(args.venv_dir)
  wheels_dir = os.path.abspath(args.wheels_dir)

  print(f"Creating virtualenv at {venv_dir}...")
  # venv.create uses the running python interpreter to create the venv.
  # Under bazel, sys.executable should point to the hermetic python interpreter
  # or the python interpreter configured for the build.
  print(f"Running interpreter: {sys.executable}")

  # Ensure parent directory of venv_dir exists
  os.makedirs(os.path.dirname(venv_dir), exist_ok=True)

  venv.create(venv_dir, with_pip=True)

  if os.name == "nt":
    venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
    venv_pip = os.path.join(venv_dir, "Scripts", "pip.exe")
  else:
    venv_python = os.path.join(venv_dir, "bin", "python")
    venv_pip = os.path.join(venv_dir, "bin", "pip")

  print(f"Using venv python: {venv_python}")

  if not os.path.exists(venv_python):
    raise RuntimeError(f"Virtualenv python not found at {venv_python}")

  print("Upgrading pip in virtualenv...")
  subprocess.run(
      [venv_python, "-m", "pip", "install", "--upgrade", "pip"], check=True
  )

  if os.path.exists(wheels_dir):
    wheels = glob.glob(os.path.join(wheels_dir, "*.whl"))
    if not wheels:
      print(f"Warning: No wheels found in {wheels_dir}")
    else:
      print(f"Found wheels: {wheels}")
      print("Installing wheels...")
      subprocess.run([venv_pip, "install"] + wheels, check=True)
  else:
    print(
        f"Warning: Wheels directory {wheels_dir} does not exist. Skipping wheel"
        " installation."
    )

  print("Virtualenv setup complete!")


if __name__ == "__main__":
  main()
