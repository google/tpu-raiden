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

#!/bin/bash

set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Point to the directory containing the compiled raw_transfer.so and source files
# We also include the workspace parent dir to map absolute 'google3.third_party...' python imports!
export PYTHONPATH="${WORKSPACE_DIR}:${WORKSPACE_DIR}/bazel-bin:${PYTHONPATH}"


# Change to the tests directory to avoid Python's local directory import shadowing
cd "${WORKSPACE_DIR}/raiden_lib/raw_transfer/jax"

echo "=== Running: raw_transfer_test.py ==="
python raw_transfer_test.py 2>&1 | tee "${WORKSPACE_DIR}/test.log"
