#!/bin/bash
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

set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Workspace root on PYTHONPATH so `api.jax._kv_cache_manager` resolves to the
# compiled .so copied next to the source by build_raw_transfer.sh.
export PYTHONPATH="${WORKSPACE_DIR}:${PYTHONPATH}"

cd "${WORKSPACE_DIR}"

echo "=== Running: test_disagg_imports.py ==="
python test_disagg_imports.py 2>&1 | tee "${WORKSPACE_DIR}/disagg_imports.log"
