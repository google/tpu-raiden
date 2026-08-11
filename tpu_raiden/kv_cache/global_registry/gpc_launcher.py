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

"""Console-script launchers for the wheel-bundled GPC control-plane binaries.

The tpu_raiden_torch wheel ships `global_registry_server` and
`raiden_orchestrator_main` as package data next to their sources, so a pip
install of the wheel is enough to run the global-prefix-cache control plane —
no separately distributed bazel-bin artifacts, and the binaries are always the
exact build the installed client library was released with. The wheel's
console scripts (`global_registry_server`, `raiden_orchestrator_main`)
resolve the bundled binary and exec it with argv passed through unchanged.
"""

import os
import pathlib
import shutil
import stat
import sys
import tempfile

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]

GLOBAL_REGISTRY_SERVER = (
    _PACKAGE_ROOT / "kv_cache" / "global_registry" / "global_registry_server"
)
RAIDEN_ORCHESTRATOR = (
    _PACKAGE_ROOT / "core" / "controller" / "raiden_orchestrator_main"
)


def _executable_path(binary: pathlib.Path) -> str:
  """Return an executable path for a bundled binary.

  Wheel extraction does not reliably preserve the execute bit, and
  site-packages may be read-only, so try in order: already executable,
  chmod in place, copy to a temp dir and chmod there.
  """
  if not binary.is_file():
    raise SystemExit(
        f"{binary.name} is not bundled in this tpu_raiden installation "
        f"(looked at {binary}). Rebuild/upgrade the tpu_raiden_torch "
        "wheel; older releases did not ship the GPC control-plane "
        "binaries."
    )
  if os.access(binary, os.X_OK):
    return str(binary)
  exec_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
  try:
    binary.chmod(binary.stat().st_mode | exec_bits)
    if os.access(binary, os.X_OK):
      return str(binary)
  except OSError:
    pass
  staged = (
      pathlib.Path(tempfile.mkdtemp(prefix="tpu_raiden_bin.")) / binary.name
  )
  shutil.copy2(binary, staged)
  staged.chmod(staged.stat().st_mode | exec_bits)
  return str(staged)


def _exec(binary: pathlib.Path) -> None:
  path = _executable_path(binary)
  os.execv(path, [path, *sys.argv[1:]])


def global_registry_main() -> None:
  _exec(GLOBAL_REGISTRY_SERVER)


def orchestrator_main() -> None:
  _exec(RAIDEN_ORCHESTRATOR)
