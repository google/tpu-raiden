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

"""Standalone Raiden transfer engine for TorchTPU KV-cache movement."""

import ctypes
import os
from pathlib import Path

import torch  # noqa: F401  # Load torch shared libraries before the extension.
import torch_tpu
from torch_tpu import _loader as _torch_tpu_loader


def _load_torch_tpu_common() -> None:
  _torch_tpu_loader.load()
  common = Path(torch_tpu.__file__).resolve().parent / "common"
  lib = common / "libpywrap_torch_tpu_common.so"
  if lib.exists():
    ctypes.CDLL(str(lib), mode=os.RTLD_GLOBAL | os.RTLD_NOW)


_load_torch_tpu_common()

from . import _raiden_transfer_engine as _impl

RaidenTransferEngine = _impl.RaidenTransferEngine

__all__ = [
    "RaidenTransferEngine",
]
