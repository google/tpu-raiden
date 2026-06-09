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

"""Preload the Raiden transfer engine at interpreter startup -- OPT-IN.

`sitecustomize` is imported automatically by CPython before any user code. In
the vllm worker we must dlopen `_transfer_engine.so` FIRST, before
tpu_inference's runner pulls in torch_xla/libtpu (loading the engine after that
stack aborts with a tcmalloc "free invalid pointer" -- two copies of the
XLA/PjRt runtime).

Gated behind RAIDEN_PRELOAD_ENGINE=1 (set by prefill.sh/decode.sh) so it does
not pollute unrelated processes that merely have this dir on PYTHONPATH.
"""

import os

if os.environ.get("RAIDEN_PRELOAD_ENGINE", "").lower() in ("1", "true", "yes"):
    try:
        from api.jax.transfer_engine import TransferEngine  # noqa: F401
    except Exception:  # pylint: disable=broad-except
        # If the engine isn't built/available, do nothing; the connector will
        # raise a clear error later if it's actually needed.
        pass
