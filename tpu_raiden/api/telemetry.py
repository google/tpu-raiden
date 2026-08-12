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

"""TPU Raiden configurable telemetry integration for Python serving.

Supports dynamic enablement/disablement and multi-backend registration
(e.g., Prometheus) with lazy library imports to avoid binary/library bloat.
"""

import threading
from typing import Any, Iterable, Optional, Sequence

PROMETHEUS = "prometheus"
_SUPPORTED_BACKENDS = frozenset({PROMETHEUS})

_active_backend_handles: dict[str, Any] = {}
_INIT_LOCK = threading.Lock()
_CACHED_TELEMETRY_MODULE: Optional[Any] = None


def _iter_candidate_modules() -> Iterable[Any]:
  """Yields candidate framework modules that may contain bound telemetry APIs."""
  # 1. PyTorch
  try:
    from tpu_raiden.api.torch import kv_cache_manager as kcm  # pylint: disable=g-import-not-at-top

    yield kcm._torch_impl()  # pylint: disable=protected-access
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  # 2. JAX
  try:
    from tpu_raiden.frameworks.jax import _tpu_raiden_jax as jax_mod  # pylint: disable=g-import-not-at-top

    yield jax_mod
  except Exception:  # pylint: disable=broad-exception-caught
    pass


def _get_telemetry_module() -> Optional[Any]:
  """Retrieves and caches the active framework extension containing bound telemetry APIs.

  Only caches non-None results so that early queries prior to framework loading
  do not permanently prevent discovery.
  """
  global _CACHED_TELEMETRY_MODULE
  if _CACHED_TELEMETRY_MODULE is not None:
    return _CACHED_TELEMETRY_MODULE

  mod = next(
      (
          m
          for m in _iter_candidate_modules()
          if hasattr(m, "configure_telemetry")
      ),
      None,
  )
  if mod is not None:
    _CACHED_TELEMETRY_MODULE = mod
  return mod


def _configure_cpp_telemetry(
    mod: Optional[Any], backends: Iterable[str]
) -> None:
  """Configures telemetry in the underlying C++ extension."""
  if mod is not None and hasattr(mod, "configure_telemetry"):
    mod.configure_telemetry(set(backends))


def _get_cpp_metrics_prometheus_text() -> str:
  """Exports Prometheus text snapshot directly from C++ memory."""
  mod = _get_telemetry_module()
  if mod is not None and hasattr(mod, "get_raiden_metrics_prometheus_text"):
    return mod.get_raiden_metrics_prometheus_text()
  return ""


def _disable_backend(backend: str, registry: Optional[Any] = None) -> None:
  """Disables an individual active backend."""
  if backend == PROMETHEUS and PROMETHEUS in _active_backend_handles:
    collector = _active_backend_handles.pop(PROMETHEUS, None)
    from tpu_raiden.api import telemetry_prometheus  # pylint: disable=g-import-not-at-top

    telemetry_prometheus.disable_prometheus_backend(
        collector, registry=registry
    )


def _disable_telemetry(registry: Optional[Any] = None) -> None:
  """Disables telemetry, unregistering C++ backends and Python collectors."""
  mod = _get_telemetry_module()
  _configure_cpp_telemetry(mod, backends=[])

  for backend in list(_active_backend_handles.keys()):
    _disable_backend(backend, registry=registry)


def init_raiden_telemetry(
    backends: Sequence[str] = (),
    registry: Optional[Any] = None,
) -> dict[str, Any]:
  """Initializes Raiden telemetry and registers exporters according to configuration.

  This function is thread-safe and idempotent. Calling it multiple times with
  the same settings will safely reuse the registered backend handles.

  Telemetry is enabled if non-empty `backends` are specified. If `backends` is
  empty, telemetry is disabled and collectors are unregistered.

  Args:
    backends: List of backend names (e.g. ['prometheus'], defaults to empty). If
      empty, telemetry is disabled.
    registry: Target Prometheus CollectorRegistry. If None, defaults to
      prometheus_client.REGISTRY.

  Returns:
    A dictionary mapping active backend names to their initialized handles
    (e.g., {'prometheus': <RaidenPrometheusCollector>}), or an empty dict if
    telemetry is disabled.

  Raises:
    ValueError: If any backend name is not recognized.
  """
  with _INIT_LOCK:
    active_backends = [b.strip().lower() for b in backends if b.strip()]

    if not active_backends:
      _disable_telemetry(registry)
      return {}

    mod = _get_telemetry_module()
    _configure_cpp_telemetry(mod, backends=active_backends)

    handles: dict[str, Any] = {}
    for backend in active_backends:
      if backend == PROMETHEUS:
        from tpu_raiden.api import telemetry_prometheus  # pylint: disable=g-import-not-at-top

        handles[backend] = telemetry_prometheus.init_prometheus_backend(
            registry=registry,
            active_handles=_active_backend_handles,
            text_snapshot_fn=_get_cpp_metrics_prometheus_text,
        )
      else:
        handles[backend] = None

    return handles
