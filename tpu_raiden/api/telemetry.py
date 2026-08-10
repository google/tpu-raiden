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

import functools
import threading
from typing import Any, Iterable, Optional, Sequence

from absl import logging

PROMETHEUS = "prometheus"

_active_backend_handles: dict[str, Any] = {}
_INIT_LOCK = threading.Lock()


def _iter_candidate_modules() -> Iterable[Any]:
  """Yields candidate framework modules that may contain bound telemetry APIs."""
  # 1. PyTorch
  try:
    from tpu_raiden.api.torch import kv_cache_manager as kcm  # pylint: disable=g-import-not-at-top

    yield kcm._torch_impl()  # pylint: disable=protected-access
  except ModuleNotFoundError:
    pass

  # 2. JAX
  try:
    from tpu_raiden.frameworks.jax import _tpu_raiden_jax as jax_mod  # pylint: disable=g-import-not-at-top

    yield jax_mod
  except ModuleNotFoundError:
    pass


@functools.cache
def _get_telemetry_module() -> Optional[Any]:
  """Retrieves and caches the active framework extension containing bound telemetry APIs."""
  return next(
      (
          m
          for m in _iter_candidate_modules()
          if hasattr(m, "configure_telemetry")
      ),
      None,
  )


def _configure_cpp_telemetry(
    mod: Optional[Any], backends: Sequence[str]
) -> None:
  """Configures telemetry in the underlying C++ extension."""
  if mod is not None and hasattr(mod, "configure_telemetry"):
    try:
      mod.configure_telemetry(list(backends))
    except Exception:  # pylint: disable=broad-exception-caught
      pass


def get_raiden_metrics_prometheus_text() -> str:
  """Exports Prometheus text snapshot directly from C++ memory."""
  mod = _get_telemetry_module()
  if mod is not None and hasattr(mod, "get_raiden_metrics_prometheus_text"):
    try:
      return mod.get_raiden_metrics_prometheus_text()
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  return ""


class RaidenPrometheusCollector:
  """Custom Prometheus collector yielding TPU Raiden C++ metrics."""

  def collect(self) -> Iterable[Any]:
    """Collects Raiden C++ Prometheus metrics and yields MetricFamily objects."""
    try:
      # pylint: disable=g-import-not-at-top
      from prometheus_client import parser as prometheus_parser
      # pylint: enable=g-import-not-at-top
      text_snapshot = get_raiden_metrics_prometheus_text()
      if not text_snapshot:
        return
      for metric_family in prometheus_parser.text_string_to_metric_families(
          text_snapshot
      ):
        yield metric_family
    except Exception:  # pylint: disable=broad-exception-caught
      # Strict exception wrapping: collection failures must never crash
      # the serving endpoint.
      logging.exception("Failed to collect TPU Raiden C++ Prometheus metrics")


def _init_prometheus_backend(
    registry: Optional[Any] = None,
) -> RaidenPrometheusCollector:
  """Initializes and registers the singleton Prometheus collector."""
  # pylint: disable=g-import-not-at-top
  import prometheus_client
  # pylint: enable=g-import-not-at-top

  if PROMETHEUS not in _active_backend_handles:
    _active_backend_handles[PROMETHEUS] = RaidenPrometheusCollector()

  collector = _active_backend_handles[PROMETHEUS]
  target_reg = registry if registry is not None else prometheus_client.REGISTRY

  try:
    target_reg.register(collector)
  except ValueError:
    pass  # Already registered in target_reg

  return collector


def _disable_telemetry(registry: Optional[Any] = None) -> None:
  """Disables telemetry, unregistering C++ backends and Python collectors."""
  # pylint: disable=g-import-not-at-top
  import prometheus_client
  # pylint: enable=g-import-not-at-top

  mod = _get_telemetry_module()
  _configure_cpp_telemetry(mod, backends=[])

  collector = _active_backend_handles.pop(PROMETHEUS, None)
  if collector is not None:
    target_reg = (
        registry if registry is not None else prometheus_client.REGISTRY
    )
    try:
      target_reg.unregister(collector)
    except (KeyError, ValueError):
      pass


def init_raiden_telemetry(
    enabled: bool = False,
    backends: Sequence[str] = (),
    registry: Optional[Any] = None,
) -> dict[str, Any]:
  """Initializes Raiden telemetry and registers exporters according to configuration.

  This function is thread-safe and idempotent. Calling it multiple times with
  the same settings will safely reuse the registered backend handles.

  Args:
    enabled: Master toggle to enable or disable telemetry (defaults to False).
    backends: List of backend names (e.g. ['prometheus'], defaults to empty).
      Has no effect if `enabled` is False.
    registry: Target Prometheus CollectorRegistry. If None, defaults to
      prometheus_client.REGISTRY.

  Returns:
    A dictionary mapping active backend names to their initialized handles
    (e.g., {'prometheus': <RaidenPrometheusCollector>}), or an empty dict if
    telemetry is disabled.
  """
  with _INIT_LOCK:
    if not enabled:
      _disable_telemetry(registry)
      return {}

    active_backends = [b.strip().lower() for b in backends if b.strip()]

    mod = _get_telemetry_module()
    _configure_cpp_telemetry(mod, backends=active_backends)

    handles: dict[str, Any] = {}
    for backend in active_backends:
      if backend == PROMETHEUS:
        handles[backend] = _init_prometheus_backend(registry)
      else:
        handles[backend] = None

    return handles
