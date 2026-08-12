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

"""TPU Raiden Prometheus metrics collector and exporter."""

import glob
import os
import threading
from typing import Any, Callable, Iterable, Optional, Set

from absl import logging
import prometheus_client
from prometheus_client import parser as prometheus_parser

PROMETHEUS = "prometheus"


class RaidenPrometheusCollector:
  """Custom Prometheus collector yielding TPU Raiden C++ metrics across multiprocess workers."""

  def __init__(
      self,
      multiproc_dir: Optional[str] = None,
      text_snapshot_fn: Optional[Callable[[], str]] = None,
  ):
    self.multiproc_dir = multiproc_dir or os.environ.get(
        "PROMETHEUS_MULTIPROC_DIR", ""
    )
    self._text_snapshot_fn = text_snapshot_fn or (lambda: "")
    self._registered_registries: Set[Any] = set()

  def register_with(self, registry: Any) -> None:
    """Registers this collector with the given Prometheus registry."""
    try:
      registry.register(self)
    except ValueError:
      pass  # Already registered
    self._registered_registries.add(registry)

  def unregister_from(self, registry: Optional[Any] = None) -> None:
    """Unregisters this collector from the specified registry or all tracked registries."""
    if registry is not None:
      try:
        registry.unregister(self)
      except (KeyError, ValueError):
        pass
      self._registered_registries.discard(registry)
    else:
      # Unregister from all tracked registries and default REGISTRY
      registries_to_clean = set(self._registered_registries)
      registries_to_clean.add(prometheus_client.REGISTRY)
      for reg in registries_to_clean:
        try:
          reg.unregister(self)
        except (KeyError, ValueError):
          pass
      self._registered_registries.clear()

  def _flush_local_snapshot(self) -> None:
    """Flushes current process C++ metrics snapshot to multiproc directory if configured."""
    if not self.multiproc_dir:
      return
    try:
      os.makedirs(self.multiproc_dir, exist_ok=True)
      filepath = os.path.join(
          self.multiproc_dir, f"raiden_metrics_pid_{os.getpid()}.prom"
      )
      text = self._text_snapshot_fn()
      if not text:
        if os.path.exists(filepath):
          os.remove(filepath)
        return
      tmppath = f"{filepath}.{threading.get_ident()}.tmp"
      with open(tmppath, "w", encoding="utf-8") as f:
        f.write(text)
      os.replace(tmppath, filepath)
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  def collect(self) -> Iterable[Any]:
    """Collects Raiden C++ Prometheus metrics and yields MetricFamily objects."""
    try:
      self._flush_local_snapshot()

      if not self.multiproc_dir:
        text_snapshot = self._text_snapshot_fn()
        if text_snapshot:
          yield from prometheus_parser.text_string_to_metric_families(
              text_snapshot
          )
        return

      # Aggregate metric families across worker files to avoid duplicate
      # MetricFamily objects
      families_by_name: dict[str, Any] = {}
      for filepath in glob.glob(
          os.path.join(self.multiproc_dir, "raiden_metrics_pid_*.prom")
      ):
        try:
          with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
          if not content:
            continue
          for family in prometheus_parser.text_string_to_metric_families(
              content
          ):
            if family.name not in families_by_name:
              families_by_name[family.name] = family
            else:
              families_by_name[family.name].samples.extend(family.samples)
        except Exception:  # pylint: disable=broad-exception-caught
          pass

      yield from families_by_name.values()
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception("Failed to collect TPU Raiden C++ Prometheus metrics")


def init_prometheus_backend(
    registry: Optional[Any] = None,
    active_handles: Optional[dict[str, Any]] = None,
    text_snapshot_fn: Optional[Callable[[], str]] = None,
) -> RaidenPrometheusCollector:
  """Initializes and registers the singleton Prometheus collector."""
  if active_handles is not None and PROMETHEUS in active_handles:
    collector = active_handles[PROMETHEUS]
  else:
    collector = RaidenPrometheusCollector(text_snapshot_fn=text_snapshot_fn)
    if active_handles is not None:
      active_handles[PROMETHEUS] = collector

  target_reg = registry if registry is not None else prometheus_client.REGISTRY
  collector.register_with(target_reg)
  return collector


def disable_prometheus_backend(
    collector: Optional[RaidenPrometheusCollector],
    registry: Optional[Any] = None,
) -> None:
  """Unregisters the Prometheus collector from target registry or all tracked registries."""
  if collector is not None:
    collector.unregister_from(registry=registry)
