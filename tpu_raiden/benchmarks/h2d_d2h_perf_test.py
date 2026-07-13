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

"""Single-host d2h/h2d KV-cache transfer benchmark (measurement only).

The transfer core lives in perf_core.measure(); this binary just parses flags,
calls it, and emits TensorBoard metrics + raw_perf_results.json. The perf gate
is a separate binary (h2d_d2h_benchmark_gating) that imports the same core.
"""

import json
import os
import sys

from absl import app
from absl import flags

from tpu_raiden.benchmarks import bap_metrics
from tpu_raiden.benchmarks import perf_core

# --- flags (dtype/layers match the DMA benchmark's @parameterized groups) ---
_CACHE_SHAPE = flags.DEFINE_string(
    'cache_shape', '16,128,8,2,128',
    'Per-layer shape "num_blocks,d1,...". dim0 = num_blocks (major dim copied). '
    'DMA test_kv_cache uses 16,128,8,2,128; large-shape uses 8,128,1024,128.')
_SHARD_AXIS = flags.DEFINE_integer('shard_axis', 2, 'Array axis sharded across devices.')
_NUM_LAYERS = flags.DEFINE_integer('num_layers', 64, 'Number of cache arrays (layers).')
_DTYPE = flags.DEFINE_string('dtype', 'float32',
                             'float32 | bfloat16 | float16 | int32 | float8_e4m3fn.')
_ITERS = flags.DEFINE_integer('iters', 10, 'Timed iterations (DMA benchmark_runs default 10).')
_WARMUP = flags.DEFINE_integer('warmup', 3, 'Warmup iterations (not timed).')
_LOCK_BUFFERS = flags.DEFINE_bool('lock_buffers', True,
    'Lock host buffers (unsafe_skip_buffer_lock=False). DMA uses locked (True here).')
_NUMA_PIN = flags.DEFINE_bool('numa_pin', False, 'Pin process to one NUMA node (experiment).')
_NUMA_NODE = flags.DEFINE_integer('numa_node', 0, 'NUMA node to pin to when --numa_pin.')


def main(_):
  if _NUMA_PIN.value:
    perf_core.bind_to_numa(_NUMA_NODE.value)
  else:
    print('[numa] pinning disabled.')

  r = perf_core.measure(
      shape=_CACHE_SHAPE.value, num_layers=_NUM_LAYERS.value, dtype=_DTYPE.value,
      shard_axis=_SHARD_AXIS.value, iters=_ITERS.value, warmup=_WARMUP.value,
      lock_buffers=_LOCK_BUFFERS.value)
  print(f"shape={tuple(r['shape'])}, layers={r['num_layers']}, dtype={r['dtype']}")
  print(f"D2H median {r['d2h_med_t']*1000:.3f} ms -> {r['d2h_gbps']:.3f} Gbps")
  print(f"H2D median {r['h2d_med_t']*1000:.3f} ms -> {r['h2d_gbps']:.3f} Gbps")

  # Tags MUST match the metrics{name: ...} entries in benchmark_registry.pbtxt.
  bap_metrics.emit({
      'd2h_time_sec': r['d2h_med_t'],
      'h2d_time_sec': r['h2d_med_t'],
      'd2h_throughput_gbps': r['d2h_gbps'],
      'h2d_throughput_gbps': r['h2d_gbps'],
  })

  art = os.environ.get('WORKLOAD_ARTIFACTS_DIR')
  if art:
    _write = {
        'shape': r['shape'], 'num_layers': r['num_layers'], 'dtype': r['dtype'],
        'lock_buffers': bool(_LOCK_BUFFERS.value), 'numa_pin': bool(_NUMA_PIN.value),
        'transferred_bytes_total': r['total_bytes'],
        'd2h_times_sec': r['d2h_times_sec'], 'h2d_times_sec': r['h2d_times_sec'],
        'd2h_gbps_all': r['d2h_gbps_all'], 'h2d_gbps_all': r['h2d_gbps_all'],
        'd2h_gbps_summary': r['d2h_gbps_summary'],
        'h2d_gbps_summary': r['h2d_gbps_summary'],
    }
    try:
      with open(os.path.join(art, 'raw_perf_results.json'), 'w') as f:
        json.dump(_write, f, indent=2)
      print('Saved raw_perf_results.json')
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'WARNING: artifact write failed: {e}', file=sys.stderr)


if __name__ == '__main__':
  app.run(main, flags_parser=lambda args: flags.FLAGS(args, known_only=True))
