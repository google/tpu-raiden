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

"""Roofline sweep: grow the transfer block size, watch d2h/h2d bandwidth.

Where bandwidth stops rising = this TPU's peak (the roofline). Configs on that
plateau are 'saturated'; configs well below it are overhead-bound (good gate
canaries). Also measures your two named gate shapes so you see where they land.

Runs against the CURRENT checkout only (no historical commits), so it works in
BAP without any partial-clone workarounds.

Env (optional): DTYPE=int32  ITERS=20  WARMUP=3
"""
import os
import sys

from tpu_raiden.benchmarks import perf_core

DTYPE = os.environ.get('DTYPE', 'int32')
ITERS = int(os.environ.get('ITERS', '20'))
WARMUP = int(os.environ.get('WARMUP', '3'))
NDEV = 8  # v5e single node = 8 chips (used only to display per-shard block size)
_ISZ = {'int32': 4, 'float32': 4, 'bfloat16': 2, 'float16': 2, 'float8_e4m3fn': 1}

# Block-size sweep: [8,128,X,128], 1 layer, shard axis 2; X grows -> block grows.
SWEEP = [([8, 128, x, 128], 1) for x in (8, 32, 128, 512, 1024, 2048, 4096)]
# Your two current gate shapes, shown for reference on the same curve.
NAMED = [('large_shape L8', [8, 128, 1024, 128], 8),
         ('kv_cache L64', [16, 128, 8, 2, 128], 64)]


def _prod(xs):
  p = 1
  for v in xs:
    p *= v
  return p


def _row(label, shape, layers):
  r = perf_core.measure(shape=shape, num_layers=layers, dtype=DTYPE,
                        shard_axis=2, iters=ITERS, warmup=WARMUP)
  isz = _ISZ.get(DTYPE, 4)
  blk = _prod(shape[1:]) * isz / NDEV / 1024          # per-shard block, KB
  tot = layers * _prod(shape) * isz / 1e6             # total moved, MB
  print(f"{label:16}{'x'.join(map(str, shape)):20}L{layers:<5}"
        f"{blk:8.0f}KB{tot:8.0f}M{r['d2h_gbps']:9.1f}{r['h2d_gbps']:9.1f}",
        flush=True)
  return r


def main():
  print(f'roofline sweep  dtype={DTYPE}  iters={ITERS}  '
        f'(shard_axis=2, {NDEV} chips)\n')
  print(f"{'label':16}{'shape':20}{'layers':6}{'blk/shard':>10}{'total':>9}"
        f"{'d2h_gbps':>9}{'h2d_gbps':>9}")
  print('-' * 92)
  for shape, layers in SWEEP:
    _row('sweep', shape, layers)
  print()
  for label, shape, layers in NAMED:
    _row(label, shape, layers)
  print('\nRead d2h/h2d top->bottom in the sweep: where they stop rising is this '
        "TPU's peak bandwidth. Configs on that plateau are 'saturated'; configs "
        'well below it are overhead-bound and make the better gate canaries.')


if __name__ == '__main__':
  main()
