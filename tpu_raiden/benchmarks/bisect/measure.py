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

"""Single-config d2h/h2d measurement used as the git-bisect judge.

Prints one parseable line:  BISECT d2h=<gbps> h2d=<gbps>
Uses ONLY perf_core.measure (present on every commit in the window) so it can be
injected into and built at each bisected commit. The config is float32 L64, the
one that regressed ~14% (d2h) / ~10% (h2d): a per-descriptor-overhead signature.
"""
import sys

from tpu_raiden.benchmarks import perf_core

r = perf_core.measure(shape=[16, 128, 8, 2, 128], num_layers=64, dtype='float32',
                      shard_axis=2, iters=15, warmup=3)  # 14% gap -> 15 iters plenty
print(f'BISECT d2h={r["d2h_gbps"]:.1f} h2d={r["h2d_gbps"]:.1f}')
sys.stdout.flush()
