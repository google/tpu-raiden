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

"""Multiprocess distributed (MCJAX MPMD) JAX KVCacheManager enqueue latency performance test."""

import multiprocessing as mp
import os
import random
import socket
import time
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from google3.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

# Set log directories BEFORE importing jax so spawned workers don't fail on
# /tmp/tpu_logs
_LOG_DIR = os.environ.get("TEST_TMPDIR", os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("TPU_LOG_DIR", _LOG_DIR)
os.environ.setdefault("GLOG_log_dir", _LOG_DIR)
os.environ.setdefault("GOOGLE_LOG_DIR", _LOG_DIR)
os.environ.setdefault("TMPDIR", _LOG_DIR)


def pick_unused_ports(count: int) -> list[int]:
  ports = []
  for _ in range(count):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    ports.append(port)
    s.close()
  return ports


def _mcjax_worker_fn(
    rank: int,
    world_size: int,
    coordinator_port: int,
    barrier,
    enable_background: bool,
    result_queue,
) -> None:
  os.environ["RAIDEN_ENABLE_ASYNC_DISPATCH"] = "1" if enable_background else "0"
  os.environ["RANK"] = str(rank)
  os.environ["WORLD_SIZE"] = str(world_size)
  os.environ["LOCAL_RANK"] = str(rank)
  os.environ["PJRT_LOCAL_PROCESS_RANK"] = str(rank)
  os.environ["JAX_PROCESS_INDEX"] = str(rank)
  os.environ["ALLOW_MULTIPLE_LIBTPU_LOAD"] = "true"

  # Stagger JAX driver initialization
  time.sleep(rank * 1.0)

  import jax
  import jax.numpy as jnp
  from tpu_raiden.api.jax import kv_cache_manager

  coord_address = f"127.0.0.1:{coordinator_port}"

  jax.distributed.initialize(
      coordinator_address=coord_address,
      num_processes=world_size,
      process_id=rank,
  )

  device = jax.local_devices()[0]
  print(f"[Rank {rank}] Initialized JAX on device {device}")

  # Serving workload configuration (LLM TP8 sharded: 62 layers, 16 tokens physical block)
  model_num_layers = 62
  device_cache_num_blocks = 32768
  kernel_block_size = 16
  num_kv_heads = 4
  head_dim = 128
  num_iterations = 20
  host_blocks_to_allocate = 24630 * kernel_block_size

  shape = (
      device_cache_num_blocks * kernel_block_size,
      num_kv_heads,
      head_dim,
  )

  kv_caches = [
      jax.device_put(jnp.zeros(shape, dtype=jnp.bfloat16), device)
      for _ in range(model_num_layers)
  ]

  manager = kv_cache_manager.KVCacheManager(
      kv_caches=kv_caches,
      local_control_port=0,
      host_blocks_to_allocate=host_blocks_to_allocate,
      parallelism=1,
  )

  # Workload Sizing Configuration:
  # - Typical LLM serving step (62 transformer layers):
  #   - H2D transfer: 512 physical blocks (32 vLLM logical blocks * 16 kernel blocks) = 248 MB total payload across 62 layers.
  #   - D2H transfer: 128 physical blocks (8 vLLM logical blocks * 16 kernel blocks) = 62 MB total payload across 62 layers.
  h2d_num_blocks = 512
  host_num_blocks = 24630
  random.seed(42)
  h2d_src_blocks = random.sample(range(host_num_blocks), h2d_num_blocks)
  h2d_dst_blocks = random.sample(range(device_cache_num_blocks), h2d_num_blocks)
  h2d_src, h2d_dst, h2d_sizes = [], [], []
  for src_b, dst_b in zip(h2d_src_blocks, h2d_dst_blocks):
    h2d_src.extend(
        range(src_b * kernel_block_size, (src_b + 1) * kernel_block_size)
    )
    h2d_dst.extend(
        range(dst_b * kernel_block_size, (dst_b + 1) * kernel_block_size)
    )
    h2d_sizes.extend([1] * kernel_block_size)

  # Setup D2H offsets
  d2h_num_blocks = 128
  random.seed(43)
  d2h_src_blocks = random.sample(range(device_cache_num_blocks), d2h_num_blocks)
  d2h_dst_blocks = random.sample(range(host_num_blocks), d2h_num_blocks)
  d2h_src, d2h_dst, d2h_sizes = [], [], []
  for src_b, dst_b in zip(d2h_src_blocks, d2h_dst_blocks):
    d2h_src.extend(
        range(src_b * kernel_block_size, (src_b + 1) * kernel_block_size)
    )
    d2h_dst.extend(
        range(dst_b * kernel_block_size, (dst_b + 1) * kernel_block_size)
    )
    d2h_sizes.extend([1] * kernel_block_size)

  # Warmup
  for _ in range(5):
    fut_d2h = manager.d2h(d2h_src, d2h_dst, d2h_sizes)
    fut_d2h.Await()
    fut_h2d = manager.h2d(h2d_src, h2d_dst, h2d_sizes)
    fut_h2d.Await()

  barrier.wait()

  # 1. Measure H2D Latency
  h2d_latencies = []
  for _ in range(num_iterations):
    t0 = time.perf_counter()
    fut = manager.h2d(h2d_src, h2d_dst, h2d_sizes)
    t1 = time.perf_counter()
    h2d_latencies.append((t1 - t0) * 1000)
    fut.Await()

  # 2. Measure D2H Latency
  d2h_latencies = []
  for _ in range(num_iterations):
    t0 = time.perf_counter()
    fut = manager.d2h(d2h_src, d2h_dst, d2h_sizes)
    t1 = time.perf_counter()
    d2h_latencies.append((t1 - t0) * 1000)
    fut.Await()

  result_queue.put({"rank": rank, "h2d": h2d_latencies, "d2h": d2h_latencies})
  barrier.wait()


class KVCacheManagerEnqueueLatencyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("background_enabled", True),
      ("background_disabled", False),
  )
  def test_mcjax_enqueue_latency(self, enable_background):
    world_size = (
        4  # Use 4 ranks to match available chips (requires-ghostfish:4)
    )
    coord_port = pick_unused_ports(1)[0]
    ctx = g3_multiprocessing.get_context(g3_multiprocessing.ABSL_SPAWN)
    barrier = ctx.Barrier(world_size)
    result_queue = ctx.Queue()
    processes = []
    for r in range(world_size):
      p = ctx.Process(
          target=_mcjax_worker_fn,
          args=(
              r,
              world_size,
              coord_port,
              barrier,
              enable_background,
              result_queue,
          ),
      )
      p.start()
      processes.append(p)
    for p in processes:
      p.join()
      self.assertEqual(p.exitcode, 0)

    results = {}
    while not result_queue.empty():
      res = result_queue.get()
      results[res["rank"]] = res
    if len(results) == world_size:
      print("\n=======================================================")
      print("   Enqueue Latency Test Results (62 Layers, 3968 Blocks)  ")
      print("=======================================================")
      h2d_means, d2h_means = [], []
      for r_idx in range(world_size):
        res = results[r_idx]
        h2d_lats = res["h2d"]
        d2h_lats = res["d2h"]
        h2d_m, d2h_m = np.mean(h2d_lats), np.mean(d2h_lats)
        h2d_means.append(h2d_m)
        d2h_means.append(d2h_m)
        print(
            f"Rank {r_idx} | H2D: Mean={h2d_m:.3f} ms"
            f" (P50={np.percentile(h2d_lats, 50):.3f}ms) | D2H:"
            f" Mean={d2h_m:.3f} ms (P50={np.percentile(d2h_lats, 50):.3f}ms)"
        )
      print("-------------------------------------------------------")
      print(
          f"Combined Concurrent H2D Latency (Mean): {np.mean(h2d_means):.3f} ms"
      )
      print(
          f"Combined Concurrent D2H Latency (Mean): {np.mean(d2h_means):.3f} ms"
      )
      print("=======================================================\n")


if __name__ == "__main__":
  mp.set_start_method("spawn", force=True)
  g3_multiprocessing.handle_test_main(absltest.main)
