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

"""Multiprocess distributed (MPMD) PyTorch KVCacheManager enqueue latency performance test."""

import os
import pathlib
import random
import socket
import time

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

# Set log directories BEFORE importing torch so spawned workers don't fail on
# /tmp/tpu_logs
_LOG_DIR = os.environ.get("TEST_TMPDIR", os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("TPU_LOG_DIR", _LOG_DIR)
os.environ.setdefault("GLOG_log_dir", _LOG_DIR)
os.environ.setdefault("GOOGLE_LOG_DIR", _LOG_DIR)
os.environ.setdefault("TMPDIR", _LOG_DIR)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tpu_raiden.api.torch import kv_cache_manager as _kv_cache_manager
from google3.pyglib.contrib.g3_multiprocessing import g3_multiprocessing

_GOOGLE_PCI_VENDOR_ID = "0x1ae0"
_TOPOLOGY_BY_TPU_PCI_DEVICE_ID = {
    "0x005e": {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,2,2"},  # TPU v4
    "0x0062": {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,2,2"},  # TPU v5p
    "0x0063": {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,2,2"},  # TPU v5e
    "0x006f": {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,4,1"},  # TPU v6e
    "0x0076": {2: "1,1,1,2", 4: "1,2,1,2", 8: "2,2,1,2"},  # TPU v7
}


def _scan_pci_tpus():
  count = 0
  topology_map = None
  pci_devices = pathlib.Path("/sys/bus/pci/devices")
  if not pci_devices.exists():
    return 0, None
  for device_path in pci_devices.iterdir():
    try:
      vendor_id = (device_path / "vendor").read_text().strip()
      if vendor_id != _GOOGLE_PCI_VENDOR_ID:
        continue
      device_id = (device_path / "device").read_text().strip()
      if device_id in _TOPOLOGY_BY_TPU_PCI_DEVICE_ID:
        try:
          group_id = (device_path / "iommu_group").readlink().name
          (pathlib.Path("/dev/vfio") / group_id).stat()
        except OSError:
          continue
        count += 1
        if topology_map is None:
          topology_map = _TOPOLOGY_BY_TPU_PCI_DEVICE_ID[device_id]
    except OSError:
      continue
  return count, topology_map


def get_tpu_topology(world_size: int) -> str:
  _, topology_map = _scan_pci_tpus()
  if topology_map and world_size in topology_map:
    return topology_map[world_size]
  return (
      "2x4"
      if world_size == 8
      else "2x2"
      if world_size == 4
      else f"1x{world_size}"
  )


def pick_unused_ports(count: int) -> list[int]:
  ports = []
  for _ in range(count):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    ports.append(port)
    s.close()
  return ports


def prepare_tpu_environment(world_size: int) -> None:
  log_dir = os.environ.get("TEST_TMPDIR", os.environ.get("TMPDIR", "/tmp"))
  os.environ["TPU_LOG_DIR"] = log_dir
  os.environ["GLOG_log_dir"] = log_dir
  os.environ["GOOGLE_LOG_DIR"] = log_dir
  os.environ["TMPDIR"] = log_dir
  if "TORCH_TPU_XPROF_SESSION_ID" not in os.environ:
    os.environ["TORCH_TPU_XPROF_SESSION_ID"] = str(time.time_ns())
  if "TORCH_TPU_SLICEBUILDER_ADDRESSES" not in os.environ:
    ports = pick_unused_ports(world_size)
    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(
        [f"localhost:{p}" for p in ports]
    )
  if "TORCH_TPU_TOPOLOGY" not in os.environ:
    os.environ["TORCH_TPU_TOPOLOGY"] = get_tpu_topology(world_size)


def _torch_worker_fn(
    rank: int, world_size: int, master_port: int, enable_background: bool
) -> None:
  os.environ["RAIDEN_ENABLE_ASYNC_DISPATCH"] = "1" if enable_background else "0"
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = str(master_port)
  os.environ["RANK"] = str(rank)
  os.environ["WORLD_SIZE"] = str(world_size)
  os.environ["LOCAL_RANK"] = str(rank)
  os.environ["PJRT_LOCAL_PROCESS_RANK"] = str(rank)
  os.environ["GROUP_RANK"] = "0"
  os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

  # Initialize PyTorch Gloo distributed process group across all 8 ranks
  dist.init_process_group(
      backend="gloo",
      init_method=f"tcp://127.0.0.1:{master_port}",
      rank=rank,
      world_size=world_size,
  )

  try:
    device = torch.device("tpu")

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
        torch.zeros(shape, dtype=torch.bfloat16, device=device)
        for _ in range(model_num_layers)
    ]

    manager = _kv_cache_manager.KVCacheManager(
        [[t] for t in kv_caches],
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
    h2d_dst_blocks = random.sample(
        range(device_cache_num_blocks), h2d_num_blocks
    )
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
    d2h_src_blocks = random.sample(
        range(device_cache_num_blocks), d2h_num_blocks
    )
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

    dist.barrier()

    # 1. Measure H2D Latency (with step-level barrier)
    h2d_latencies = []
    for _ in range(num_iterations):
      t0 = time.perf_counter()
      fut = manager.h2d(h2d_src, h2d_dst, h2d_sizes)
      t1 = time.perf_counter()
      h2d_latencies.append((t1 - t0) * 1000)
      fut.Await()

    # 2. Measure D2H Latency (with step-level barrier)
    d2h_latencies = []
    for _ in range(num_iterations):
      t0 = time.perf_counter()
      fut = manager.d2h(d2h_src, d2h_dst, d2h_sizes)
      t1 = time.perf_counter()
      d2h_latencies.append((t1 - t0) * 1000)
      fut.Await()

    res = {"rank": rank, "h2d": h2d_latencies, "d2h": d2h_latencies}
    gather_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_list, res)
    if rank == 0:
      print("\n=======================================================")
      print("   Enqueue Latency Test Results (62 Layers, 3968 Blocks)  ")
      print("=======================================================")
      h2d_means, d2h_means = [], []
      for r_idx in range(world_size):
        res_r = gather_list[r_idx]
        h2d_lats = res_r["h2d"]
        d2h_lats = res_r["d2h"]
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

  finally:
    dist.barrier()
    dist.destroy_process_group()


class KVCacheManagerEnqueueLatencyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("background_enabled", True),
      ("background_disabled", False),
  )
  def test_enqueue_latency(self, enable_background):
    world_size = 8
    prepare_tpu_environment(world_size)
    master_port = pick_unused_ports(1)[0]
    mp.spawn(
        _torch_worker_fn,
        args=(world_size, master_port, enable_background),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn", force=True)
  g3_multiprocessing.handle_test_main(absltest.main)
