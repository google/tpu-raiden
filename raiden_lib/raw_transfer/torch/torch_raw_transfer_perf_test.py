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

import gc
import os
import pathlib
import socket
import time

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
import torch_tpu

from google3.pyglib.contrib.g3_multiprocessing import g3_multiprocessing
from raiden_lib.raw_transfer.torch import torch_raw_transfer

_GOOGLE_PCI_VENDOR_ID = "0x1ae0"
_TOPOLOGY_BY_TPU_PCI_DEVICE_ID = {
    "0x005e": {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,2,2"},  # TPU v4
    "0x0062": {1: "1,1,1", 2: "1,2,1", 4: "2,2,1", 8: "2,2,2"},  # TPU v5p
    "0x0063": {1: "1,1,1", 4: "2,2,1", 8: "2,2,2"},  # TPU v5e
    "0x006f": {1: "1,1,1", 4: "2,2,1", 8: "2,4,1"},  # TPU v6e
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


def get_tpu_device_count() -> int:
  count, _ = _scan_pci_tpus()
  return count


def get_tpu_topology(world_size: int) -> str:
  _, topology_map = _scan_pci_tpus()
  if topology_map and world_size in topology_map:
    return topology_map[world_size]
  raise ValueError(f"No TPU topology found for count: {world_size}")


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
  if "TORCH_TPU_XPROF_SESSION_ID" not in os.environ:
    os.environ["TORCH_TPU_XPROF_SESSION_ID"] = str(time.time_ns())
  if "TORCH_TPU_SLICEBUILDER_ADDRESSES" not in os.environ:
    ports = pick_unused_ports(world_size)
    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(
        [f"localhost:{p}" for p in ports]
    )
  if "TORCH_TPU_TOPOLOGY" not in os.environ:
    os.environ["TORCH_TPU_TOPOLOGY"] = get_tpu_topology(world_size)


def _worker_fn(rank: int, world_size: int, master_port: int, fn, args, kwargs):
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = str(master_port)
  os.environ["RANK"] = str(rank)
  os.environ["WORLD_SIZE"] = str(world_size)
  os.environ["LOCAL_RANK"] = str(rank)
  os.environ["GROUP_RANK"] = "0"
  os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
  fn(*args, **kwargs)


def dist_run(world_size: int, fn, *args, **kwargs):
  prepare_tpu_environment(world_size)
  master_port = pick_unused_ports(1)[0]
  mp.spawn(
      _worker_fn,
      args=(world_size, master_port, fn, args, kwargs),
      nprocs=world_size,
      join=True,
  )


SUPPORTED_DTYPES = {
    torch.float8_e4m3fn: "fp8",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
}


def _run_kv_cache_perf_compare(dtype, num_layers, num_blocks):
  if dtype not in SUPPORTED_DTYPES:
    return
  dtype_str = SUPPORTED_DTYPES[dtype]

  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  device = torch.device("tpu")

  # (num_blocks, 128, 8, 2, 128) for num_blocks per layer
  shape = (num_blocks, 128, 8, 2, 128)

  src_arrs = []
  for j in range(num_layers):
    arr = torch.randn(shape).to(dtype)
    chunks = torch.chunk(arr, world_size, dim=2)
    src_arrs.append(chunks[rank].to(device))
  torch.tpu.synchronize()

  dst_arrs = []
  for src in src_arrs:
    dst_arrs.append(torch.zeros_like(src, device="cpu", pin_memory=True))

  num_iterations = 10 if num_layers >= 1024 else 20

  tpu_dst_arrs = []
  for src in src_arrs:
    tpu_dst_arrs.append(torch.zeros_like(src, device="cpu").to(device))
  torch.tpu.synchronize()

  # Benchmark our library (batch, optimized)
  d2h_times = []
  h2d_times = []

  for i in range(num_iterations):
    gc.disable()
    dist.barrier()
    start = time.time()
    futures = torch_raw_transfer.transfer_d2h_batch_async(src_arrs, dst_arrs)
    futures.Await()
    d2h_times.append(time.time() - start)

    gc.enable()
    gc.collect()
    gc.disable()

    dist.barrier()
    start = time.time()
    futures = torch_raw_transfer.transfer_h2d_batch_async(
        dst_arrs, tpu_dst_arrs
    )
    futures.Await()
    torch.tpu.synchronize()
    h2d_times.append(time.time() - start)

    gc.enable()
    gc.collect()
    if i == 0:
      # Verify H2D
      for j in range(len(src_arrs)):
        torch.testing.assert_close(tpu_dst_arrs[j].cpu(), src_arrs[j].cpu())
      if rank == 0:
        print("Library H2D verification passed")

  if rank == 0:
    print(f"[{dtype_str}] Library D2H times: {d2h_times}")
    print(f"[{dtype_str}] Library H2D times: {h2d_times}")
    print(
        f"[{dtype_str}] Library D2H time: {np.median(d2h_times):.6f} s (median)"
        f" / {np.mean(d2h_times):.6f} s (mean)"
    )
    print(
        f"[{dtype_str}] Library H2D time: {np.median(h2d_times):.6f} s (median)"
        f" / {np.mean(h2d_times):.6f} s (mean)"
    )

  # Benchmark our library (Sync)
  sync_d2h_times = []
  sync_h2d_times = []

  for i in range(num_iterations):
    gc.disable()
    dist.barrier()
    start = time.time()
    torch_raw_transfer.transfer_d2h_batch(src_arrs, dst_arrs)
    sync_d2h_times.append(time.time() - start)

    gc.enable()
    gc.collect()
    gc.disable()

    dist.barrier()
    start = time.time()
    torch_raw_transfer.transfer_h2d_batch(dst_arrs, tpu_dst_arrs)
    torch.tpu.synchronize()
    sync_h2d_times.append(time.time() - start)

    gc.enable()
    gc.collect()
    if i == 0:
      # Verify H2D
      for j in range(len(src_arrs)):
        torch.testing.assert_close(tpu_dst_arrs[j].cpu(), src_arrs[j].cpu())
      if rank == 0:
        print("Library Sync H2D verification passed")

  # Benchmark Native Torch
  native_d2h_times = []
  native_h2d_times = []
  cpu_tensors = None
  for _ in range(num_iterations):
    gc.disable()
    dist.barrier()
    start = time.time()
    cpu_tensors = [tpu.cpu() for tpu in src_arrs]
    native_d2h_times.append(time.time() - start)

    gc.enable()
    gc.collect()
    gc.disable()

    dist.barrier()
    start = time.time()
    _ = [cpu.to(device) for cpu in cpu_tensors]
    torch.tpu.synchronize()
    native_h2d_times.append(time.time() - start)

    gc.enable()
    gc.collect()

  if rank == 0:
    # Calculate bandwidth
    element_size = torch.tensor([], dtype=dtype).element_size()
    total_bytes = np.prod(shape) * element_size * num_layers

    lib_d2h_bw_median = (
        total_bytes / np.median(d2h_times) / (1024 * 1024 * 1024)
    )
    lib_h2d_bw_median = (
        total_bytes / np.median(h2d_times) / (1024 * 1024 * 1024)
    )
    native_d2h_bw_median = (
        total_bytes / np.median(native_d2h_times) / (1024 * 1024 * 1024)
    )
    native_h2d_bw_median = (
        total_bytes / np.median(native_h2d_times) / (1024 * 1024 * 1024)
    )
    lib_sync_d2h_bw = (
        total_bytes / np.median(sync_d2h_times) / (1024 * 1024 * 1024)
    )
    lib_sync_h2d_bw = (
        total_bytes / np.median(sync_h2d_times) / (1024 * 1024 * 1024)
    )

    print(
        f"[{dtype}, {num_layers} layers, shape={shape}]"
        f" transfer_d2h_batch_async bandwidth: {lib_d2h_bw_median:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}]"
        f" transfer_h2d_batch_async bandwidth: {lib_h2d_bw_median:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] transfer_d2h_batch"
        f" bandwidth: {lib_sync_d2h_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] transfer_h2d_batch"
        f" bandwidth: {lib_sync_h2d_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] tensor.cpu() D2H"
        f" bandwidth: {native_d2h_bw_median:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] tensor.to(tpu) H2D"
        f" bandwidth: {native_h2d_bw_median:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Relative Sync D2H"
        " bandwidth (Lib_Sync/Torch):"
        f" {lib_sync_d2h_bw / native_d2h_bw_median:.2f}x"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Relative Sync H2D"
        " bandwidth (Lib_Sync/Torch):"
        f" {lib_sync_h2d_bw / native_h2d_bw_median:.2f}x"
    )

  del src_arrs
  del dst_arrs
  del tpu_dst_arrs
  del cpu_tensors
  gc.collect()


def _run_large_shape_perf_compare(num_layers, shape, dtype):
  if dtype not in SUPPORTED_DTYPES:
    return

  dtype_str = SUPPORTED_DTYPES[dtype]
  shape_str = "_".join([str(s) for s in shape])
  test_str = f"l{num_layers}_s{shape_str}_d{dtype_str}"

  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  device = torch.device("tpu")

  src_arrs = []
  for j in range(num_layers):
    arr = torch.randn(shape).to(dtype)
    chunks = torch.chunk(arr, world_size, dim=3)
    src_arrs.append(chunks[rank].to(device))
  torch.tpu.synchronize()

  dst_arrs = []
  for src in src_arrs:
    dst_arrs.append(torch.zeros_like(src, device="cpu", pin_memory=True))

  num_iterations = 2 if num_layers >= 64 else 10

  tpu_dst_arrs = []
  for src in src_arrs:
    tpu_dst_arrs.append(torch.zeros_like(src, device="cpu").to(device))
  torch.tpu.synchronize()

  # Benchmark our library (batch, optimized)
  opt_batched_d2h_times = []
  opt_batched_h2d_times = []

  for i in range(num_iterations):
    gc.disable()
    dist.barrier()
    start = time.time()
    futures = torch_raw_transfer.transfer_d2h_batch_async(src_arrs, dst_arrs)
    futures.Await()
    opt_batched_d2h_times.append(time.time() - start)

    gc.enable()
    gc.collect()
    gc.disable()

    dist.barrier()
    start = time.time()
    futures = torch_raw_transfer.transfer_h2d_batch_async(
        dst_arrs, tpu_dst_arrs
    )
    futures.Await()
    torch.tpu.synchronize()
    opt_batched_h2d_times.append(time.time() - start)

    gc.enable()
    gc.collect()
    if i == 0:
      for j in range(len(src_arrs)):
        torch.testing.assert_close(tpu_dst_arrs[j].cpu(), src_arrs[j].cpu())
      if rank == 0:
        print("Library H2D verification passed")

  # Benchmark non-batched async
  non_batch_d2h_times = []
  non_batch_h2d_times = []

  for _ in range(num_iterations):
    gc.disable()
    dist.barrier()
    start = time.time()
    all_futures = []
    for j in range(len(src_arrs)):
      futures = torch_raw_transfer.transfer_d2h_async(src_arrs[j], dst_arrs[j])
      all_futures.append(futures)
    for f in all_futures:
      f.Await()
    non_batch_d2h_times.append(time.time() - start)

    gc.enable()
    gc.collect()
    gc.disable()

    dist.barrier()
    start = time.time()
    all_futures = []
    for j in range(len(src_arrs)):
      futures = torch_raw_transfer.transfer_h2d_async(
          dst_arrs[j], tpu_dst_arrs[j]
      )
      all_futures.append(futures)
    for f in all_futures:
      f.Await()
    torch.tpu.synchronize()
    non_batch_h2d_times.append(time.time() - start)

    gc.enable()
    gc.collect()

  # Benchmark Native Torch
  native_d2h_times = []
  native_h2d_times = []
  cpu_tensors = None
  for _ in range(num_iterations):
    gc.disable()
    dist.barrier()
    start = time.time()
    cpu_tensors = [tpu.cpu() for tpu in src_arrs]
    native_d2h_times.append(time.time() - start)

    gc.enable()
    gc.collect()
    gc.disable()

    dist.barrier()
    start = time.time()
    _ = [cpu.to(device) for cpu in cpu_tensors]
    torch.tpu.synchronize()
    native_h2d_times.append(time.time() - start)

    gc.enable()
    gc.collect()

  if rank == 0:
    # Calculate bandwidth
    element_size = torch.tensor([], dtype=dtype).element_size()
    total_bytes = np.prod(shape) * element_size * num_layers

    print(f"[{test_str}] Total bytes: {total_bytes / (1024*1024*1024):.3f} GB")

    def report_perf(name, times):
      med_time = np.median(times)
      avg_time = np.mean(times)
      med_bw = total_bytes / med_time / (1024 * 1024 * 1024)
      avg_bw = total_bytes / avg_time / (1024 * 1024 * 1024)
      print(
          f"[{test_str}] {name} time: {med_time:.6f} s (median), {avg_time:.6f}"
          " s (mean)"
      )
      print(
          f"[{test_str}] {name} BW: {med_bw:.3f} GB/s (median), {avg_bw:.3f}"
          " GB/s (mean)"
      )

    report_perf("Library non-batch D2H", non_batch_d2h_times)
    report_perf("Library non-batch H2D", non_batch_h2d_times)
    report_perf("Library optimized batch D2H", opt_batched_d2h_times)
    report_perf("Library optimized batch H2D", opt_batched_h2d_times)
    report_perf("Torch native D2H", native_d2h_times)
    report_perf("Torch native H2D", native_h2d_times)

  del src_arrs
  del dst_arrs
  del tpu_dst_arrs
  del cpu_tensors
  gc.collect()


class TorchRawTransferPerfTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("fp8", torch.float8_e4m3fn, 64, 16),
      ("bf16", torch.bfloat16, 64, 16),
      ("f32", torch.float32, 64, 16),
  )
  def test_kv_cache_perf_compare(self, dtype, num_layers, num_blocks):
    if dtype not in SUPPORTED_DTYPES:
      self.skipTest(f"Unsupported dtype: {dtype}")
    device_count = get_tpu_device_count()
    dist_run(
        device_count,
        _run_kv_cache_perf_compare,
        dtype,
        num_layers,
        num_blocks,
    )

  @parameterized.named_parameters(
      ("1_layers_fp8", 1, (8, 128, 1024, 128), torch.float8_e4m3fn),
      ("1_layers_bf16", 1, (8, 128, 1024, 128), torch.bfloat16),
      ("1_layers_fp32", 1, (8, 128, 1024, 128), torch.float32),
      ("2_layer_fp8", 2, (8, 128, 1024, 128), torch.float8_e4m3fn),
      ("4_layer_fp8", 4, (8, 128, 1024, 128), torch.float8_e4m3fn),
      ("8_layer_fp8", 8, (8, 128, 1024, 128), torch.float8_e4m3fn),
  )
  def test_large_shape_perf_compare(self, num_layers, shape, dtype):
    if dtype not in SUPPORTED_DTYPES:
      self.skipTest("Unsupported dtype")
    device_count = get_tpu_device_count()
    dist_run(
        device_count,
        _run_large_shape_perf_compare,
        num_layers,
        shape,
        dtype,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
