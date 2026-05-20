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

import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
import torch_tpu

from raiden_lib.raw_transfer.torch import torch_raw_transfer

os.environ["TORCH_TPU_TOPOLOGY"] = "1x1"

SUPPORTED_DTYPES = {
    torch.float8_e4m3fn: "fp8",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
}


class TorchRawTransferTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu:0")
    # Emulate realistic KV cache block dimensions as requested by user
    num_blocks = 4
    self.shape = (num_blocks, 128, 8, 2, 128)

  @parameterized.named_parameters(
      ("fp8", torch.float8_e4m3fn),
      ("bf16", torch.bfloat16),
      ("fp32", torch.float32),
  )
  def test_single_transfers(self, dtype):
    ref_arr = torch.randn(self.shape).to(dtype)
    tpu_arr = ref_arr.to(self.device)
    torch.tpu.synchronize()

    dst_d2h = torch.zeros_like(ref_arr)
    torch_raw_transfer.transfer_d2h(tpu_arr, dst_d2h)

    dst_h2d = torch.zeros_like(tpu_arr, device="cpu").to(self.device)
    torch_raw_transfer.transfer_h2d(dst_d2h, dst_h2d)
    torch.tpu.synchronize()
    torch.testing.assert_close(dst_h2d.cpu(), ref_arr)

  @parameterized.named_parameters(
      ("fp8", torch.float8_e4m3fn),
      ("bf16", torch.bfloat16),
      ("fp32", torch.float32),
  )
  def test_single_async_transfers(self, dtype):
    ref_arr = torch.randn(self.shape).to(dtype)
    tpu_arr = ref_arr.to(self.device)
    torch.tpu.synchronize()

    dst_d2h = torch.zeros_like(ref_arr)
    torch_raw_transfer.transfer_d2h_async(tpu_arr, dst_d2h).Await()

    dst_h2d = torch.zeros_like(tpu_arr, device="cpu").to(self.device)
    torch_raw_transfer.transfer_h2d_async(dst_d2h, dst_h2d).Await()
    torch.tpu.synchronize()
    torch.testing.assert_close(dst_h2d.cpu(), ref_arr)

  @parameterized.named_parameters(
      ("sync_fp8", "sync", torch.float8_e4m3fn),
      ("sync_bf16", "sync", torch.bfloat16),
      ("sync_fp32", "sync", torch.float32),
      ("async_fp8", "async", torch.float8_e4m3fn),
      ("async_bf16", "async", torch.bfloat16),
      ("async_fp32", "async", torch.float32),
  )
  def test_batch_transfers(self, mode, dtype):
    n_layers = 2
    ref_arrs = []
    tpu_arrs = []

    for i in range(n_layers):
      base = torch.randn(self.shape).to(dtype)
      ref_arrs.append(base)
      tpu_arrs.append(base.to(self.device))

    torch.tpu.synchronize()

    host_arrs = [torch.zeros_like(ref) for ref in ref_arrs]
    tpu_dst_arrs = [torch.zeros_like(tpu, device="cpu").to(self.device) for tpu in tpu_arrs]

    if mode == "sync":
      torch_raw_transfer.transfer_d2h_batch(tpu_arrs, host_arrs)

      torch_raw_transfer.transfer_h2d_batch(host_arrs, tpu_dst_arrs)
      torch.tpu.synchronize()
      for i in range(n_layers):
        torch.testing.assert_close(tpu_dst_arrs[i].cpu(), ref_arrs[i])

    elif mode == "async":
      torch_raw_transfer.transfer_d2h_batch_async(tpu_arrs, host_arrs).Await()

      torch_raw_transfer.transfer_h2d_batch_async(
          host_arrs, tpu_dst_arrs
      ).Await()
      torch.tpu.synchronize()
      for i in range(n_layers):
        torch.testing.assert_close(tpu_dst_arrs[i].cpu(), ref_arrs[i])

  def test_single_async_transfers_is_ready(self):
    dtype = torch.bfloat16
    ref_arr = torch.randn(self.shape).to(dtype)
    tpu_arr = ref_arr.to(self.device)
    torch.tpu.synchronize()

    dst_d2h = torch.zeros_like(ref_arr)
    future_d2h = torch_raw_transfer.transfer_d2h_async(tpu_arr, dst_d2h)

    self.assertIsInstance(future_d2h.IsReady(), bool)
    self.assertTrue(torch_raw_transfer.is_ready(future_d2h) in [True, False])
    self.assertTrue(torch_raw_transfer.is_ready([future_d2h]) in [True, False])

    future_d2h.Await()

    self.assertTrue(future_d2h.IsReady())
    self.assertTrue(torch_raw_transfer.is_ready(future_d2h))
    self.assertTrue(torch_raw_transfer.is_ready([future_d2h]))

    dst_h2d = torch.zeros_like(tpu_arr, device="cpu").to(self.device)
    future_h2d = torch_raw_transfer.transfer_h2d_async(dst_d2h, dst_h2d)

    self.assertIsInstance(future_h2d.IsReady(), bool)
    self.assertTrue(torch_raw_transfer.is_ready(future_h2d) in [True, False])
    self.assertTrue(torch_raw_transfer.is_ready([future_h2d]) in [True, False])

    future_h2d.Await()
    torch.tpu.synchronize()

    self.assertTrue(future_h2d.IsReady())
    self.assertTrue(torch_raw_transfer.is_ready(future_h2d))
    self.assertTrue(torch_raw_transfer.is_ready([future_h2d]))
    torch.testing.assert_close(dst_h2d.cpu(), ref_arr)

  def test_batch_transfers_is_ready(self):
    dtype = torch.bfloat16
    n_layers = 2
    ref_arrs = []
    tpu_arrs = []

    for i in range(n_layers):
      base = torch.randn(self.shape).to(dtype)
      ref_arrs.append(base)
      tpu_arrs.append(base.to(self.device))

    torch.tpu.synchronize()

    host_arrs = [torch.zeros_like(ref) for ref in ref_arrs]
    tpu_dst_arrs = [torch.zeros_like(tpu, device="cpu").to(self.device) for tpu in tpu_arrs]

    future_d2h = torch_raw_transfer.transfer_d2h_batch_async(
        tpu_arrs, host_arrs
    )
    self.assertIsInstance(future_d2h.IsReady(), bool)
    self.assertTrue(torch_raw_transfer.is_ready(future_d2h) in [True, False])
    self.assertTrue(torch_raw_transfer.is_ready([future_d2h]) in [True, False])

    future_d2h.Await()

    self.assertTrue(future_d2h.IsReady())
    self.assertTrue(torch_raw_transfer.is_ready(future_d2h))
    self.assertTrue(torch_raw_transfer.is_ready([future_d2h]))

    future_h2d = torch_raw_transfer.transfer_h2d_batch_async(
        host_arrs, tpu_dst_arrs
    )
    self.assertIsInstance(future_h2d.IsReady(), bool)
    self.assertTrue(torch_raw_transfer.is_ready(future_h2d) in [True, False])
    self.assertTrue(torch_raw_transfer.is_ready([future_h2d]) in [True, False])

    future_h2d.Await()
    torch.tpu.synchronize()

    self.assertTrue(future_h2d.IsReady())
    self.assertTrue(torch_raw_transfer.is_ready(future_h2d))
    self.assertTrue(torch_raw_transfer.is_ready([future_h2d]))
    for i in range(n_layers):
      torch.testing.assert_close(tpu_dst_arrs[i].cpu(), ref_arrs[i])

  def test_prepared_transfer(self):
    dtype = torch.bfloat16
    ref_arr = torch.randn(self.shape).to(dtype)
    tpu_arr = ref_arr.to(self.device)
    torch.tpu.synchronize()

    # 2 bytes per bfloat16
    size_bytes = ref_arr.numel() * 2
    host_buffer = torch_raw_transfer.RawHostBuffer(size_bytes)
    prepared = torch_raw_transfer.PreparedTorchRawTransfer(tpu_arr, host_buffer)

    # Perform D2H
    prepared.d2h()

    # Perform H2D
    dst_tpu_arr = torch.zeros_like(tpu_arr, device="cpu").to(self.device)
    prepared_dst = torch_raw_transfer.PreparedTorchRawTransfer(
        dst_tpu_arr, host_buffer
    )
    prepared_dst.h2d()
    torch.tpu.synchronize()

    torch.testing.assert_close(dst_tpu_arr.cpu(), ref_arr)

  def test_perf_compare(self):
    import time
    import gc

    num_layers = 4
    num_iterations = 10
    dtype = torch.bfloat16

    ref_arrs = []
    tpu_arrs = []

    for i in range(num_layers):
      base = torch.randn(self.shape).to(dtype)
      ref_arrs.append(base)
      tpu_arrs.append(base.to(self.device))

    torch.tpu.synchronize()

    host_arrs = [torch.zeros_like(ref) for ref in ref_arrs]
    tpu_dst_arrs = [torch.zeros_like(tpu, device="cpu").to(self.device) for tpu in tpu_arrs]

    # Benchmark Stateless batch async
    batched_d2h_times = []
    batched_h2d_times = []

    for i in range(num_iterations):
      gc.disable()
      start = time.time()
      futures = torch_raw_transfer.transfer_d2h_batch_async(tpu_arrs, host_arrs)
      futures.Await()
      batched_d2h_times.append(time.time() - start)

      gc.enable()
      gc.collect()
      gc.disable()

      start = time.time()
      futures = torch_raw_transfer.transfer_h2d_batch_async(
          host_arrs, tpu_dst_arrs
      )
      futures.Await()
      torch.tpu.synchronize()
      batched_h2d_times.append(time.time() - start)

      gc.enable()
      gc.collect()

    # Benchmark Prepared (emulating batch using loop)
    prepared_d2h_times = []
    prepared_h2d_times = []

    size_bytes = ref_arrs[0].numel() * 2
    host_buffers = [
        torch_raw_transfer.RawHostBuffer(size_bytes) for _ in range(num_layers)
    ]
    prepared_d2h = [
        torch_raw_transfer.PreparedTorchRawTransfer(
            tpu_arrs[i], host_buffers[i]
        )
        for i in range(num_layers)
    ]
    prepared_h2d = [
        torch_raw_transfer.PreparedTorchRawTransfer(
            tpu_dst_arrs[i], host_buffers[i]
        )
        for i in range(num_layers)
    ]

    for i in range(num_iterations):
      gc.disable()
      start = time.time()
      futures = [p.d2h_async() for p in prepared_d2h]
      for f in futures:
        f.Await()
      prepared_d2h_times.append(time.time() - start)

      gc.enable()
      gc.collect()
      gc.disable()

      start = time.time()
      futures = [p.h2d_async() for p in prepared_h2d]
      for f in futures:
        f.Await()
      torch.tpu.synchronize()
      prepared_h2d_times.append(time.time() - start)

      gc.enable()
      gc.collect()

    # Benchmark Native Torch/XLA .cpu() and .to()
    native_d2h_times = []
    native_h2d_times = []

    for i in range(num_iterations):
      gc.disable()
      start = time.time()
      cpu_tensors = [tpu.cpu() for tpu in tpu_arrs]
      native_d2h_times.append(time.time() - start)

      gc.enable()
      gc.collect()
      gc.disable()

      start = time.time()
      tpu_res = [cpu.to(self.device) for cpu in cpu_tensors]
      torch.tpu.synchronize()
      native_h2d_times.append(time.time() - start)

      gc.enable()
      gc.collect()

    element_size = 2
    total_bytes = np.prod(self.shape) * element_size * num_layers

    def report_perf(name, times):
      med_time = np.median(times)
      med_bw = total_bytes / med_time / (1024 * 1024 * 1024)
      print(
          f"PERF_REPORT | {name} BW: {med_bw:.3f} GB/s (median time:"
          f" {med_time:.6f}s)"
      )

    print("\n--- Performance Comparison (Torch TPU vs Native) ---")
    report_perf("Stateless Batch D2H", batched_d2h_times)
    report_perf("Stateless Batch H2D", batched_h2d_times)
    report_perf("Prepared Raw D2H", prepared_d2h_times)
    report_perf("Prepared Raw H2D", prepared_h2d_times)
    report_perf("Native torch.cpu() D2H", native_d2h_times)
    report_perf("Native tensor.to(tpu) H2D", native_h2d_times)
    print("----------------------------------------------------\n")


if __name__ == "__main__":
  absltest.main()
