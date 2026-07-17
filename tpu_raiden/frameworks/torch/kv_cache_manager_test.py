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

"""E2E physical unit tests for PyTorch KVCacheManager on XLA TPUs."""

import faulthandler

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
from torch_tpu._internal import execution_mode
from torch_tpu._internal import sync

from tpu_raiden.frameworks.torch import _tpu_raiden_torch as _kv_cache_manager

EagerMode = execution_mode.EagerMode

# Upper bound on how long constructing a manager over lazy KV caches may take.
# The constructor holds the GIL, so a regressed deadlock freezes the whole
# process and no Python-thread watchdog can fire; faulthandler dumps every
# thread stack and hard-exits the process past this bound so a regression fails
# loudly instead of hanging the test runner.
_CONSTRUCTION_DEADLOCK_TIMEOUT_S = 60.0


class KVCacheManagerTorchTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Initialize PyTorch XLA accelerator device E2E
    self.device = torch.device("tpu")
    self.num_layers = 2
    self.num_shards = 1
    self.block_size = 2
    self.slice_byte_size = 16384 // 4  # float32 capacity

  def _new_manager(self, tensors, num_blocks):
    """Constructs a manager, hard-failing if construction hangs.

    Construction unpacks each cache and awaits its device buffer; a cache still
    carrying a deferred op whose graph is never materialized would block here
    forever. The constructor holds the GIL, so a hang freezes the process and no
    Python-thread watchdog could fire -- guard it with faulthandler instead.
    """
    faulthandler.dump_traceback_later(
        _CONSTRUCTION_DEADLOCK_TIMEOUT_S, exit=True
    )
    try:
      return _kv_cache_manager.KVCacheManager(
          tensors,
          local_port=0,
          host_blocks_to_allocate=num_blocks * self.block_size,
          parallelism=1,
      )
    finally:
      faulthandler.cancel_dump_traceback_later()

  @parameterized.named_parameters(
      ("fp32", torch.float32, False),
      ("int32", torch.int32, False),
      ("fp32_deferred", torch.float32, True),
      ("int32_deferred", torch.int32, True),
  )
  def test_e2e_distributed_staging_offloads_and_socket_transceives(
      self, dtype, defer
  ):
    num_blocks = 2
    shape = (num_blocks * self.block_size, 128, 8)  # 2 blocks capacity

    if defer:
      # Create and register the caches while they still carry a pending deferred
      # op (vLLM's DEFER_AND_FUSE mode), exercising the in-place materialization
      # that construction performs. If that materialization forked a separate
      # buffer instead of filling the live one, the H2d in step 3 would write
      # into the copy and the parity check in step 4 would fail.
      cm = execution_mode.set_eager_mode(EagerMode.DEFER_AND_FUSE)
      cm.__enter__()
      self.addCleanup(cm.__exit__, None, None, None)

    # 1. Allocate sharded eager tensors directly on the physical XLA TPU devices!
    # Layer 0 and Layer 1 for Source (Trainer)
    src_tensors = []
    for l in range(self.num_layers):
      shards = []
      for sh in range(self.num_shards):
        # Distinct layer val patterns: Layer 0=1.0, Layer 1=2.0
        val = float(l + 1.0)
        t = torch.full(shape, fill_value=val, dtype=dtype, device=self.device)
        shards.append(t)
      src_tensors.append(shards)

    # Layer 0 and Layer 1 for Destination (Inference, initially zeroed)
    dst_tensors = []
    for l in range(self.num_layers):
      shards = []
      for sh in range(self.num_shards):
        t = torch.zeros(shape, dtype=dtype, device=self.device)
        shards.append(t)
      dst_tensors.append(shards)

    if defer:
      # Guard: the caches must still be deferred here, or this variant would not
      # exercise construction-time materialization at all.
      self.assertFalse(sync.is_materialized(src_tensors[0][0]))
      self.assertFalse(sync.is_materialized(dst_tensors[0][0]))

    # 2. Instantiate two real managers locally on ephemeral loopback ports!
    ws_source = self._new_manager(src_tensors, num_blocks)
    ws_dest = self._new_manager(dst_tensors, num_blocks)

    self.assertIsNotNone(ws_source.local_port)
    self.assertIsNotNone(ws_dest.local_port)

    peer_source = ws_source.get_local_endpoints()[0]["endpoint"]
    peer_dest = ws_dest.get_local_endpoints()[0]["endpoint"]

    # Validate baseline state (destination tensors on TPU are zeroed)
    for l in range(self.num_layers):
      for sh in range(self.num_shards):
        np.testing.assert_allclose(
            dst_tensors[l][sh].cpu().numpy(), 0.0, atol=1e-5
        )

    # ==========================================================================
    # Step 1 (Source): Offload caches from Device TPU to Host CPU staging buffer E2E!
    # ==========================================================================
    d2h_future = ws_source.D2h(
        src_offsets_major_dim=[0],
        dst_offsets_major_dim=[0],
        copy_sizes_major_dim=[self.block_size],
    )
    d2h_future.Await()

    # ==========================================================================
    # Step 2 (Network): Push and Pull E2E!
    # ==========================================================================
    # Source pushes block page 0 to destination peer server!
    # This will allocate block 0 on ws_dest.
    write_ids, write_future = ws_source.H2hWrite(
        peer_dest, src_block_ids=[0, 1]
    )
    write_future.Await()
    self.assertEqual(write_ids, [0, 1])

    # Destination pulls block page 0 from source peer!
    # This will allocate block 1 on ws_dest (since block 0 is locked).
    read_ids, read_future = ws_dest.H2hRead(peer_source, src_block_ids=[0, 1])
    read_future.Await()
    self.assertEqual(read_ids, [2, 3])

    # ==========================================================================
    # Step 3 (Destination): Stage received weights from Host onto Device TPU HBM E2E!
    # ==========================================================================
    # Copy pushed data (host block 0, offset 0) to device block 0 (offset 0)
    h2d_future_push = ws_dest.H2d(
        src_offsets_major_dim=[0],
        dst_offsets_major_dim=[0],
        copy_sizes_major_dim=[self.block_size],
    )
    h2d_future_push.Await()

    # Copy pulled data (host block 1, offset 2) to device block 1 (offset 2)
    h2d_future_pull = ws_dest.H2d(
        src_offsets_major_dim=[2],
        dst_offsets_major_dim=[2],
        copy_sizes_major_dim=[self.block_size],
    )
    h2d_future_pull.Await()

    # ==========================================================================
    # Step 4: Verify complete numerical parity on the destination TPU devices!
    # ==========================================================================
    for l in range(self.num_layers):
      for sh in range(self.num_shards):
        expected_val = float(l + 1.0)
        actual_data = dst_tensors[l][sh].cpu().numpy()
        # Verify both blocks have the expected values
        np.testing.assert_allclose(actual_data[0:2], expected_val, atol=1e-5)
        np.testing.assert_allclose(actual_data[2:4], expected_val, atol=1e-5)

  def test_construction_over_lazy_kv_caches_does_not_deadlock(self):
    """Constructing a manager over still-deferred KV caches must not hang.

    Under DEFER_AND_FUSE (vLLM's execution mode) a KV cache registered right
    after allocation still carries a pending deferred op. UnpackTorchTensor
    resolves each tensor's base buffer by awaiting it, and the await only
    completes once the deferred graph has been executed. The unpack forces that
    execution before awaiting; without it the constructor waits on a graph that
    only this (now-blocked) thread could flush.
    """
    num_blocks = 2
    shape = (num_blocks * self.block_size, 128, 8)

    with execution_mode.set_eager_mode(EagerMode.DEFER_AND_FUSE):
      lazy_tensors = [
          [torch.ones(shape, dtype=torch.float32, device=self.device)]
          for _ in range(self.num_layers)
      ]
      # The regression only exists for unmaterialized tensors; assert the
      # tensors are actually deferred so a future eager-materialization change
      # can't silently turn this into a no-op test.
      for layer in lazy_tensors:
        for shard in layer:
          self.assertFalse(sync.is_materialized(shard))

      manager = self._new_manager(lazy_tensors, num_blocks)

    # Unpack awaited each base buffer, so the tensors are materialized now.
    for layer in lazy_tensors:
      for shard in layer:
        self.assertTrue(sync.is_materialized(shard))
    self.assertIsNotNone(manager.local_port)

  def test_construction_over_host_transferred_kv_caches(self):
    """KV caches created as `cpu_tensor.to(device)` must unpack.

    vLLM allocates KV caches as `torch.zeros(shape, dtype).to(device)`. Under
    DEFER_AND_FUSE the resulting tensor's TensorImpl may not expose a device
    until the transfer materializes, so unpack must not gate on
    `tensor.device()` before resolving and materializing the base buffer.
    """
    num_blocks = 2
    shape = (num_blocks * self.block_size, 128, 8)

    with execution_mode.set_eager_mode(EagerMode.DEFER_AND_FUSE):
      transferred = [
          [torch.zeros(shape, dtype=torch.float32).to(self.device)]
          for _ in range(self.num_layers)
      ]
      manager = self._new_manager(transferred, num_blocks)

    for layer in transferred:
      for shard in layer:
        self.assertTrue(sync.is_materialized(shard))
    self.assertIsNotNone(manager.local_port)


if __name__ == "__main__":
  absltest.main()
