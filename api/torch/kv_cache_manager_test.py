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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch

from api.torch.kv_cache_manager import KVCacheManager


class KVCacheManagerTorchTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Initialize PyTorch XLA accelerator device E2E
    self.device = torch.device("tpu")
    self.num_layers = 2
    self.num_shards = 1
    self.block_size = 2
    self.slice_byte_size = 16384 // 4  # float32 capacity

  @parameterized.named_parameters(
      ("fp32", torch.float32),
      ("int32", torch.int32),
  )
  def test_e2e_distributed_staging_offloads_and_socket_transceives(self, dtype):
    num_blocks = 2
    shape = (num_blocks * self.block_size, 128, 8)  # 2 blocks capacity

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

    # 2. Instantiate two real managers locally on ephemeral loopback ports!
    ws_source = KVCacheManager(
        src_tensors,
        block_size=self.block_size,
        local_port=0,
        host_blocks_to_allocate=num_blocks,
        parallelism=1,
    )
    ws_dest = KVCacheManager(
        dst_tensors,
        block_size=self.block_size,
        local_port=0,
        host_blocks_to_allocate=num_blocks,
        parallelism=1,
    )

    self.assertIsNotNone(ws_source.local_port)
    self.assertIsNotNone(ws_dest.local_port)

    peer_source = f"localhost:{ws_source.local_port}"
    peer_dest = f"localhost:{ws_dest.local_port}"

    # Validate baseline state (destination tensors on TPU are zeroed)
    for l in range(self.num_layers):
      for sh in range(self.num_shards):
        np.testing.assert_allclose(
            dst_tensors[l][sh].cpu().numpy(), 0.0, atol=1e-5
        )

    # ==========================================================================
    # Step 1 (Source): Offload caches from Device TPU to Host CPU staging buffer E2E!
    # ==========================================================================
    d2h_future = ws_source.d2h(
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
    write_ids, write_future = ws_source.h2h_write(peer_dest, src_block_ids=[0])
    write_future.Await()
    self.assertEqual(write_ids, [0])

    # Destination pulls block page 0 from source peer!
    # This will allocate block 1 on ws_dest (since block 0 is locked).
    read_ids, read_future = ws_dest.h2h_read(peer_source, src_block_ids=[0])
    read_future.Await()
    self.assertEqual(read_ids, [1])

    # ==========================================================================
    # Step 3 (Destination): Stage received weights from Host onto Device TPU HBM E2E!
    # ==========================================================================
    # Copy pushed data (host block 0, offset 0) to device block 0 (offset 0)
    h2d_future_push = ws_dest.h2d(
        src_offsets_major_dim=[0],
        dst_offsets_major_dim=[0],
        copy_sizes_major_dim=[self.block_size],
    )
    h2d_future_push.Await()

    # Copy pulled data (host block 1, offset 2) to device block 1 (offset 2)
    h2d_future_pull = ws_dest.h2d(
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


if __name__ == "__main__":
  absltest.main()
