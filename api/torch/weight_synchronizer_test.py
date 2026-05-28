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

"""E2E physical integration tests for PyTorch WeightSynchronizer on XLA TPUs."""

import os
import time

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
import torch_tpu

from api.torch.weight_synchronizer import WeightSynchronizer


class WeightSynchronizerTorchTest(parameterized.TestCase):

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
  def test_e2e_3node_distributed_weight_push_and_pull(self, dtype):
    shape = (self.block_size, 128, 8)  # 16384 bytes capacity per layer shard

    # 1. Allocate source (Trainer) weights on Device TPU
    src_tensors = []
    for l in range(self.num_layers):
      shards = []
      for sh in range(self.num_shards):
        t = torch.zeros(shape, dtype=dtype, device=self.device)
        shards.append(t)
      src_tensors.append(shards)

    # Allocate destination 1 (Inference Peer 1) weights
    dst1_tensors = []
    for l in range(self.num_layers):
      shards = []
      for sh in range(self.num_shards):
        t = torch.zeros(shape, dtype=dtype, device=self.device)
        shards.append(t)
      dst1_tensors.append(shards)

    # Allocate destination 2 (Inference Peer 2) weights
    dst2_tensors = []
    for l in range(self.num_layers):
      shards = []
      for sh in range(self.num_shards):
        t = torch.zeros(shape, dtype=dtype, device=self.device)
        shards.append(t)
      dst2_tensors.append(shards)

    # 2. Instantiate destination WeightSynchronizers on ephemeral ports!
    ws_dest1 = WeightSynchronizer(dst1_tensors, local_port=0, parallelism=1)
    ws_dest2 = WeightSynchronizer(dst2_tensors, local_port=0, parallelism=1)

    self.assertIsNotNone(ws_dest1.local_port)
    self.assertIsNotNone(ws_dest2.local_port)

    peer_dest1 = f"localhost:{ws_dest1.local_port}"
    peer_dest2 = f"localhost:{ws_dest2.local_port}"

    # ==========================================================================
    # Scenario A: Test the Push API E2E (1 Source pushes to 2 Destinations!)
    # ==========================================================================
    # Trainer fills source weights with distinct values per layer
    for l in range(self.num_layers):
      for sh in range(self.num_shards):
        val = float(l + 10.0)  # Layer 0=10.0, Layer 1=11.0
        src_tensors[l][sh].fill_(val)

    # Recreate/Instantiate ws_source to capture filled buffers!
    ws_source = WeightSynchronizer(src_tensors, local_port=0, parallelism=1)
    self.assertIsNotNone(ws_source.local_port)
    peer_source = f"localhost:{ws_source.local_port}"

    # Source pushes weights to both dest1 and dest2 socket servers E2E!
    ws_source.push_weights([peer_dest1, peer_dest2])

    # Assert both destinations have received the trainer's weights on their TPU HBM!
    for l in range(self.num_layers):
      for sh in range(self.num_shards):
        expected_val = float(l + 10.0)
        np.testing.assert_allclose(
            dst1_tensors[l][sh].cpu().numpy(), expected_val, atol=1e-5
        )
        np.testing.assert_allclose(
            dst2_tensors[l][sh].cpu().numpy(), expected_val, atol=1e-5
        )

    # ==========================================================================
    # Scenario B: Test the Pull API E2E (2 Destinations pull from 1 Source!)
    # ==========================================================================
    # Trainer updates source weights with brand new values
    for l in range(self.num_layers):
      for sh in range(self.num_shards):
        val = float(l + 20.0)  # Layer 0=20.0, Layer 1=21.0
        src_tensors[l][sh].fill_(val)

    # Recreate ws_source to capture new buffers!
    ws_source = WeightSynchronizer(src_tensors, local_port=0, parallelism=1)
    self.assertIsNotNone(ws_source.local_port)
    peer_source = f"localhost:{ws_source.local_port}"

    # Self-push to populate host buffer!
    ws_source.push_weights([peer_source])

    # Dest1 and Dest2 independently pull weights from the trainer source!
    ws_dest1.pull_weights(peer_source)
    ws_dest2.pull_weights(peer_source)

    # Assert both destinations have successfully pulled the trainer's new weights E2E!
    for l in range(self.num_layers):
      for sh in range(self.num_shards):
        expected_val = float(l + 20.0)
        np.testing.assert_allclose(
            dst1_tensors[l][sh].cpu().numpy(), expected_val, atol=1e-5
        )
        np.testing.assert_allclose(
            dst2_tensors[l][sh].cpu().numpy(), expected_val, atol=1e-5
        )


if __name__ == "__main__":
  absltest.main()
