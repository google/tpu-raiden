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

"""E2E physical unit tests for KVCacheManager on XLA TPUs."""

import time

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch

from tpu_raiden.api.torch import kv_cache_manager

BufferSpec = kv_cache_manager.BufferSpec
KVCacheManager = kv_cache_manager.KVCacheManager


class KVCacheManagerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Initialize PyTorch XLA accelerator device E2E
    self.device = torch.device("tpu")
    self.num_layers = 1
    self.block_size = 1

  def test_initialization(self):
    device = self.device
    shape = (4, 128, 8)
    kv_caches = [torch.zeros(shape, device=device)]

    manager = KVCacheManager(
        kv_caches=kv_caches,
        node_id=0,
        local_control_port=0,
        max_blocks=4,
        num_slots=2,
    )
    self.assertIsNotNone(manager)

  def test_e2e_transfer_polling(self):
    num_blocks = 2
    shape = (num_blocks, 128, 8)

    src_caches = []
    for _ in range(self.num_layers):
      src_caches.append(
          torch.full(
              shape, fill_value=1.0, dtype=torch.float32, device=self.device
          )
      )

    dst_caches = []
    for _ in range(self.num_layers):
      dst_caches.append(
          torch.zeros(shape, dtype=torch.float32, device=self.device)
      )

    producer = KVCacheManager(
        kv_caches=src_caches,
        node_id=0,
        local_control_port=0,
        max_blocks=2,
        num_slots=2,
    )

    consumer = KVCacheManager(
        kv_caches=dst_caches,
        node_id=0,
        local_control_port=0,
        max_blocks=2,
        num_slots=2,
    )

    port = producer.local_control_port
    self.assertGreater(port, 0)

    req_id = "test_req_poll"
    uuid = 12345
    producer.register_read(req_id, uuid, [0, 1])

    remote_endpoint = f"127.0.0.1:{port}"
    consumer.start_read(
        req_id=req_id,
        uuid=uuid,
        remote_endpoint=remote_endpoint,
        remote_block_ids=[0, 1],
        local_block_ids=[0, 1],
    )

    # Poll until consumer is done receiving
    done = False
    for _ in range(50):
      _, done_recving, failed_recving = consumer.poll_stats()
      if req_id in failed_recving:
        self.fail("Transfer failed")
      if req_id in done_recving:
        done = True
        break
      time.sleep(0.1)

    self.assertTrue(done, "Consumer did not finish transfer in time")

    # Check that consumer correctly loaded the values
    for t in dst_caches:
      np.testing.assert_allclose(t.cpu().numpy(), 1.0, atol=1e-5)

    # Poll producer until it's done sending
    done_prod = False
    for _ in range(50):
      done_sending, _, _ = producer.poll_stats()
      if req_id in done_sending:
        done_prod = True
        break
      time.sleep(0.1)

    self.assertTrue(done_prod, "Producer did not finish sending in time")

  def test_parallel_pull(self):
    num_blocks = 2
    shape = (num_blocks, 128, 8)

    src_caches = []
    for _ in range(self.num_layers):
      src_caches.append(
          torch.full(
              shape, fill_value=2.0, dtype=torch.float32, device=self.device
          )
      )

    dst_caches = []
    for _ in range(self.num_layers):
      dst_caches.append(
          torch.zeros(shape, dtype=torch.float32, device=self.device)
      )

    producer = KVCacheManager(
        kv_caches=src_caches,
        node_id=0,
        local_control_port=0,
        max_blocks=2,
        num_slots=2,
    )

    consumer = KVCacheManager(
        kv_caches=dst_caches,
        node_id=0,
        local_control_port=0,
        max_blocks=2,
        num_slots=2,
    )

    port = producer.local_control_port
    self.assertGreater(port, 0)

    req_id = "test_req_parallel"
    uuid = 99999
    producer.register_read(req_id, uuid, [0, 1])

    remote_endpoint = f"127.0.0.1:{port}"
    consumer.start_read(
        req_id=req_id,
        uuid=uuid,
        remote_endpoint=remote_endpoint,
        remote_block_ids=[0, 1],
        local_block_ids=[0, 1],
        parallelism=2,
    )

    done = False
    for _ in range(50):
      _, done_recving, failed_recving = consumer.poll_stats()
      if req_id in failed_recving:
        self.fail("Transfer failed")
      if req_id in done_recving:
        done = True
        break
      time.sleep(0.1)

    self.assertTrue(done, "Consumer did not finish transfer in time")
    time.sleep(0.5)

    for t in dst_caches:
      np.testing.assert_allclose(t.cpu().numpy(), 2.0, atol=1e-5)

    done_prod = False
    for _ in range(50):
      done_sending, _, _ = producer.poll_stats()
      if req_id in done_sending:
        done_prod = True
        break
      time.sleep(0.1)

    self.assertTrue(done_prod, "Producer did not finish sending in time")

  def test_unified_pool_requires_slice_byte_size(self):
    """A flat 1D tensor without explicit geometry is rejected."""
    pool = torch.zeros(4096, dtype=torch.int8, device=self.device)

    with self.assertRaisesRegex(ValueError, "slice_byte_size"):
      KVCacheManager(
          kv_caches=[pool],
          node_id=0,
          local_control_port=0,
          max_blocks=4,
          num_slots=2,
      )

  def test_unified_pool_e2e_transfer_polling(self):
    """Byte-exact block pull between two flat unified pools."""
    num_blocks = 2
    slice_byte_size = 1024
    total = num_blocks * slice_byte_size

    # Producer pool: block 0 = 0,1,2,... (mod 127), block 1 = zeros.
    src_host = torch.zeros(total, dtype=torch.int8)
    src_host[:slice_byte_size] = (
        torch.arange(slice_byte_size, dtype=torch.int64)
        .remainder(127)
        .to(torch.int8)
    )
    src_pool = src_host.to(self.device)
    dst_pool = torch.zeros(total, dtype=torch.int8, device=self.device)

    producer = KVCacheManager(
        kv_caches=[src_pool],
        node_id=0,
        local_control_port=0,
        max_blocks=2,
        num_slots=2,
        buffer_spec=BufferSpec(slice_byte_size=slice_byte_size),
    )
    consumer = KVCacheManager(
        kv_caches=[dst_pool],
        node_id=0,
        local_control_port=0,
        max_blocks=2,
        num_slots=2,
        buffer_spec=BufferSpec(slice_byte_size=slice_byte_size),
    )

    port = producer.local_control_port
    self.assertGreater(port, 0)

    req_id = "test_req_unified_pool"
    uuid = 424242
    producer.register_read(req_id, uuid, [0])

    # Pull producer block 0 into consumer block 1: exercises non-identity
    # block addressing (offset = block_id * slice_byte_size) on both sides.
    consumer.start_read(
        req_id=req_id,
        uuid=uuid,
        remote_endpoint=f"127.0.0.1:{port}",
        remote_block_ids=[0],
        local_block_ids=[1],
    )

    done = False
    for _ in range(50):
      _, done_recving, failed_recving = consumer.poll_stats()
      if req_id in failed_recving:
        self.fail("Transfer failed")
      if req_id in done_recving:
        done = True
        break
      time.sleep(0.1)

    self.assertTrue(done, "Consumer did not finish transfer in time")

    got = dst_pool.cpu().numpy()
    expected_block = src_host[:slice_byte_size].numpy()
    np.testing.assert_array_equal(
        got[slice_byte_size:],
        expected_block,
        err_msg="consumer block 1 must be byte-identical to producer block 0",
    )
    np.testing.assert_array_equal(
        got[:slice_byte_size],
        np.zeros(slice_byte_size, dtype=np.int8),
        err_msg="consumer block 0 must be untouched",
    )


if __name__ == "__main__":
  absltest.main()
