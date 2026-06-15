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
        tp_rank=0,
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
        tp_rank=0,
        local_control_port=0,
        max_blocks=2,
        num_slots=2,
    )

    consumer = KVCacheManager(
        kv_caches=dst_caches,
        tp_rank=0,
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
        tp_rank=0,
        local_control_port=0,
        max_blocks=2,
        num_slots=2,
    )

    consumer = KVCacheManager(
        kv_caches=dst_caches,
        tp_rank=0,
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


if __name__ == "__main__":
  absltest.main()
