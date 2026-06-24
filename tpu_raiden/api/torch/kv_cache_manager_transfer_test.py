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

"""E2E physical unit tests for KVCacheManager transfer on XLA TPUs."""

import socket
import threading
import time

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch

from tpu_raiden.api.torch.kv_cache_manager import KVCacheManager
from tpu_raiden.rpc import raiden_service_pb2


class KVCacheManagerTransferTest(parameterized.TestCase):

  def _send_control_request(self, port, req):
    payload = req.SerializeToString()
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM, 0)
    sock.connect(("::1", port))
    sock.sendall(len(payload).to_bytes(4, "big") + payload)
    resp_len = int.from_bytes(sock.recv(4), "big")
    resp_bytes = sock.recv(resp_len)
    resp = raiden_service_pb2.ControlResponse()
    resp.ParseFromString(resp_bytes)
    sock.close()
    return resp

  def setUp(self):
    super().setUp()
    # Initialize PyTorch XLA accelerator device E2E
    self.device = torch.device("tpu")
    self.num_layers = 1
    self.block_size = 1

  def test_initialization(self):
    shape = (4, 128, 8)
    kv_caches = [torch.zeros(shape, device=self.device)]

    engine = KVCacheManager(
        kv_caches=kv_caches,
        node_id=0,
        local_control_port=0,
        max_blocks=4,
        num_slots=2,
    )
    self.assertIsNotNone(engine)

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

    # Use getattr just in case local_control_port was completely hidden
    port = getattr(producer, "local_control_port", 0)
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
      done_sending, done_recving, failed_recving = consumer.poll_stats()
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
      done_sending, done_recving, failed_recving = producer.poll_stats()
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

    port = getattr(producer, "local_control_port", 0)
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
      done_sending, done_recving, failed_recving = consumer.poll_stats()
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
      done_sending, done_recving, failed_recving = producer.poll_stats()
      if req_id in done_sending:
        done_prod = True
        break
      time.sleep(0.1)

    self.assertTrue(done_prod, "Producer did not finish sending in time")

  def test_e2e_transfer_polling_controller(self):
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
        local_control_port=-1,
        max_blocks=2,
        num_slots=2,
    )

    consumer = KVCacheManager(
        kv_caches=dst_caches,
        node_id=0,
        local_control_port=-1,
        max_blocks=2,
        num_slots=2,
    )

    self.assertNotEmpty(producer.listener_addresses)
    self.assertNotEmpty(consumer.listener_addresses)
    producer_port = int(producer.listener_addresses[0].split(":")[-1])
    consumer_port = int(consumer.listener_addresses[0].split(":")[-1])

    req_id = "test_req_poll"
    uuid = 12345
    block_size = np.prod(src_caches[0].shape[1:]) * src_caches[0].element_size()

    consumer_eps = consumer.get_local_endpoints()
    self.assertLen(consumer_eps, 1)
    dst_peer = consumer_eps[0]["endpoint"]

    # Construct StartTransferRequest
    transfer_req = raiden_service_pb2.StartTransferRequest(
        uuid=uuid,
        dst_mem_type=raiden_service_pb2.MEMORY_TYPE_HBM,
    )
    schedule = transfer_req.shard_push_schedules[0]
    for bid in [0, 1]:
      entry = schedule.entries.add()
      entry.dst_peer = dst_peer
      entry.dst_shard_idx = 0
      entry.src_block_id = bid
      entry.dst_block_id = bid
      entry.src_offset_bytes = bid * block_size
      entry.dst_offset_bytes = bid * block_size
      entry.size_bytes = block_size

    # 1. Prepare Consumer (Receiver)
    consumer_req = raiden_service_pb2.ControlRequest(
        command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
    )
    consumer_req.start_transfer_request.CopyFrom(transfer_req)
    consumer_req.start_transfer_request.is_sender = False
    resp = self._send_control_request(consumer_port, consumer_req)
    self.assertTrue(resp.success)

    # 2. Trigger Producer (Sender)
    producer_req = raiden_service_pb2.ControlRequest(
        command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
    )
    producer_req.start_transfer_request.CopyFrom(transfer_req)
    producer_req.start_transfer_request.is_sender = True
    resp = self._send_control_request(producer_port, producer_req)
    self.assertTrue(resp.success)

    # Poll until consumer is done receiving
    expected_req_id = f"resharded_transfer_{uuid}"
    done = False
    for _ in range(50):
      _, done_recving, failed_recving = consumer.poll_stats()
      if expected_req_id in failed_recving:
        self.fail("Transfer failed")
      if expected_req_id in done_recving:
        done = True
        break
      time.sleep(0.1)

    self.assertTrue(done, "Consumer did not finish transfer in time")

    # Check that consumer correctly loaded the values
    for t in dst_caches:
      np.testing.assert_allclose(t.cpu().numpy(), 1.0, atol=1e-5)

  def test_parallel_pull_controller(self):
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
        local_control_port=-1,
        max_blocks=2,
        num_slots=2,
    )

    consumer = KVCacheManager(
        kv_caches=dst_caches,
        node_id=0,
        local_control_port=-1,
        max_blocks=2,
        num_slots=2,
    )

    self.assertNotEmpty(producer.listener_addresses)
    self.assertNotEmpty(consumer.listener_addresses)
    producer_port = int(producer.listener_addresses[0].split(":")[-1])
    consumer_port = int(consumer.listener_addresses[0].split(":")[-1])

    req_id = "test_req_parallel"
    uuid = 99999
    block_size = np.prod(src_caches[0].shape[1:]) * src_caches[0].element_size()

    consumer_eps = consumer.get_local_endpoints()
    self.assertLen(consumer_eps, 1)
    dst_peer = consumer_eps[0]["endpoint"]

    # Construct StartTransferRequest
    transfer_req = raiden_service_pb2.StartTransferRequest(
        uuid=uuid,
        dst_mem_type=raiden_service_pb2.MEMORY_TYPE_HBM,
    )
    schedule = transfer_req.shard_push_schedules[0]
    for bid in [0, 1]:
      entry = schedule.entries.add()
      entry.dst_peer = dst_peer
      entry.dst_shard_idx = 0
      entry.src_block_id = bid
      entry.dst_block_id = bid
      entry.src_offset_bytes = bid * block_size
      entry.dst_offset_bytes = bid * block_size
      entry.size_bytes = block_size

    # 1. Prepare Consumer (Receiver)
    consumer_req = raiden_service_pb2.ControlRequest(
        command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
    )
    consumer_req.start_transfer_request.CopyFrom(transfer_req)
    consumer_req.start_transfer_request.is_sender = False
    resp = self._send_control_request(consumer_port, consumer_req)
    self.assertTrue(resp.success)

    # 2. Trigger Producer (Sender)
    producer_req = raiden_service_pb2.ControlRequest(
        command=raiden_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
    )
    producer_req.start_transfer_request.CopyFrom(transfer_req)
    producer_req.start_transfer_request.is_sender = True
    resp = self._send_control_request(producer_port, producer_req)
    self.assertTrue(resp.success)

    expected_req_id = f"resharded_transfer_{uuid}"
    done = False
    for _ in range(50):
      _, done_recving, failed_recving = consumer.poll_stats()
      if expected_req_id in failed_recving:
        self.fail("Transfer failed")
      if expected_req_id in done_recving:
        done = True
        break
      time.sleep(0.1)

    self.assertTrue(done, "Consumer did not finish transfer in time")
    time.sleep(0.5)

    for t in dst_caches:
      np.testing.assert_allclose(t.cpu().numpy(), 2.0, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
