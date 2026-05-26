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
import socket
import time

from absl.testing import absltest
import torch
import torch_tpu  # noqa: F401

from kv_cache import raiden_transfer_engine

os.environ["TORCH_TPU_TOPOLOGY"] = "1x1"

_PORT_OFFSET = 0


def _ports():
  global _PORT_OFFSET
  for _ in range(1000):
    base = 24000 + (os.getpid() % 1000) * 32 + _PORT_OFFSET
    _PORT_OFFSET += 4
    ports = (base, base + 1, base + 2, base + 3)
    if all(_port_available(port) for port in ports):
      return base, base + 2
  raise RuntimeError("failed to find free Raiden test ports")


def _port_available(port):
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
      sock.bind(("0.0.0.0", port))
    except OSError:
      return False
  return True


class RaidenTransferEngineTest(absltest.TestCase):

  def test_round_trip_arbitrary_blocks(self):
    device = torch.device("tpu:0")
    dtype = torch.bfloat16
    shape = (4, 128, 8, 2, 128)
    num_layers = 2
    refs = [torch.randn(shape).to(dtype) + layer for layer in range(num_layers)]
    src = [ref.to(device) for ref in refs]
    dst = [torch.zeros_like(ref, device="cpu").to(device) for ref in refs]
    torch.tpu.synchronize()

    producer_port, _ = _ports()
    engine = raiden_transfer_engine.RaidenTransferEngine(
        src, 0, producer_port, 2, 2, 30.0
    )

    self.assertTrue(engine.uses_prepared_tpu_buffers)
    self.assertEqual(engine.register_kv_cache(src), [0, 1])
    self.assertEqual(engine._count_copy_segments_for_testing([0, 1, 2, 6, 7, 9]),
                     3)
    self.assertEqual(
        engine._count_canonical_send_copy_segments_for_testing(
            [3, 2, 1, 0, 7, 6]
        ),
        2,
    )
    self.assertEqual(
        engine._count_canonical_load_copy_segments_for_testing(
            [3, 2, 1, 0, 7, 6], [13, 12, 11, 10, 23, 22]
        ),
        2,
    )
    send_plan = engine._send_copy_plan_for_testing([3, 2, 1, 0, 7, 6])
    self.assertEqual(send_plan["producer_remote_block_ids"], [0, 1, 2, 3, 6, 7])
    self.assertEqual(send_plan["d2h_copy"]["src_offsets"], [0, 6])
    self.assertEqual(send_plan["d2h_copy"]["dst_offsets"], [0, 4])
    self.assertEqual(send_plan["d2h_copy"]["sizes"], [4, 2])

    load_plan = engine._load_copy_plan_for_testing([3, 1], [0, 2])
    self.assertEqual(load_plan["producer_remote_block_ids"], [1, 3])
    self.assertEqual(load_plan["h2d_local_block_ids"], [0, 2])
    self.assertEqual(load_plan["host_dst_to_src"], [1, 0])
    self.assertTrue(load_plan["requires_host_reorder"])
    self.assertEqual(load_plan["h2d_copy"]["src_offsets"], [0, 1])
    self.assertEqual(load_plan["h2d_copy"]["dst_offsets"], [0, 2])
    self.assertEqual(load_plan["h2d_copy"]["sizes"], [1, 1])

    future, src_refs, host_views, total_bytes = engine.stage_d2h(
        slot_idx=0, num_blocks=2, block_ids=[3, 1]
    )
    self.assertLen(src_refs, num_layers)
    self.assertLen(host_views, num_layers)
    self.assertGreater(total_bytes, 0)
    self.assertTrue(all(view.is_pinned() for view in host_views))
    future.wait()

    self.assertEqual(engine.register_kv_cache(dst), [0, 1])
    issue_ms, wait_ms, total_ms, h2d_bytes = engine.commit_h2d(
        slot_idx=0, num_blocks=2, local_block_ids=[0, 2]
    )
    self.assertGreaterEqual(issue_ms, 0)
    self.assertGreaterEqual(wait_ms, 0)
    self.assertGreaterEqual(total_ms, 0)
    self.assertEqual(h2d_bytes, total_bytes)

    torch.tpu.synchronize()
    for layer_idx in range(num_layers):
      got = dst[layer_idx].cpu()
      torch.testing.assert_close(got[0:1], refs[layer_idx][3:4])
      torch.testing.assert_close(got[2:3], refs[layer_idx][1:2])

  def test_register_send_submit_load_over_socket_transport(self):
    device = torch.device("tpu:0")
    dtype = torch.bfloat16
    shape = (4, 128, 8, 2, 128)
    num_layers = 2
    refs = [torch.randn(shape).to(dtype) + layer for layer in range(num_layers)]
    src = [ref.to(device) for ref in refs]
    dst = [torch.zeros_like(ref, device="cpu").to(device) for ref in refs]
    torch.tpu.synchronize()

    producer_port, consumer_port = _ports()
    producer = raiden_transfer_engine.RaidenTransferEngine(
        src, 0, producer_port, 2, 2, 30.0
    )
    consumer = raiden_transfer_engine.RaidenTransferEngine(
        dst, 0, consumer_port, 2, 2, 30.0
    )

    uuid = 123456
    producer.register_send("prefill-req", uuid, [3, 1])
    consumer.submit_load("decode-req", uuid, f"127.0.0.1:{producer_port}",
                         [3, 1], [0, 2])

    deadline = time.time() + 30
    done_recv = set()
    done_send = set()
    while time.time() < deadline:
      sent, _, _ = producer.poll_finished()
      _, recv, failed = consumer.poll_finished()
      self.assertEmpty(failed)
      done_send.update(sent)
      done_recv.update(recv)
      if "prefill-req" in done_send and "decode-req" in done_recv:
        break
      time.sleep(0.05)
    self.assertIn("prefill-req", done_send)
    self.assertIn("decode-req", done_recv)

    torch.tpu.synchronize()
    for layer_idx in range(num_layers):
      got = dst[layer_idx].cpu()
      torch.testing.assert_close(got[0:1], refs[layer_idx][3:4])
      torch.testing.assert_close(got[2:3], refs[layer_idx][1:2])

  def test_socket_transport_loads_registered_subset(self):
    device = torch.device("tpu:0")
    dtype = torch.bfloat16
    shape = (4, 128, 8, 2, 128)
    num_layers = 2
    refs = [torch.randn(shape).to(dtype) + layer for layer in range(num_layers)]
    src = [ref.to(device) for ref in refs]
    dst = [torch.zeros_like(ref, device="cpu").to(device) for ref in refs]
    torch.tpu.synchronize()

    producer_port, consumer_port = _ports()
    producer = raiden_transfer_engine.RaidenTransferEngine(
        src, 0, producer_port, 2, 2, 30.0
    )
    consumer = raiden_transfer_engine.RaidenTransferEngine(
        dst, 0, consumer_port, 2, 2, 30.0
    )

    uuid = 223344
    producer.register_send("prefill-req", uuid, [0, 1, 2, 3])
    consumer.submit_load("decode-req", uuid, f"127.0.0.1:{producer_port}",
                         [3], [1])

    deadline = time.time() + 30
    done_recv = set()
    done_send = set()
    while time.time() < deadline:
      sent, _, _ = producer.poll_finished()
      _, recv, failed = consumer.poll_finished()
      self.assertEmpty(failed)
      done_send.update(sent)
      done_recv.update(recv)
      if "prefill-req" in done_send and "decode-req" in done_recv:
        break
      time.sleep(0.05)
    self.assertIn("prefill-req", done_send)
    self.assertIn("decode-req", done_recv)

    torch.tpu.synchronize()
    for layer_idx in range(num_layers):
      got = dst[layer_idx].cpu()
      torch.testing.assert_close(got[1:2], refs[layer_idx][3:4])
      torch.testing.assert_close(got[0:1], torch.zeros_like(got[0:1]))
      torch.testing.assert_close(got[2:3], torch.zeros_like(got[2:3]))

  def test_empty_submit_load_ack_does_not_report_recv_done(self):
    device = torch.device("tpu:0")
    shape = (4, 128, 8, 2, 128)
    src = [torch.randn(shape).to(torch.bfloat16).to(device)]
    dst = [torch.zeros(shape, dtype=torch.bfloat16).to(device)]
    torch.tpu.synchronize()

    producer_port, consumer_port = _ports()
    producer = raiden_transfer_engine.RaidenTransferEngine(
        src, 0, producer_port, 1, 2, 30.0
    )
    consumer = raiden_transfer_engine.RaidenTransferEngine(
        dst, 0, consumer_port, 1, 2, 30.0
    )

    uuid = 654321
    producer.register_send("prefill-req", uuid, [0])
    consumer.submit_load("decode-req", uuid, f"127.0.0.1:{producer_port}",
                         [], [])

    deadline = time.time() + 30
    done_send = set()
    done_recv = set()
    failed_recv = set()
    while time.time() < deadline:
      sent, _, _ = producer.poll_finished()
      _, recv, failed = consumer.poll_finished()
      done_send.update(sent)
      done_recv.update(recv)
      failed_recv.update(failed)
      if "prefill-req" in done_send:
        break
      time.sleep(0.05)
    self.assertIn("prefill-req", done_send)
    self.assertNotIn("decode-req", done_recv)
    self.assertEmpty(failed_recv)


if __name__ == "__main__":
  absltest.main()
