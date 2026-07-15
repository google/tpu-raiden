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

# Copyright 2026 The TPU Raiden Authors. All Rights Reserved.
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
"""Tests for Raiden Controller high-level transfer API under rpc/."""

import asyncio
import concurrent.futures
import math
import socket
import threading
from unittest import mock
from absl.testing import absltest
from tpu_raiden.rpc import raiden_controller
from tpu_raiden.rpc import raiden_service_pb2


class DummyWorkerRpcClient(raiden_controller.WorkerRpcClient):

  async def start_transfer(self, target_id, transfer_plan) -> None:
    pass


class RecordingWorkerRpcClient(raiden_controller.WorkerRpcClient):

  def __init__(self, event_log=None, label=""):
    super().__init__()
    self.calls = []
    self.event_log = event_log
    self.label = label

  async def start_transfer(self, target_id, transfer_plan) -> None:
    self.calls.append((target_id, transfer_plan))
    if self.event_log is not None:
      self.event_log.append((self.label, target_id))


_FA_TAG = "fa"


def _pool_manifest(
    page_tokens,
    *,
    fingerprint_dtype="fp8",
    padded_fa_bytes=0,
    include_gdn=False,
):
  pools = []
  tags = [_FA_TAG] * 15
  if include_gdn:
    tags += ["gdn.conv"] * 45 + ["gdn.ssm"] * 45
  for pool_idx, tag in enumerate(tags):
    if tag == _FA_TAG:
      live_bytes = page_tokens * 1024
      stride_bytes = live_bytes + padded_fa_bytes
      unit_bytes = 1024
      num_units = page_tokens
    else:
      live_bytes = 64
      stride_bytes = 64
      unit_bytes = 64
      num_units = 1
    pool = raiden_service_pb2.PoolSpecProto(
        tag=tag,
        storage_index=pool_idx,
        block_stride_bytes=stride_bytes,
        num_blocks=4096,
        dtype_tag=fingerprint_dtype,
    )
    pool.regions.add(
        name=tag,
        offset_bytes=0,
        stride_bytes=unit_bytes,
        unit_bytes=unit_bytes,
        num_units=num_units,
        units_per_stride=1,
    )
    pools.append(pool)
  return pools


class RaidenControllerTest(absltest.TestCase):

  def _stage3_fixture(
      self,
      *,
      num_tokens=65536,
      dst_page_tokens=1024,
      src_page_slice_tokens=4096,
      src_fingerprint="fingerprint",
      dst_fingerprint="fingerprint",
      dst_padding=0,
      include_gdn=False,
      register_blocks=True,
  ):
    client = RecordingWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10000, worker_rpc_client=client
    )
    src_units = []
    for rank in range(8):
      unit = raiden_controller.RaidenId(
          job_name="prefill",
          job_replica_id=f"engine-rank{rank}",
          data_name="kv.fa",
          data_replica_idx=0,
      )
      src_units.append(unit)
      controller.register_work_unit(
          unit,
          [f"10.0.0.{rank + 1}:8000"],
          control_plane_rpc_address=f"10.0.0.{rank + 1}:9000",
          pool_manifest=_pool_manifest(4096, include_gdn=include_gdn),
          layout_fingerprint=src_fingerprint,
          page_tokens=4096,
          page_slice_tokens=src_page_slice_tokens,
          transfer_parallelism=8,
          transfer_rank=rank,
      )
      if register_blocks:
        first_rank_token = rank * src_page_slice_tokens
        cycle_tokens = src_page_slice_tokens * 8
        owned_cycles = (
            0
            if num_tokens <= first_rank_token
            else 1 + (num_tokens - 1 - first_rank_token) // cycle_tokens
        )
        chunks_per_block = 4096 // src_page_slice_tokens
        owned_blocks = math.ceil(owned_cycles / chunks_per_block)
        block_ids = [1000 + rank * 100 + i for i in range(owned_blocks)]
        controller.register_request_blocks("request", 123, unit, block_ids)

    dst_unit = raiden_controller.RaidenId(
        job_name="decode",
        job_replica_id="engine",
        data_name="kv.fa",
        data_replica_idx=3,
    )
    controller.register_work_unit(
        dst_unit,
        ["10.1.0.1:8000"],
        control_plane_rpc_address="10.1.0.1:9000",
        pool_manifest=_pool_manifest(
            dst_page_tokens,
            padded_fa_bytes=dst_padding,
            include_gdn=include_gdn,
        ),
        layout_fingerprint=dst_fingerprint,
        page_tokens=dst_page_tokens,
        page_slice_tokens=dst_page_tokens,
        transfer_parallelism=8,
        transfer_rank=0,
    )
    dst_ids = list(range(2000, 2000 + math.ceil(num_tokens / dst_page_tokens)))
    return controller, client, src_units, dst_unit, dst_ids

  def _build_stage3_plan(self, **kwargs):
    controller, client, src_units, dst_unit, dst_ids = self._stage3_fixture(
        **kwargs
    )
    plan = controller._build_fa_reshard_plan(
        src_units=src_units,
        dst_units=[dst_unit],
        src_metadata=controller._get_local_metadata(src_units),
        dst_metadata=controller._get_local_metadata([dst_unit]),
        req_id="request",
        uuid=123,
        dst_device_block_ids=dst_ids,
        num_tokens=kwargs.get("num_tokens", 65536),
        parallelism=8,
    )
    return controller, client, src_units, dst_unit, dst_ids, plan

  def test_stage3_registration_metadata_round_trip_and_replacement(self):
    bind_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    bind_sock.bind(("127.0.0.1", 0))
    port = bind_sock.getsockname()[1]
    bind_sock.close()

    controller = raiden_controller.RaidenController(port=port)
    server = raiden_controller.RaidenControllerServer(controller)
    server.start()
    facade = raiden_controller.RaidenControllerClientFacade(f"127.0.0.1:{port}")
    unit = raiden_controller.RaidenId("prefill", "engine-rank0", "kv.fa", 0)
    try:
      facade.register_work_unit(
          unit,
          ["127.0.0.1:8100"],
          control_plane_rpc_address="127.0.0.1:9100",
          pool_manifest=_pool_manifest(4096),
          layout_fingerprint="layout-v1",
          page_tokens=4096,
          page_slice_tokens=256,
          transfer_parallelism=8,
          transfer_rank=0,
      )
      metadata = facade.get_metadata()
      self.assertLen(metadata, 1)
      self.assertEqual(
          raiden_controller._raiden_id_from_proto(metadata[0].unit), unit
      )
      self.assertLen(metadata[0].pools, 15)
      self.assertEqual(metadata[0].layout_fingerprint, "layout-v1")
      self.assertEqual(metadata[0].page_tokens, 4096)
      self.assertEqual(metadata[0].page_slice_tokens, 256)
      self.assertEqual(metadata[0].transfer_parallelism, 8)
      self.assertEqual(metadata[0].transfer_rank, 0)

      facade.register_request_blocks("roundtrip", 44, unit, [7, 9])
      self.assertEqual(
          controller._lookup_request_blocks("roundtrip", 44, [unit]),
          {unit: [7, 9]},
      )
      facade.complete_request_blocks("roundtrip", 44, unit=unit)
      with self.assertRaisesRegex(ValueError, "Missing producer block"):
        controller._lookup_request_blocks("roundtrip", 44, [unit])

      facade.register_request_blocks("cancel-roundtrip", 45, unit, [11])
      self.assertTrue(
          facade.cancel_request_blocks_if_unclaimed("cancel-roundtrip", 45)
      )
      self.assertTrue(
          facade.cancel_request_blocks_if_unclaimed("cancel-roundtrip", 45)
      )
      with self.assertRaisesRegex(RuntimeError, "cancelled"):
        facade.register_request_blocks("cancel-roundtrip", 45, unit, [11])

      # Force release clears the cancellation tombstone. A subsequent lookup
      # claims the new snapshot, after which cancellation must lose.
      facade.release_request_blocks("cancel-roundtrip", 45)
      facade.register_request_blocks("cancel-roundtrip", 45, unit, [12])
      self.assertEqual(
          controller._lookup_request_blocks("cancel-roundtrip", 45, [unit]),
          {unit: [12]},
      )
      self.assertFalse(
          facade.cancel_request_blocks_if_unclaimed("cancel-roundtrip", 45)
      )
      facade.release_request_blocks("cancel-roundtrip", 45)

      # Registration is replacement, not a patch. Legacy re-registration
      # removes every optional Stage-3 field and stale control endpoint.
      facade.register_work_unit(unit, ["127.0.0.1:8200"])
      replaced = facade.get_metadata()[0]
      self.assertEmpty(replaced.pools)
      self.assertEmpty(replaced.layout_fingerprint)
      self.assertEqual(replaced.page_tokens, 0)
      self.assertEqual(replaced.page_slice_tokens, 0)
      self.assertEmpty(replaced.control_plane_rpc_address)
    finally:
      server.stop()
      server._thread.join(timeout=2)

  def test_stage3_golden_pcp8_to_dp8_plan_and_encoder(self):
    controller, client, src_units, dst_unit, dst_ids = self._stage3_fixture(
        src_page_slice_tokens=256
    )
    future = controller.start_transfer(
        src_units=src_units,
        dst_units=[dst_unit],
        req_id="request",
        dst_device_block_ids=dst_ids,
        dst_mem_type=raiden_controller.RaidenMemoryType.HBM,
        use_block_chunks=True,
        uuid=123,
        num_tokens=65536,
        parallelism=8,
    )
    asyncio.run(future.wait())
    plan = controller.get_plan("request")

    self.assertLen(plan.src_units, 8)
    self.assertLen(plan.transfer_pool_indices, 15)
    self.assertEqual(plan.transfer_pool_indices, list(range(15)))
    self.assertEqual(plan.expected_pushes_per_pool, 64)
    self.assertEqual(plan.dst_device_block_ids, dst_ids)
    self.assertEqual(client.calls[0][0], dst_unit)
    self.assertEqual({call[0] for call in client.calls[1:]}, set(src_units))

    entries = []
    for rank, unit in enumerate(src_units):
      unit_entries = plan.shard_push_schedules[unit][0]
      self.assertLen(unit_entries, 32)
      self.assertEqual(
          [entry[3] for entry in unit_entries],
          [offset * 256 * 1024 for offset in range(16)] * 2,
      )
      self.assertEqual(
          [entry[5] for entry in unit_entries],
          [1000 + rank * 100] * 16 + [1001 + rank * 100] * 16,
      )
      entries.extend(unit_entries)
    self.assertLen(entries, 256)
    for entry in entries:
      self.assertEqual(entry[4], 256 * 1024)
      self.assertEqual(entry[7:], (0, 0, 1))
    entries_by_dst_block = {
        block_id: [entry for entry in entries if entry[6] == block_id]
        for block_id in dst_ids
    }
    for page_entries in entries_by_dst_block.values():
      self.assertLen(page_entries, 4)
      self.assertEqual(
          sorted(entry[2] for entry in page_entries),
          [offset * 256 * 1024 for offset in range(4)],
      )

    sender_req = raiden_service_pb2.ControlRequest()
    sender_req.ParseFromString(
        client._encode_start_transfer(src_units[0], plan)
    )
    encoded = sender_req.start_transfer_request
    self.assertEqual(encoded.expected_pushes_per_pool, 64)
    self.assertEqual(list(encoded.transfer_pool_indices), list(range(15)))
    self.assertEqual(list(encoded.src_block_ids), [1000, 1001])
    self.assertEqual(list(encoded.dst_device_block_ids), dst_ids)
    self.assertEqual(encoded.parallelism, 8)
    self.assertEqual(encoded.num_tokens, 65536)
    self.assertEqual(encoded.dst_mem_type, raiden_service_pb2.MEMORY_TYPE_HBM)
    self.assertEqual(set(encoded.shard_push_schedules), {0})

    receiver_req = raiden_service_pb2.ControlRequest()
    receiver_req.ParseFromString(client._encode_start_transfer(dst_unit, plan))
    self.assertEqual(
        set(receiver_req.start_transfer_request.shard_push_schedules),
        set(range(8)),
    )

  def test_stage3_golden_pcp8_interleave256_tail_reconstructs_tokens(self):
    num_tokens = 65_023
    _, _, src_units, _, dst_ids, plan = self._build_stage3_plan(
        num_tokens=num_tokens, src_page_slice_tokens=256
    )

    token_bytes = 1024
    reconstructed = [None] * num_tokens
    entries = []
    for rank, unit in enumerate(src_units):
      for entry in plan.shard_push_schedules[unit][0]:
        entries.append(entry)
        src_block_id = entry[5]
        local_block = src_block_id - (1000 + rank * 100)
        self.assertIn(local_block, (0, 1))
        src_token_offset = entry[3] // token_bytes
        dst_page = entry[6] - dst_ids[0]
        dst_token_offset = entry[2] // token_bytes
        copied_tokens = entry[4] // token_bytes
        self.assertEqual(entry[3] % token_bytes, 0)
        self.assertEqual(entry[2] % token_bytes, 0)
        self.assertEqual(entry[4] % token_bytes, 0)
        self.assertGreater(copied_tokens, 0)
        self.assertLessEqual(copied_tokens, 256)

        chunk_in_block, offset_in_chunk = divmod(src_token_offset, 256)
        global_start = (
            local_block * (4096 * 8)
            + chunk_in_block * (256 * 8)
            + rank * 256
            + offset_in_chunk
        )
        destination_start = dst_page * 1024 + dst_token_offset
        self.assertEqual(global_start, destination_start)
        for offset in range(copied_tokens):
          destination_token = destination_start + offset
          self.assertIsNone(reconstructed[destination_token])
          reconstructed[destination_token] = global_start + offset

    self.assertLen(entries, 254)
    self.assertEqual(reconstructed, list(range(num_tokens)))
    self.assertEqual(
        [entry[4] for entry in entries].count(255 * token_bytes), 1
    )
    self.assertEqual(plan.expected_pushes_per_pool, 64)

  def test_stage3_rejects_invalid_page_slice_geometry(self):
    controller = raiden_controller.RaidenController(port=10000)
    unit = raiden_controller.RaidenId("prefill", "rank0", "kv.fa")
    with self.assertRaisesRegex(
        ValueError, "page_slice_tokens must divide page_tokens"
    ):
      controller.register_work_unit(
          unit,
          ["127.0.0.1:8000"],
          control_plane_rpc_address="127.0.0.1:9000",
          pool_manifest=_pool_manifest(4096),
          layout_fingerprint="fingerprint",
          page_tokens=4096,
          page_slice_tokens=300,
          transfer_parallelism=8,
          transfer_rank=0,
      )

    with self.assertRaisesRegex(ValueError, "less than transfer_parallelism"):
      controller.register_work_unit(
          unit,
          ["127.0.0.1:8000"],
          control_plane_rpc_address="127.0.0.1:9000",
          pool_manifest=_pool_manifest(4096),
          layout_fingerprint="fingerprint",
          page_tokens=4096,
          page_slice_tokens=256,
          transfer_parallelism=8,
          transfer_rank=8,
      )

  def test_stage3_two_controllers_arm_receiver_before_push(self):
    num_tokens = 65_023
    events = []
    src_controller, src_client, src_units, dst_unit, dst_ids = (
        self._stage3_fixture(num_tokens=num_tokens, src_page_slice_tokens=256)
    )
    src_client.event_log = events
    src_client.label = "source"

    dst_client = RecordingWorkerRpcClient(events, "destination")
    dst_controller = raiden_controller.RaidenController(
        port=0, worker_rpc_client=dst_client
    )
    dst_controller.register_work_unit(
        dst_unit,
        ["10.1.0.1:8000"],
        control_plane_rpc_address="10.1.0.1:9000",
        pool_manifest=_pool_manifest(1024),
        layout_fingerprint="fingerprint",
        page_tokens=1024,
        page_slice_tokens=1024,
        transfer_parallelism=8,
        transfer_rank=0,
    )

    def unused_port():
      sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      sock.bind(("127.0.0.1", 0))
      result = sock.getsockname()[1]
      sock.close()
      return result

    src_controller.port = unused_port()
    dst_controller.port = unused_port()
    src_server = raiden_controller.RaidenControllerServer(src_controller)
    dst_server = raiden_controller.RaidenControllerServer(dst_controller)
    src_server.start()
    dst_server.start()
    src_address = f"127.0.0.1:{src_controller.port}"
    dst_address = f"127.0.0.1:{dst_controller.port}"
    try:
      facade = raiden_controller.RaidenControllerClientFacade(dst_address)
      self.assertTrue(
          facade.coordinate_transfer(
              src_units=src_units,
              dst_units=[dst_unit],
              req_id="request",
              use_block_chunks=True,
              is_sender=False,
              uuid=123,
              src_controller_address=src_address,
              dst_controller_address=dst_address,
              dst_mem_type=raiden_controller.RaidenMemoryType.HBM,
              dst_device_block_ids=dst_ids,
              num_tokens=num_tokens,
              parallelism=8,
          )
      )
      self.assertLen(dst_client.calls, 1)
      self.assertLen(src_client.calls, 8)
      self.assertEqual(events[0], ("destination", dst_unit))
      self.assertTrue(all(label == "source" for label, _ in events[1:]))
      self.assertIsNotNone(src_controller.get_plan("request"))
      dst_plan = dst_controller.get_plan("request")
      self.assertIsNotNone(dst_plan)
      entries = [
          entry
          for unit_schedules in dst_plan.shard_push_schedules.values()
          for entry in unit_schedules[0]
      ]
      self.assertLen(entries, 254)
      self.assertTrue(any(entry[2] > 0 for entry in entries))
      self.assertEqual(dst_plan.expected_pushes_per_pool, 64)
    finally:
      src_server.stop()
      dst_server.stop()
      src_server._thread.join(timeout=2)
      dst_server._thread.join(timeout=2)

  def test_stage3_receiver_rejects_invalid_interleave_coverage(self):
    controller, _, _, _, dst_ids, plan = self._build_stage3_plan(
        num_tokens=65_023, src_page_slice_tokens=256
    )

    def schedules_with_first_entry_change(
        *, dst_offset_bytes=None, dst_offset_delta=0, size_delta=0
    ):
      schedules = {
          unit: {0: list(unit_schedules[0])}
          for unit, unit_schedules in plan.shard_push_schedules.items()
      }
      for unit, unit_schedules in schedules.items():
        for entry_idx, entry in enumerate(unit_schedules[0]):
          if entry[6] == dst_ids[0] and entry[2] == 0:
            changed = list(entry)
            changed[2] = (
                changed[2] + dst_offset_delta
                if dst_offset_bytes is None
                else dst_offset_bytes
            )
            changed[4] += size_delta
            unit_schedules[0][entry_idx] = tuple(changed)
            return schedules
      self.fail("Interleaved plan has no entry at destination offset zero")

    def register(schedules):
      return controller.register_fa_receiver_plan(
          src_units=plan.src_units,
          dst_units=plan.dst_units,
          req_id=plan.req_id,
          uuid=plan.uuid,
          shard_push_schedules=schedules,
          expected_pushes_per_pool=plan.expected_pushes_per_pool,
          transfer_pool_indices=plan.transfer_pool_indices,
          pool_dtype_tags=plan.pool_dtype_tags,
          dst_device_block_ids=plan.dst_device_block_ids,
          src_schedule_keys=plan.src_schedule_keys,
          parallelism=plan.parallelism,
          num_tokens=plan.num_tokens,
      )

    token_bytes = 1024
    with self.subTest("gap"):
      with self.assertRaisesRegex(ValueError, "coverage gap"):
        register(
            schedules_with_first_entry_change(
                dst_offset_delta=token_bytes, size_delta=-token_bytes
            )
        )
    with self.subTest("overlap"):
      with self.assertRaisesRegex(ValueError, "coverage overlap"):
        register(schedules_with_first_entry_change(size_delta=token_bytes))
    with self.subTest("out_of_range"):
      with self.assertRaisesRegex(ValueError, "exceeds its destination page"):
        register(
            schedules_with_first_entry_change(
                dst_offset_bytes=1023 * token_bytes
            )
        )

  def test_stage3_receiver_rejects_tampered_push_count_before_arm(self):
    controller, client, _, _, _, plan = self._build_stage3_plan(
        num_tokens=65_023, src_page_slice_tokens=256
    )

    with self.assertRaisesRegex(
        ValueError,
        "expected_pushes_per_pool does not match the received schedules",
    ):
      controller.register_fa_receiver_plan(
          src_units=plan.src_units,
          dst_units=plan.dst_units,
          req_id=plan.req_id,
          uuid=plan.uuid,
          shard_push_schedules=plan.shard_push_schedules,
          expected_pushes_per_pool=plan.expected_pushes_per_pool + 1,
          transfer_pool_indices=plan.transfer_pool_indices,
          pool_dtype_tags=plan.pool_dtype_tags,
          dst_device_block_ids=plan.dst_device_block_ids,
          src_schedule_keys=plan.src_schedule_keys,
          parallelism=plan.parallelism,
          num_tokens=plan.num_tokens,
      )

    self.assertEmpty(client.calls)
    self.assertIsNone(controller.get_plan(plan.req_id))

  def test_stage3_geometry_edges(self):
    _, _, _, _, dst_ids, tail_plan = self._build_stage3_plan(num_tokens=65024)
    self.assertLen(dst_ids, 64)
    tail_entry = next(
        entry
        for schedules in tail_plan.shard_push_schedules.values()
        for entry in schedules[0]
        if entry[6] == dst_ids[-1]
    )
    self.assertEqual(tail_entry[3], 3 * 1024**2)
    # 65,024 ends 512 tokens into its final decode page. The 3,584-token
    # quantity is the tail within the source page, not the final wire entry.
    self.assertEqual(tail_entry[4], 512 * 1024)
    self.assertEqual(tail_entry[7:], (0, 0, 1))

    _, _, n2_src_units, _, n2_ids, n2_plan = self._build_stage3_plan(
        dst_page_tokens=2048, src_page_slice_tokens=256
    )
    self.assertLen(n2_ids, 32)
    n2_entries = [
        entry
        for schedules in n2_plan.shard_push_schedules.values()
        for entry in schedules[0]
    ]
    self.assertLen(n2_entries, 256)
    self.assertTrue(all(entry[4] == 256 * 1024 for entry in n2_entries))
    for unit in n2_src_units:
      self.assertLen(n2_plan.shard_push_schedules[unit][0], 32)
    for dst_id in n2_ids:
      page_entries = [entry for entry in n2_entries if entry[6] == dst_id]
      self.assertLen(page_entries, 8)
      self.assertEqual(
          sorted(entry[2] for entry in page_entries),
          [offset * 256 * 1024 for offset in range(8)],
      )
    self.assertEqual(n2_plan.expected_pushes_per_pool, 64)

    for num_tokens, expected_size in ((1024, 1024**2), (511, 511 * 1024)):
      _, _, _, _, small_ids, small_plan = self._build_stage3_plan(
          num_tokens=num_tokens
      )
      self.assertLen(small_ids, 1)
      self.assertLen(small_plan.src_units, 1)
      only_entry = small_plan.shard_push_schedules[small_plan.src_units[0]][0][
          0
      ]
      self.assertEqual(only_entry[4], expected_size)
      self.assertEqual(small_plan.expected_pushes_per_pool, 1)

  def test_stage3_fingerprint_mismatch_precedes_worker_rpc(self):
    controller, client, src_units, dst_unit, dst_ids = self._stage3_fixture(
        dst_fingerprint="different"
    )
    future = controller.start_transfer(
        src_units=src_units,
        dst_units=[dst_unit],
        req_id="request",
        dst_device_block_ids=dst_ids,
        dst_mem_type=raiden_controller.RaidenMemoryType.HBM,
        use_block_chunks=True,
        uuid=123,
        num_tokens=65536,
    )
    with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
      asyncio.run(future.wait())
    self.assertEmpty(client.calls)
    self.assertIsNone(controller.get_plan("request"))

  def test_stage3_rejects_padded_decode_pool_with_gather_reference(self):
    controller, client, src_units, dst_unit, dst_ids = self._stage3_fixture(
        dst_padding=256
    )
    future = controller.start_transfer(
        src_units=src_units,
        dst_units=[dst_unit],
        req_id="request",
        dst_device_block_ids=dst_ids,
        dst_mem_type=raiden_controller.RaidenMemoryType.HBM,
        use_block_chunks=True,
        uuid=123,
        num_tokens=65536,
    )
    with self.assertRaisesRegex(
        ValueError, "RESHARD_STAGE3_P0_2_FP8_HEAD_GATHER.md"
    ):
      asyncio.run(future.wait())
    self.assertEmpty(client.calls)
    self.assertIsNone(controller.get_plan("request"))

  def test_stage3_fa_filter_and_gdn_skip_accounting(self):
    _, _, _, _, _, plan = self._build_stage3_plan(include_gdn=True)
    self.assertEqual(plan.transfer_pool_indices, list(range(15)))
    self.assertEqual(plan.skipped_pool_counts, {"gdn.conv": 45, "gdn.ssm": 45})
    self.assertLen(plan.pool_dtype_tags, 105)

  def test_stage3_block_argument_validation_is_atomic(self):
    controller, client, src_units, dst_unit, dst_ids = self._stage3_fixture(
        register_blocks=False
    )
    with self.assertRaisesRegex(ValueError, "block_ids.*non-negative"):
      controller.register_request_blocks("bad", 123, src_units[0], [-1])
    with self.assertRaisesRegex(ValueError, "duplicates"):
      controller.register_request_blocks("bad", 123, src_units[0], [1, 1])
    controller.register_request_blocks("bad", 123, src_units[0], [1])
    with self.assertRaisesRegex(ValueError, "Conflicting"):
      controller.register_request_blocks("bad", 123, src_units[0], [2])

    cases = (
        ([-1] + dst_ids[1:], "non-negative"),
        ([dst_ids[0], dst_ids[0]] + dst_ids[2:], "duplicates"),
        (dst_ids[:-1], "page count"),
    )
    for bad_ids, message in cases:
      future = controller.start_transfer(
          src_units=src_units,
          dst_units=[dst_unit],
          req_id="request",
          dst_device_block_ids=bad_ids,
          dst_mem_type=raiden_controller.RaidenMemoryType.HBM,
          use_block_chunks=True,
          uuid=123,
          num_tokens=65536,
      )
      with self.assertRaisesRegex(ValueError, message):
        asyncio.run(future.wait())
      self.assertIsNone(controller.get_plan("request"))
      self.assertEmpty(client.calls)

    future = controller.start_transfer(
        src_units=src_units,
        dst_units=[dst_unit],
        req_id="request",
        dst_device_block_ids=dst_ids,
        dst_mem_type=raiden_controller.RaidenMemoryType.HBM,
        use_block_chunks=True,
        uuid=123,
        num_tokens=65536,
    )
    with self.assertRaisesRegex(ValueError, "Missing producer block"):
      asyncio.run(future.wait())
    self.assertIsNone(controller.get_plan("request"))
    self.assertEmpty(client.calls)

  def test_stage3_planner_failure_abandons_its_request_block_claim(self):
    controller, _, src_units, dst_unit, dst_ids = self._stage3_fixture(
        register_blocks=False
    )
    for rank, unit in enumerate(src_units):
      block_ids = [1000 + rank * 100, 1001 + rank * 100]
      if rank == 0:
        block_ids.pop()
      controller.register_request_blocks("request", 123, unit, block_ids)

    with self.assertRaisesRegex(ValueError, "Source block-id count"):
      controller._build_fa_reshard_plan(
          src_units=src_units,
          dst_units=[dst_unit],
          src_metadata=controller._get_local_metadata(src_units),
          dst_metadata=controller._get_local_metadata([dst_unit]),
          req_id="request",
          uuid=123,
          dst_device_block_ids=dst_ids,
          num_tokens=65536,
          parallelism=8,
      )

    self.assertTrue(
        controller.cancel_request_blocks_if_unclaimed("request", 123)
    )

  def test_duplicate_planner_cannot_abandon_another_attempts_claim(self):
    controller, _, src_units, dst_unit, dst_ids, first_plan = (
        self._build_stage3_plan()
    )

    with self.assertRaisesRegex(ValueError, "another planning attempt"):
      controller._build_fa_reshard_plan(
          src_units=src_units,
          dst_units=[dst_unit],
          src_metadata=controller._get_local_metadata(src_units),
          dst_metadata=controller._get_local_metadata([dst_unit]),
          req_id="request",
          uuid=123,
          dst_device_block_ids=dst_ids,
          num_tokens=65536,
          parallelism=8,
      )

    self.assertFalse(
        controller.cancel_request_blocks_if_unclaimed("request", 123)
    )
    self.assertEqual(
        controller._lookup_request_blocks(
            "request",
            123,
            src_units,
            claim_owner=first_plan.request_block_claim_owner,
        ),
        first_plan.src_block_ids,
    )

  def test_receiver_arm_failure_abandons_claim_before_sender_dispatch(self):
    calls = []

    class FailingReceiverWorkerRpcClient(raiden_controller.WorkerRpcClient):

      async def start_transfer(self, target_id, transfer_plan) -> None:
        calls.append(target_id)
        if target_id in transfer_plan.dst_units:
          raise RuntimeError("receiver arm failed")

    controller, original_client, src_units, dst_unit, dst_ids = (
        self._stage3_fixture()
    )
    controller.worker_rpc_client = FailingReceiverWorkerRpcClient(
        endpoint_addresses=original_client.get_worker_endpoints()
    )
    future = controller.start_transfer(
        src_units=src_units,
        dst_units=[dst_unit],
        req_id="request",
        dst_device_block_ids=dst_ids,
        dst_mem_type=raiden_controller.RaidenMemoryType.HBM,
        use_block_chunks=True,
        uuid=123,
        num_tokens=65536,
    )

    with self.assertRaisesRegex(RuntimeError, "receiver arm failed"):
      asyncio.run(future.wait())

    self.assertEqual(calls, [dst_unit])
    self.assertIsNone(controller.get_plan("request"))
    self.assertTrue(
        controller.cancel_request_blocks_if_unclaimed("request", 123)
    )

  def test_raw_legacy_release_without_force_is_rejected(self):
    controller = raiden_controller.RaidenController(port=0)
    server = raiden_controller.RaidenControllerServer(controller)
    server.start()
    facade = raiden_controller.RaidenControllerClientFacade(
        f"127.0.0.1:{server.port}"
    )
    request = raiden_service_pb2.ControlRequest(
        command=(
            raiden_service_pb2.ControlRequest.COMMAND_RELEASE_REQUEST_BLOCKS
        ),
        release_request_blocks_request=(
            raiden_service_pb2.ReleaseRequestBlocksRequest(
                req_id="legacy-release", uuid=123
            )
        ),
    )
    try:
      with self.assertRaisesRegex(
          RuntimeError, "Legacy rank-local release is unsafe"
      ):
        facade._send_raiden_protobuf_rpc(request)
    finally:
      server.stop()

  def test_request_block_registry_ttl_and_worker_restart_cleanup(self):
    controller = raiden_controller.RaidenController(
        port=10000, request_registry_ttl_s=1.0
    )
    unit = raiden_controller.RaidenId("prefill", "rank0", "kv.fa")
    controller.register_work_unit(unit, ["127.0.0.1:8000"])
    with mock.patch.object(
        raiden_controller.time, "monotonic", side_effect=[10.0, 12.0]
    ):
      controller.register_request_blocks("expired", 77, unit, [1])
      with self.assertRaisesRegex(ValueError, "Missing producer block"):
        controller._lookup_request_blocks("expired", 77, [unit])

    with mock.patch.object(
        raiden_controller.time, "monotonic", return_value=20.0
    ):
      controller.register_request_blocks("restart", 78, unit, [2])
      controller.register_work_unit(unit, ["127.0.0.1:8001"])
      with self.assertRaisesRegex(ValueError, "Missing producer block"):
        controller._lookup_request_blocks("restart", 78, [unit])

  def test_worker_restart_cannot_replace_a_claimed_snapshot(self):
    controller = raiden_controller.RaidenController(port=10000)
    unit = raiden_controller.RaidenId("prefill", "rank0", "kv.fa")
    controller.register_work_unit(unit, ["127.0.0.1:8000"])
    controller.register_request_blocks("claimed-restart", 79, unit, [10])
    self.assertEqual(
        controller._lookup_request_blocks("claimed-restart", 79, [unit]),
        {unit: [10]},
    )

    controller.register_work_unit(unit, ["127.0.0.1:8001"])
    with self.assertRaisesRegex(ValueError, "already claimed"):
      controller.register_request_blocks("claimed-restart", 79, unit, [99])
    with self.assertRaisesRegex(ValueError, "Missing producer block"):
      controller._lookup_request_blocks("claimed-restart", 79, [unit])

  def test_request_block_cancellation_wins_and_is_idempotent(self):
    controller = raiden_controller.RaidenController(port=10000)
    unit = raiden_controller.RaidenId("prefill", "rank0", "kv.fa")
    controller.register_work_unit(unit, ["127.0.0.1:8000"])
    controller.register_request_blocks("cancelled", 81, unit, [1, 2])

    self.assertTrue(
        controller.cancel_request_blocks_if_unclaimed("cancelled", 81)
    )
    self.assertTrue(
        controller.cancel_request_blocks_if_unclaimed("cancelled", 81)
    )
    with self.assertRaisesRegex(ValueError, "cancelled"):
      controller._lookup_request_blocks("cancelled", 81, [unit])
    self.assertEqual(controller.release_request_blocks("cancelled", 81), 0)

  def test_request_block_cancellation_seals_late_registration(self):
    controller = raiden_controller.RaidenController(port=10000)
    unit = raiden_controller.RaidenId("prefill", "rank0", "kv.fa")
    controller.register_work_unit(unit, ["127.0.0.1:8000"])

    self.assertTrue(controller.cancel_request_blocks_if_unclaimed("late", 82))
    with self.assertRaisesRegex(ValueError, "cancelled"):
      controller.register_request_blocks("late", 82, unit, [3])

    # Force release explicitly ends the sealed lifecycle generation.
    self.assertEqual(controller.release_request_blocks("late", 82), 0)
    controller.register_request_blocks("late", 82, unit, [3])
    self.assertEqual(
        controller._lookup_request_blocks("late", 82, [unit]),
        {unit: [3]},
    )

  def test_request_block_lookup_claim_wins_over_cancellation(self):
    controller = raiden_controller.RaidenController(port=10000)
    unit = raiden_controller.RaidenId("prefill", "rank0", "kv.fa")
    controller.register_work_unit(unit, ["127.0.0.1:8000"])
    controller.register_request_blocks("claimed", 83, unit, [4])

    self.assertEqual(
        controller._lookup_request_blocks("claimed", 83, [unit]),
        {unit: [4]},
    )
    self.assertFalse(
        controller.cancel_request_blocks_if_unclaimed("claimed", 83)
    )
    self.assertEqual(
        controller._lookup_request_blocks("claimed", 83, [unit]),
        {unit: [4]},
    )
    self.assertEqual(controller.release_request_blocks("claimed", 83), 1)

    # Force release clears the claim without changing normal deletion count.
    controller.register_request_blocks("claimed", 83, unit, [5])
    self.assertTrue(
        controller.cancel_request_blocks_if_unclaimed("claimed", 83)
    )

  def test_request_block_release_waits_for_all_claimed_units(self):
    controller = raiden_controller.RaidenController(port=10000)
    units = [
        raiden_controller.RaidenId("prefill", f"rank{rank}", "kv.fa")
        for rank in range(2)
    ]
    for rank, unit in enumerate(units):
      controller.register_work_unit(unit, [f"127.0.0.1:{8000 + rank}"])
      controller.register_request_blocks("aggregate", 86, unit, [rank])

    self.assertEqual(
        controller._lookup_request_blocks("aggregate", 86, units),
        {units[0]: [0], units[1]: [1]},
    )
    self.assertEqual(
        controller.complete_request_blocks("aggregate", 86, unit=units[0]),
        0,
    )
    self.assertFalse(
        controller.cancel_request_blocks_if_unclaimed("aggregate", 86)
    )
    self.assertEqual(
        controller._lookup_request_blocks("aggregate", 86, units),
        {units[0]: [0], units[1]: [1]},
    )
    self.assertEqual(
        controller.complete_request_blocks("aggregate", 86, unit=units[1]),
        2,
    )
    with self.assertRaisesRegex(ValueError, "Missing producer block"):
      controller._lookup_request_blocks("aggregate", 86, units)

  def test_request_block_terminal_vote_before_claim_keeps_d5_snapshot(self):
    controller = raiden_controller.RaidenController(port=10000)
    units = [
        raiden_controller.RaidenId("prefill", f"rank{rank}", "kv.fa")
        for rank in range(2)
    ]
    for rank, unit in enumerate(units):
      controller.register_work_unit(unit, [f"127.0.0.1:{8100 + rank}"])
      controller.register_request_blocks(
          "preclaim-terminal", 87, unit, [] if rank == 0 else [4]
      )

    self.assertEqual(
        controller.complete_request_blocks(
            "preclaim-terminal", 87, unit=units[0]
        ),
        0,
    )
    self.assertEqual(
        controller._lookup_request_blocks("preclaim-terminal", 87, units),
        {units[0]: [], units[1]: [4]},
    )
    self.assertEqual(
        controller.complete_request_blocks(
            "preclaim-terminal", 87, unit=units[1]
        ),
        2,
    )

  def test_nonempty_request_block_cannot_complete_before_claim(self):
    controller = raiden_controller.RaidenController(port=10000)
    unit = raiden_controller.RaidenId("prefill", "rank0", "kv.fa")
    controller.register_work_unit(unit, ["127.0.0.1:8000"])
    controller.register_request_blocks("premature-terminal", 89, unit, [3])

    with self.assertRaisesRegex(ValueError, "empty producer registration"):
      controller.complete_request_blocks("premature-terminal", 89, unit)
    self.assertEqual(
        controller._lookup_request_blocks("premature-terminal", 89, [unit]),
        {unit: [3]},
    )

  def test_claim_refreshes_preclaim_terminal_vote_ttl(self):
    controller = raiden_controller.RaidenController(
        port=10000, request_registry_ttl_s=1.0
    )
    units = [
        raiden_controller.RaidenId("prefill", f"rank{rank}", "kv.fa")
        for rank in range(2)
    ]
    for rank, unit in enumerate(units):
      controller.register_work_unit(unit, [f"127.0.0.1:{8200 + rank}"])
      controller.register_request_blocks(
          "late-claim", 88, unit, [] if rank == 0 else [rank]
      )

    with mock.patch.object(
        raiden_controller.time, "monotonic", return_value=10.0
    ):
      self.assertEqual(
          controller.complete_request_blocks("late-claim", 88, unit=units[0]), 0
      )
    with mock.patch.object(
        raiden_controller.time, "monotonic", return_value=10.9
    ):
      self.assertEqual(
          controller._lookup_request_blocks("late-claim", 88, units),
          {
              units[0]: [],
              units[1]: [1],
          },
      )
    with mock.patch.object(
        raiden_controller.time, "monotonic", return_value=11.1
    ):
      self.assertEqual(
          controller.complete_request_blocks("late-claim", 88, unit=units[1]), 2
      )

  def test_request_block_lifecycle_markers_expire(self):
    controller = raiden_controller.RaidenController(
        port=10000, request_registry_ttl_s=1.0
    )
    unit = raiden_controller.RaidenId("prefill", "rank0", "kv.fa")
    controller.register_work_unit(unit, ["127.0.0.1:8000"])

    with mock.patch.object(
        raiden_controller.time,
        "monotonic",
        side_effect=[10.0, 10.5, 12.0, 12.1],
    ):
      self.assertTrue(
          controller.cancel_request_blocks_if_unclaimed("cancel-expiry", 84)
      )
      with self.assertRaisesRegex(ValueError, "cancelled"):
        controller.register_request_blocks("cancel-expiry", 84, unit, [6])
      controller.register_request_blocks("cancel-expiry", 84, unit, [6])
      self.assertEqual(
          controller._lookup_request_blocks("cancel-expiry", 84, [unit]),
          {unit: [6]},
      )

    with mock.patch.object(
        raiden_controller.time,
        "monotonic",
        side_effect=[20.0, 20.1, 22.0],
    ):
      controller.register_request_blocks("claim-expiry", 85, unit, [7])
      controller._lookup_request_blocks("claim-expiry", 85, [unit])
      self.assertTrue(
          controller.cancel_request_blocks_if_unclaimed("claim-expiry", 85)
      )

  def test_request_block_lookup_cancel_race_has_single_winner(self):
    controller = raiden_controller.RaidenController(port=10000)
    unit = raiden_controller.RaidenId("prefill", "rank0", "kv.fa")
    controller.register_work_unit(unit, ["127.0.0.1:8000"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
      for generation in range(25):
        req_id = f"race-{generation}"
        uuid = 100 + generation
        controller.register_request_blocks(req_id, uuid, unit, [generation])
        barrier = threading.Barrier(3)

        def lookup():
          barrier.wait(timeout=5)
          try:
            return controller._lookup_request_blocks(req_id, uuid, [unit])
          except ValueError as error:
            if "cancelled" not in str(error):
              raise
            return None

        def cancel():
          barrier.wait(timeout=5)
          return controller.cancel_request_blocks_if_unclaimed(req_id, uuid)

        lookup_future = executor.submit(lookup)
        cancel_future = executor.submit(cancel)
        barrier.wait(timeout=5)
        lookup_result = lookup_future.result(timeout=5)
        cancel_result = cancel_future.result(timeout=5)

        if cancel_result:
          self.assertIsNone(lookup_result)
        else:
          self.assertEqual(lookup_result, {unit: [generation]})
        controller.release_request_blocks(req_id, uuid)

  def test_dynamic_balancing_and_overlap_planner(self):
    dummy_client = DummyWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10000, worker_rpc_client=dummy_client
    )

    src_unit_0 = raiden_controller.RaidenId(
        job_name="sampler",
        job_replica_id="0",
        data_name="kv_cache",
    )
    src_unit_1 = raiden_controller.RaidenId(
        job_name="sampler",
        job_replica_id="1",
        data_name="kv_cache",
    )
    target_unit = raiden_controller.RaidenId(
        job_name="inference_server",
        job_replica_id="215",
        data_name="kv_cache",
    )

    controller.register_work_unit(
        src_unit_0, ["10.0.0.1:8000", "10.0.0.2:8000"]
    )
    controller.register_work_unit(
        src_unit_1, ["10.0.0.3:8000", "10.0.0.4:8000"]
    )
    controller.register_work_unit(
        target_unit, ["10.0.0.5:8000", "10.0.0.6:8000"]
    )

    # First transfer routes to src_unit_0
    future_1 = controller.start_transfer(
        src_units=[src_unit_0, src_unit_1],
        dst_units=[target_unit],
    )
    asyncio.run(future_1.wait())

    self.assertTrue(future_1.done())
    self.assertEqual(future_1.session_id, 0)

    plan_1 = controller.get_plan("req_0")
    self.assertEqual(plan_1.src_units[0].job_replica_id, "0")

    # Verify generalized NDSlice fully qualified overlap push schedule dict
    self.assertIn(src_unit_0, plan_1.plan)
    unit_0_plan = plan_1.plan[src_unit_0]
    self.assertLen(unit_0_plan, 2)
    self.assertEqual(unit_0_plan[0], [(target_unit, 0, [[(0, 2)]])])
    self.assertEqual(unit_0_plan[1], [(target_unit, 1, [[(0, 2)]])])

    # Second transfer routes dynamically to least-loaded src_unit_1
    future_2 = controller.start_transfer(
        src_units=[src_unit_0, src_unit_1],
        dst_units=[target_unit],
    )
    self.assertEqual(future_2.session_id, 1)

    plan_2 = controller.get_plan("req_1")
    self.assertEqual(plan_2.src_units[0].job_replica_id, "1")
    asyncio.run(future_2.wait())

  def test_fan_out_multiple_targets(self):
    controller = raiden_controller.RaidenController(
        port=10001, worker_rpc_client=DummyWorkerRpcClient()
    )

    src = raiden_controller.RaidenId(
        job_name="trainer",
        job_replica_id="0",
        data_name="layer0.weights",
    )
    target_0 = raiden_controller.RaidenId(
        job_name="inference_server",
        job_replica_id="10",
        data_name="layer0.weights",
    )
    target_1 = raiden_controller.RaidenId(
        job_name="inference_server",
        job_replica_id="11",
        data_name="layer0.weights",
    )

    controller.register_work_unit(src, ["10.0.0.1:8000"])
    controller.register_work_unit(target_0, ["10.0.0.2:8000"])
    controller.register_work_unit(target_1, ["10.0.0.3:8000"])

    future_multi = controller.start_transfer(
        src_units=[src],
        dst_units=[target_0, target_1],
    )
    self.assertEqual(future_multi.session_id, 0)

    plan_multi = controller.get_plan("req_0")
    self.assertLen(plan_multi.dst_units, 2)
    asyncio.run(future_multi.wait())

  def test_rpc_client_push_coordination(self):
    recorded_actions = []

    class MockWorkerClient(raiden_controller.WorkerRpcClient):

      async def start_transfer(self, orchestrator_id, plan) -> None:
        recorded_actions.append(("start", [orchestrator_id]))

    mock_client = MockWorkerClient()
    controller = raiden_controller.RaidenController(
        port=10002, worker_rpc_client=mock_client
    )

    src = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )
    dst = raiden_controller.RaidenId(
        job_name="sampler", job_replica_id="0", data_name="weights"
    )
    controller.register_work_unit(src, ["10.0.0.1:8000"])
    controller.register_work_unit(dst, ["10.0.0.2:8000"])

    future = controller.start_transfer(
        src_units=[src],
        dst_units=[dst],
    )

    asyncio.run(future.wait())

    self.assertEqual(
        recorded_actions,
        [
            ("start", [dst]),
            ("start", [src]),
        ],
    )

  def test_enforce_metadata_completeness(self):
    controller = raiden_controller.RaidenController(port=10003)
    unit = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )

    # 1. Failure if some shape/layout fields are missing
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda e: "all of them must be provided" in str(e)
    ):
      controller.register_work_unit(
          unit,
          ["10.0.0.1:8000"],
          mesh_shape=[1, 1, 4, 1, 1],
          # layout and global_shape are missing
      )

    # 2. Failure if itemsize is missing when metadata is provided
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda e: "itemsize must be provided" in str(e)
    ):
      controller.register_work_unit(
          unit,
          ["10.0.0.1:8000"],
          mesh_shape=[1, 1, 4, 1, 1],
          layout=[4, 3, 2, 1, 0],
          global_shape=[128, 16, 8, 2, 128],
          # itemsize is missing
      )

    # 3. Success if all are provided
    controller.register_work_unit(
        unit,
        ["10.0.0.1:8000"],
        mesh_shape=[1, 1, 4, 1, 1],
        layout=[4, 3, 2, 1, 0],
        global_shape=[128, 16, 8, 2, 128],
        itemsize=4,
    )

  def test_multi_shard_worker_resharding(self):
    dummy_client = DummyWorkerRpcClient()
    controller = raiden_controller.RaidenController(
        port=10004, worker_rpc_client=dummy_client
    )
    src = raiden_controller.RaidenId("trainer", "0", "weights")
    dst = raiden_controller.RaidenId("sampler", "1", "weights")

    controller.register_work_unit(
        src,
        ["10.0.0.1:8000", "10.0.0.1:8001", "10.0.0.1:8002", "10.0.0.1:8003"],
        mesh_shape=(2, 2),
        layout=(1, 0),
        global_shape=(128, 1024),
        itemsize=4,
    )
    controller.register_work_unit(
        dst,
        ["10.0.0.2:8000", "10.0.0.2:8001", "10.0.0.2:8002", "10.0.0.2:8003"],
        mesh_shape=(1, 4),
        layout=(1, 0),
        global_shape=(128, 1024),
        itemsize=4,
    )

    future = controller.start_transfer(
        src_units=[src],
        dst_units=[dst],
        use_block_chunks=True,
    )
    asyncio.run(future.wait())
    plan = controller.get_plan("req_0")
    self.assertIn(src, plan.shard_push_schedules)
    schedules = plan.shard_push_schedules[src]
    self.assertSetEqual(set(schedules.keys()), {0, 1, 2, 3})

  def test_greedy_tree_broadcast(self):
    recorded_calls = []

    class MockBroadcastWorkerClient(raiden_controller.WorkerRpcClient):

      def __init__(self):
        super().__init__()
        self.endpoints = {}

      def get_worker_endpoints(self):
        return self.endpoints

      async def start_transfer(self, target_id, transfer_plan) -> None:
        is_sender = transfer_plan.is_sender
        recorded_calls.append((target_id, is_sender))

    mock_client = MockBroadcastWorkerClient()
    controller = raiden_controller.RaidenController(
        port=10004, worker_rpc_client=mock_client
    )
    controller.broadcast_k = 2

    src = raiden_controller.RaidenId(
        job_name="trainer", job_replica_id="0", data_name="weights"
    )
    target_0 = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="0", data_name="weights"
    )
    target_1 = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="1", data_name="weights"
    )
    target_2 = raiden_controller.RaidenId(
        job_name="inference_server", job_replica_id="2", data_name="weights"
    )

    controller.register_work_unit(
        src,
        ["10.0.0.1:8000"],
        mesh_shape=[1, 1],
        layout=[1, 0],
        global_shape=[128, 128],
        itemsize=4,
        control_plane_rpc_address="10.0.0.1:9000",
    )
    controller.register_work_unit(
        target_0,
        ["10.0.0.2:8000"],
        mesh_shape=[1, 1],
        layout=[1, 0],
        global_shape=[128, 128],
        itemsize=4,
        control_plane_rpc_address="10.0.0.2:9000",
    )
    controller.register_work_unit(
        target_1,
        ["10.0.0.3:8000"],
        mesh_shape=[1, 1],
        layout=[1, 0],
        global_shape=[128, 128],
        itemsize=4,
        control_plane_rpc_address="10.0.0.3:9000",
    )
    controller.register_work_unit(
        target_2,
        ["10.0.0.4:8000"],
        mesh_shape=[1, 1],
        layout=[1, 0],
        global_shape=[128, 128],
        itemsize=4,
        control_plane_rpc_address="10.0.0.4:9000",
    )

    future = controller.start_transfer(
        src_units=[src],
        dst_units=[target_0, target_1, target_2],
        use_block_chunks=True,
    )
    asyncio.run(future.wait())

    self.assertTrue(future.done())
    self.assertGreater(len(recorded_calls), 0)


if __name__ == "__main__":
  absltest.main()
