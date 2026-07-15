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
"""Permanent host-only integration coverage for Stage-3 FA resharding."""

from __future__ import annotations

import asyncio
import concurrent.futures
import gc
import math
import threading
import time

from absl.testing import absltest
from tpu_raiden.api.torch import kv_cache_manager
from tpu_raiden.api.torch import pool_layout
from tpu_raiden.rpc import raiden_controller


_PCP_RANKS = 8
_PREFILL_PAGE_TOKENS = 8
_PREFILL_PAGE_SLICE_TOKENS = 2
_DECODE_PAGE_TOKENS = 2
_TOKEN_BYTES = 4
_SOURCE_BLOCK_BYTES = _PREFILL_PAGE_TOKENS * _TOKEN_BYTES
_DESTINATION_BLOCK_BYTES = _DECODE_PAGE_TOKENS * _TOKEN_BYTES
_SOURCE_BLOCKS = 4
_DESTINATION_BLOCKS = 128
_DECODE_MANAGERS = 2
_LAYOUT_FINGERPRINT = "stage3-host-pcp8-fa-v1"
_FA_SENTINEL = bytes([0xA0]) * _DESTINATION_BLOCK_BYTES
_GDN_SENTINELS = {
    1: bytes([0xC1]) * _DESTINATION_BLOCK_BYTES,
    2: bytes([0xD2]) * _DESTINATION_BLOCK_BYTES,
}


def _dense_pool(
    tag: str,
    storage_index: int,
    block_bytes: int,
    num_blocks: int,
) -> pool_layout.PoolSpec:
  return pool_layout.PoolSpec(
      tag=tag,
      storage_index=storage_index,
      base_offset_bytes=0,
      block_stride_bytes=block_bytes,
      num_blocks=num_blocks,
      regions=(
          pool_layout.RegionSpec(
              name="payload",
              offset_bytes=0,
              stride_bytes=block_bytes,
              unit_bytes=block_bytes,
              num_units=1,
          ),
      ),
      dtype_tag="host-bytes-v1",
  )


def _pool_manifest(block_bytes: int, num_blocks: int):
  return (
      _dense_pool("fa", 0, block_bytes, num_blocks),
      _dense_pool("gdn.conv", 1, block_bytes, num_blocks),
      _dense_pool("gdn.ssm", 2, block_bytes, num_blocks),
  )


def _token_payload(seed: int, token: int) -> bytes:
  return bytes(
      (seed * 37 + token * 11 + byte_index * 3) & 0xFF
      for byte_index in range(_TOKEN_BYTES)
  )


class _Stage3HostCluster:
  """Owns eight PCP managers, two decode managers, and two controllers."""

  def __init__(
      self,
      source_timeout_s: float = 120.0,
      destination_timeout_s: float = 120.0,
  ):
    self.source_timeout_s = source_timeout_s
    self.destination_timeout_s = destination_timeout_s
    self.source_managers = []
    self.source_units = []
    self.decode_managers = []
    self.destination_units = []
    self.extra_managers = []
    self.source_server = None
    self.destination_server = None

  def __enter__(self):
    self.source_manifest = _pool_manifest(_SOURCE_BLOCK_BYTES, _SOURCE_BLOCKS)
    self.destination_manifest = _pool_manifest(
        _DESTINATION_BLOCK_BYTES, _DESTINATION_BLOCKS
    )

    for rank in range(_PCP_RANKS):
      manager = self.create_source_manager(rank)
      if not manager.transfer_address or not manager.listener_address:
        raise AssertionError("host manager did not advertise both endpoints")
      if manager.local_port == manager.listener_port:
        raise AssertionError(
            "data and protobuf listener ports must be distinct"
        )
      self.source_managers.append(manager)
      self.source_units.append(
          raiden_controller.RaidenId(
              job_name="prefill",
              job_replica_id=f"engine-rank{rank}",
              data_name="kv.fa",
              data_replica_idx=0,
          )
      )

    for decode_index in range(_DECODE_MANAGERS):
      manager = kv_cache_manager.KVCacheManager.create_host_only(
          num_layers=3,
          num_shards=1,
          slice_byte_size=_DESTINATION_BLOCK_BYTES,
          node_id=100 + decode_index,
          host_blocks=_DESTINATION_BLOCKS,
          parallelism=_PCP_RANKS,
          listener_port=0,
          timeout_s=self.destination_timeout_s,
      )
      manager.register_pools(self.destination_manifest)
      self.decode_managers.append(manager)
      self.destination_units.append(
          raiden_controller.RaidenId(
              job_name="decode",
              job_replica_id=f"engine-{decode_index}",
              data_name="kv.fa",
              data_replica_idx=0,
          )
      )
      for block_id in range(_DESTINATION_BLOCKS):
        manager.write_block_bytes(0, block_id, _FA_SENTINEL)
        for layer_idx, sentinel in _GDN_SENTINELS.items():
          manager.write_block_bytes(layer_idx, block_id, sentinel)

    self.source_controller = raiden_controller.RaidenController(0)
    self.source_server = raiden_controller.RaidenControllerServer(
        self.source_controller
    )
    source_port = self.source_server.port
    self.destination_controller = raiden_controller.RaidenController(0)
    self.destination_server = raiden_controller.RaidenControllerServer(
        self.destination_controller
    )
    destination_port = self.destination_server.port
    self.destination_controller_port = destination_port
    self.source_server.start()
    self.destination_server.start()
    self.source_controller_address = f"127.0.0.1:{source_port}"
    self.destination_controller_address = f"127.0.0.1:{destination_port}"
    self.source_facade = raiden_controller.RaidenControllerClientFacade(
        self.source_controller_address
    )
    self.destination_facade = raiden_controller.RaidenControllerClientFacade(
        self.destination_controller_address
    )

    for rank, (unit, manager) in enumerate(
        zip(self.source_units, self.source_managers)
    ):
      self.source_facade.register_work_unit(
          unit,
          [manager.transfer_address],
          control_plane_rpc_address=manager.listener_address,
          pool_manifest=self.source_manifest,
          layout_fingerprint=_LAYOUT_FINGERPRINT,
          page_tokens=_PREFILL_PAGE_TOKENS,
          page_slice_tokens=_PREFILL_PAGE_SLICE_TOKENS,
          transfer_parallelism=_PCP_RANKS,
          transfer_rank=rank,
      )
    self.register_destination_units()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    # No controller command may outlive the native listener it addresses.
    for server in (self.source_server, self.destination_server):
      if server is not None:
        server.stop()
    for server in (self.source_server, self.destination_server):
      thread = getattr(server, "_thread", None)
      if thread is not None:
        thread.join(timeout=2.0)

    managers = self.source_managers + self.extra_managers
    decode_managers = self.decode_managers
    self.source_managers = []
    self.extra_managers = []
    self.decode_managers = []
    del managers
    del decode_managers
    gc.collect()

  def create_source_manager(self, node_id: int, admit_pools: bool = True):
    """Creates a real source manager, optionally without pool admission."""
    manager = kv_cache_manager.KVCacheManager.create_host_only(
        num_layers=3,
        num_shards=1,
        slice_byte_size=_SOURCE_BLOCK_BYTES,
        node_id=node_id,
        host_blocks=_SOURCE_BLOCKS,
        parallelism=_PCP_RANKS,
        listener_port=0,
        timeout_s=self.source_timeout_s,
    )
    if admit_pools:
      manager.register_pools(self.source_manifest)
    return manager

  def replace_source_manager(self, rank: int) -> None:
    """Replaces and re-registers one producer identity after sender loss."""
    self.extra_managers.append(self.source_managers[rank])
    # BlockTransport carries node_id as the source schedule key on the wire,
    # so a restarted PCP process must retain its transfer rank here.
    manager = self.create_source_manager(rank)
    self.source_managers[rank] = manager
    self.source_facade.register_work_unit(
        self.source_units[rank],
        [manager.transfer_address],
        control_plane_rpc_address=manager.listener_address,
        pool_manifest=self.source_manifest,
        layout_fingerprint=_LAYOUT_FINGERPRINT,
        page_tokens=_PREFILL_PAGE_TOKENS,
        page_slice_tokens=_PREFILL_PAGE_SLICE_TOKENS,
        transfer_parallelism=_PCP_RANKS,
        transfer_rank=rank,
    )

  def register_destination_units(self) -> None:
    """Registers every live decode manager on the current controller-D."""
    for unit, manager in zip(self.destination_units, self.decode_managers):
      self.destination_facade.register_work_unit(
          unit,
          [manager.transfer_address],
          control_plane_rpc_address=manager.listener_address,
          pool_manifest=self.destination_manifest,
          layout_fingerprint=_LAYOUT_FINGERPRINT,
          page_tokens=_DECODE_PAGE_TOKENS,
          page_slice_tokens=_DECODE_PAGE_TOKENS,
          transfer_parallelism=_PCP_RANKS,
          transfer_rank=0,
      )

  def restart_destination_controller(self) -> None:
    """Replaces controller-D at the same address with an empty registry."""
    self.destination_server.stop()
    thread = getattr(self.destination_server, "_thread", None)
    if thread is not None:
      thread.join(timeout=2.0)
    self.destination_controller = raiden_controller.RaidenController(
        self.destination_controller_port
    )
    self.destination_server = raiden_controller.RaidenControllerServer(
        self.destination_controller
    )
    self.destination_server.start()
    self.destination_facade = raiden_controller.RaidenControllerClientFacade(
        self.destination_controller_address
    )

  def prepare_request(
      self,
      *,
      req_id: str,
      uuid: int,
      num_tokens: int,
      seed: int,
      source_block_ids_by_rank: list[list[int]],
      destination_block_ids: list[int],
      destination_index: int = 0,
  ) -> dict[int, bytes]:
    chunks_per_source_block = _PREFILL_PAGE_TOKENS // _PREFILL_PAGE_SLICE_TOKENS
    interleave_cycle_tokens = _PREFILL_PAGE_SLICE_TOKENS * _PCP_RANKS
    expected_source_counts = []
    for rank in range(_PCP_RANKS):
      first_rank_token = rank * _PREFILL_PAGE_SLICE_TOKENS
      if num_tokens <= first_rank_token:
        owned_interleave_cycles = 0
      else:
        owned_interleave_cycles = 1 + (
            (num_tokens - 1 - first_rank_token) // interleave_cycle_tokens
        )
      expected_source_counts.append(
          math.ceil(owned_interleave_cycles / chunks_per_source_block)
      )
    if [len(ids) for ids in source_block_ids_by_rank] != expected_source_counts:
      raise AssertionError("test source block IDs do not match PCP ownership")

    all_token_bytes = b"".join(
        _token_payload(seed, token) for token in range(num_tokens)
    )
    for rank, (unit, manager, physical_ids) in enumerate(
        zip(
            self.source_units,
            self.source_managers,
            source_block_ids_by_rank,
        )
    ):
      for local_block, physical_id in enumerate(physical_ids):
        payload = bytearray([0xEE]) * _SOURCE_BLOCK_BYTES
        for chunk_in_block in range(chunks_per_source_block):
          interleave_cycle = (
              local_block * chunks_per_source_block + chunk_in_block
          )
          global_token_start = (
              interleave_cycle * interleave_cycle_tokens
              + rank * _PREFILL_PAGE_SLICE_TOKENS
          )
          if global_token_start >= num_tokens:
            continue
          global_token_end = min(
              global_token_start + _PREFILL_PAGE_SLICE_TOKENS,
              num_tokens,
          )
          source_byte_start = global_token_start * _TOKEN_BYTES
          source_byte_end = global_token_end * _TOKEN_BYTES
          block_byte_start = (
              chunk_in_block * _PREFILL_PAGE_SLICE_TOKENS * _TOKEN_BYTES
          )
          block_byte_end = block_byte_start + (
              source_byte_end - source_byte_start
          )
          payload[block_byte_start:block_byte_end] = all_token_bytes[
              source_byte_start:source_byte_end
          ]
        manager.write_block_bytes(0, physical_id, bytes(payload))
      self.source_facade.register_request_blocks(
          req_id, uuid, unit, physical_ids
      )

    expected_page_count = math.ceil(num_tokens / _DECODE_PAGE_TOKENS)
    if len(destination_block_ids) != expected_page_count:
      raise AssertionError("test destination block count is incorrect")
    decode_manager = self.decode_managers[destination_index]
    expected = {}
    for ordinal, physical_id in enumerate(destination_block_ids):
      byte_start = ordinal * _DESTINATION_BLOCK_BYTES
      payload = all_token_bytes[
          byte_start : byte_start + _DESTINATION_BLOCK_BYTES
      ]
      payload += _FA_SENTINEL[len(payload) :]
      expected[physical_id] = payload
      decode_manager.write_block_bytes(0, physical_id, _FA_SENTINEL)
    return expected

  def coordinate(
      self,
      *,
      req_id: str,
      uuid: int,
      num_tokens: int,
      destination_block_ids: list[int],
      destination_index: int = 0,
  ) -> bool:
    # Enter through the destination controller. It asks the source controller
    # to plan, which arms this receiver before dispatching any source push.
    facade = raiden_controller.RaidenControllerClientFacade(
        self.destination_controller_address
    )
    return facade.coordinate_transfer(
        src_units=self.source_units,
        dst_units=[self.destination_units[destination_index]],
        req_id=req_id,
        use_block_chunks=True,
        is_sender=False,
        uuid=uuid,
        src_controller_address=self.source_controller_address,
        dst_controller_address=self.destination_controller_address,
        dst_mem_type=raiden_controller.RaidenMemoryType.HBM,
        dst_device_block_ids=destination_block_ids,
        num_tokens=num_tokens,
        parallelism=_PCP_RANKS,
    )

  def wait_for_destination(
      self,
      expected: dict[int, bytes],
      destination_index: int = 0,
      timeout_s: float = 15.0,
  ) -> None:
    decode_manager = self.decode_managers[destination_index]
    deadline = time.monotonic() + timeout_s
    last_mismatches = []
    while time.monotonic() < deadline:
      last_mismatches = [
          block_id
          for block_id, payload in expected.items()
          if decode_manager.read_block_bytes(0, block_id) != payload
      ]
      if not last_mismatches:
        # Let the final transport acknowledgement retire sender state before
        # native-manager teardown begins.
        time.sleep(0.05)
        return
      time.sleep(0.01)
    raise AssertionError(
        "destination did not converge; mismatched block IDs: "
        f"{last_mismatches[:12]}"
    )

  def wait_for_source_completion(
      self,
      req_id: str,
      ranks: list[int],
      timeout_s: float = 5.0,
  ) -> None:
    """Waits until each selected real producer has retired the request."""
    pending = set(ranks)
    deadline = time.monotonic() + timeout_s
    while pending and time.monotonic() < deadline:
      for rank in list(pending):
        done_sending, _, failed = self.source_managers[rank].poll_stats()
        if req_id in done_sending:
          pending.remove(rank)
        elif req_id in failed:
          raise AssertionError(
              f"source rank {rank} failed while retiring {req_id}"
          )
      if pending:
        time.sleep(0.01)
    if pending:
      raise AssertionError(
          f"source ranks did not retire {req_id}: {sorted(pending)}"
      )

  def assert_gdn_unchanged(self) -> None:
    for decode_index, manager in enumerate(self.decode_managers):
      for layer_idx, sentinel in _GDN_SENTINELS.items():
        for block_id in range(_DESTINATION_BLOCKS):
          actual = manager.read_block_bytes(layer_idx, block_id)
          if actual != sentinel:
            raise AssertionError(
                f"decode {decode_index} GDN layer {layer_idx} block "
                f"{block_id} was modified"
            )


class RaidenControllerStage3HostTest(absltest.TestCase):

  def test_pcp8_controller_reshards_fa_and_leaves_gdn_unchanged(self):
    num_tokens = 82
    destination_ids = list(range(math.ceil(num_tokens / 2)))
    # Six interleave cycles reach rank 0 and five reach every other rank. Four
    # rank-local 2-token chunks fit in each 8-token source page, so every rank
    # owns two pages. Reversed, nonzero physical IDs prove the controller
    # consumes D5 registration order rather than sorting physical block IDs.
    source_ids = [[3, 1] for _ in range(_PCP_RANKS)]

    with _Stage3HostCluster() as cluster:
      expected = cluster.prepare_request(
          req_id="h1-stage3",
          uuid=31001,
          num_tokens=num_tokens,
          seed=7,
          source_block_ids_by_rank=source_ids,
          destination_block_ids=destination_ids,
      )

      self.assertTrue(
          cluster.coordinate(
              req_id="h1-stage3",
              uuid=31001,
              num_tokens=num_tokens,
              destination_block_ids=destination_ids,
          )
      )
      cluster.wait_for_destination(expected)
      cluster.assert_gdn_unchanged()
      self.assertEqual(
          cluster.decode_managers[0].poll_stats(),
          ([], ["h1-stage3"], []),
      )
      self.assertEqual(cluster.decode_managers[0].poll_stats(), ([], [], []))
      self.assertEqual(cluster.decode_managers[1].poll_stats(), ([], [], []))
      for manager in cluster.source_managers:
        self.assertEqual(manager.poll_stats(), (["h1-stage3"], [], []))
        self.assertEqual(manager.poll_stats(), ([], [], []))

      plan = cluster.source_controller.get_plan("h1-stage3")
      self.assertIsNotNone(plan)
      self.assertEqual(plan.transfer_pool_indices, [0])
      self.assertEqual(plan.skipped_pool_counts, {"gdn.conv": 1, "gdn.ssm": 1})
      self.assertLen(plan.src_units, _PCP_RANKS)
      self.assertEqual(plan.dst_device_block_ids, destination_ids)
      for schedules in plan.shard_push_schedules.values():
        for entry in schedules[0]:
          self.assertEqual(entry[7:], (0, 0, 1))

  def test_four_concurrent_uuids_across_two_decoders_remain_isolated(self):
    num_tokens = 64
    requests = [
        {
            "req_id": f"h2-request-{request_index}",
            "uuid": 32001 + request_index,
            "seed": 13 + request_index * 17,
            "source_ids": [[request_index] for _ in range(_PCP_RANKS)],
            "destination_index": request_index % _DECODE_MANAGERS,
            "destination_ids": list(
                range(request_index * 32, (request_index + 1) * 32)
            ),
        }
        for request_index in range(4)
    ]
    all_destination_ids = [
        block_id
        for request in requests
        for block_id in request["destination_ids"]
    ]
    self.assertLen(set(all_destination_ids), 128)

    with _Stage3HostCluster() as cluster:
      expected_by_request = {}
      for request in requests:
        expected_by_request[request["req_id"]] = cluster.prepare_request(
            req_id=request["req_id"],
            uuid=request["uuid"],
            num_tokens=num_tokens,
            seed=request["seed"],
            source_block_ids_by_rank=request["source_ids"],
            destination_block_ids=request["destination_ids"],
            destination_index=request["destination_index"],
        )

      with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                cluster.coordinate,
                req_id=request["req_id"],
                uuid=request["uuid"],
                num_tokens=num_tokens,
                destination_block_ids=request["destination_ids"],
                destination_index=request["destination_index"],
            )
            for request in requests
        ]
        for future in futures:
          self.assertTrue(future.result(timeout=20.0))

      for request in requests:
        cluster.wait_for_destination(
            expected_by_request[request["req_id"]],
            destination_index=request["destination_index"],
        )
      cluster.assert_gdn_unchanged()
      for decode_index, manager in enumerate(cluster.decode_managers):
        done_sending, done_recving, failed_recving = manager.poll_stats()
        expected_done = [
            request["req_id"]
            for request in requests
            if request["destination_index"] == decode_index
        ]
        self.assertEqual(done_sending, [])
        self.assertCountEqual(done_recving, expected_done)
        self.assertLen(done_recving, len(expected_done))
        self.assertEqual(failed_recving, [])
        self.assertEqual(manager.poll_stats(), ([], [], []))
      expected_source_done = [request["req_id"] for request in requests]
      for manager in cluster.source_managers:
        done_sending, done_recving, failed_recving = manager.poll_stats()
        self.assertCountEqual(done_sending, expected_source_done)
        self.assertLen(done_sending, len(expected_source_done))
        self.assertEqual(done_recving, [])
        self.assertEqual(failed_recving, [])
        self.assertEqual(manager.poll_stats(), ([], [], []))

      for request in requests:
        source_plan = cluster.source_controller.get_plan(request["req_id"])
        receiver_plan = cluster.destination_controller.get_plan(
            request["req_id"]
        )
        self.assertEqual(source_plan.uuid, request["uuid"])
        self.assertEqual(receiver_plan.uuid, request["uuid"])
        self.assertEqual(source_plan.expected_pushes_per_pool, 32)
        self.assertEqual(receiver_plan.expected_pushes_per_pool, 32)
        self.assertEqual(source_plan.expected_block_count, 32)
        self.assertEqual(source_plan.transfer_pool_indices, [0])
        self.assertEqual(receiver_plan.transfer_pool_indices, [0])
        self.assertEqual(
            source_plan.skipped_pool_counts,
            {"gdn.conv": 1, "gdn.ssm": 1},
        )
        self.assertEqual(
            {
                entry[6]
                for schedules in source_plan.shard_push_schedules.values()
                for entry in schedules[0]
            },
            set(request["destination_ids"]),
        )
        self.assertEqual(
            {
                entry[0]
                for schedules in source_plan.shard_push_schedules.values()
                for entry in schedules[0]
            },
            {
                cluster.decode_managers[
                    request["destination_index"]
                ].transfer_address
            },
        )
        for unit, source_ids in zip(
            cluster.source_units, request["source_ids"]
        ):
          self.assertEqual(source_plan.src_block_ids[unit], source_ids)

  def test_sender_loss_times_out_and_same_uuid_is_reusable(self):
    num_tokens = 64
    uuid = 33001
    failed_destination_ids = list(range(32))
    reused_destination_ids = list(range(32, 64))

    with _Stage3HostCluster(destination_timeout_s=0.05) as cluster:
      cluster.prepare_request(
          req_id="h3-sender-loss",
          uuid=uuid,
          num_tokens=num_tokens,
          seed=31,
          source_block_ids_by_rank=[[0] for _ in range(_PCP_RANKS)],
          destination_block_ids=failed_destination_ids,
          destination_index=0,
      )

      # The replacement listener is a real host manager but deliberately has
      # no admitted pools. Once the receiver is armed, rank 0 is shut down and
      # its identity is redirected to this replacement. The native listener
      # rejects the controller plan synchronously, so the controller future
      # must report failure while the receiver remains short of its barrier.
      rejected_sender = cluster.create_source_manager(900, admit_pools=False)
      cluster.extra_managers.append(rejected_sender)
      victim = cluster.source_units[0]
      victim_listener = cluster.source_managers[0].listener_address
      worker_client = cluster.source_controller.worker_rpc_client
      original_start_transfer = worker_client.start_transfer

      async def lose_sender_after_receiver_arm(target_id, transfer_plan):
        if target_id == victim:
          response = await worker_client._send_rpc(
              victim_listener, worker_client._encode_shutdown()
          )
          worker_client._verify_response(response)
          worker_client.register_worker_endpoint(
              victim, rejected_sender.listener_address
          )
        await original_start_transfer(target_id, transfer_plan)

      worker_client.start_transfer = lose_sender_after_receiver_arm
      try:
        with self.assertRaisesRegex(
            RuntimeError, "explicitly registered pools"
        ):
          cluster.coordinate(
              req_id="h3-sender-loss",
              uuid=uuid,
              num_tokens=num_tokens,
              destination_block_ids=failed_destination_ids,
              destination_index=0,
          )
      finally:
        worker_client.start_transfer = original_start_transfer

      # The controller propagates rank 0's synchronous failure while the
      # other seven real listeners continue their already-dispatched pushes.
      # Observe every surviving sender's transport acknowledgement before
      # reclaiming the receiver generation; otherwise late streams from the
      # failed generation could race the intentional UUID reuse below.
      cluster.wait_for_source_completion(
          "h3-sender-loss", list(range(1, _PCP_RANKS))
      )
      time.sleep(0.10)
      done_sending, done_recving, failed_recving = cluster.decode_managers[
          0
      ].poll_stats()
      self.assertEqual(done_sending, [])
      self.assertNotIn("h3-sender-loss", done_recving)
      self.assertEqual(failed_recving, ["h3-sender-loss"])
      with self.assertRaisesRegex(RuntimeError, "not registered"):
        cluster.decode_managers[0].unregister_active_plan(uuid)
      self.assertEqual(cluster.decode_managers[0].poll_stats(), ([], [], []))

      # Polling the deadline removed receiver and transport progress. Replace
      # and re-register the lost producer, then reuse the exact UUID on the
      # same decode manager. A complete bit-exact transfer proves that neither
      # the receiver entry nor per-UUID stream counts leaked.
      cluster.replace_source_manager(0)
      expected = cluster.prepare_request(
          req_id="h3-uuid-reused",
          uuid=uuid,
          num_tokens=num_tokens,
          seed=37,
          source_block_ids_by_rank=[[1] for _ in range(_PCP_RANKS)],
          destination_block_ids=reused_destination_ids,
          destination_index=0,
      )
      self.assertTrue(
          cluster.coordinate(
              req_id="h3-uuid-reused",
              uuid=uuid,
              num_tokens=num_tokens,
              destination_block_ids=reused_destination_ids,
              destination_index=0,
          )
      )
      cluster.wait_for_destination(expected, destination_index=0)
      self.assertEqual(
          cluster.decode_managers[0].poll_stats(),
          ([], ["h3-uuid-reused"], []),
      )
      self.assertEqual(cluster.decode_managers[0].poll_stats(), ([], [], []))
      self.assertEqual(cluster.decode_managers[1].poll_stats(), ([], [], []))
      cluster.assert_gdn_unchanged()
      self.assertEqual(
          cluster.source_controller.get_plan("h3-uuid-reused").uuid, uuid
      )

  def test_destination_controller_restart_fails_inflight_then_recovers(self):
    num_tokens = 64
    failed_destination_ids = list(range(32))
    fresh_destination_ids = list(range(32, 64))

    with _Stage3HostCluster() as cluster:
      cluster.prepare_request(
          req_id="h4-inflight",
          uuid=34001,
          num_tokens=num_tokens,
          seed=41,
          source_block_ids_by_rank=[[0] for _ in range(_PCP_RANKS)],
          destination_block_ids=failed_destination_ids,
          destination_index=0,
      )

      # Pause only at the boundary before the real source controller queries
      # controller-D. The request has already traversed both real controller
      # servers over TCP; after release, the original method performs the real
      # metadata RPC against the restarted, empty controller.
      query_entered = threading.Event()
      release_query = threading.Event()
      original_query = cluster.source_controller._query_remote_metadata

      async def query_after_restart(address):
        query_entered.set()
        loop = asyncio.get_running_loop()
        released = await loop.run_in_executor(
            None, lambda: release_query.wait(timeout=5.0)
        )
        if not released:
          raise TimeoutError("test did not release metadata query")
        return await original_query(address)

      cluster.source_controller._query_remote_metadata = query_after_restart
      with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        in_flight = executor.submit(
            cluster.coordinate,
            req_id="h4-inflight",
            uuid=34001,
            num_tokens=num_tokens,
            destination_block_ids=failed_destination_ids,
            destination_index=0,
        )
        try:
          self.assertTrue(query_entered.wait(timeout=5.0))
          cluster.restart_destination_controller()
          release_query.set()
          with self.assertRaisesRegex(
              RuntimeError, "Missing registration metadata"
          ):
            in_flight.result(timeout=10.0)
        finally:
          release_query.set()
          cluster.source_controller._query_remote_metadata = original_query

      self.assertIsNone(cluster.source_controller.get_plan("h4-inflight"))
      self.assertIsNone(cluster.destination_controller.get_plan("h4-inflight"))
      for block_id in failed_destination_ids:
        self.assertEqual(
            cluster.decode_managers[0].read_block_bytes(0, block_id),
            _FA_SENTINEL,
        )

      # A restarted controller owns an empty registry. Re-register both live
      # decode workers, then prove a fresh request traverses the same managers,
      # listeners, transports, and controller address successfully.
      cluster.register_destination_units()
      self.assertLen(cluster.destination_facade.get_metadata(), 2)
      expected = cluster.prepare_request(
          req_id="h4-fresh",
          uuid=34002,
          num_tokens=num_tokens,
          seed=53,
          source_block_ids_by_rank=[[1] for _ in range(_PCP_RANKS)],
          destination_block_ids=fresh_destination_ids,
          destination_index=0,
      )
      self.assertTrue(
          cluster.coordinate(
              req_id="h4-fresh",
              uuid=34002,
              num_tokens=num_tokens,
              destination_block_ids=fresh_destination_ids,
              destination_index=0,
          )
      )
      cluster.wait_for_destination(expected, destination_index=0)
      cluster.assert_gdn_unchanged()
      self.assertEqual(
          cluster.decode_managers[0].poll_stats(),
          ([], ["h4-fresh"], []),
      )
      self.assertEqual(cluster.decode_managers[0].poll_stats(), ([], [], []))
      self.assertEqual(
          cluster.source_controller.get_plan("h4-fresh").uuid, 34002
      )
      self.assertEqual(
          cluster.destination_controller.get_plan("h4-fresh").uuid, 34002
      )


if __name__ == "__main__":
  absltest.main()
