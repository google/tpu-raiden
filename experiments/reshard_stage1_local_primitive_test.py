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
"""Stage 1 local primitive tests for host-only Raiden KV resharded push.

These tests validate one source manager pushing one destination head over the
Stage 1 API. They are the local primitive success criterion for:

  Python surface -> RegisterActivePlan -> GetBlockChunks -> Push(op=6)
  -> WriteVExact/ReadVExact -> bit-exact host destination bytes.

The full PCP8 -> TP2 topology is exercised by reshard_stage1_e2e.py.
"""

from absl.testing import absltest

from tpu_raiden.experimental import kv_cache_manager_host
from tpu_raiden.rpc import raiden_service_pb2

NUM_FA_LAYERS = 15
TOKENS_PER_BLOCK = 1024
HEAD_PAIR_BYTES = 512
TOKEN_STRIDE_BYTES = 1024
SRC_SLOT_BYTES = 1_067_008
DST_SLOT_BYTES = TOKENS_PER_BLOCK * HEAD_PAIR_BYTES
SOURCE_NODE_ID = 0
SOURCE_BLOCK_ID = 0
DEST_BLOCK_ID = 0
UUID = 0x51E135
POISON = 0xAA


def _pattern_byte(layer: int, page: int, token: int, head: int) -> int:
  return (layer * 37 + head * 101 + (page % 16) * 13 + (token % 256)) % 251


def _source_payload(layer: int, page: int, live_tokens: int) -> bytes:
  payload = bytearray(SRC_SLOT_BYTES)
  for token in range(live_tokens):
    token_base = token * TOKEN_STRIDE_BYTES
    for head in (0, 1):
      offset = token_base + head * HEAD_PAIR_BYTES
      value = _pattern_byte(layer, page, token, head)
      payload[offset : offset + HEAD_PAIR_BYTES] = (
          bytes([value]) * HEAD_PAIR_BYTES
      )
  return bytes(payload)


def _expected_dest_payload(
    layer: int, page: int, live_tokens: int, head: int
) -> bytes:
  payload = bytearray([POISON]) * DST_SLOT_BYTES
  for token in range(live_tokens):
    offset = token * HEAD_PAIR_BYTES
    value = _pattern_byte(layer, page, token, head)
    payload[offset : offset + HEAD_PAIR_BYTES] = (
        bytes([value]) * HEAD_PAIR_BYTES
    )
  return bytes(payload)


def _fill_source(manager, page: int, live_tokens: int) -> None:
  for layer in range(NUM_FA_LAYERS):
    manager.write_block_bytes(
        layer, SOURCE_BLOCK_ID, _source_payload(layer, page, live_tokens)
    )


def _poison_dest(manager) -> None:
  poison = bytes([POISON]) * DST_SLOT_BYTES
  for layer in range(NUM_FA_LAYERS):
    manager.write_block_bytes(layer, DEST_BLOCK_ID, poison)


def _new_request(uuid: int, is_sender: bool):
  req = raiden_service_pb2.StartTransferRequest()
  req.uuid = uuid
  req.is_sender = is_sender
  req.dst_mem_type = raiden_service_pb2.MEMORY_TYPE_DRAM
  req.use_block_chunks = True
  req.req_id = f"stage1_test_{uuid}"
  return req


def _add_entry(
    schedule,
    *,
    peer: str,
    src_block_id: int,
    dst_block_id: int,
    dst_shard_idx: int,
    src_offset: int,
    dst_offset: int,
    size: int,
    src_stride: int = 0,
    dst_stride: int = 0,
    count: int = 1,
) -> None:
  entry = schedule.entries.add()
  entry.dst_peer = peer
  entry.dst_shard_idx = dst_shard_idx
  entry.src_block_id = src_block_id
  entry.dst_block_id = dst_block_id
  entry.src_offset_bytes = src_offset
  entry.dst_offset_bytes = dst_offset
  entry.size_bytes = size
  entry.src_stride_bytes = src_stride
  entry.dst_stride_bytes = dst_stride
  entry.count = count


def _sender_request(
    uuid: int,
    peer: str,
    *,
    head: int,
    live_tokens: int,
    include_other_peer: bool,
):
  req = _new_request(uuid, is_sender=True)
  schedule = req.shard_push_schedules[0]
  _add_entry(
      schedule,
      peer=peer,
      src_block_id=SOURCE_BLOCK_ID,
      dst_block_id=DEST_BLOCK_ID,
      dst_shard_idx=0,
      src_offset=head * HEAD_PAIR_BYTES,
      dst_offset=0,
      size=HEAD_PAIR_BYTES,
      src_stride=TOKEN_STRIDE_BYTES,
      count=live_tokens,
  )
  if include_other_peer:
    # Real Stage 1 sender plans contain entries for both decode peers. The push
    # call is per peer, so sender-side GetBlockChunks must filter by dst_peer.
    _add_entry(
        schedule,
        peer="127.0.0.1:1",
        src_block_id=SOURCE_BLOCK_ID,
        dst_block_id=DEST_BLOCK_ID,
        dst_shard_idx=0,
        src_offset=(1 - head) * HEAD_PAIR_BYTES,
        dst_offset=0,
        size=HEAD_PAIR_BYTES,
        src_stride=TOKEN_STRIDE_BYTES,
        count=live_tokens,
    )
  return req


def _receiver_request(uuid: int, peer: str, *, live_tokens: int):
  req = _new_request(uuid, is_sender=False)
  schedule = req.shard_push_schedules[SOURCE_NODE_ID]
  _add_entry(
      schedule,
      peer=peer,
      src_block_id=SOURCE_BLOCK_ID,
      dst_block_id=DEST_BLOCK_ID,
      dst_shard_idx=0,
      src_offset=0,
      dst_offset=0,
      size=live_tokens * HEAD_PAIR_BYTES,
      count=1,
  )
  return req


class ReshardStage1LocalPrimitiveTest(absltest.TestCase):

  def _new_source_manager(self):
    return kv_cache_manager_host.HostKVCacheManager(
        num_layers=NUM_FA_LAYERS,
        num_shards=1,
        slice_byte_size=SRC_SLOT_BYTES,
        node_id=SOURCE_NODE_ID,
        local_port=0,
        host_blocks=1,
        parallelism=4,
    )

  def _new_dest_manager(self):
    return kv_cache_manager_host.HostKVCacheManager(
        num_layers=NUM_FA_LAYERS,
        num_shards=1,
        slice_byte_size=DST_SLOT_BYTES,
        node_id=100,
        local_port=0,
        host_blocks=1,
        parallelism=4,
    )

  def _register_and_push(
      self,
      source,
      dest,
      *,
      uuid: int,
      page: int,
      head: int,
      live_tokens: int,
      include_other_peer: bool = True,
  ) -> None:
    _fill_source(source, page, live_tokens)
    _poison_dest(dest)

    peer = dest.transfer_address
    self.assertTrue(peer)
    source.register_active_plan(
        uuid,
        _sender_request(
            uuid,
            peer,
            head=head,
            live_tokens=live_tokens,
            include_other_peer=include_other_peer,
        ),
        is_sender=True,
    )
    dest.register_active_plan(
        uuid,
        _receiver_request(uuid, peer, live_tokens=live_tokens),
        is_sender=False,
    )

    source.push_registered_plan(
        uuid,
        peer,
        [SOURCE_BLOCK_ID],
        [DEST_BLOCK_ID],
        layer_idx=-1,
        parallelism=1,
    )

    for layer in range(NUM_FA_LAYERS):
      actual = dest.read_block_bytes(layer, DEST_BLOCK_ID)
      expected = _expected_dest_payload(layer, page, live_tokens, head)
      self.assertEqual(
          actual,
          expected,
          msg=f"mismatch layer={layer} page={page} head={head}",
      )

  def test_strided_push_full_block_tail_and_uuid_reuse(self):
    source = self._new_source_manager()
    dest = self._new_dest_manager()

    self._register_and_push(
        source,
        dest,
        uuid=UUID,
        page=3,
        head=1,
        live_tokens=TOKENS_PER_BLOCK,
    )
    source.unregister_active_plan(UUID)
    dest.unregister_active_plan(UUID)

    # Reuse the same UUID with a partial tail page. This proves plan cleanup,
    # count < 1024, and that unwritten destination bytes remain untouched.
    self._register_and_push(
        source,
        dest,
        uuid=UUID,
        page=9,
        head=0,
        live_tokens=TOKENS_PER_BLOCK // 2,
    )
    source.unregister_active_plan(UUID)
    dest.unregister_active_plan(UUID)

  def test_size_mismatch_fails_without_modifying_destination(self):
    source = self._new_source_manager()
    dest = self._new_dest_manager()
    _fill_source(source, page=11, live_tokens=8)
    _poison_dest(dest)

    peer = dest.transfer_address
    source.register_active_plan(
        UUID,
        _sender_request(
            UUID, peer, head=1, live_tokens=8, include_other_peer=False
        ),
        is_sender=True,
    )
    # Receiver expects one fewer token than the sender advertises.
    dest.register_active_plan(
        UUID,
        _receiver_request(UUID, peer, live_tokens=7),
        is_sender=False,
    )

    with self.assertRaisesRegex(
        RuntimeError,
        "size mismatch|Push verification failed|Broken pipe|Connection|failed",
    ):
      source.push_registered_plan(
          UUID,
          peer,
          [SOURCE_BLOCK_ID],
          [DEST_BLOCK_ID],
          layer_idx=0,
          parallelism=1,
      )

    self.assertEqual(
        dest.read_block_bytes(0, DEST_BLOCK_ID),
        bytes([POISON]) * DST_SLOT_BYTES,
    )


if __name__ == "__main__":
  absltest.main()
