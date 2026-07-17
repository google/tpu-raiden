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

"""Stage-3 pool reshard D-series: real-chip executor byte path.

Runs on one host with >= 5 TPU chips (the local 8x TPU7x VM):

  D1/D2  multi-manager interleave fan-in: k source managers on distinct chips
         declare spans for a rank-major interleaved prompt; one destination
         manager receives via two in-process controllers (sender-side
         planning, receiver-side independent validation, arm-before-dispatch)
         and the pool executor's D2hPoolBlocks -> wire -> H2dPoolBlocks path.
         Destination bytes are compared bit-exactly against the global
         logical token sequence read back through torch_tpu, unselected
         pools must remain byte-identical (no-leak), the device data_ptr
         must not change, and the poison tail beyond num_tokens must be
         intact.
  D3     two uuids in flight concurrently with disjoint destination blocks.
  D4     failure injection: a source whose listener is gone fails the
         dispatch; the armed receiver times out into failed_recving and the
         uuid is reusable afterwards.

Spans make reduced-P fan-in a first-class geometry, so k < 8 exercises the
identical raiden layer as PCP8 (doc section 12.Q-F).
"""

from __future__ import annotations

import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch_tpu  # noqa: F401,E402
from torch_tpu._internal.batch_transfer import batch_transfer_d2h_sync  # noqa: E402

from tpu_raiden.api.torch.kv_cache_manager import KVCacheManager  # noqa: E402
from tpu_raiden.rpc import raiden_controller  # noqa: E402

TOKEN_BYTES = 1024
SRC_PAGE_TOKENS = 256
DST_PAGE_TOKENS = 64
INTERLEAVE = 32
PARALLELISM = 4
SRC_NUM_BLOCKS = 4
DST_NUM_BLOCKS = 64
SKIP_BLOCK_BYTES = 4096
SKIP_NUM_BLOCKS = 4
FINGERPRINT = "stage3-device-test-fingerprint"
POISON = 0xEE


def _token_record(token: int) -> np.ndarray:
  # Values 1..126 only: fp8_e4m3fn NaN encodings would be canonicalized by
  # the logical device write that places the source pattern.
  cols = np.arange(TOKEN_BYTES, dtype=np.uint32)
  return ((token * 7 + cols * 13 + 5) % 126 + 1).astype(np.uint8)


def _owned_ranges(rank: int, num_tokens: int):
  cycle = INTERLEAVE * PARALLELISM
  start = rank * INTERLEAVE
  while start < num_tokens:
    yield start, min(start + INTERLEAVE, num_tokens)
    start += cycle


def _spans_for_rank(rank: int, num_tokens: int):
  spans = []
  local_cursor = 0
  for start, end in _owned_ranges(rank, num_tokens):
    cursor = start
    while cursor < end:
      ordinal, offset = divmod(local_cursor, SRC_PAGE_TOKENS)
      run = min(end - cursor, SRC_PAGE_TOKENS - offset)
      spans.append(
          raiden_controller.SourceSpan(
              dst_token_start=cursor,
              token_count=run,
              src_block_ordinal=ordinal,
              src_block_token_offset=offset,
          )
      )
      local_cursor += run
      cursor += run
  return spans, local_cursor


def _pool_manifest_dicts():
  return [
      {
          "tag": "fa",
          "storage_index": 0,
          "base_offset_bytes": 0,
          "block_stride_bytes": SRC_PAGE_TOKENS * TOKEN_BYTES,
          "num_blocks": SRC_NUM_BLOCKS,
          "dtype_tag": "u8",
          "regions": [{
              "name": "fa",
              "offset_bytes": 0,
              "stride_bytes": SRC_PAGE_TOKENS * TOKEN_BYTES,
              "unit_bytes": SRC_PAGE_TOKENS * TOKEN_BYTES,
              "num_units": 1,
              "units_per_stride": 1,
          }],
      },
      {
          "tag": "gdn.conv",
          "storage_index": 1,
          "base_offset_bytes": 0,
          "block_stride_bytes": SKIP_BLOCK_BYTES,
          "num_blocks": SKIP_NUM_BLOCKS,
          "dtype_tag": "bf16",
          "regions": [{
              "name": "conv",
              "offset_bytes": 0,
              "stride_bytes": SKIP_BLOCK_BYTES,
              "unit_bytes": SKIP_BLOCK_BYTES,
              "num_units": 1,
              "units_per_stride": 1,
          }],
      },
  ]


def _dst_pool_manifest_dicts():
  dicts = _pool_manifest_dicts()
  dicts[0]["block_stride_bytes"] = DST_PAGE_TOKENS * TOKEN_BYTES
  dicts[0]["num_blocks"] = DST_NUM_BLOCKS
  dicts[0]["regions"][0]["stride_bytes"] = DST_PAGE_TOKENS * TOKEN_BYTES
  dicts[0]["regions"][0]["unit_bytes"] = DST_PAGE_TOKENS * TOKEN_BYTES
  return dicts


class Node:
  """One manager on one chip, with its filled device storages.

  The fa storage uses the E0'-calibrated fp8 KV shape
  [blocks, page_tokens, 1, 4, 256]: the XLA tile [[4,128],[4,1]] permutes
  bytes only within one token's 1,024-byte record and pages are
  block-contained, which is the physical premise the raw pool transfer rests
  on. (A flat u8 array would NOT satisfy it: XLA tiles 1-D arrays across
  records.) Verification therefore reads the destination back through the
  logical view, where the within-record permutation cancels.
  """

  def __init__(self, *, device_idx: int, rank: int, page_tokens: int,
               num_blocks: int, fa_host: np.ndarray):
    self.rank = rank
    skip_host = np.full(SKIP_BLOCK_BYTES * SKIP_NUM_BLOCKS,
                        0x30 + rank,
                        dtype=np.uint8)
    device = f"tpu:{device_idx}"
    self.fa_shape = (num_blocks, page_tokens, 1, 4, 256)
    self.fa = (torch.from_numpy(fa_host).view(torch.float8_e4m3fn).reshape(
        self.fa_shape).to(device))
    self.skip = torch.from_numpy(skip_host).to(device)
    self.fa_data_ptr = self.fa.data_ptr()
    self.manager = KVCacheManager(
        kv_caches=[self.fa, self.skip],
        node_id=rank,
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=1,
        timeout_s=12.0,
        listener_port=0,
        parallelism=2,
    )
    manifest = (_pool_manifest_dicts()
                if page_tokens == SRC_PAGE_TOKENS else
                _dst_pool_manifest_dicts())
    summary = self.manager.register_pools(manifest)
    assert summary["admitted"], summary
    self.manifest = manifest

  @property
  def transfer_address(self) -> str:
    return self.manager.transfer_address

  @property
  def listener_address(self) -> str:
    return self.manager.listener_address

  def read_fa_device_bytes(self) -> np.ndarray:
    # Logical readback: the within-record device permutation applied at both
    # ends cancels, so tokens compare directly against logical records.
    return (self.fa.cpu().reshape(-1).view(torch.uint8).numpy().copy())

  def read_skip_device_bytes(self) -> np.ndarray:
    host = torch.empty(self.skip.nbytes, dtype=torch.uint8)
    batch_transfer_d2h_sync([self.skip], [host])
    return host.numpy().copy()


def _build_sources(num_tokens: int):
  nodes = []
  block_ids_by_rank = {}
  spans_by_rank = {}
  for rank in range(PARALLELISM):
    spans, owned_tokens = _spans_for_rank(rank, num_tokens)
    owned_blocks = -(-owned_tokens // SRC_PAGE_TOKENS)
    fa_host = np.full(SRC_PAGE_TOKENS * TOKEN_BYTES * SRC_NUM_BLOCKS,
                      POISON,
                      dtype=np.uint8)
    local_cursor = 0
    for start, end in _owned_ranges(rank, num_tokens):
      for token in range(start, end):
        fa_host[local_cursor * TOKEN_BYTES:(local_cursor + 1) *
                TOKEN_BYTES] = _token_record(token)
        local_cursor += 1
    node = Node(device_idx=rank,
                rank=rank,
                page_tokens=SRC_PAGE_TOKENS,
                num_blocks=SRC_NUM_BLOCKS,
                fa_host=fa_host)
    nodes.append(node)
    block_ids_by_rank[rank] = list(range(owned_blocks))
    spans_by_rank[rank] = spans
  return nodes, block_ids_by_rank, spans_by_rank


def _unit(rank: int) -> raiden_controller.RaidenId:
  return raiden_controller.RaidenId(
      job_name="prefill",
      job_replica_id=f"engine-rank{rank}",
      data_name="kv.fa",
      data_replica_idx=0,
  )


DST_UNIT = raiden_controller.RaidenId(
    job_name="decode",
    job_replica_id="engine",
    data_name="kv.fa",
    data_replica_idx=0,
)


def _register_sources(facade, nodes):
  for node in nodes:
    facade.register_work_unit(
        unit=_unit(node.rank),
        shards=[node.transfer_address],
        control_plane_rpc_address=node.listener_address,
        pool_manifest=node.manifest,
        layout_fingerprint=FINGERPRINT,
        page_tokens=SRC_PAGE_TOKENS,
        transfer_parallelism=PARALLELISM,
        transfer_rank=node.rank,
    )


def _register_destination(facade, node):
  facade.register_work_unit(
      unit=DST_UNIT,
      shards=[node.transfer_address],
      control_plane_rpc_address=node.listener_address,
      pool_manifest=node.manifest,
      layout_fingerprint=FINGERPRINT,
      page_tokens=DST_PAGE_TOKENS,
      transfer_parallelism=PARALLELISM,
      transfer_rank=0,
  )


class _StatPoller:
  """Accumulates poll_stats results: the native poll clears on read."""

  def __init__(self, manager):
    self._manager = manager
    self.seen = {
        "done_sending": set(),
        "done_recving": set(),
        "failed_recving": set(),
    }

  def wait_for(self, want, deadline_s=30.0):
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
      done_sending, done_recving, failed = self._manager.poll_stats()
      self.seen["done_sending"].update(done_sending)
      self.seen["done_recving"].update(done_recving)
      self.seen["failed_recving"].update(failed)
      if want[1] in self.seen[want[0]]:
        return self.seen[want[0]]
      if (want[0] != "failed_recving"
          and want[1] in self.seen["failed_recving"]):
        raise AssertionError(
            f"transfer failed instead: {sorted(self.seen.items())}")
      time.sleep(0.05)
    raise AssertionError(
        f"timed out waiting for {want}; saw {sorted(self.seen.items())}")


def _expected_dst_fa(dst_ids, num_tokens: int) -> dict[int, np.ndarray]:
  expected = {}
  for page_ordinal, block_id in enumerate(dst_ids):
    page_start = page_ordinal * DST_PAGE_TOKENS
    page = np.full(DST_PAGE_TOKENS * TOKEN_BYTES, POISON, dtype=np.uint8)
    for offset in range(DST_PAGE_TOKENS):
      token = page_start + offset
      if token >= num_tokens:
        break
      page[offset * TOKEN_BYTES:(offset + 1) *
           TOKEN_BYTES] = _token_record(token)
    expected[block_id] = page
  return expected


def _transfer(dst_facade, src_units, dst_ids, req_id, uuid, num_tokens,
              src_addr, dst_addr):
  return dst_facade.start_transfer(
      src_units=src_units,
      dst_units=[DST_UNIT],
      req_id=req_id,
      dst_device_block_ids=list(dst_ids),
      dst_mem_type=raiden_controller.RaidenMemoryType.HBM,
      use_block_chunks=True,
      is_sender=False,
      uuid=uuid,
      num_tokens=num_tokens,
      transfer_pool_tags=["fa"],
      src_controller_address=src_addr,
      dst_controller_address=dst_addr,
  )


def main() -> int:
  # 2,500 tokens: partial final interleave slice (2,500 % 32 = 4 tokens on
  # rank 2) and a partial final destination page (2,500 % 64 = 4).
  num_tokens = 2500

  src_controller = raiden_controller.RaidenController(port=0)
  src_server = raiden_controller.RaidenControllerServer(src_controller)
  src_server.start()
  dst_controller = raiden_controller.RaidenController(port=0)
  dst_server = raiden_controller.RaidenControllerServer(dst_controller)
  dst_server.start()
  src_addr = f"127.0.0.1:{src_server.port}"
  dst_addr = f"127.0.0.1:{dst_server.port}"
  src_facade = raiden_controller.RaidenControllerClientFacade(src_addr)
  dst_facade = raiden_controller.RaidenControllerClientFacade(dst_addr)

  print(f"[D] building {PARALLELISM} source managers + 1 destination "
        f"across {PARALLELISM + 1} chips")
  src_nodes, block_ids_by_rank, spans_by_rank = _build_sources(num_tokens)
  dst_host = np.full(DST_PAGE_TOKENS * TOKEN_BYTES * DST_NUM_BLOCKS,
                     POISON,
                     dtype=np.uint8)
  dst_node = Node(device_idx=PARALLELISM,
                  rank=0,
                  page_tokens=DST_PAGE_TOKENS,
                  num_blocks=DST_NUM_BLOCKS,
                  fa_host=dst_host)
  dst_skip_before = dst_node.read_skip_device_bytes()
  dst_poll = _StatPoller(dst_node.manager)
  src_polls = {node.rank: _StatPoller(node.manager) for node in src_nodes}

  _register_sources(src_facade, src_nodes)
  _register_destination(dst_facade, dst_node)

  # ---- D1/D2: single transfer, bit-exact interleave fan-in ----
  req_id, uuid = "device-req-1", 424242
  for node in src_nodes:
    src_facade.register_request_blocks(
        req_id=req_id,
        uuid=uuid,
        unit=_unit(node.rank),
        block_ids=block_ids_by_rank[node.rank],
        spans=spans_by_rank[node.rank],
    )
  dst_ids = list(range(3, 3 + -(-num_tokens // DST_PAGE_TOKENS)))
  assert _transfer(dst_facade, [_unit(r) for r in range(PARALLELISM)],
                   dst_ids, req_id, uuid, num_tokens, src_addr,
                   dst_addr) is True
  dst_poll.wait_for(("done_recving", req_id))
  for node in src_nodes:
    src_polls[node.rank].wait_for(("done_sending", req_id))

  fa_bytes = dst_node.read_fa_device_bytes()
  expected = _expected_dst_fa(dst_ids, num_tokens)
  page_bytes = DST_PAGE_TOKENS * TOKEN_BYTES
  final_block = dst_ids[-1]
  live_tail_bytes = (num_tokens - (len(dst_ids) - 1) * DST_PAGE_TOKENS
                     ) * TOKEN_BYTES
  for block_id in range(DST_NUM_BLOCKS):
    got = fa_bytes[block_id * page_bytes:(block_id + 1) * page_bytes]
    want = expected.get(
        block_id, np.full(page_bytes, POISON, dtype=np.uint8))
    # Whole-block H2D makes the final destination block's dead region beyond
    # num_tokens unspecified (those bytes belong to tokens decode writes
    # later). Every block outside dst_ids must stay untouched poison.
    compare_bytes = live_tail_bytes if block_id == final_block else page_bytes
    if not np.array_equal(got[:compare_bytes], want[:compare_bytes]):
      first = int(np.argmax(got[:compare_bytes] != want[:compare_bytes]))
      raise AssertionError(
          f"D2 destination block {block_id} differs at byte {first}: "
          f"got={got[first]}, want={want[first]}")
  assert dst_node.fa.data_ptr() == dst_node.fa_data_ptr, "data_ptr changed"
  assert np.array_equal(dst_node.read_skip_device_bytes(), dst_skip_before), (
      "unselected pool bytes changed (leak)")
  print(f"[D1/D2] PASS: {len(dst_ids)} destination pages bit-exact, "
        f"poison tail + unselected pool intact, data_ptr unchanged")

  # ---- D3: two uuids in flight, disjoint destinations ----
  reqs = [("device-req-2", 515151, 1024, 43),
          ("device-req-3", 616161, 256, 60)]
  for r_req, r_uuid, r_tokens, dst_base in reqs:
    for node in src_nodes:
      spans, owned = _spans_for_rank(node.rank, r_tokens)
      src_facade.register_request_blocks(
          req_id=r_req,
          uuid=r_uuid,
          unit=_unit(node.rank),
          block_ids=list(range(-(-owned // SRC_PAGE_TOKENS))),
          spans=spans,
      )
  results = []
  for r_req, r_uuid, r_tokens, dst_base in reqs:
    r_ids = list(range(dst_base, dst_base + -(-r_tokens // DST_PAGE_TOKENS)))
    assert _transfer(dst_facade, [_unit(r) for r in range(PARALLELISM)],
                     r_ids, r_req, r_uuid, r_tokens, src_addr,
                     dst_addr) is True
    results.append((r_req, r_ids, r_tokens))
  for r_req, r_ids, r_tokens in results:
    dst_poll.wait_for(("done_recving", r_req))
  fa_bytes = dst_node.read_fa_device_bytes()
  for r_req, r_ids, r_tokens in results:
    # Sources reuse block 0.. for every request, so the source pattern for a
    # shorter request is the prefix of the 2,500-token fill. Recompute.
    for page_ordinal, block_id in enumerate(r_ids):
      got = fa_bytes[block_id * page_bytes:(block_id + 1) * page_bytes]
      page_start = page_ordinal * DST_PAGE_TOKENS
      want = np.full(page_bytes, POISON, dtype=np.uint8)
      for offset in range(DST_PAGE_TOKENS):
        token = page_start + offset
        if token >= r_tokens:
          break
        want[offset * TOKEN_BYTES:(offset + 1) *
             TOKEN_BYTES] = _token_record(token)
      assert np.array_equal(got, want), (
          f"D3 {r_req} block {block_id} mismatch")
  print("[D3] PASS: two concurrent uuids isolated and bit-exact")

  # ---- D4: dispatch failure -> receiver timeout -> uuid reusable ----
  req4, uuid4 = "device-req-4", 717171
  broken_rank = PARALLELISM - 1
  # Point the broken rank's control endpoint at a dead port so the sender
  # dispatch fails after the receiver is armed.
  broken = src_nodes[broken_rank]
  src_facade.register_work_unit(
      unit=_unit(broken_rank),
      shards=[broken.transfer_address],
      control_plane_rpc_address="127.0.0.1:1",
      pool_manifest=broken.manifest,
      layout_fingerprint=FINGERPRINT,
      page_tokens=SRC_PAGE_TOKENS,
      transfer_parallelism=PARALLELISM,
      transfer_rank=broken_rank,
  )
  for node in src_nodes:
    spans, owned = _spans_for_rank(node.rank, 512)
    src_facade.register_request_blocks(
        req_id=req4,
        uuid=uuid4,
        unit=_unit(node.rank),
        block_ids=list(range(-(-owned // SRC_PAGE_TOKENS))),
        spans=spans,
    )
  ids4 = list(range(20, 20 + -(-512 // DST_PAGE_TOKENS)))
  dispatch_failed = False
  try:
    _transfer(dst_facade, [_unit(r) for r in range(PARALLELISM)], ids4, req4,
              uuid4, 512, src_addr, dst_addr)
  except RuntimeError as exc:
    dispatch_failed = True
    print(f"[D4] dispatch failed as injected: {exc}")
  assert dispatch_failed, "expected sender dispatch failure"
  failed = dst_poll.wait_for(("failed_recving", req4), deadline_s=30.0)
  print(f"[D4] receiver deadline surfaced failed_recving={sorted(failed)}")

  # Heal the endpoint, release the consumed D5 generation, and reuse the
  # SAME uuid to prove receive-progress reclamation.
  src_facade.register_work_unit(
      unit=_unit(broken_rank),
      shards=[broken.transfer_address],
      control_plane_rpc_address=broken.listener_address,
      pool_manifest=broken.manifest,
      layout_fingerprint=FINGERPRINT,
      page_tokens=SRC_PAGE_TOKENS,
      transfer_parallelism=PARALLELISM,
      transfer_rank=broken_rank,
  )
  src_facade.release_request_blocks(req4, uuid4)
  req4b = "device-req-4-retry"
  for node in src_nodes:
    spans, owned = _spans_for_rank(node.rank, 512)
    src_facade.register_request_blocks(
        req_id=req4b,
        uuid=uuid4,
        unit=_unit(node.rank),
        block_ids=list(range(-(-owned // SRC_PAGE_TOKENS))),
        spans=spans,
    )
  assert _transfer(dst_facade, [_unit(r) for r in range(PARALLELISM)], ids4,
                   req4b, uuid4, 512, src_addr, dst_addr) is True
  dst_poll.wait_for(("done_recving", req4b))
  fa_bytes = dst_node.read_fa_device_bytes()
  for page_ordinal, block_id in enumerate(ids4):
    got = fa_bytes[block_id * page_bytes:(block_id + 1) * page_bytes]
    want = np.full(page_bytes, POISON, dtype=np.uint8)
    live = 0
    for offset in range(DST_PAGE_TOKENS):
      token = page_ordinal * DST_PAGE_TOKENS + offset
      if token >= 512:
        break
      want[offset * TOKEN_BYTES:(offset + 1) *
           TOKEN_BYTES] = _token_record(token)
      live += TOKEN_BYTES
    assert np.array_equal(got[:live], want[:live]), (
        f"D4 retry block {block_id} mismatch")
  print("[D4] PASS: injected failure surfaced, uuid reused successfully")

  src_server.stop()
  dst_server.stop()
  print("RAIDEN_STAGE3_D_SERIES=PASS")
  return 0


if __name__ == "__main__":
  sys.exit(main())
