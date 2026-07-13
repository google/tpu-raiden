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
"""Standalone Stage 1 PCP8 -> DPxTP2 host resharding runner.

This script is intentionally host-only. It drives the same registered
StartTransferRequest plans used by the Stage 1 local primitive test, then sends
bytes
with KVCacheManager.push_registered_plan(), which lowers to BlockTransport::Push
op=6 and RawBufferTransport::WriteVExact/ReadVExact.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import socket
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tpu_raiden.api.torch import kv_cache_manager_host
from tpu_raiden.rpc import raiden_service_pb2

NUM_FA_LAYERS = 15
TOKENS_PER_BLOCK = 1024
HEAD_PAIR_BYTES = 512
TOKEN_STRIDE_BYTES = 1024
SRC_SLOT_BYTES = 1_067_008
DST_SLOT_BYTES = TOKENS_PER_BLOCK * HEAD_PAIR_BYTES
POISON = 0xAA
DEFAULT_NUM_TOKENS = 64 * 1024
DEFAULT_UUID_BASE = 0x51E135


def _positive_int(value: str) -> int:
  result = int(value, 0)
  if result <= 0:
    raise argparse.ArgumentTypeError(f"{value!r} must be positive")
  return result


def _nonnegative_int(value: str) -> int:
  result = int(value, 0)
  if result < 0:
    raise argparse.ArgumentTypeError(f"{value!r} must be non-negative")
  return result


def _resolve_shape(args: argparse.Namespace) -> Tuple[int, int]:
  if args.num_tokens is None and args.num_pages is None:
    num_tokens = DEFAULT_NUM_TOKENS
    num_pages = DEFAULT_NUM_TOKENS // TOKENS_PER_BLOCK
  elif args.num_tokens is None:
    num_pages = args.num_pages
    num_tokens = num_pages * TOKENS_PER_BLOCK
  elif args.num_pages is None:
    num_tokens = args.num_tokens
    num_pages = math.ceil(num_tokens / TOKENS_PER_BLOCK)
  else:
    num_tokens = args.num_tokens
    num_pages = args.num_pages
    expected_pages = math.ceil(num_tokens / TOKENS_PER_BLOCK)
    if num_pages != expected_pages:
      raise ValueError(
          "--num_pages must equal ceil(--num_tokens / 1024) when both are set:"
          f" got {num_pages}, expected {expected_pages}"
      )

  if num_tokens <= 0 or num_pages <= 0:
    raise ValueError("num_tokens and num_pages must both be positive")
  return num_tokens, num_pages


def _tokens_in_page(page: int, num_tokens: int) -> int:
  remaining = num_tokens - page * TOKENS_PER_BLOCK
  return max(0, min(TOKENS_PER_BLOCK, remaining))


def _owned_pages(rank: int, num_sources: int, num_pages: int) -> List[int]:
  return list(range(rank, num_pages, num_sources))


def _owned_block_ids(
    rank: int, num_sources: int, num_pages: int
) -> Tuple[List[int], List[int]]:
  pages = _owned_pages(rank, num_sources, num_pages)
  src_ids = [page // num_sources for page in pages]
  dst_ids = pages
  return src_ids, dst_ids


def _host_blocks_for_rank(rank: int, num_sources: int, num_pages: int) -> int:
  return max(1, len(_owned_pages(rank, num_sources, num_pages)))


def _format_host_port(host: str, port: int) -> str:
  if ":" in host and not host.startswith("["):
    return f"[{host}]:{port}"
  return f"{host}:{port}"


def _dest_peer(
    dest_host: str,
    dest_base_port: int,
    dests_per_group: int,
    group: int,
    dest_rank: int,
) -> str:
  port = dest_base_port + group * dests_per_group + dest_rank
  return _format_host_port(dest_host, port)


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


def _new_request(uuid: int, is_sender: bool) -> Any:
  req = raiden_service_pb2.StartTransferRequest()
  req.uuid = uuid
  req.is_sender = is_sender
  req.dst_mem_type = raiden_service_pb2.MEMORY_TYPE_DRAM
  req.use_block_chunks = True
  req.req_id = f"stage1_reshard_{uuid}"
  return req


def _add_entry(
    schedule: Any,
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
    peers: Sequence[str],
    rank: int,
    num_sources: int,
    num_pages: int,
    num_tokens: int,
) -> Any:
  req = _new_request(uuid, is_sender=True)
  schedule = req.shard_push_schedules[0]
  for page in _owned_pages(rank, num_sources, num_pages):
    live_tokens = _tokens_in_page(page, num_tokens)
    if live_tokens <= 0:
      continue
    src_block_id = page // num_sources
    for head, peer in enumerate(peers):
      _add_entry(
          schedule,
          peer=peer,
          src_block_id=src_block_id,
          dst_block_id=page,
          dst_shard_idx=0,
          src_offset=head * HEAD_PAIR_BYTES,
          dst_offset=0,
          size=HEAD_PAIR_BYTES,
          src_stride=TOKEN_STRIDE_BYTES,
          count=live_tokens,
      )
  return req


def _receiver_request(
    uuid: int,
    peers: Sequence[str],
    dest_rank: int,
    num_sources: int,
    num_pages: int,
    num_tokens: int,
) -> Any:
  req = _new_request(uuid, is_sender=False)
  for rank in range(num_sources):
    schedule = req.shard_push_schedules[rank]
    for page in _owned_pages(rank, num_sources, num_pages):
      live_tokens = _tokens_in_page(page, num_tokens)
      if live_tokens <= 0:
        continue
      _add_entry(
          schedule,
          peer=peers[dest_rank],
          src_block_id=page // num_sources,
          dst_block_id=page,
          dst_shard_idx=0,
          src_offset=0,
          dst_offset=0,
          size=live_tokens * HEAD_PAIR_BYTES,
          count=1,
      )
  return req


def _validate_args(
    args: argparse.Namespace, num_tokens: int, num_pages: int
) -> None:
  del num_tokens, num_pages
  if args.num_sources != 8:
    raise ValueError("Stage 1 topology expects --num_sources=8")
  if args.dests_per_group != 2:
    raise ValueError("Stage 1 geometry has exactly two destination heads")
  if args.push_parallelism <= 0:
    raise ValueError("--push_parallelism must be positive")


def _control_request(
    host: str, port: int, payload: Dict[str, Any], timeout_s: float
) -> Dict[str, Any]:
  with socket.create_connection((host, port), timeout=timeout_s) as sock:
    sock.settimeout(timeout_s)
    with sock.makefile("rwb") as f:
      f.write(json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n")
      f.flush()
      line = f.readline()
  if not line:
    raise RuntimeError(f"control request {payload.get('cmd')} got no response")
  response = json.loads(line.decode("utf-8"))
  if not response.get("ok", False):
    raise RuntimeError(
        f"control request {payload.get('cmd')} failed: {response}"
    )
  return response


def _make_source_managers(
    args: argparse.Namespace, num_pages: int
) -> List[Any]:
  managers = []
  for rank in range(args.num_sources):
    managers.append(
        kv_cache_manager_host.HostKVCacheManager(
            num_layers=NUM_FA_LAYERS,
            num_shards=1,
            slice_byte_size=SRC_SLOT_BYTES,
            node_id=rank,
            local_port=args.source_base_port + rank,
            host_blocks=_host_blocks_for_rank(
                rank, args.num_sources, num_pages
            ),
            parallelism=args.manager_parallelism,
        )
    )
  return managers


def _make_destination_managers(
    args: argparse.Namespace, num_pages: int
) -> List[Any]:
  managers = []
  manager_count = args.groups * args.dests_per_group
  for idx in range(manager_count):
    managers.append(
        kv_cache_manager_host.HostKVCacheManager(
            num_layers=NUM_FA_LAYERS,
            num_shards=1,
            slice_byte_size=DST_SLOT_BYTES,
            node_id=100 + idx,
            local_port=args.dest_base_port + idx,
            host_blocks=num_pages,
            parallelism=args.manager_parallelism,
        )
    )
  return managers


def _fill_sources(
    managers: Sequence[Any], num_sources: int, num_pages: int, num_tokens: int
) -> None:
  for rank, manager in enumerate(managers):
    for page in _owned_pages(rank, num_sources, num_pages):
      live_tokens = _tokens_in_page(page, num_tokens)
      for layer in range(NUM_FA_LAYERS):
        manager.write_block_bytes(
            layer,
            page // num_sources,
            _source_payload(layer, page, live_tokens),
        )


def _poison_destinations(managers: Sequence[Any], num_pages: int) -> None:
  poison = bytes([POISON]) * DST_SLOT_BYTES
  for manager in managers:
    for layer in range(NUM_FA_LAYERS):
      for block_id in range(num_pages):
        manager.write_block_bytes(layer, block_id, poison)


def _manager_index(group: int, dest_rank: int, dests_per_group: int) -> int:
  return group * dests_per_group + dest_rank


def _register_sender_plans(
    managers: Sequence[Any],
    args: argparse.Namespace,
    num_pages: int,
    num_tokens: int,
    dest_host: str,
    active_uuids: List[int],
) -> None:
  for group in range(args.groups):
    uuid = args.uuid_base + group
    peers = [
        _dest_peer(
            dest_host,
            args.dest_base_port,
            args.dests_per_group,
            group,
            dest_rank,
        )
        for dest_rank in range(args.dests_per_group)
    ]
    for rank, manager in enumerate(managers):
      manager.register_active_plan(
          uuid,
          _sender_request(
              uuid, peers, rank, args.num_sources, num_pages, num_tokens
          ),
          is_sender=True,
      )
    active_uuids.append(uuid)


def _unregister_sender_plans(
    managers: Sequence[Any], active_uuids: Sequence[int]
) -> None:
  for manager in managers:
    for uuid in active_uuids:
      with contextlib.suppress(Exception):
        manager.unregister_active_plan(uuid)


def _run_pushes(
    managers: Sequence[Any],
    args: argparse.Namespace,
    num_pages: int,
    dest_host: str,
) -> Tuple[float, List[Dict[str, Any]]]:
  results = []
  results_lock = threading.Lock()
  threads = []

  def push_one(
      group: int,
      rank: int,
      dest_rank: int,
      peer: str,
      src_ids: Sequence[int],
      dst_ids: Sequence[int],
  ) -> None:
    uuid = args.uuid_base + group
    start = time.perf_counter()
    ok = True
    error = ""
    try:
      managers[rank].push_registered_plan(
          uuid,
          peer,
          src_ids,
          dst_ids,
          layer_idx=-1,
          parallelism=args.push_parallelism,
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      ok = False
      error = repr(exc)
    elapsed_s = time.perf_counter() - start
    with results_lock:
      results.append({
          "ok": ok,
          "error": error,
          "group": group,
          "source_rank": rank,
          "dest_rank": dest_rank,
          "peer": peer,
          "blocks": len(src_ids),
          "elapsed_s": elapsed_s,
      })

  start_all = time.perf_counter()
  for group in range(args.groups):
    for rank in range(args.num_sources):
      src_ids, dst_ids = _owned_block_ids(rank, args.num_sources, num_pages)
      if not src_ids:
        continue
      for dest_rank in range(args.dests_per_group):
        peer = _dest_peer(
            dest_host,
            args.dest_base_port,
            args.dests_per_group,
            group,
            dest_rank,
        )
        thread = threading.Thread(
            target=push_one,
            args=(group, rank, dest_rank, peer, src_ids, dst_ids),
            daemon=True,
        )
        threads.append(thread)
        thread.start()

  for thread in threads:
    thread.join()
  elapsed_s = time.perf_counter() - start_all
  return elapsed_s, sorted(
      results,
      key=lambda item: (item["group"], item["source_rank"], item["dest_rank"]),
  )


def _payload_bytes(
    groups: int, dests_per_group: int, num_pages: int, num_tokens: int
) -> int:
  per_dest_layer = 0
  for page in range(num_pages):
    per_dest_layer += _tokens_in_page(page, num_tokens) * HEAD_PAIR_BYTES
  return groups * dests_per_group * NUM_FA_LAYERS * per_dest_layer


def _print_source_report(
    elapsed_s: float,
    results: Sequence[Dict[str, Any]],
    groups: int,
    dests_per_group: int,
    num_pages: int,
    num_tokens: int,
    verify_response: Dict[str, Any],
) -> None:
  total_bytes = _payload_bytes(groups, dests_per_group, num_pages, num_tokens)
  gbps = total_bytes * 8 / elapsed_s / 1e9 if elapsed_s > 0 else 0.0
  print(
      json.dumps(
          {
              "ok": True,
              "elapsed_s": elapsed_s,
              "aggregate_gbps": gbps,
              "bytes": total_bytes,
              "flows": len(results),
              "verify": verify_response,
              "flow_results": results,
          },
          sort_keys=True,
      ),
      flush=True,
  )


class DestinationServer:
  """JSON control server around destination-side host managers."""

  def __init__(self, args: argparse.Namespace, num_tokens: int, num_pages: int):
    self._args = args
    self._num_tokens = num_tokens
    self._num_pages = num_pages
    self._managers = _make_destination_managers(args, num_pages)
    self._active_uuids: List[int] = []
    self._stop = threading.Event()
    self._handler_threads: List[threading.Thread] = []
    print(
        json.dumps(
            {
                "event": "destination_ready",
                "control": _format_host_port(args.bind_host, args.control_port),
                "data_ports": [
                    args.dest_base_port + idx
                    for idx in range(args.groups * args.dests_per_group)
                ],
                "groups": args.groups,
                "num_pages": num_pages,
                "num_tokens": num_tokens,
            },
            sort_keys=True,
        ),
        flush=True,
    )

  def serve(self) -> None:
    try:
      with socket.create_server(
          (self._args.bind_host, self._args.control_port),
          backlog=16,
          reuse_port=False,
      ) as server:
        server.settimeout(0.25)
        while not self._stop.is_set():
          try:
            conn, _ = server.accept()
          except socket.timeout:
            continue
          thread = threading.Thread(target=self._handle_conn, args=(conn,))
          self._handler_threads.append(thread)
          thread.start()
    finally:
      for thread in self._handler_threads:
        thread.join()
      self._unregister_active()
      self._managers.clear()

  def _handle_conn(self, conn: socket.socket) -> None:
    with conn:
      with conn.makefile("rwb") as f:
        line = f.readline()
        if not line:
          return
        try:
          request = json.loads(line.decode("utf-8"))
          response = self._dispatch(request)
        except Exception as exc:  # pylint: disable=broad-exception-caught
          response = {"ok": False, "error": repr(exc)}
        f.write(json.dumps(response, sort_keys=True).encode("utf-8") + b"\n")
        f.flush()

  def _dispatch(self, request: Dict[str, Any]) -> Dict[str, Any]:
    cmd = request.get("cmd")
    if cmd == "reset":
      return self._reset(request)
    if cmd == "verify":
      return self._verify(request)
    if cmd == "cleanup":
      self._unregister_active()
      return {"ok": True}
    if cmd == "shutdown":
      self._unregister_active()
      self._stop.set()
      return {"ok": True}
    return {"ok": False, "error": f"unknown command {cmd!r}"}

  def _request_config(
      self, request: Dict[str, Any]
  ) -> Tuple[int, int, int, str]:
    groups = int(request.get("groups", self._args.groups))
    num_tokens = int(request.get("num_tokens", self._num_tokens))
    num_pages = int(request.get("num_pages", self._num_pages))
    dest_host = str(request.get("dest_host", self._args.advertise_host))
    if groups < 1 or groups > self._args.groups:
      raise ValueError(
          f"groups must be in [1, {self._args.groups}], got {groups}"
      )
    if num_tokens != self._num_tokens or num_pages != self._num_pages:
      raise ValueError(
          "destination was started for a fixed shape:"
          f" got num_tokens={num_tokens}, num_pages={num_pages};"
          f" expected {self._num_tokens}, {self._num_pages}"
      )
    return groups, num_tokens, num_pages, dest_host

  def _unregister_active(self) -> None:
    active_uuids = list(dict.fromkeys(self._active_uuids))
    self._active_uuids.clear()
    for manager in self._managers:
      for uuid in active_uuids:
        with contextlib.suppress(Exception):
          manager.unregister_active_plan(uuid)

  def _reset(self, request: Dict[str, Any]) -> Dict[str, Any]:
    groups, num_tokens, num_pages, dest_host = self._request_config(request)
    self._unregister_active()
    _poison_destinations(
        self._managers[: groups * self._args.dests_per_group], num_pages
    )
    for group in range(groups):
      uuid = int(request.get("uuid_base", self._args.uuid_base)) + group
      peers = [
          _dest_peer(
              dest_host,
              self._args.dest_base_port,
              self._args.dests_per_group,
              group,
              dest_rank,
          )
          for dest_rank in range(self._args.dests_per_group)
      ]
      for dest_rank in range(self._args.dests_per_group):
        manager = self._managers[
            _manager_index(group, dest_rank, self._args.dests_per_group)
        ]
        manager.register_active_plan(
            uuid,
            _receiver_request(
                uuid,
                peers,
                dest_rank,
                self._args.num_sources,
                num_pages,
                num_tokens,
            ),
            is_sender=False,
        )
      self._active_uuids.append(uuid)
    return {
        "ok": True,
        "registered_uuids": list(self._active_uuids),
        "poisoned_blocks": groups * self._args.dests_per_group * num_pages,
    }

  def _verify(self, request: Dict[str, Any]) -> Dict[str, Any]:
    groups, num_tokens, num_pages, _ = self._request_config(request)
    mismatches = 0
    examples = []
    start = time.perf_counter()
    for group in range(groups):
      for dest_rank in range(self._args.dests_per_group):
        manager = self._managers[
            _manager_index(group, dest_rank, self._args.dests_per_group)
        ]
        for layer in range(NUM_FA_LAYERS):
          for page in range(num_pages):
            live_tokens = _tokens_in_page(page, num_tokens)
            actual = manager.read_block_bytes(layer, page)
            expected = _expected_dest_payload(
                layer, page, live_tokens, dest_rank
            )
            if actual == expected:
              continue
            mismatches += 1
            if len(examples) < 10:
              examples.append(
                  self._describe_mismatch(
                      group,
                      dest_rank,
                      layer,
                      page,
                      live_tokens,
                      actual,
                      expected,
                  )
              )
    elapsed_s = time.perf_counter() - start
    return {
        "ok": mismatches == 0,
        "mismatches": mismatches,
        "examples": examples,
        "verify_elapsed_s": elapsed_s,
    }

  def _describe_mismatch(
      self,
      group: int,
      dest_rank: int,
      layer: int,
      page: int,
      live_tokens: int,
      actual: bytes,
      expected: bytes,
  ) -> Dict[str, Any]:
    offset = next(
        (
            idx
            for idx, pair in enumerate(zip(actual, expected))
            if pair[0] != pair[1]
        ),
        min(len(actual), len(expected)),
    )
    token = offset // HEAD_PAIR_BYTES
    in_token_offset = offset % HEAD_PAIR_BYTES
    actual_byte = actual[offset] if offset < len(actual) else None
    expected_byte = expected[offset] if offset < len(expected) else None
    return {
        "group": group,
        "dest_rank": dest_rank,
        "layer": layer,
        "page": page,
        "live_tokens": live_tokens,
        "offset": offset,
        "token": token,
        "in_token_offset": in_token_offset,
        "actual": actual_byte,
        "expected": expected_byte,
    }


def _run_destination(args: argparse.Namespace) -> int:
  num_tokens, num_pages = _resolve_shape(args)
  _validate_args(args, num_tokens, num_pages)
  DestinationServer(args, num_tokens, num_pages).serve()
  return 0


def _run_source(args: argparse.Namespace) -> int:
  num_tokens, num_pages = _resolve_shape(args)
  _validate_args(args, num_tokens, num_pages)
  dest_host = args.dest_host or args.peer
  if not dest_host:
    raise ValueError("--dest_host or --peer is required for --role=source")

  managers = _make_source_managers(args, num_pages)
  print(
      json.dumps(
          {
              "event": "source_managers_ready",
              "source_ports": [
                  args.source_base_port + rank
                  for rank in range(args.num_sources)
              ],
              "groups": args.groups,
              "num_pages": num_pages,
              "num_tokens": num_tokens,
          },
          sort_keys=True,
      ),
      flush=True,
  )
  _fill_sources(managers, args.num_sources, num_pages, num_tokens)

  for run_idx in range(args.repeat):
    active_uuids: List[int] = []
    try:
      _register_sender_plans(
          managers, args, num_pages, num_tokens, dest_host, active_uuids
      )
      reset_response = _control_request(
          dest_host,
          args.control_port,
          {
              "cmd": "reset",
              "uuid_base": args.uuid_base,
              "groups": args.groups,
              "num_tokens": num_tokens,
              "num_pages": num_pages,
              "dest_host": dest_host,
          },
          args.control_timeout_s,
      )
      print(
          json.dumps(
              {
                  "event": "destination_reset",
                  "run": run_idx,
                  "response": reset_response,
              },
              sort_keys=True,
          ),
          flush=True,
      )

      elapsed_s, results = _run_pushes(managers, args, num_pages, dest_host)
      failures = [result for result in results if not result["ok"]]
      if failures:
        raise RuntimeError(f"push failures: {failures}")

      verify_response = _control_request(
          dest_host,
          args.control_port,
          {
              "cmd": "verify",
              "uuid_base": args.uuid_base,
              "groups": args.groups,
              "num_tokens": num_tokens,
              "num_pages": num_pages,
              "dest_host": dest_host,
          },
          args.control_timeout_s,
      )
      _print_source_report(
          elapsed_s,
          results,
          args.groups,
          args.dests_per_group,
          num_pages,
          num_tokens,
          verify_response,
      )
    finally:
      _unregister_sender_plans(managers, active_uuids)
      with contextlib.suppress(Exception):
        _control_request(
            dest_host,
            args.control_port,
            {
                "cmd": "cleanup",
                "uuid_base": args.uuid_base,
                "groups": args.groups,
                "num_tokens": num_tokens,
                "num_pages": num_pages,
                "dest_host": dest_host,
            },
            args.control_timeout_s,
        )

  if args.shutdown_destination:
    with contextlib.suppress(Exception):
      _control_request(
          dest_host,
          args.control_port,
          {"cmd": "shutdown"},
          args.control_timeout_s,
      )
  return 0


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--role", choices=("source", "destination"), required=True
  )
  parser.add_argument("--num_tokens", type=_positive_int, default=None)
  parser.add_argument("--num_pages", type=_positive_int, default=None)
  parser.add_argument("--num_sources", type=_positive_int, default=8)
  parser.add_argument("--groups", type=_positive_int, default=1)
  parser.add_argument("--dests_per_group", type=_positive_int, default=2)
  parser.add_argument(
      "--uuid_base", type=_nonnegative_int, default=DEFAULT_UUID_BASE
  )
  parser.add_argument("--source_base_port", type=_positive_int, default=19000)
  parser.add_argument("--dest_base_port", type=_positive_int, default=19100)
  parser.add_argument("--control_port", type=_positive_int, default=18700)
  parser.add_argument("--manager_parallelism", type=_positive_int, default=4)
  parser.add_argument("--push_parallelism", type=_positive_int, default=1)
  parser.add_argument("--control_timeout_s", type=float, default=300.0)
  parser.add_argument("--bind_host", default="0.0.0.0")
  parser.add_argument("--advertise_host", default="127.0.0.1")
  parser.add_argument("--dest_host", default=None)
  parser.add_argument("--peer", default=None, help="Alias for --dest_host.")
  parser.add_argument("--repeat", type=_positive_int, default=1)
  parser.add_argument("--shutdown_destination", action="store_true")
  return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
  args = _parse_args(argv)
  if args.role == "destination":
    return _run_destination(args)
  return _run_source(args)


if __name__ == "__main__":
  sys.exit(main())
