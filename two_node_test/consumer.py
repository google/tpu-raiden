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

"""Consumer side of the two-node Raiden TransferEngine benchmark.

Runs on the DECODE host. Builds zero-filled KV caches with the real spec,
constructs a TransferEngine, issues `--num-requests` `start_read` pulls against
the producer's `--remote-endpoint`, polls `complete_read()` until all are done
(or failed/timeout), then reports aggregate bandwidth and (for a single request)
verifies the pulled bytes against the deterministic source.

The engine emits per-request `RAIDEN_TIMING event=consumer_pipeline ...` lines
to stderr (control_setup_ms ~= producer d2h_issue; h2h_ms / h2d_* are the
network + host->device stages).
"""

import argparse
import sys
import time

from api.jax.transfer_engine import TransferEngine
import kv_spec


def log(msg: str) -> None:
    print(msg, flush=True)


_ZERO_SAMPLE = 16  # untouched blocks sampled per layer for the zero invariant


def verify(dst_caches, seed: int, kv_cache_num_blocks: int, maps,
           completed_idxs, full: bool) -> bool:
    """Byte-exact verification of the pulled KV across all completed requests.

    Per layer, two invariants:
      1. data:  dst[local[i]] == src[requested_remote[i]]  (the mapping landed)
      2. zero:  blocks never written stay all-zero (no over-write)

    Only the specific blocks being checked are moved to host (the pool is large),
    and source bytes are regenerated from `seed` (never received). `full=True`
    checks all 64 layers and every transferred block; otherwise it samples
    first/mid/last per request. The zero invariant is always sampled
    (`_ZERO_SAMPLE` untouched blocks/layer) since the pool is mostly zeros.
    """
    layers = (list(range(kv_spec.NUM_LAYERS)) if full
              else sorted({0, kv_spec.NUM_LAYERS // 2, kv_spec.NUM_LAYERS - 1}))

    # (dst_block, src_block) pairs to byte-compare, across completed requests.
    pairs = []
    touched = set()
    for r in completed_idxs:
        _, requested_remote, local = maps[r]
        touched.update(local)
        rpairs = list(zip(local, requested_remote))
        if not full and rpairs:
            idxs = sorted({0, len(rpairs) // 2, len(rpairs) - 1})
            rpairs = [rpairs[i] for i in idxs]
        pairs.extend(rpairs)

    # Sampled untouched blocks spread across the pool for the zero check.
    untouched = [b for b in range(kv_cache_num_blocks) if b not in touched]
    if untouched:
        step = max(1, len(untouched) // _ZERO_SAMPLE)
        zero_ids = untouched[::step][:_ZERO_SAMPLE]
    else:
        zero_ids = []

    dst_ids = [p[0] for p in pairs]
    src_ids = [p[1] for p in pairs]

    ok = True
    n_data = n_zero = 0
    for layer in layers:
        if pairs:
            got = kv_spec.read_blocks_uint8(dst_caches[layer], dst_ids)
            want = kv_spec.expected_blocks_uint8(seed, layer, src_ids)
            for j in range(len(pairs)):
                n_data += 1
                if not (got[j] == want[j]).all():
                    mism = int((got[j] != want[j]).sum())
                    log(f"VERIFY FAIL data layer={layer} dst_block={dst_ids[j]} "
                        f"src_block={src_ids[j]} mismatch={mism}/{got[j].size}")
                    ok = False
        if zero_ids:
            gotz = kv_spec.read_blocks_uint8(dst_caches[layer], zero_ids)
            for j, b in enumerate(zero_ids):
                n_zero += 1
                if gotz[j].any():
                    nz = int((gotz[j] != 0).sum())
                    log(f"VERIFY FAIL nonzero-untouched layer={layer} block={b} "
                        f"nonzero={nz}/{gotz[j].size}")
                    ok = False
    log(f"VERIFY checked layers={len(layers)} requests={len(completed_idxs)} "
        f"data_slices={n_data} zero_slices={n_zero} full={full}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--remote-endpoint", type=str, required=True,
                   help="Producer control endpoint host:port.")
    p.add_argument("--control-port", type=int, default=9201,
                   help="This engine's own control port (unused inbound).")
    p.add_argument("--num-blocks", type=int, default=64)
    p.add_argument("--max-blocks", type=int, default=72)
    p.add_argument("--kv-cache-num-blocks", type=int, default=1024,
                   help="Cache dim0: the full block pool (must match producer).")
    p.add_argument("--num-requests", type=int, default=1)
    p.add_argument("--mapping", type=str, default="identity",
                   choices=["identity", "reorder", "sparse"],
                   help="Block mapping (must match the producer).")
    p.add_argument("--num-slots", type=int, default=0,
                   help="0 => max(num_requests, 8). Set == num_requests to "
                        "reproduce the N-way D2H contention curve.")
    p.add_argument("--parallelism", type=int, default=4,
                   help="H2H transport sockets per pull (real run uses 4).")
    p.add_argument("--num-devices", type=int, default=8)
    p.add_argument("--req-id-prefix", type=str, default="two_node_req_")
    p.add_argument("--base-uuid", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--timeout-s", type=float, default=600.0)
    p.add_argument("--verify", action="store_true", default=False)
    p.add_argument("--verify-full", action="store_true", default=False,
                   help="Check every layer/block (not just sampled slices).")
    p.add_argument("--skip-buffer-lock", action="store_true", default=True)
    args = p.parse_args()

    num_slots = args.num_slots or max(args.num_requests, 8)

    import jax
    log(f"CONSUMER devices={len(jax.devices('tpu'))} "
        f"remote={args.remote_endpoint} requests={args.num_requests} "
        f"kv_cache_num_blocks={args.kv_cache_num_blocks} "
        f"num_slots={num_slots} parallelism={args.parallelism}")

    sharding = kv_spec.build_sharding(args.num_devices)
    dst_caches = kv_spec.make_dst_caches(args.kv_cache_num_blocks, sharding)

    engine = TransferEngine(
        kv_caches=dst_caches,
        local_control_port=args.control_port,
        max_blocks=args.max_blocks,
        num_slots=num_slots,
        unsafe_skip_buffer_lock=args.skip_buffer_lock,
    )

    # Distinct, non-overlapping block window per request (matches the producer).
    maps = [
        kv_spec.request_block_map(args.mapping, args.num_blocks,
                                  args.kv_cache_num_blocks, r)
        for r in range(args.num_requests)
    ]
    transferred_blocks = len(maps[0][2])
    req_ids = [f"{args.req_id_prefix}{i}" for i in range(args.num_requests)]
    _, req0_remote, req0_local = maps[0]
    log(f"CONSUMER mapping={args.mapping} transferred_blocks={transferred_blocks}"
        f" req0_remote[:4]={req0_remote[:4]} req0_local[:4]={req0_local[:4]}")

    # Issue all pulls up front; the slot pool (num_slots) bounds real
    # concurrency -- this reproduces the prefill->decode burst.
    req_index = {req_id: i for i, req_id in enumerate(req_ids)}
    t_start = time.perf_counter()
    for i, req_id in enumerate(req_ids):
        _, requested_remote, local = maps[i]
        engine.start_read(
            req_id=req_id,
            uuid=args.base_uuid + i,
            remote_endpoint=args.remote_endpoint,
            remote_block_ids=requested_remote,
            local_block_ids=local,
            parallelism=args.parallelism,
        )

    pending = set(req_ids)
    failed = set()
    completed = []
    deadline = time.perf_counter() + args.timeout_s
    while pending and time.perf_counter() < deadline:
        _, done_recving, failed_recving = engine.complete_read()
        for r in failed_recving:
            if r in pending:
                failed.add(r)
                pending.discard(r)
                log(f"CONSUMER FAILED req={r}")
        for r in done_recving:
            if r in pending:
                pending.discard(r)
                completed.append(req_index[r])
        time.sleep(0.02)
    wall = time.perf_counter() - t_start

    n_done = args.num_requests - len(pending) - len(failed)
    bytes_per_request = kv_spec.bytes_for(transferred_blocks)
    total_bytes = bytes_per_request * n_done
    gbps = (total_bytes / wall / 1e9) if wall > 0 else 0.0

    log("==================== TWO-NODE TRANSFER SUMMARY ====================")
    log(f"requests={args.num_requests} done={n_done} failed={len(failed)} "
        f"pending={len(pending)} mapping={args.mapping}")
    log(f"bytes_per_request={bytes_per_request} "
        f"({bytes_per_request/2**30:.2f} GiB)")
    log(f"wall_s={wall:.3f} aggregate_GB_s={gbps:.2f} "
        f"per_request_s={wall/max(n_done,1):.3f}")
    log("==================================================================")

    rc = 0
    if pending:
        log(f"CONSUMER_TIMEOUT pending={sorted(pending)}")
        rc = 1
    if failed:
        log(f"CONSUMER_FAILED failed={sorted(failed)}")
        rc = 1

    if args.verify and completed:
        if verify(dst_caches, args.seed, args.kv_cache_num_blocks, maps,
                  sorted(completed), args.verify_full):
            log("VERIFY OK (byte-exact"
                + (" full" if args.verify_full else " sampled") + ")")
        else:
            log("VERIFY FAILED")
            rc = 1
    elif args.verify:
        log("VERIFY SKIPPED (no completed request)")

    return rc


if __name__ == "__main__":
    sys.exit(main())
