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

"""Producer side of the two-node Raiden TransferEngine benchmark.

Runs on the PREFILL host. Builds the real Qwen3-32B TP=8 fp8 KV cache (random,
deterministic), constructs a TransferEngine, registers `--num-requests` send
entries with `notify_for_read`, prints `PRODUCER_READY <host>:<port>` so the
orchestrator can launch the consumer, then polls `complete_read()` until every
request has been pulled (done_sending) or the timeout fires.

The engine emits per-request `RAIDEN_TIMING event=producer_pipeline ...` lines
to stderr (d2h_issue_ms is the bottleneck under study).
"""

import argparse
import sys
import time

from api.jax.transfer_engine import TransferEngine
import kv_spec


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--control-port", type=int, default=9200,
                   help="Fixed control port so the consumer can connect.")
    p.add_argument("--num-blocks", type=int, default=64,
                   help="Blocks transferred per request (64 => 2 GiB).")
    p.add_argument("--max-blocks", type=int, default=72,
                   help="Engine per-request staging budget (>= num-blocks).")
    p.add_argument("--kv-cache-num-blocks", type=int, default=1024,
                   help="Cache dim0: the full block pool (real run ~15022). "
                        "Each request uses a distinct window of it.")
    p.add_argument("--num-requests", type=int, default=1)
    p.add_argument("--mapping", type=str, default="identity",
                   choices=["identity", "reorder", "sparse"],
                   help="Block mapping (must match the consumer).")
    p.add_argument("--num-slots", type=int, default=0,
                   help="Host staging slots; 0 => max(num_requests, 8).")
    p.add_argument("--num-devices", type=int, default=8)
    p.add_argument("--req-id-prefix", type=str, default="two_node_req_")
    p.add_argument("--base-uuid", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--timeout-s", type=float, default=600.0)
    p.add_argument("--skip-buffer-lock", action="store_true", default=True)
    args = p.parse_args()

    num_slots = args.num_slots or max(args.num_requests, 8)

    if args.num_blocks > args.max_blocks:
        raise ValueError(
            f"num_blocks ({args.num_blocks}) must be <= max_blocks "
            f"({args.max_blocks}, the engine staging budget)")

    import jax
    log(f"PRODUCER devices={len(jax.devices('tpu'))} "
        f"num_layers={kv_spec.NUM_LAYERS} num_blocks={args.num_blocks} "
        f"kv_cache_num_blocks={args.kv_cache_num_blocks} "
        f"bytes={kv_spec.bytes_for(args.num_blocks)} "
        f"dtype={kv_spec.DTYPE.__name__}")

    # Distinct, non-overlapping block window per request (see request_block_map).
    maps = [
        kv_spec.request_block_map(args.mapping, args.num_blocks,
                                  args.kv_cache_num_blocks, r)
        for r in range(args.num_requests)
    ]
    all_registered = sorted({b for (reg, _, _) in maps for b in reg})

    sharding = kv_spec.build_sharding(args.num_devices)
    t0 = time.perf_counter()
    src_caches = kv_spec.make_src_caches(
        args.seed, args.kv_cache_num_blocks, all_registered, sharding)
    log(f"PRODUCER built src caches in {time.perf_counter() - t0:.2f}s "
        f"(per-layer shape={src_caches[0].shape}, "
        f"filled_blocks={len(all_registered)})")

    engine = TransferEngine(
        kv_caches=src_caches,
        local_control_port=args.control_port,
        max_blocks=args.max_blocks,
        num_slots=num_slots,
        unsafe_skip_buffer_lock=args.skip_buffer_lock,
    )
    control_port = getattr(engine, "local_control_port", args.control_port)
    data_port = getattr(engine, "local_data_port", None)

    # Each request registers its own distinct window of the pool.
    req_ids = [f"{args.req_id_prefix}{i}" for i in range(args.num_requests)]
    for i, req_id in enumerate(req_ids):
        registered_i = maps[i][0]
        engine.notify_for_read(req_id, args.base_uuid + i, registered_i)
    sample_reg = maps[0][0]
    log(f"PRODUCER mapping={args.mapping} per_request_registered={len(sample_reg)} "
        f"req0_blocks=[{sample_reg[0]}..{sample_reg[-1]}] "
        f"total_distinct={len(all_registered)}")

    # The orchestrator greps for this exact token.
    log(f"PRODUCER_READY control_port={control_port} data_port={data_port} "
        f"num_slots={num_slots} requests={args.num_requests}")

    pending = set(req_ids)
    deadline = time.perf_counter() + args.timeout_s
    while pending and time.perf_counter() < deadline:
        done_sending, _, _ = engine.complete_read()
        if done_sending:
            for r in done_sending:
                pending.discard(r)
            log(f"PRODUCER done_sending={sorted(done_sending)} "
                f"remaining={len(pending)}")
        time.sleep(0.05)

    if pending:
        log(f"PRODUCER_TIMEOUT still_pending={sorted(pending)}")
        return 1
    log("PRODUCER_DONE all requests pulled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
