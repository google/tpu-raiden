# Two-node Raiden `TransferEngine` benchmark

A **true two-node** (separate hosts, separate processes) exercise of the JAX
`TransferEngine` pull path — one producer (prefill host) and one consumer
(decode host) — using the **real Qwen3-32B TP=8 fp8 KV-cache spec** from a
production disaggregated run. Adapted from the single-process
`api/jax/transfer_engine_test.py`.

It isolates the KV transfer (D2H → cross-host H2H → H2D) from vLLM so we can
measure and optimize TTFT (see `current_work/TTFT_OPTIMIZATION_PLAN.md`) without
a full model serve.

## Real spec (from `prefill.log` `Init kv-cache`, see `kv_spec.py`)

| field | value |
|---|---|
| layers | 64 arrays |
| per-layer shape | `(num_blocks, 128, 8, 4, 128)` |
| dtype | `float8_e4m3fn` |
| sharding | `P('data', None, 'model')` on mesh `(data=1, model=8)` |
| block_size | 128 |
| max_blocks / req (staging budget) | 72 |
| kv cache dim0 (block pool) | large — real run ~15022; default **1024** here |
| transfer size | 64 blocks → **2 GiB** (= production `h2h_bytes`) |

Like the real run, the cache `dim0` is a large pool and each request transfers a
small (64-block) slice at **large, distinct** block ids — every request takes a
non-overlapping window of the pool, so no two requests register/read/write the
same blocks (`KV_CACHE_NUM_BLOCKS`, default 1024; auto-grown for multi-request
runs so the windows stay distinct). `max_blocks` (72) is just the per-request
staging budget, not the pool size.

## Files

- `kv_spec.py` — the real spec + deterministic cache builders / verification.
- `producer.py` — builds random src caches, `notify_for_read`, waits for pulls.
- `consumer.py` — builds zero dst caches, `start_read` × N, measures BW, verifies.
- `run_two_node_test.sh` — orchestrator (launch producer local + consumer over ssh).

## Run

```bash
cd /mnt/disks/jcgu/code/ullm/raiden/tpu-raiden/two_node_test

# Single 2 GiB pull, byte-verified (sampled):
bash run_two_node_test.sh

# Exhaustive byte-verification of every layer/block:
VERIFY_FULL=1 bash run_two_node_test.sh

# Correctness of the engine's gather/scatter (non-identity mappings):
MAPPING=reorder VERIFY_FULL=1 bash run_two_node_test.sh   # full reversed permutation
MAPPING=sparse  VERIFY_FULL=1 bash run_two_node_test.sh   # non-contiguous + reorder + zero-check

# Reproduce the D2H contention curve that drives TTFT:
NUM_REQUESTS=8  NUM_SLOTS=8  bash run_two_node_test.sh   # ~2.7s regime
NUM_REQUESTS=30 NUM_SLOTS=30 bash run_two_node_test.sh   # ~9.9s regime
```

### Verification

The consumer regenerates the source bytes from `--seed` (identical to the
producer, never received) and byte-compares against what landed in its
destination caches. Per layer it checks two invariants:

- **data:** `dst[local[i]] == src[requested_remote[i]]` (the mapping landed),
- **zero:** every local block *not* written stays all-zero (no over-write).

`MAPPING` selects the block map (`identity` = the real contiguous path;
`reorder` = full reversed permutation; `sparse` = non-contiguous even remote
ids, reversed, into compact local — exercises gather + scatter + the zero
invariant). `VERIFY_FULL=1` checks all 64 layers and every block; the default
samples first/mid/last to stay fast. Verify runs only for `NUM_REQUESTS=1`.

Per-stage numbers come from the engine's `RAIDEN_TIMING event=producer_pipeline`
(`d2h_issue_ms`) and `event=consumer_pipeline` (`control_setup_ms`, `h2h_ms`,
`h2d_*`) lines, surfaced at the end of the run plus the harness aggregate GB/s.

## Notes / gotchas

- Each host runs an independent single-host JAX program on its 8 local TPUs;
  the two processes talk **only** over the Raiden engine's TCP sockets (no
  multi-host JAX). Both hosts must have the 8 TPUs free (no vLLM running).
- The consumer host needs the tpu-raiden engine `.so` + `api/`/`frameworks/`
  already present (it does on aman); the orchestrator only syncs the test files.
- `num_slots` bounds real concurrency. Set `NUM_SLOTS == NUM_REQUESTS` to
  reproduce N-way contention; setting `NUM_REQUESTS > NUM_SLOTS` will trip the
  `host slot pool exhausted` throw (the known regression) — surfaced as
  `CONSUMER FAILED`, not a crash.
- Verification is byte-exact on sampled (layer, block) slices and only runs for
  `NUM_REQUESTS=1` (concurrent pulls reuse the same dst blocks).
