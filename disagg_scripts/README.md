# Two-node `DisaggKVCacheManager` test harness

Drives the `raiden_oss_0528` `DisaggKVCacheManager` across **two physical TPU
hosts** for real disaggregation testing — prefill on the LOCAL host, decode on
REMOTE_HOST (`10.128.1.8` by default). This is the manager/transport engine
itself, **not** the vLLM serving stack in [`../../scripts/`](../../scripts) (that
one runs `vllm serve` + a toy proxy router; this one calls the manager directly).

The upstream E2E test (`raiden_oss_0528/api/jax/disagg_kv_cache_manager_test.py`)
runs both managers *in one process* on `127.0.0.1`. Here they are separate
processes on separate hosts, so they rendezvous over the ZMQ discovery proxy
(`kv_cache/disagg_proxy.py`).

## Files
| File | Role |
|---|---|
| `disagg_env.sh` | Shared config (hosts, paths, conda env, transfer plan). Override via env. |
| `sync_oss_to_remote.sh` | scp + md5-verify the `.so`, python wrappers and this harness to REMOTE. |
| `disagg_node_runner.py` | Runs ONE role (`--role prefill\|decode`). **Extend this for your tests.** |
| `run_two_node_disagg_test.sh` | Orchestrates clean → sync → proxy → decode(remote)+prefill(local) → PASS/FAIL. |
| `kill_disagg_nodes.sh` | Kill runner/proxy on both hosts (does not touch vLLM). |

## Prerequisites
1. Same OSS tree path on both hosts (`/mnt/disks/jcgu/code/ullm/raiden/raiden_oss_0528`).
2. Build the extension locally first: `cd raiden_oss_0528 && ./build_raw_transfer.sh`
   (produces `api/jax/_kv_cache_manager.so`), then it gets synced to REMOTE.
3. Conda env `raiden1` on both hosts; passwordless ssh to REMOTE_HOST.

## Run
```bash
cd raiden_oss_0528/disagg_scripts
./run_two_node_disagg_test.sh                 # default: tpu, int32, 2 layers, 1 request
NUM_REQUESTS=4 ./run_two_node_disagg_test.sh  # 4 concurrent requests
NUM_REQUESTS=4 WORKER_PARALLELISM=2 ./run_two_node_disagg_test.sh      # 2 concurrent H2H workers
TRANSPORT_PARALLELISM=2 ./run_two_node_disagg_test.sh                  # split each transfer over 2 TCP streams
MODE=pull ./run_two_node_disagg_test.sh                               # decode pulls from prefill (default: push)
DTYPE=bf16 N_LAYERS=4 ./run_two_node_disagg_test.sh

./run_all_two_node_tests.sh   # the full push+pull matrix with a PASS/FAIL summary
```

`MODE` selects the transfer direction: `push` (default — prefill pushes to
decode) or `pull` (decode pulls from prefill). See §4 of
`../disagg_kv_cache_manager.md` for the per-step information exchange of each.

`NUM_REQUESTS=N` fires N concurrent requests; request `i` is shifted by
`i*sum(SIZES)` on both src and dst so the requests touch **disjoint, independently
verified** regions, and the major dim auto-grows to fit them. The decode side
queues all N receives, waits for every callback, then verifies all chunks.

There are **two independent H2H parallelism knobs** (see
`api/jax/disagg_kv_cache_manager.py`): `TRANSPORT_PARALLELISM` (parallel TCP
streams per single Push/Pull — the BlockTransport parallelism) and
`WORKER_PARALLELISM` (concurrent H2H worker threads / transfers in flight).
Both `>1` are supported (verified with up to 8 requests / parallelism=4
across two hosts). Earlier a concurrency bug corrupted data when the receiver
allocated blocks non-sequentially; it was fixed (2026-05-29) in
`disagg_kv_cache_manager_base.cc` (notify the peer with the receiver's allocated
block ids, not local staging ids) plus a `block_manager_mutex_` guard in
`kv_cache_manager_base.h`. The regression test is upstream
`disagg_kv_cache_manager_test.py::test_e2e_disagg_push_multi_request_concurrent`.

> **TPU only.** `DEVICE=cpu` exists but currently **segfaults inside the
> manager's PJRT D2H path** — this is a pre-existing limitation (the upstream
> `disagg_kv_cache_manager_test.py --device_type=cpu` core-dumps the same way),
> not a harness bug. Each engine needs exclusive ownership of its host's 8 TPU
> chips, which is why two separate hosts are required (two TPU processes can't
> share one host).

## How rendezvous works (ephemeral ports)
`zmq_control_port()` / `local_port()` are ephemeral, so we can't hardcode them:
1. Decode queues its `DECODE_H2D` receive, **then** `REGISTER`s `{ip,zmq,trans}`
   with the proxy — registration therefore means "ready to receive".
2. Prefill `RESOLVE`s `decode` (blocks until step 1), `register_peer`s it, then
   submits `PREFILL_D2H`. No sleep-based race.
3. Data flows P-device →(D2H)→ P-host →(H2H TCP push)→ D-host →(H2D)→ D-device;
   prefill notifies decode over ZMQ `NOTIFY_COMPLETE` to trigger the H2D.

## Writing your own tests
Edit `disagg_node_runner.py`:
- `build_reference_layer(shape, dtype, layer_idx)` — the deterministic payload
  both hosts compute independently (default: `arange + layer_idx`, int32).
- `verify_decode(...)` — assertions on what landed on the decode device.
- transfer plan via `--src-offsets/--dst-offsets/--sizes` (major-dim units).

The orchestration (manager lifecycle, proxy rendezvous, peer registration,
callbacks, exit codes) is reused, so a new scenario is usually just new
payload + verification. For multi-request / concurrency, mirror the patterns in
`disagg_kv_cache_manager_test.py::test_e2e_disagg_push_concurrent`.
