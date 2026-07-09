# H2H Single-Host Test

A presubmit gate that verifies the **correctness** of the host-to-host (H2H)
KV-cache push path on a single machine.

## Purpose

On a single host the H2H transfer runs over **loopback**, so its throughput
reflects the kernel/memcpy path, not a real NIC — it is not a product bandwidth
number. What *is* meaningful on a single host is **correctness**: whether the
bytes that arrive are exactly the bytes that were sent.

> **This gate is byte-integrity only.** Throughput is printed for observability
> but never decides pass/fail. Perf gating belongs on real hardware, not here.

## Setup

A thin Python driver (`h2h_cpp_gate.py`) spawns the C++ runner
(`//examples/microbenchmarks:h2h_benchmark_runner`) as two processes on the same
machine and reads its stdout.

| Aspect | Setting |
|---|---|
| Interface | loopback (`--data_interface=lo`, `127.0.0.1`), single IP |
| Placement | sender on NUMA node 0, receiver on NUMA node 1 (forces the copy across the socket interconnect) |
| Transport op | `op=6` (explicit destination blocks — the working H2H push path) |
| Fixed shape | 32 layers × 1 shard (runner constants) |

**Correctness check.** The sender fills each byte with a position-dependent
pattern `(layer + block + byte_index + interface) & 0xFF`, so every byte is
distinct; the receiver then byte-compares the whole buffer against the same
formula. Any misplacement, swapped block, wrong offset, partial write, or
dropped/zeroed data fails the check — the data needs no variation, only the
transfer *structure* does.

## Config choices

Configs cover **distinct code paths / bug classes**, not bandwidth saturation.
Each `(block_size, num_blocks, parallelism)` reaches something the others cannot.

| block_size | num_blocks | parallelism | Bug class it catches |
|---|---|---|---|
| 1 MiB | 64 | **1** | baseline block mapping / offset / copy correctness |
| 1 MiB | 64 | **8** | **races, write interleaving, ordering, stream partitioning** — impossible at P1 |
| 1 MiB − 3 B | 64 | **4** | **alignment / boundary / partial-write** off-by-one that round sizes hide |

- **parallelism (P1 → P8) is the primary axis.** The push splits blocks across
  `N = parallelism` streams; concurrency bugs only appear with `P > 1`. P1 proves
  the serial path, P8 applies concurrency stress (and matches the 8-way shape).
- **num_blocks = 64** exercises the block-id → offset mapping (`block_id *
  block_size`, explicit `dst_block_ids`) with many distinct offsets.
- **block_size = 1 MiB** crosses page boundaries yet keeps memory bounded at
  `32 × 64 × 1 MiB = 2 GiB`/process (≈ 4 GiB total), avoiding the OOM that
  16/128 MiB configs caused.
- **non-round size (1 MiB − 3 B)** probes tail/partial handling that
  power-of-two sizes mask. Verified safe: the transport uses `readv`/`writev` of
  arbitrary byte counts and `posix_memalign(64, size)`, which aligns the pointer
  but does not constrain the size.

Deliberately excluded: large blocks / size sweeps (same path, and OOM), dtype
(H2H moves raw bytes), and `op=1` dynamic allocation (unused by the runner).

Each config runs only a few iterations (`--iters`): correctness just needs to
exercise the path, not the many samples a perf benchmark collects.

## Verdict

A config passes **iff** its receiver reports `Data integrity verification
PASSED`; any failure fails the gate (non-zero exit). No baseline file is
needed — the check is self-contained, so the gate is stateless.

## Running it

```bash
bazel run -c opt --config=oss --config=ci \
  //tpu_raiden/benchmarks:h2h_cpp_gate
```

In CI it runs as the `h2h_cpp_gate` workload in `benchmark_registry.pbtxt` via
the `run_benchmarks` workflow.

## Scope

This validates the **transport software path** (serialization, block mapping,
stream fan-out, concurrent writes), which is shared with the real NIC path. It
does **not** cover NIC-specific behavior (RDMA, multi-NIC scaling, offload) or
real network bandwidth — those need multi-host runs on real hardware.
