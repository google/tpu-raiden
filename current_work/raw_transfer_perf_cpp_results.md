# Raiden Raw-Transfer C++ Perf Test Results

Benchmark: `core/raw_transfer_perf_test.cc` (`//core:raw_transfer_perf_test`)

| Field | Value |
|---|---|
| Device | TPU7x |
| PJRT | C-API dynamic load (`TPU_LIBRARY_PATH` → libtpu.so); PJRT plugin API 0.103 / framework 0.110 |
| Staging | DMA-mapped host buffers (`HostMemoryAllocator::AllocateDmaMapped`, `client->DmaMap`) |
| Iterations | 50 timed runs/case (+3 warmup), median reported |
| Perf gate | 24 GB/s (D2H and H2D) |
| Transfer pattern | Odd blocks only (8 of 16 blocks) → 50% of data, fragmented copy |
| KV shape | 64 layers, blocks=16, block_size∈{128,256,512}, heads=8, kv=2, head_dim=128 |
| Result | **13 passed, 1 failed**, 73.7 s total |
| Raw log | [`raw_transfer_perf_cpp_run.log`](./raw_transfer_perf_cpp_run.log) |

Two scenarios:
- **Scenario A — Fragmented Batch:** 64 independent per-layer device buffers; D2H/H2D issued as 64 separate transfers, each copying its odd blocks.
- **Scenario B — Baked-in Tensor:** a single device buffer with the layer dimension baked in; one transfer covering all odd blocks.

## Scenario A — Fragmented Batch

| dtype | block_size | xfer size | D2H median (GB/s) | H2D median (GB/s) | gate |
|---|---|---|---|---|---|
| BF16 | 128 | 256 MB | 43.32 | 43.21 | ✅ |
| F32 | 128 | 512 MB | 51.80 | 52.03 | ✅ |
| F8E4M3FN | 128 | 128 MB | 22.52 | 21.71 | ❌ |
| BF16 | 256 | 512 MB | 51.83 | 52.02 | ✅ |
| BF16 | 512 | 1024 MB | 52.25 | 52.47 | ✅ |
| F8E4M3FN | 256 | 256 MB | 43.21 | 43.17 | ✅ |
| F8E4M3FN | 512 | 512 MB | 51.82 | 52.03 | ✅ |

## Scenario B — Baked-in Tensor

| dtype | block_size | xfer size | D2H median (GB/s) | H2D median (GB/s) | gate |
|---|---|---|---|---|---|
| BF16 | 128 | 256 MB | 51.77 | 51.97 | ✅ |
| F32 | 128 | 512 MB | 52.25 | 52.44 | ✅ |
| F8E4M3FN | 128 | 128 MB | 50.79 | 51.12 | ✅ |
| BF16 | 256 | 512 MB | 52.21 | 52.44 | ✅ |
| BF16 | 512 | 1024 MB | 52.49 | 52.62 | ✅ |
| F8E4M3FN | 256 | 256 MB | 51.78 | 51.98 | ✅ |
| F8E4M3FN | 512 | 512 MB | 52.25 | 52.44 | ✅ |

## num_layers = 1024 sweep (block_size = 128, all dtypes)

`kNumLayers` is configurable (default 64). These cases set it to 1024 to stress
larger working sets. Raw log: [`raw_transfer_perf_cpp_l1024.log`](./raw_transfer_perf_cpp_l1024.log).

### Scenario A — Fragmented Batch (1024 separate layer buffers)

| dtype | total size | xfer size | D2H median (GB/s) | H2D median (GB/s) | gate |
|---|---|---|---|---|---|
| BF16 | 8192 MB | 4096 MB | 39.58 | 40.41 | ✅ |
| F32 | 16384 MB | 8192 MB | 52.64 | 52.85 | ✅ |
| F8E4M3FN | 4096 MB | 2048 MB | 22.79 | 22.78 | ❌ |

### Scenario B — Baked-in Tensor (single buffer, layer dim = 1024)

| dtype | total size | xfer size | D2H median (GB/s) | H2D median (GB/s) | gate |
|---|---|---|---|---|---|
| BF16 | 8192 MB | 4096 MB | 52.57 | 52.79 | ✅ |
| F32 | 16384 MB | 8192 MB | 52.62 | 52.84 | ✅ |
| F8E4M3FN | 4096 MB | 2048 MB | 52.49 | 52.77 | ✅ |

Notes:
- F32/L1024 allocates a **16 GB device buffer** (Scenario B is a single 16 GB
  buffer); fits comfortably on TPU7x HBM. These are the largest/slowest cases
  (F32 Scenario B ≈ 157 s for 50 iters).
- fp8/bs128 reproduces at scale (~22.8 GB/s, vs ~22.5 at 64 layers) — bandwidth
  is bound by per-fragment size, not total size. It is the only L1024 failure
  (below the 24 GB/s gate).
- BF16 fragmented dips 43 → 40 GB/s going 64 → 1024 layers (8× working set /
  8192 block-copies per transfer); Scenario B stays at the ~52 GB/s plateau.

## Observations

- **Bandwidth tracks per-fragment transfer size, not dtype.** In Scenario A the fp8 sweep scales 22.5 → 43.2 → 51.8 GB/s for block_size 128 → 256 → 512, and matches the BF16 number at the *same transfer size*: fp8/bs256 (256 MB → 43.2) ≈ bf16/bs128 (256 MB → 43.3); fp8/bs512 (512 MB → 51.8) ≈ bf16/bs256 (512 MB → 51.8). Once each contiguous odd-block run is large enough, per-transfer dispatch overhead is amortized and all dtypes converge to the ~52 GB/s plateau.
- **Scenario B is at the plateau everywhere.** A single baked-in buffer has no per-layer fragmentation, so even fp8/bs128 (128 MB) reaches ~51 GB/s.
- **The only failing case** is `ScenarioA_FragmentedBatch_F8E4M3FN` (bs=128): 22.5 / 21.7 GB/s, below the 24 GB/s gate (asserted at `raw_transfer_perf_test.cc:271` and `:274`). This is genuine small-fragment overhead (fp8 = 1 byte → only 4 MB/layer, 128 MB total), not a code defect. The gate was intentionally left at 24 GB/s so this configuration stays flagged as below-target.

## How to reproduce

```bash
cd tpu-raiden
# build (see build_raw_transfer.sh for full flag set)
bazel build -c opt //core:raw_transfer_perf_test
# run (TPU must be free)
TPU_LIBRARY_PATH=/mnt/disks/jcgu/anaconda3/envs/raiden1/lib/python3.12/site-packages/libtpu/libtpu.so \
  ./bazel-bin/core/raw_transfer_perf_test
```
