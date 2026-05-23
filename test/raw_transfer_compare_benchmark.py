#!/usr/bin/env python3
# Copyright 2026 Google LLC
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

"""Compare Raiden/JAX/TorchTPU raw H2D and D2H transfer throughput.

The driver launches one subprocess per benchmark point so the JAX and TorchTPU
PJRT runtimes do not share a process. For TorchTPU 8-DMA runs it uses torchrun
with one worker per local TPU chip; each worker issues one DMA.
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any


DEFAULT_SIZES_MIB = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
DEFAULT_CONCURRENCIES = [1, 8]
REQUESTED_BACKENDS = [
    "raiden_torch",
    "raiden_torch_prepared",
    "jax_raiden",
    "torch_tpu_batch",
]
TORCH_BACKENDS = {"raiden_torch", "torch_tpu_batch"}
PREPARED_TORCH_BACKENDS = {"raiden_torch_prepared"}
RESULT_PREFIX = "RESULT_JSON "
GIB = 1024.0**3


def _repo_paths() -> tuple[Path, Path, Path]:
  raiden_root = Path(__file__).resolve().parents[1]
  workspace_root = raiden_root.parent
  bazel_bin = raiden_root / "bazel-bin"
  return workspace_root, raiden_root, bazel_bin


def _child_env() -> dict[str, str]:
  _, raiden_root, bazel_bin = _repo_paths()
  env = os.environ.copy()
  paths = [str(bazel_bin), str(raiden_root)]
  if env.get("PYTHONPATH"):
    paths.append(env["PYTHONPATH"])
  env["PYTHONPATH"] = os.pathsep.join(paths)
  env["PYTHONUNBUFFERED"] = "1"
  return env


def _add_torch_singlehost_env(env: dict[str, str]) -> dict[str, str]:
  if env.get("TORCH_TPU_TOPOLOGY") and env.get(
      "TORCH_TPU_SLICEBUILDER_ADDRESSES"
  ):
    return env

  proc = subprocess.run(
      [
          sys.executable,
          "-m",
          "torch_tpu._internal.distributed.launchers.singlehost_wrapper",
      ],
      cwd=str(_repo_paths()[0]),
      env=env,
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      timeout=120,
      check=True,
  )
  for line in proc.stdout.splitlines():
    if not line.strip() or "=" not in line:
      continue
    name, value = line.split("=", 1)
    name = name.strip()
    value = shlex.split(value.strip())[0] if value.strip() else ""
    env[name] = value
  return env


def _parse_csv_ints(value: str) -> list[int]:
  return [int(part.strip()) for part in value.split(",") if part.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, sort_keys=True)
    f.write("\n")
    f.flush()
    os.fsync(f.fileno())


def _rank_result_path(sync_dir: Path, rank: int) -> Path:
  return sync_dir / f"rank_{rank}.json"


def _file_barrier(sync_dir: Path, name: str, rank: int, world_size: int) -> None:
  if world_size <= 1:
    return
  sync_dir.mkdir(parents=True, exist_ok=True)
  token = sync_dir / f"{name}.rank_{rank}"
  token.write_text("ready\n", encoding="utf-8")
  deadline = time.monotonic() + 600
  pattern = str(sync_dir / f"{name}.rank_*")
  while len(glob.glob(pattern)) < world_size:
    if time.monotonic() > deadline:
      raise TimeoutError(
          f"Timed out in barrier {name!r}: "
          f"{len(glob.glob(pattern))}/{world_size} workers arrived"
      )
    time.sleep(0.01)


def _wait_future(future: Any) -> None:
  if future is None:
    return
  if hasattr(future, "Await"):
    future.Await()
    return
  if hasattr(future, "wait"):
    future.wait()
    return
  raise TypeError(f"Unsupported future object: {type(future)!r}")


def _wait_jax_result(result: Any) -> None:
  if hasattr(result, "Await") or hasattr(result, "wait"):
    _wait_future(result)
    return
  import jax

  jax.block_until_ready(result)


def _torch_synchronize(torch_mod: Any) -> None:
  tpu = getattr(torch_mod, "tpu", None)
  if tpu is not None and hasattr(tpu, "synchronize"):
    tpu.synchronize()


def _torch_backend_ops(backend: str) -> tuple[Any, Any]:
  if backend == "raiden_torch":
    from raiden_lib.raw_transfer.torch import torch_raw_transfer

    def d2h(src: Any, dst: Any) -> Any:
      return torch_raw_transfer.transfer_d2h_async(src, dst)

    def h2d(src: Any, dst: Any) -> Any:
      return torch_raw_transfer.transfer_h2d_async(src, dst)

    return d2h, h2d

  if backend == "torch_tpu_batch":
    from torch_tpu._internal import batch_transfer

    def d2h(src: Any, dst: Any) -> Any:
      return batch_transfer.batch_transfer_d2h([src], [dst])

    def h2d(src: Any, dst: Any) -> Any:
      return batch_transfer.batch_transfer_h2d([src], [dst])

    return d2h, h2d

  raise ValueError(f"Unknown Torch backend {backend!r}")


def _run_torch_worker(args: argparse.Namespace) -> None:
  import torch
  import torch_tpu  # noqa: F401  # Registers the torch PrivateUse1 backend.

  rank = int(os.environ.get("RANK", "0"))
  local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
  world_size = int(os.environ.get("WORLD_SIZE", "1"))
  sync_dir = Path(args.sync_dir)

  try:
    size_bytes = args.size_mib * 1024 * 1024
    if size_bytes % 4:
      raise ValueError(f"size_bytes must be float32-aligned, got {size_bytes}")

    numel = size_bytes // 4
    device = torch.device("tpu")
    tpu_buf = torch.empty(numel, dtype=torch.float32, device=device)
    if args.backend in PREPARED_TORCH_BACKENDS:
      from raiden_lib.raw_transfer.torch import torch_raw_transfer

      host_stage = torch_raw_transfer.RawHostBuffer(
          size_bytes + args.host_padding_bytes
      )
      prepared = torch_raw_transfer.PreparedTorchRawTransfer(
          tpu_buf, host_stage
      )

      def d2h(src: Any, dst: Any) -> Any:
        del src, dst
        return prepared.d2h_async()

      def h2d(src: Any, dst: Any) -> Any:
        del src, dst
        return prepared.h2d_async()

    else:
      d2h, h2d = _torch_backend_ops(args.backend)
      host_stage = torch.empty(
          size_bytes + args.host_padding_bytes,
          dtype=torch.uint8,
          pin_memory=not args.no_pin_memory,
      )
    _torch_synchronize(torch)
    _file_barrier(sync_dir, "allocated", rank, world_size)

    for warmup in range(args.warmups):
      _file_barrier(sync_dir, f"warmup_d2h_{warmup}", rank, world_size)
      _wait_future(d2h(tpu_buf, host_stage))
      _file_barrier(sync_dir, f"warmup_h2d_{warmup}", rank, world_size)
      _wait_future(h2d(host_stage, tpu_buf))
    _torch_synchronize(torch)
    _file_barrier(sync_dir, "warmups_done", rank, world_size)

    d2h_total_s = []
    h2d_total_s = []
    for i in range(args.iterations):
      _file_barrier(sync_dir, f"iter_{i}_d2h_start", rank, world_size)
      t0 = time.perf_counter()
      future = d2h(tpu_buf, host_stage)
      _wait_future(future)
      d2h_total_s.append(time.perf_counter() - t0)

      _file_barrier(sync_dir, f"iter_{i}_h2d_start", rank, world_size)
      t0 = time.perf_counter()
      future = h2d(host_stage, tpu_buf)
      _wait_future(future)
      h2d_total_s.append(time.perf_counter() - t0)

    payload = {
        "status": "ok",
        "backend": args.backend,
        "rank": rank,
        "local_rank": local_rank,
        "current_device": (
            torch.tpu.current_device()
            if hasattr(torch, "tpu") and hasattr(torch.tpu, "current_device")
            else None
        ),
        "world_size": world_size,
        "size_mib": args.size_mib,
        "size_bytes": size_bytes,
        "iterations": args.iterations,
        "warmups": args.warmups,
        "d2h_total_s": d2h_total_s,
        "h2d_total_s": h2d_total_s,
    }
    _write_json(_rank_result_path(sync_dir, rank), payload)
    print(RESULT_PREFIX + json.dumps(payload, sort_keys=True), flush=True)
    _file_barrier(sync_dir, "results_written", rank, world_size)
    if args.fast_exit:
      os._exit(0)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    payload = {
        "status": "error",
        "backend": args.backend,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "size_mib": args.size_mib,
        "error": repr(exc),
        "traceback": traceback.format_exc(),
    }
    _write_json(_rank_result_path(sync_dir, rank), payload)
    print(RESULT_PREFIX + json.dumps(payload, sort_keys=True), flush=True)
    raise


def _jax_shardings(concurrency: int) -> tuple[Any, Any, tuple[int, int]]:
  import jax
  import numpy as np

  devices = jax.devices("tpu")
  if len(devices) < concurrency:
    raise RuntimeError(
        f"Need {concurrency} TPU devices for JAX, found {len(devices)}"
    )
  mesh = jax.sharding.Mesh(np.array(devices[:concurrency]), ("x",))
  spec = jax.sharding.PartitionSpec("x", None)
  tpu_sharding = jax.sharding.NamedSharding(mesh, spec)
  host_sharding = jax.sharding.NamedSharding(
      mesh, spec, memory_kind="pinned_host"
  )
  return tpu_sharding, host_sharding, (concurrency,)


def _make_jax_array(shape: tuple[int, int], sharding: Any) -> Any:
  import jax
  import jax.numpy as jnp

  allocate = jax.jit(
      lambda: jnp.zeros(shape, dtype=jnp.float32), out_shardings=sharding
  )
  arr = allocate()
  jax.block_until_ready(arr)
  return arr


def _run_jax_worker(args: argparse.Namespace) -> None:
  sync_dir = Path(args.sync_dir)
  rank = 0
  try:
    import jax

    if args.backend == "jax_raiden":
      try:
        raw_transfer = importlib.import_module(args.jax_raw_module)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        payload = {
            "status": "skipped",
            "backend": args.backend,
            "rank": rank,
            "world_size": args.concurrency,
            "size_mib": args.size_mib,
            "reason": (
                f"Could not import JAX Raiden raw module "
                f"{args.jax_raw_module!r}: {exc!r}"
            ),
        }
        _write_json(_rank_result_path(sync_dir, rank), payload)
        print(RESULT_PREFIX + json.dumps(payload, sort_keys=True), flush=True)
        return

      def d2h(src: Any, dst: Any) -> Any:
        return raw_transfer.transfer_d2h_batch_async([src], [dst])

      def h2d(src: Any, dst: Any) -> Any:
        return raw_transfer.transfer_h2d_batch_async([src], [dst])

    elif args.backend == "jax_device_put":

      def d2h(src: Any, dst: Any) -> Any:
        del dst
        return jax.device_put(src, host_sharding)

      def h2d(src: Any, dst: Any) -> Any:
        del dst
        return jax.device_put(src, tpu_sharding)

    else:
      raise ValueError(f"Unknown JAX backend {args.backend!r}")

    size_bytes = args.size_mib * 1024 * 1024
    if size_bytes % 4:
      raise ValueError(f"size_bytes must be float32-aligned, got {size_bytes}")
    shape = (args.concurrency, size_bytes // 4)
    tpu_sharding, host_sharding, _ = _jax_shardings(args.concurrency)
    tpu_src = _make_jax_array(shape, tpu_sharding)
    host_stage = _make_jax_array(shape, host_sharding)
    tpu_dst = _make_jax_array(shape, tpu_sharding)

    for _ in range(args.warmups):
      future = d2h(tpu_src, host_stage)
      _wait_jax_result(future)
      future = h2d(host_stage, tpu_dst)
      _wait_jax_result(future)

    d2h_total_s = []
    h2d_total_s = []
    for _ in range(args.iterations):
      t0 = time.perf_counter()
      future = d2h(tpu_src, host_stage)
      _wait_jax_result(future)
      d2h_total_s.append(time.perf_counter() - t0)

      t0 = time.perf_counter()
      future = h2d(host_stage, tpu_dst)
      _wait_jax_result(future)
      h2d_total_s.append(time.perf_counter() - t0)

    payload = {
        "status": "ok",
        "backend": args.backend,
        "rank": rank,
        "world_size": args.concurrency,
        "size_mib": args.size_mib,
        "size_bytes": size_bytes,
        "iterations": args.iterations,
        "warmups": args.warmups,
        "d2h_total_s": d2h_total_s,
        "h2d_total_s": h2d_total_s,
    }
    _write_json(_rank_result_path(sync_dir, rank), payload)
    print(RESULT_PREFIX + json.dumps(payload, sort_keys=True), flush=True)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    payload = {
        "status": "error",
        "backend": args.backend,
        "rank": rank,
        "world_size": args.concurrency,
        "size_mib": args.size_mib,
        "error": repr(exc),
        "traceback": traceback.format_exc(),
    }
    _write_json(_rank_result_path(sync_dir, rank), payload)
    print(RESULT_PREFIX + json.dumps(payload, sort_keys=True), flush=True)
    raise


def _median(values: list[float]) -> float:
  return float(statistics.median(values))


def _mean(values: list[float]) -> float:
  return float(statistics.mean(values))


def _percentile(values: list[float], pct: float) -> float:
  if not values:
    return math.nan
  ordered = sorted(values)
  if len(ordered) == 1:
    return float(ordered[0])
  pos = (len(ordered) - 1) * pct
  lo = math.floor(pos)
  hi = math.ceil(pos)
  if lo == hi:
    return float(ordered[lo])
  frac = pos - lo
  return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _aggregate_success(
    payloads: list[dict[str, Any]], child_returncode: int
) -> list[dict[str, Any]]:
  first = payloads[0]
  world_size = int(first["world_size"])
  size_bytes = int(first["size_bytes"])
  total_gib = (size_bytes * world_size) / GIB
  rows = []
  for direction in ("d2h", "h2d"):
    total_by_iter = list(zip(*[p[f"{direction}_total_s"] for p in payloads]))
    totals = [max(iter_times) for iter_times in total_by_iter]
    gib_s = [total_gib / seconds for seconds in totals]
    rows.append({
        "status": "ok",
        "backend": first["backend"],
        "size_mib": first["size_mib"],
        "concurrency": world_size,
        "direction": direction,
        "iterations": first["iterations"],
        "warmups": first["warmups"],
        "median_total_s": _median(totals),
        "mean_total_s": _mean(totals),
        "min_total_s": min(totals),
        "p90_total_s": _percentile(totals, 0.90),
        "median_gib_s": _median(gib_s),
        "mean_gib_s": _mean(gib_s),
        "max_gib_s": max(gib_s),
        "min_gib_s": min(gib_s),
        "child_returncode": child_returncode,
        "notes": "",
    })
  return rows


def _aggregate_payloads(
    payloads: list[dict[str, Any]],
    backend: str,
    size_mib: int,
    concurrency: int,
    child_returncode: int,
) -> list[dict[str, Any]]:
  if not payloads:
    return [{
        "status": "error",
        "backend": backend,
        "size_mib": size_mib,
        "concurrency": concurrency,
        "direction": "both",
        "iterations": "",
        "warmups": "",
        "median_total_s": "",
        "mean_total_s": "",
        "min_total_s": "",
        "p90_total_s": "",
        "median_gib_s": "",
        "mean_gib_s": "",
        "max_gib_s": "",
        "min_gib_s": "",
        "child_returncode": child_returncode,
        "notes": "No worker result files were produced.",
    }]

  statuses = {p.get("status") for p in payloads}
  if "error" in statuses:
    note = " | ".join(
        str(p.get("error", "")) for p in payloads if p.get("status") == "error"
    )
    return [{
        "status": "error",
        "backend": backend,
        "size_mib": size_mib,
        "concurrency": concurrency,
        "direction": "both",
        "iterations": "",
        "warmups": "",
        "median_total_s": "",
        "mean_total_s": "",
        "min_total_s": "",
        "p90_total_s": "",
        "median_gib_s": "",
        "mean_gib_s": "",
        "max_gib_s": "",
        "min_gib_s": "",
        "child_returncode": child_returncode,
        "notes": note,
    }]

  if statuses == {"skipped"}:
    note = " | ".join(str(p.get("reason", "")) for p in payloads)
    return [{
        "status": "skipped",
        "backend": backend,
        "size_mib": size_mib,
        "concurrency": concurrency,
        "direction": "both",
        "iterations": "",
        "warmups": "",
        "median_total_s": "",
        "mean_total_s": "",
        "min_total_s": "",
        "p90_total_s": "",
        "median_gib_s": "",
        "mean_gib_s": "",
        "max_gib_s": "",
        "min_gib_s": "",
        "child_returncode": child_returncode,
        "notes": note,
    }]

  ok_payloads = [p for p in payloads if p.get("status") == "ok"]
  return _aggregate_success(ok_payloads, child_returncode)


def _read_worker_results(sync_dir: Path) -> list[dict[str, Any]]:
  payloads = []
  for path in sorted(sync_dir.glob("rank_*.json")):
    with path.open(encoding="utf-8") as f:
      payloads.append(json.load(f))
  return payloads


def _write_outputs(rows: list[dict[str, Any]], out_prefix: Path) -> None:
  out_prefix.parent.mkdir(parents=True, exist_ok=True)
  jsonl_path = out_prefix.with_suffix(".jsonl")
  csv_path = out_prefix.with_suffix(".csv")
  with jsonl_path.open("w", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(row, sort_keys=True) + "\n")
  fieldnames = [
      "status",
      "backend",
      "size_mib",
      "concurrency",
      "direction",
      "iterations",
      "warmups",
      "median_total_s",
      "mean_total_s",
      "min_total_s",
      "p90_total_s",
      "median_gib_s",
      "mean_gib_s",
      "max_gib_s",
      "min_gib_s",
      "child_returncode",
      "notes",
  ]
  with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow(row)


def _worker_command(
    backend: str,
    size_mib: int,
    concurrency: int,
    args: argparse.Namespace,
    sync_dir: Path,
) -> list[str]:
  script = Path(__file__).resolve()
  common = [
      str(script),
      "--worker",
      "--backend",
      backend,
      "--size-mib",
      str(size_mib),
      "--concurrency",
      str(concurrency),
      "--iterations",
      str(args.iterations),
      "--warmups",
      str(args.warmups),
      "--sync-dir",
      str(sync_dir),
      "--jax-raw-module",
      args.jax_raw_module,
      "--host-padding-bytes",
      str(args.host_padding_bytes),
  ]
  if args.no_pin_memory:
    common.append("--no-pin-memory")
  if args.fast_exit:
    common.append("--fast-exit")

  if backend in TORCH_BACKENDS.union(PREPARED_TORCH_BACKENDS) and concurrency == 8:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
    ] + common
  return [sys.executable] + common


def _run_one(
    backend: str,
    size_mib: int,
    concurrency: int,
    args: argparse.Namespace,
    log_file: Any,
) -> list[dict[str, Any]]:
  if (
      backend in TORCH_BACKENDS.union(PREPARED_TORCH_BACKENDS)
      and concurrency not in (1, 8)
  ):
    raise ValueError("Torch backends currently support concurrency 1 or 8.")

  with tempfile.TemporaryDirectory(
      prefix=f"raiden_raw_{backend}_{size_mib}m_{concurrency}x_"
  ) as tmp:
    sync_dir = Path(tmp)
    cmd = _worker_command(backend, size_mib, concurrency, args, sync_dir)
    print(
        f"RUN backend={backend} size={size_mib}MiB concurrency={concurrency}",
        flush=True,
    )
    print("$ " + " ".join(cmd), file=log_file, flush=True)
    child_env = _child_env()
    if backend in TORCH_BACKENDS.union(PREPARED_TORCH_BACKENDS) and concurrency == 8:
      child_env = _add_torch_singlehost_env(child_env)
    proc = subprocess.run(
        cmd,
        cwd=str(_repo_paths()[0]),
        env=child_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout_s,
        check=False,
    )
    print(proc.stdout, file=log_file, flush=True)
    payloads = _read_worker_results(sync_dir)
    rows = _aggregate_payloads(
        payloads, backend, size_mib, concurrency, proc.returncode
    )
    for row in rows:
      if row["status"] == "ok":
        print(
            "  "
            f"{row['direction']} median={row['median_gib_s']:.2f} GiB/s "
            f"time={row['median_total_s']:.6f}s",
            flush=True,
        )
      else:
        print(f"  {row['status']}: {row['notes']}", flush=True)
    return rows


def _driver(args: argparse.Namespace) -> None:
  workspace_root, _, _ = _repo_paths()
  backends = [b.strip() for b in args.backends.split(",") if b.strip()]
  sizes_mib = _parse_csv_ints(args.sizes_mib)
  concurrencies = _parse_csv_ints(args.concurrencies)
  run_logs = workspace_root / "run_logs"
  timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
  out_prefix = run_logs / f"raw_transfer_compare_{timestamp}"
  all_rows = []
  with out_prefix.with_suffix(".log").open("w", encoding="utf-8") as log_file:
    for backend in backends:
      for concurrency in concurrencies:
        for size_mib in sizes_mib:
          try:
            all_rows.extend(
                _run_one(backend, size_mib, concurrency, args, log_file)
            )
          except subprocess.TimeoutExpired as exc:
            all_rows.append({
                "status": "error",
                "backend": backend,
                "size_mib": size_mib,
                "concurrency": concurrency,
                "direction": "both",
                "iterations": "",
                "warmups": "",
                "median_total_s": "",
                "mean_total_s": "",
                "min_total_s": "",
                "p90_total_s": "",
                "median_gib_s": "",
                "mean_gib_s": "",
                "max_gib_s": "",
                "min_gib_s": "",
                "child_returncode": "",
                "notes": f"Timed out after {exc.timeout}s",
            })
            print(f"  error: timed out after {exc.timeout}s", flush=True)
          _write_outputs(all_rows, out_prefix)
  _write_outputs(all_rows, out_prefix)
  print(f"Wrote {out_prefix.with_suffix('.csv')}", flush=True)
  print(f"Wrote {out_prefix.with_suffix('.jsonl')}", flush=True)
  print(f"Wrote {out_prefix.with_suffix('.log')}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--worker", action="store_true")
  parser.add_argument("--backend", default=",".join(REQUESTED_BACKENDS))
  parser.add_argument("--backends", default=",".join(REQUESTED_BACKENDS))
  parser.add_argument(
      "--sizes-mib",
      default=",".join(str(x) for x in DEFAULT_SIZES_MIB),
      help="Comma-separated per-DMA sizes in MiB.",
  )
  parser.add_argument(
      "--concurrencies",
      default=",".join(str(x) for x in DEFAULT_CONCURRENCIES),
      help="Comma-separated concurrent DMA counts.",
  )
  parser.add_argument("--size-mib", type=int, default=64)
  parser.add_argument("--concurrency", type=int, default=1)
  parser.add_argument("--iterations", type=int, default=3)
  parser.add_argument("--warmups", type=int, default=1)
  parser.add_argument("--sync-dir", default="")
  parser.add_argument(
      "--jax-raw-module",
      default="raiden_lib.raw_transfer.jax.raw_transfer",
      help="Import path for the JAX Raiden raw transfer module.",
  )
  parser.add_argument("--timeout-s", type=int, default=1800)
  parser.add_argument("--host-padding-bytes", type=int, default=4096)
  parser.add_argument("--no-pin-memory", action="store_true")
  parser.add_argument(
      "--fast-exit",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Use os._exit(0) after successful Torch workers to avoid teardown.",
  )
  parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
  return parser


def main() -> None:
  args = _build_parser().parse_args()
  if args.worker:
    if not args.sync_dir:
      raise ValueError("--sync-dir is required in worker mode")
    if args.backend in TORCH_BACKENDS.union(PREPARED_TORCH_BACKENDS):
      _run_torch_worker(args)
    elif args.backend in {"jax_raiden", "jax_device_put"}:
      _run_jax_worker(args)
    else:
      raise ValueError(f"Unknown backend {args.backend!r}")
  else:
    _driver(args)


if __name__ == "__main__":
  main()
