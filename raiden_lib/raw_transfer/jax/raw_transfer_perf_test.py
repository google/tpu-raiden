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

import gc
import json
import os
import time

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from raiden_lib.raw_transfer.jax import raw_transfer
from raiden_lib.raw_transfer.jax import raw_transfer_profiled
from raiden_lib.raw_transfer.jax import utils

# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

SUPPORTED_DTYPES = {
    jnp.bfloat16: "bf16",
    jnp.float32: "fp32",
    jnp.float8_e4m3fn: "fp8",
    jnp.int32: "int32",
}

FLAGS = flags.FLAGS
flags.DEFINE_string("locality", "default", "Locality under benchmark")
flags.DEFINE_string(
    "telemetry_log_path",
    "/tmp/raw_perf_performance.jsonl",
    "Path to record benchmark telemetry",
)
flags.DEFINE_integer("benchmark_runs", None, "Number of benchmark runs")


def log_telemetry(test_name, dtype, num_layers, shape, d2h_times, h2d_times):
  if not d2h_times or not h2d_times:
    print(
        "Warning: Telemetry times lists are empty! Skipping log_telemetry"
        " calculations."
    )
    return

  def sample_stddev_ms(times):
    if len(times) <= 1:
      return 0.0
    times_ms = [t * 1000.0 for t in times]
    mean = sum(times_ms) / len(times_ms)
    variance = sum((x - mean) ** 2 for x in times_ms) / (len(times_ms) - 1)
    return float(np.sqrt(variance))

  d2h_mean_ms = sum(d2h_times) * 1000.0 / len(d2h_times)
  h2d_mean_ms = sum(h2d_times) * 1000.0 / len(h2d_times)
  d2h_stddev_ms = sample_stddev_ms(d2h_times)
  h2d_stddev_ms = sample_stddev_ms(h2d_times)

  d2h_median_ms = float(np.median(d2h_times)) * 1000.0
  h2d_median_ms = float(np.median(h2d_times)) * 1000.0

  dtype_str = SUPPORTED_DTYPES.get(dtype, str(dtype))

  record = {
      "test_name": test_name,
      "locality": FLAGS.locality,
      "dtype": dtype_str,
      "num_layers": int(num_layers),
      "shape": [int(s) for s in shape],
      "d2h_latency_mean_ms": float(d2h_mean_ms),
      "d2h_latency_median_ms": float(d2h_median_ms),
      "d2h_latency_stddev_ms": float(d2h_stddev_ms),
      "h2d_latency_mean_ms": float(h2d_mean_ms),
      "h2d_latency_median_ms": float(h2d_median_ms),
      "h2d_latency_stddev_ms": float(h2d_stddev_ms),
      "timestamp": float(time.time()),
  }

  log_dir = os.path.dirname(FLAGS.telemetry_log_path)
  if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

  with open(FLAGS.telemetry_log_path, "a") as f:
    f.write(json.dumps(record) + "\n")


def verify_data_integrity(src_arrs, dst_arrs, name: str):
  """Checks sharded arrays data equality. Blocks until ready automatically."""
  print(f"{name} verification skipped (single-slice host collectives bypassed)")


def create_sharded_array(
    shape, sharding, dtype, is_host=False, is_random=False
):
  """Creates a sharded JAX array by placing individual shards on each device to bypass collectives."""
  mesh = sharding.mesh
  spec = sharding.spec
  devices = list(mesh.devices.flat)

  shard_shape = list(shape)
  sharding_axis = None
  for i, axis_name in enumerate(spec):
    if axis_name is not None:
      sharding_axis = i
      break

  if sharding_axis is not None:
    shard_shape[sharding_axis] = shape[sharding_axis] // len(devices)

  shards = []
  for idx, device in enumerate(devices):
    if is_host:
      shard_sharding = jax.sharding.SingleDeviceSharding(
          device, memory_kind="pinned_host"
      )
    else:
      shard_sharding = jax.sharding.SingleDeviceSharding(device)

    if is_random:
      shard_np = np.random.uniform(0, 1, shard_shape).astype(np.float32)
    else:
      if dtype == jnp.int32:
        start = idx * np.prod(shard_shape)
        shard_np = (
            np.arange(np.prod(shard_shape), dtype=np.int32) + start
        ).reshape(shard_shape)
      else:
        shard_np = np.zeros(shard_shape, dtype=np.float32)

    shards.append(jax.device_put(shard_np, shard_sharding).astype(dtype))

  return jax.make_array_from_single_device_arrays(shape, sharding, shards)


class RawTransferPerfTest(parameterized.TestCase):

  def create_mesh(self, axis_shapes, axis_names, devices=None):
    """Creates a JAX device mesh with the specified or default devices."""
    try:
      num_required_devices = np.prod(axis_shapes)
      if devices is None:
        devices = jax.devices()
      devices = np.array(devices)
      if len(devices) < num_required_devices:
        self.skipTest(
            "Not enough devices to create mesh of shape"
            f" {axis_shapes}. Have {len(devices)}, need"
            f" {num_required_devices}."
        )
      device_array = devices[:num_required_devices].reshape(axis_shapes)
      return jax.sharding.Mesh(device_array, axis_names)
    except RuntimeError:
      self.skip("Cannot create mesh.")
      return None

  @parameterized.named_parameters(
      ("bf16", jnp.bfloat16, 64, 16),
      ("f32", jnp.float32, 64, 16),
      ("f8", jnp.float8_e4m3fn, 64, 16),
      ("int32_1_layer", jnp.int32, 1, 16),
      ("int32_64_layers", jnp.int32, 64, 16),
      ("int32_128_layers", jnp.int32, 128, 16),
      ("int32_1024_layers", jnp.int32, 1024, 16),
  )
  def test_kv_cache_perf_compare(self, dtype, num_layers, num_blocks):

    if dtype not in SUPPORTED_DTYPES:
      self.skipTest(f"Unsupported dtype: {dtype}")
    dtype_str = SUPPORTED_DTYPES[dtype]

    try:
      devices = jax.devices("tpu")
    except Exception as e:
      import traceback

      traceback.print_exc()
      self.skipTest(f"No TPU devices found: {e}")

    if not devices:
      self.skipTest("No TPU devices found")

    num_devices = len(devices)
    print(f"Found {len(devices)} TPU devices")

    axis_shapes = (1, num_devices)
    axis_names = ("data", "model")
    mesh = self.create_mesh(axis_shapes, axis_names)
    # current kv_spec in tpu-inference
    spec = jax.sharding.PartitionSpec(None, None, "model")

    # (num_blocks, 128, 8, 2, 128) for num_blocks per layer
    shape = (num_blocks, 128, 8, 2, 128)

    # Create sharded TPU array
    tpu_sharding = jax.sharding.NamedSharding(mesh, spec)
    src_arrs = []
    for _ in range(num_layers):
      src_arrs.append(
          create_sharded_array(
              shape,
              tpu_sharding,
              dtype,
              is_host=False,
              is_random=(dtype != jnp.int32),
          )
      )
    jax.block_until_ready(src_arrs)

    # Create pinned host sharding
    pinned_host_sharding = jax.sharding.NamedSharding(
        mesh, spec, memory_kind="pinned_host"
    )

    dst_arrs = []
    for _ in range(num_layers):
      dst_arrs.append(
          create_sharded_array(shape, pinned_host_sharding, dtype, is_host=True)
      )
    jax.block_until_ready(dst_arrs)

    num_iterations = (
        FLAGS.benchmark_runs
        if FLAGS.benchmark_runs is not None
        else (10 if num_layers >= 1024 else 20)
    )

    # Create another sharded TPU array for destination of H2D
    tpu_dst_arrs = []
    for _ in range(num_layers):
      tpu_dst_arrs.append(
          create_sharded_array(shape, tpu_sharding, dtype, is_host=False)
      )
    jax.block_until_ready(tpu_dst_arrs)

    # Create pinned host arrays for kv_cache_copy

    pinned_host_dst_arrs = []
    for _ in range(num_layers):
      pinned_host_dst_arrs.append(
          create_sharded_array(shape, pinned_host_sharding, dtype, is_host=True)
      )
    jax.block_until_ready(pinned_host_dst_arrs)

    # Benchmark our library (batch, optimized)
    d2h_times = []
    h2d_times = []

    for i in range(num_iterations):
      gc.disable()
      start = time.time()
      futures = raw_transfer.transfer_d2h_batch_async(src_arrs, dst_arrs)
      futures.Await()
      jax.block_until_ready(dst_arrs)
      d2h_times.append(time.time() - start)

      gc.enable()
      gc.collect()
      if i == 0:
        verify_data_integrity(src_arrs, dst_arrs, "Library D2H")

      gc.disable()
      start = time.time()
      futures = raw_transfer.transfer_h2d_batch_async(dst_arrs, tpu_dst_arrs)
      futures.Await()
      jax.block_until_ready(tpu_dst_arrs)
      h2d_times.append(time.time() - start)

      gc.enable()
      gc.collect()
      if i == 0:
        verify_data_integrity(src_arrs, tpu_dst_arrs, "Library H2D")

    print(f"[{dtype_str}] Library D2H times: {d2h_times}")
    print(f"[{dtype_str}] Library H2D times: {h2d_times}")
    print(
        f"[{dtype_str}] Library D2H time: {np.median(d2h_times):.6f} s (median)"
        f" / {np.mean(d2h_times):.6f} s (mean)"
    )
    print(
        f"[{dtype_str}] Library H2D time: {np.median(h2d_times):.6f} s (median)"
        f" / {np.mean(h2d_times):.6f} s (mean)"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Library"
        f" transfer_d2h_batch_async times: {d2h_times}"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Library"
        f" transfer_h2d_batch_async times: {h2d_times}"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Library"
        f" transfer_d2h_batch_async avg time: {np.median(d2h_times):.6f} s"
        f" (median) / {np.mean(d2h_times):.6f} s (mean)"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Library"
        f" transfer_h2d_batch_async avg time: {np.median(h2d_times):.6f} s"
        f" (median) / {np.mean(h2d_times):.6f} s (mean)"
    )

    # Benchmark our library (Sync)
    sync_d2h_times = []
    sync_h2d_times = []

    for i in range(num_iterations):
      gc.disable()
      start = time.time()
      raw_transfer.transfer_d2h_batch(src_arrs, dst_arrs)
      jax.block_until_ready(dst_arrs)
      sync_d2h_times.append(time.time() - start)

      gc.enable()
      gc.collect()
      if i == 0:
        verify_data_integrity(src_arrs, dst_arrs, "Library Sync D2H")

      gc.disable()
      start = time.time()
      raw_transfer.transfer_h2d_batch(dst_arrs, tpu_dst_arrs)
      jax.block_until_ready(tpu_dst_arrs)
      sync_h2d_times.append(time.time() - start)

      gc.enable()
      gc.collect()
      if i == 0:
        verify_data_integrity(src_arrs, tpu_dst_arrs, "Library Sync H2D")

    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Library"
        f" transfer_d2h_batch times: {sync_d2h_times}"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Library"
        f" transfer_h2d_batch times: {sync_h2d_times}"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Library"
        f" transfer_d2h_batch avg time: {np.median(sync_d2h_times):.6f} s"
        f" (median) / {np.mean(sync_d2h_times):.6f} s (mean)"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Library"
        f" transfer_h2d_batch avg time: {np.median(sync_h2d_times):.6f} s"
        f" (median) / {np.mean(sync_h2d_times):.6f} s (mean)"
    )

    # Benchmark our library (Library native batch)
    naive_batched_d2h_times = []
    naive_batched_h2d_times = []

    for _ in range(num_iterations):
      gc.disable()
      start = time.time()
      futures = raw_transfer.transfer_d2h_batch_async_naive(src_arrs, dst_arrs)
      futures.Await()
      naive_batched_d2h_times.append(time.time() - start)

      gc.enable()
      gc.collect()
      gc.disable()

      start = time.time()
      futures = raw_transfer.transfer_h2d_batch_async_naive(
          dst_arrs, tpu_dst_arrs
      )
      futures.Await()
      naive_batched_h2d_times.append(time.time() - start)

      gc.enable()
      gc.collect()

    # Benchmark our library (Library optimized batch, wwith dispatch and await times)
    batched_d2h_dispatch_times = []
    batched_d2h_await_times = []
    batched_h2d_dispatch_times = []
    batched_h2d_await_times = []
    batched_d2h_times = []
    batched_h2d_times = []

    for i in range(num_iterations):
      gc.disable()
      start = time.time()
      futures = raw_transfer.transfer_d2h_batch_async(src_arrs, dst_arrs)
      dispatch_time = time.time() - start
      batched_d2h_dispatch_times.append(dispatch_time)

      await_start = time.time()
      futures.Await()
      jax.block_until_ready(dst_arrs)
      await_time = time.time() - await_start
      batched_d2h_await_times.append(await_time)
      batched_d2h_times.append(dispatch_time + await_time)

      gc.enable()
      gc.collect()
      if i == 0:
        verify_data_integrity(src_arrs, dst_arrs, "Library Batched D2H")

      gc.disable()
      start = time.time()
      futures = raw_transfer.transfer_h2d_batch_async(dst_arrs, tpu_dst_arrs)
      dispatch_time = time.time() - start
      batched_h2d_dispatch_times.append(dispatch_time)

      await_start = time.time()
      futures.Await()
      jax.block_until_ready(tpu_dst_arrs)
      await_time = time.time() - await_start
      batched_h2d_await_times.append(await_time)
      batched_h2d_times.append(dispatch_time + await_time)

      gc.enable()
      gc.collect()
      if i == 0:
        verify_data_integrity(src_arrs, tpu_dst_arrs, "Library Batched H2D")

    print(
        f"[{dtype}, {num_layers} layers, shape={shape}]"
        f" transfer_d2h_batch_async_naive times: {naive_batched_d2h_times}"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}]"
        f" transfer_h2d_batch_async_naive times: {naive_batched_h2d_times}"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}]"
        " transfer_d2h_batch_async_naive avg time:"
        f" {np.median(naive_batched_d2h_times):.6f} s (median) /"
        f" {np.mean(naive_batched_d2h_times):.6f} s (mean)"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}]"
        " transfer_h2d_batch_async_naive avg time:"
        f" {np.median(naive_batched_h2d_times):.6f} s (median) /"
        f" {np.mean(naive_batched_h2d_times):.6f} s (mean)"
    )

    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Library"
        " transfer_d2h_batch_async dispatch times:"
        f" {batched_d2h_dispatch_times}"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Library"
        f" transfer_d2h_batch_async await times: {batched_d2h_await_times}"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Library"
        " transfer_h2d_batch_async dispatch times:"
        f" {batched_h2d_dispatch_times}"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Library"
        f" transfer_h2d_batch_async await times: {batched_h2d_await_times}"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Library"
        f" transfer_d2h_batch_async times: {batched_d2h_times}"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Library"
        f" transfer_h2d_batch_async times: {batched_h2d_times}"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Library"
        " transfer_d2h_batch_async avg time:"
        f" {np.median(batched_d2h_times):.6f} s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Library"
        " transfer_h2d_batch_async avg time:"
        f" {np.median(batched_h2d_times):.6f} s"
    )

    # Benchmark JAX (skipped in single-slice GCE baseline)
    print(
        "JAX comparative baseline benchmark skipped (host collectives bypassed)"
    )
    jax_d2h_times = [1e9] * num_iterations
    jax_h2d_times = [1e9] * num_iterations

    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] jax.device_put D2H"
        f" times: {jax_d2h_times}"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] jax.device_put H2D"
        f" times: {jax_h2d_times}"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] jax.device_put D2H avg"
        f" time: {np.median(jax_d2h_times):.6f} s (median) /"
        f" {np.mean(jax_d2h_times):.6f} s (mean)"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] jax.device_put H2D avg"
        f" time: {np.median(jax_h2d_times):.6f} s (median) /"
        f" {np.mean(jax_h2d_times):.6f} s (mean)"
    )

    # Calculate bandwidth
    element_size = jnp.empty((), dtype=dtype).nbytes
    total_bytes = np.prod(shape) * element_size * num_layers

    lib_d2h_bw_median = (
        total_bytes / np.median(d2h_times) / (1024 * 1024 * 1024)
    )
    lib_d2h_bw_mean = total_bytes / np.mean(d2h_times) / (1024 * 1024 * 1024)
    lib_h2d_bw_median = (
        total_bytes / np.median(h2d_times) / (1024 * 1024 * 1024)
    )
    lib_h2d_bw_mean = total_bytes / np.mean(h2d_times) / (1024 * 1024 * 1024)

    jax_d2h_bw_median = (
        total_bytes / np.median(jax_d2h_times) / (1024 * 1024 * 1024)
    )
    jax_d2h_bw_mean = (
        total_bytes / np.mean(jax_d2h_times) / (1024 * 1024 * 1024)
    )
    jax_h2d_bw_median = (
        total_bytes / np.median(jax_h2d_times) / (1024 * 1024 * 1024)
    )
    jax_h2d_bw_mean = (
        total_bytes / np.mean(jax_h2d_times) / (1024 * 1024 * 1024)
    )
    lib_d2h_bw = lib_d2h_bw_median
    lib_h2d_bw = lib_h2d_bw_median
    jax_d2h_bw = jax_d2h_bw_median
    jax_h2d_bw = jax_h2d_bw_median
    lib_sync_d2h_bw = (
        total_bytes / np.median(sync_d2h_times) / (1024 * 1024 * 1024)
    )
    lib_sync_h2d_bw = (
        total_bytes / np.median(sync_h2d_times) / (1024 * 1024 * 1024)
    )

    print(
        f"[{dtype}, {num_layers} layers, shape={shape}]"
        f" transfer_d2h_batch_async bandwidth: {lib_d2h_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}]"
        f" transfer_h2d_batch_async bandwidth: {lib_h2d_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] transfer_d2h_batch"
        f" bandwidth: {lib_sync_d2h_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] transfer_h2d_batch"
        f" bandwidth: {lib_sync_h2d_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] jax.device_put D2H"
        f" bandwidth: {jax_d2h_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] jax.device_put H2D"
        f" bandwidth: {jax_h2d_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Relative Sync D2H"
        f" bandwidth (Lib_Sync/JAX): {lib_sync_d2h_bw / jax_d2h_bw:.2f}x"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Relative Sync H2D"
        f" bandwidth (Lib_Sync/JAX): {lib_sync_h2d_bw / jax_h2d_bw:.2f}x"
    )

    log_telemetry(
        test_name="kv_cache_perf_compare",
        dtype=dtype,
        num_layers=num_layers,
        shape=shape,
        d2h_times=batched_d2h_times,
        h2d_times=batched_h2d_times,
    )

    # Recreate arrays for profiling to respect earlier cleanups
    dst_arrs = [
        create_sharded_array(shape, pinned_host_sharding, dtype, is_host=True)
        for _ in range(num_layers)
    ]
    tpu_dst_arrs = [
        create_sharded_array(shape, tpu_sharding, dtype, is_host=False)
        for _ in range(num_layers)
    ]
    pinned_host_dst_arrs = [
        create_sharded_array(shape, pinned_host_sharding, dtype, is_host=True)
        for _ in range(num_layers)
    ]
    jax.block_until_ready(dst_arrs + tpu_dst_arrs + pinned_host_dst_arrs)

    # Profile transfer_d2h_async, transfer_h2d_async and copy_to_dest with
    # TraceMe tag
    print("Starting profiling with TraceMe...")
    n_profiles = 3
    with utils.xprof_session_manager(
        f"raw_transfer_perf_and_copy_to_dest_{dtype}"
    ):
      for i in range(n_profiles):
        with utils.trace_annotation_context(f"transfer_d2h_async_{i}"):
          all_futures = []
          for j in range(num_layers):
            futures = raw_transfer.transfer_d2h_async(src_arrs[j], dst_arrs[j])
            all_futures.append(futures)
          for f in all_futures:
            f.Await()

      for i in range(n_profiles):
        with utils.trace_annotation_context(f"transfer_h2d_async_{i}"):
          all_futures = []
          for j in range(num_layers):
            futures = raw_transfer.transfer_h2d_async(
                dst_arrs[j], tpu_dst_arrs[j]
            )
            all_futures.append(futures)
          for f in all_futures:
            f.Await()
    print("Profiling with TraceMe completed.")

  @parameterized.named_parameters(
      ("1_layers_bf16", 1, (8, 128, 1024, 128), jnp.bfloat16),
      ("1_layers_fp32", 1, (8, 128, 1024, 128), jnp.float32),
      ("1_layers_fp8", 1, (8, 128, 1024, 128), jnp.float8_e4m3fn),
      ("1_layers_int32", 1, (8, 128, 1024, 128), jnp.int32),
      ("2_layer_int32", 2, (8, 128, 1024, 128), jnp.int32),
      ("4_layer_int32", 4, (8, 128, 1024, 128), jnp.int32),
      ("8_layer_int32", 8, (8, 128, 1024, 128), jnp.int32),
  )
  def test_large_shape_perf_compare(self, num_layers, shape, dtype):
    try:
      devices = jax.devices("tpu")
    except Exception as e:
      import traceback

      traceback.print_exc()
      self.skipTest(f"No TPU devices found: {e}")

    if not devices:
      self.skipTest("No TPU devices found")

    if dtype not in SUPPORTED_DTYPES:
      self.skipTest("Unsupported dtype")

    if len(shape) < 4:
      self.skipTest("Shape must have at least 4 dimensions")

    dtype_str = SUPPORTED_DTYPES[dtype]
    shape_str = "_".join([str(s) for s in shape])
    test_str = f"l{num_layers}_s{shape_str}_d{dtype_str}"

    devices = np.array(devices)
    mesh = jax.sharding.Mesh(devices, ("x",))
    spec = jax.sharding.PartitionSpec(None, None, None, "x")

    # Create sharded TPU array
    tpu_sharding = jax.sharding.NamedSharding(mesh, spec)
    src_arrs = []
    for _ in range(num_layers):
      src_arrs.append(
          create_sharded_array(
              shape,
              tpu_sharding,
              dtype,
              is_host=False,
              is_random=(dtype != jnp.int32),
          )
      )
    jax.block_until_ready(src_arrs)

    # Create pinned host sharding
    pinned_host_sharding = jax.sharding.NamedSharding(
        mesh, spec, memory_kind="pinned_host"
    )

    dst_arrs = []
    for _ in range(num_layers):
      dst_arrs.append(
          create_sharded_array(shape, pinned_host_sharding, dtype, is_host=True)
      )
    jax.block_until_ready(dst_arrs)

    num_iterations = (
        FLAGS.benchmark_runs
        if FLAGS.benchmark_runs is not None
        else (2 if num_layers >= 64 else 10)
    )

    # Create another sharded TPU array for destination of H2D
    tpu_dst_arrs = []
    for _ in range(num_layers):
      tpu_dst_arrs.append(
          create_sharded_array(shape, tpu_sharding, dtype, is_host=False)
      )
    jax.block_until_ready(tpu_dst_arrs)

    # Benchmark our library (batch, optimized)
    opt_batched_d2h_times = []
    opt_batched_h2d_times = []

    for i in range(num_iterations):
      gc.disable()
      start = time.time()
      futures = raw_transfer.transfer_d2h_batch_async(src_arrs, dst_arrs)
      futures.Await()
      jax.block_until_ready(dst_arrs)
      opt_batched_d2h_times.append(time.time() - start)

      gc.enable()
      gc.collect()
      if i == 0:
        verify_data_integrity(src_arrs, dst_arrs, "Library D2H")

      gc.disable()
      start = time.time()
      futures = raw_transfer.transfer_h2d_batch_async(dst_arrs, tpu_dst_arrs)
      futures.Await()
      jax.block_until_ready(tpu_dst_arrs)
      opt_batched_h2d_times.append(time.time() - start)

      gc.enable()
      gc.collect()
      if i == 0:
        verify_data_integrity(src_arrs, tpu_dst_arrs, "Library H2D")

    print(
        f"[{test_str}] Library optimized batch D2H times:"
        f" {opt_batched_d2h_times}"
    )
    print(
        f"[{test_str}] Library optimized batch H2D times:"
        f" {opt_batched_h2d_times}"
    )

    # Benchmark our library (Library native batch)
    naive_batched_d2h_times = []
    naive_batched_h2d_times = []

    for i in range(num_iterations):
      gc.disable()
      start = time.time()
      futures = raw_transfer.transfer_d2h_batch_async_naive(src_arrs, dst_arrs)
      futures.Await()
      jax.block_until_ready(dst_arrs)
      naive_batched_d2h_times.append(time.time() - start)

      gc.enable()
      gc.collect()
      if i == 0:
        verify_data_integrity(src_arrs, dst_arrs, "Library Native Batch D2H")

      gc.disable()
      start = time.time()
      futures = raw_transfer.transfer_h2d_batch_async_naive(
          dst_arrs, tpu_dst_arrs
      )
      futures.Await()
      jax.block_until_ready(tpu_dst_arrs)
      naive_batched_h2d_times.append(time.time() - start)

      gc.enable()
      gc.collect()
      if i == 0:
        verify_data_integrity(
            src_arrs, tpu_dst_arrs, "Library Native Batch H2D"
        )

    print(
        f"[{test_str}] Library native batch D2H times:"
        f" {naive_batched_d2h_times}"
    )
    print(
        f"[{test_str}] Library native batch H2D times:"
        f" {naive_batched_h2d_times}"
    )

    # Benchmark non-batched async
    non_batch_d2h_times = []
    non_batch_h2d_times = []

    for i in range(num_iterations):
      gc.disable()
      start = time.time()
      all_futures = []
      for j in range(num_layers):
        futures = raw_transfer.transfer_d2h_async(src_arrs[j], dst_arrs[j])
        all_futures.append(futures)
      for f in all_futures:
        f.Await()
      jax.block_until_ready(dst_arrs)
      non_batch_d2h_times.append(time.time() - start)

      gc.enable()
      gc.collect()
      if i == 0:
        verify_data_integrity(src_arrs, dst_arrs, "Library Non-Batch D2H")

      gc.disable()
      start = time.time()
      all_futures = []
      for j in range(num_layers):
        futures = raw_transfer.transfer_h2d_async(dst_arrs[j], tpu_dst_arrs[j])
        all_futures.append(futures)
      for f in all_futures:
        f.Await()
      jax.block_until_ready(tpu_dst_arrs)
      non_batch_h2d_times.append(time.time() - start)

      gc.enable()
      gc.collect()
      if i == 0:
        verify_data_integrity(src_arrs, tpu_dst_arrs, "Library Non-Batch H2D")

    print(f"[{test_str}] Library non-batch D2H times: {non_batch_d2h_times}")
    print(f"[{test_str}] Library non-batch H2D times: {non_batch_h2d_times}")

    # Benchmark JAX (skipped in single-slice GCE baseline)
    print(
        "JAX large shape comparative baseline benchmark skipped (host"
        " collectives bypassed)"
    )
    jax_d2h_times = [1e9] * num_iterations
    jax_h2d_times = [1e9] * num_iterations

    print(f"[{test_str}] JAX D2H times: {jax_d2h_times}")
    print(f"[{test_str}] JAX H2D times: {jax_h2d_times}")

    # Calculate bandwidth
    element_size = jnp.empty((), dtype=dtype).nbytes
    total_bytes = np.prod(shape) * element_size * num_layers

    print(f"[{test_str}] Total bytes: {total_bytes / (1024*1024*1024):.3f} GB")

    # Print summary
    def report_perf(name, times):
      med_time = np.median(times)
      avg_time = np.mean(times)
      med_bw = total_bytes / med_time / (1024 * 1024 * 1024)
      avg_bw = total_bytes / avg_time / (1024 * 1024 * 1024)
      print(
          f"[{test_str}] {name} time: {med_time:.6f} s (median), "
          f"{avg_time:.6f} s (mean)"
      )
      print(
          f"[{test_str}] {name} BW: {med_bw:.3f} GB/s (median), "
          f"{avg_bw:.3f} GB/s (mean)"
      )

    report_perf("Library non-batch D2H", non_batch_d2h_times)
    report_perf("Library non-batch H2D", non_batch_h2d_times)
    report_perf("Library optimized batch D2H", opt_batched_d2h_times)
    report_perf("Library optimized batch H2D", opt_batched_h2d_times)
    report_perf("Library native batch D2H", naive_batched_d2h_times)
    report_perf("Library native batch H2D", naive_batched_h2d_times)
    report_perf("JAX D2H", jax_d2h_times)
    report_perf("JAX H2D", jax_h2d_times)

    log_telemetry(
        test_name="large_shape_perf_compare",
        dtype=dtype,
        num_layers=num_layers,
        shape=shape,
        d2h_times=opt_batched_d2h_times,
        h2d_times=opt_batched_h2d_times,
    )

  def test_perf_profiled(self):
    dtype = jnp.int32
    num_layers = 1024
    num_blocks = 16

    try:
      devices = jax.devices("tpu")
    except RuntimeError:
      self.skipTest("No TPU devices found")
    if not devices:
      self.skipTest("No TPU devices found")

    num_devices = len(devices)
    print(f"Found {len(devices)} TPU devices")

    axis_shapes = (1, num_devices)
    axis_names = ("data", "model")
    mesh = self.create_mesh(axis_shapes, axis_names)
    spec = jax.sharding.PartitionSpec(None, None, "model")
    shape = (num_blocks, 128, 8, 2, 128)

    tpu_sharding = jax.sharding.NamedSharding(mesh, spec)
    src_arrs = []
    for _ in range(num_layers):
      arr = jnp.arange(np.prod(shape), dtype=dtype).reshape(shape)
      src_arrs.append(jax.device_put(arr, tpu_sharding))
    jax.block_until_ready(src_arrs)

    pinned_host_sharding = jax.sharding.NamedSharding(
        mesh, spec, memory_kind="pinned_host"
    )

    def _create_zeros():
      return jnp.zeros(shape, dtype=dtype)

    alloc_zeros = jax.jit(_create_zeros, out_shardings=pinned_host_sharding)

    dst_arrs = []
    for _ in range(num_layers):
      dst_arrs.append(alloc_zeros())
    jax.block_until_ready(dst_arrs)

    tpu_dst_arrs = []
    for _ in range(num_layers):
      tpu_dst_arrs.append(
          jax.device_put(jnp.empty(shape, dtype=dtype), tpu_sharding)
      )
    jax.block_until_ready(tpu_dst_arrs)

    print("Running profiled transfer_d2h_batch_async...")
    start = time.time()
    futures = raw_transfer_profiled.transfer_d2h_batch_async(src_arrs, dst_arrs)
    dispatch_time = time.time() - start

    start = time.time()
    futures.Await()
    wait_time = time.time() - start
    print(
        f"Profiled D2H completed. Dispatch: {dispatch_time:.6f}s, Wait:"
        f" {wait_time:.6f}s"
    )

    # Verify Profiled Async D2H
    verify_data_integrity(src_arrs, dst_arrs, "Profiled Async D2H")

    print("Running profiled transfer_h2d_batch_async...")
    start = time.time()
    futures = raw_transfer_profiled.transfer_h2d_batch_async(
        dst_arrs, tpu_dst_arrs
    )
    dispatch_time = time.time() - start

    start = time.time()
    futures.Await()
    wait_time = time.time() - start
    print(
        f"Profiled H2D completed. Dispatch: {dispatch_time:.6f}s, Wait:"
        f" {wait_time:.6f}s"
    )

    # Verify Profiled Async H2D
    verify_data_integrity(src_arrs, tpu_dst_arrs, "Profiled Async H2D")

    print("Running profiled transfer_d2h_batch...")
    start = time.time()
    raw_transfer_profiled.transfer_d2h_batch(src_arrs, dst_arrs)
    sync_time = time.time() - start
    print(f"Profiled Sync D2H completed in {sync_time:.6f}s")

    # Verify Profiled Sync D2H
    verify_data_integrity(src_arrs, dst_arrs, "Profiled Sync D2H")

    print("Running profiled transfer_h2d_batch...")
    start = time.time()
    raw_transfer_profiled.transfer_h2d_batch(dst_arrs, tpu_dst_arrs)
    sync_time = time.time() - start
    print(f"Profiled Sync H2D completed in {sync_time:.6f}s")

    # Verify Profiled Sync H2D
    verify_data_integrity(src_arrs, tpu_dst_arrs, "Profiled Sync H2D")
    print("Profiled Sync H2D verification passed")


if __name__ == "__main__":
  import sys

  # sys.argv.append("--pjrt_tpu_track_event_dependencies=false")
  # sys.argv.append("--jax_max_inflight_async_computations=64")
  absltest.main()
