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

from tpu_raiden.frameworks.jax import _tpu_raiden_jax as _kv_cache_manager

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
    "/tmp/kv_cache_manager_perf_performance.jsonl",
    "Path to record benchmark telemetry",
)
flags.DEFINE_integer("benchmark_runs", None, "Number of benchmark runs")


def calculate_stats_and_ci(times):
  times = np.array(times)
  mean_val = np.mean(times)
  std_val = np.std(times, ddof=1)  # sample standard deviation
  n = len(times)
  if n <= 1:
    return mean_val, 0.0, mean_val, mean_val
  t_table = {
      1: 12.706,
      2: 4.303,
      3: 3.182,
      4: 2.776,
      5: 2.571,
      6: 2.447,
      7: 2.365,
      8: 2.306,
      9: 2.262,
      10: 2.228,
      11: 2.201,
      12: 2.179,
      13: 2.160,
      14: 2.145,
      15: 2.131,
      16: 2.120,
      17: 2.110,
      18: 2.101,
      19: 2.093,
      20: 2.086,
      21: 2.080,
      22: 2.074,
      23: 2.069,
      24: 2.064,
      25: 2.060,
      26: 2.056,
      27: 2.052,
      28: 2.048,
      29: 2.045,
      30: 2.042,
  }
  t_val = t_table.get(n - 1, 1.960) if n - 1 <= 30 else 1.960
  margin_of_error = t_val * (std_val / np.sqrt(n))
  return (
      mean_val,
      std_val,
      mean_val - margin_of_error,
      mean_val + margin_of_error,
  )


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


def verify_data_integrity(unused_src_arrs, unused_dst_arrs, name: str):
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


class KVCacheManagerPerfTest(parameterized.TestCase):

  def tearDown(self):
    for attr in [
        "src_arrs",
        "dst_arrs",
        "tpu_dst_arrs",
        "pinned_host_dst_arrs",
        "jax_pinned_dst_arrs",
        "jax_pinned_tpu_dst_arrs",
        "jax_std_dst_arrs",
        "jax_std_tpu_dst_arrs",
        "std_host_numpy_arrs",
    ]:
      if hasattr(self, attr):
        delattr(self, attr)
    gc.collect()
    try:
      jax.effects_barrier()
    except Exception:  # pylint: disable=broad-except
      pass
    super().tearDown()

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
  )
  def test_kv_cache_perf_compare(self, dtype, num_layers, num_blocks):
    if dtype not in SUPPORTED_DTYPES:
      self.skipTest(f"Unsupported dtype: {dtype}")

    try:
      devices = jax.devices("tpu")
    except Exception as e:
      self.skipTest(f"No TPU devices found: {e}")

    if not devices:
      self.skipTest("No TPU devices found")

    num_devices = len(devices)
    print(f"Found {len(devices)} TPU devices")

    axis_shapes = (1, num_devices)
    axis_names = ("data", "model")
    mesh = self.create_mesh(axis_shapes, axis_names)
    spec = jax.sharding.PartitionSpec(None, None, "model")

    # (num_blocks, 128, 8, 2, 128) for num_blocks per layer
    shape = (num_blocks, 128, 8, 2, 128)

    # Create sharded TPU array
    tpu_sharding = jax.sharding.NamedSharding(mesh, spec)
    self.src_arrs = []
    for _ in range(num_layers):
      self.src_arrs.append(
          create_sharded_array(
              shape,
              tpu_sharding,
              dtype,
              is_host=False,
              is_random=(dtype != jnp.int32),
          )
      )
    jax.block_until_ready(self.src_arrs)

    # Force shadow page invalidation on PJRT/IFRT backend
    tpu_mutate_fn = jax.jit(
        lambda x: x
        + jnp.array(1 if x.dtype == jnp.int32 else 0.01, dtype=x.dtype)
    )

    # Create pinned host sharding
    pinned_host_sharding = jax.sharding.NamedSharding(
        mesh, spec, memory_kind="pinned_host"
    )

    self.dst_arrs = []
    for _ in range(num_layers):
      self.dst_arrs.append(
          create_sharded_array(shape, pinned_host_sharding, dtype, is_host=True)
      )
    jax.block_until_ready(self.dst_arrs)

    num_iterations = (
        FLAGS.benchmark_runs if FLAGS.benchmark_runs is not None else 10
    )
    print(f"Running benchmark with {num_iterations} iterations")

    # Benchmark KVCacheManager
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Benchmarking"
        " KVCacheManager..."
    )

    # Initialize KVCacheManager
    manager = _kv_cache_manager.KVCacheManager(
        device_arrays=self.src_arrs,
        host_blocks_to_allocate=num_blocks,
        unsafe_skip_buffer_lock=False,
    )

    d2h_times = []
    h2d_times = []

    # We copy the full major dimension
    offsets = [0]
    sizes = [num_blocks]

    for _ in range(num_iterations):
      # Invalidate local shadow copies by mutating array on device
      self.src_arrs = [tpu_mutate_fn(arr) for arr in self.src_arrs]
      jax.block_until_ready(self.src_arrs)

      gc.disable()
      start = time.time()
      future = manager.d2h(
          src_offsets_major_dim=offsets,
          dst_offsets_major_dim=offsets,
          copy_sizes_major_dim=sizes,
      )
      future.Await()
      d2h_times.append(time.time() - start)
      gc.enable()
      gc.collect()

      gc.disable()
      start = time.time()
      future = manager.h2d(
          src_offsets_major_dim=offsets,
          dst_offsets_major_dim=offsets,
          copy_sizes_major_dim=sizes,
      )
      future.Await()
      h2d_times.append(time.time() - start)
      gc.enable()
      gc.collect()

    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] KVCacheManager"
        f" d2h avg time: {np.median(d2h_times):.6f} s (median)"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] KVCacheManager"
        f" h2d avg time: {np.median(h2d_times):.6f} s (median)"
    )

    # Benchmark JAX Pinned Host Baseline
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Benchmarking JAX Pinned"
        " Host Baseline..."
    )
    jax_pinned_d2h_times = []
    jax_pinned_h2d_times = []
    for _ in range(num_iterations):
      gc.disable()
      start = time.time()
      self.jax_pinned_dst_arrs = []
      for j in range(num_layers):
        self.jax_pinned_dst_arrs.append(
            jax.device_put(self.src_arrs[j], pinned_host_sharding)
        )
      jax.block_until_ready(self.jax_pinned_dst_arrs)
      jax_pinned_d2h_times.append(time.time() - start)
      gc.enable()
      gc.collect()

      gc.disable()
      start = time.time()
      self.jax_pinned_tpu_dst_arrs = []
      for j in range(num_layers):
        self.jax_pinned_tpu_dst_arrs.append(
            jax.device_put(self.dst_arrs[j], tpu_sharding)
        )
      jax.block_until_ready(self.jax_pinned_tpu_dst_arrs)
      jax_pinned_h2d_times.append(time.time() - start)
      gc.enable()
      gc.collect()

    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] JAX Pinned D2H avg"
        f" time: {np.median(jax_pinned_d2h_times):.6f} s (median)"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] JAX Pinned H2D avg"
        f" time: {np.median(jax_pinned_h2d_times):.6f} s (median)"
    )

    # Benchmark JAX Standard Baseline (non-pinned NumPy)
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Benchmarking JAX"
        " Standard Baseline..."
    )
    jax_std_d2h_times = []
    jax_std_h2d_times = []
    self.std_host_numpy_arrs = []
    for _ in range(num_layers):
      self.std_host_numpy_arrs.append(
          np.zeros(shape, dtype=np.float32).astype(dtype)
      )
    for _ in range(num_iterations):
      self.src_arrs = [tpu_mutate_fn(arr) for arr in self.src_arrs]
      jax.block_until_ready(self.src_arrs)
      gc.disable()
      start = time.time()
      self.jax_std_dst_arrs = []
      for j in range(num_layers):
        self.jax_std_dst_arrs.append(jax.device_get(self.src_arrs[j]))
      jax_std_d2h_times.append(time.time() - start)
      gc.enable()
      gc.collect()

      gc.disable()
      start = time.time()
      self.jax_std_tpu_dst_arrs = []
      for j in range(num_layers):
        self.jax_std_tpu_dst_arrs.append(
            jax.device_put(self.std_host_numpy_arrs[j], tpu_sharding)
        )
      jax.block_until_ready(self.jax_std_tpu_dst_arrs)
      jax_std_h2d_times.append(time.time() - start)
      gc.enable()
      gc.collect()

    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] JAX Standard D2H avg"
        f" time: {np.median(jax_std_d2h_times):.6f} s (median)"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] JAX Standard H2D avg"
        f" time: {np.median(jax_std_h2d_times):.6f} s (median)"
    )

    # Calculate bandwidth
    element_size = jnp.empty((), dtype=dtype).nbytes
    total_bytes = np.prod(shape) * element_size * num_layers

    mgr_d2h_bw = total_bytes / np.median(d2h_times) / (1024 * 1024 * 1024)
    mgr_h2d_bw = total_bytes / np.median(h2d_times) / (1024 * 1024 * 1024)

    jax_pinned_d2h_bw = (
        total_bytes / np.median(jax_pinned_d2h_times) / (1024 * 1024 * 1024)
    )
    jax_pinned_h2d_bw = (
        total_bytes / np.median(jax_pinned_h2d_times) / (1024 * 1024 * 1024)
    )
    jax_std_d2h_bw = (
        total_bytes / np.median(jax_std_d2h_times) / (1024 * 1024 * 1024)
    )
    jax_std_h2d_bw = (
        total_bytes / np.median(jax_std_h2d_times) / (1024 * 1024 * 1024)
    )

    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] KVCacheManager"
        f" D2H bandwidth: {mgr_d2h_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] KVCacheManager"
        f" H2D bandwidth: {mgr_h2d_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] JAX Pinned D2H"
        f" bandwidth: {jax_pinned_d2h_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] JAX Pinned H2D"
        f" bandwidth: {jax_pinned_h2d_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] JAX Standard D2H"
        f" bandwidth: {jax_std_d2h_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] JAX Standard H2D"
        f" bandwidth: {jax_std_h2d_bw:.3f} GB/s"
    )

    log_telemetry(
        "kv_cache_manager_perf_test",
        dtype,
        num_layers,
        shape,
        d2h_times,
        h2d_times,
    )

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
    if dtype not in SUPPORTED_DTYPES:
      self.skipTest(f"Unsupported dtype: {dtype}")

    try:
      devices = jax.devices("tpu")
    except Exception as e:
      self.skipTest(f"No TPU devices found: {e}")

    if not devices:
      self.skipTest("No TPU devices found")

    num_devices = len(devices)
    print(f"Found {len(devices)} TPU devices")

    axis_shapes = (1, num_devices)
    axis_names = ("data", "model")
    mesh = self.create_mesh(axis_shapes, axis_names)
    spec = jax.sharding.PartitionSpec(None, None, "model")

    tpu_sharding = jax.sharding.NamedSharding(mesh, spec)
    self.src_arrs = []
    for _ in range(num_layers):
      self.src_arrs.append(
          create_sharded_array(
              shape,
              tpu_sharding,
              dtype,
              is_host=False,
              is_random=(dtype != jnp.int32),
          )
      )
    jax.block_until_ready(self.src_arrs)

    tpu_mutate_fn = jax.jit(
        lambda x: x
        + jnp.array(1 if x.dtype == jnp.int32 else 0.01, dtype=x.dtype)
    )

    pinned_host_sharding = jax.sharding.NamedSharding(
        mesh, spec, memory_kind="pinned_host"
    )

    self.dst_arrs = []
    for _ in range(num_layers):
      self.dst_arrs.append(
          create_sharded_array(shape, pinned_host_sharding, dtype, is_host=True)
      )
    jax.block_until_ready(self.dst_arrs)

    num_iterations = (
        FLAGS.benchmark_runs if FLAGS.benchmark_runs is not None else 10
    )
    print(f"Running benchmark with {num_iterations} iterations")

    # Benchmark KVCacheManager
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Benchmarking"
        " KVCacheManager..."
    )

    num_blocks = shape[0]
    manager = _kv_cache_manager.KVCacheManager(
        device_arrays=self.src_arrs,
        host_blocks_to_allocate=num_blocks,
        unsafe_skip_buffer_lock=False,
    )

    d2h_times = []
    h2d_times = []
    offsets = [0]
    sizes = [num_blocks]

    for _ in range(num_iterations):
      self.src_arrs = [tpu_mutate_fn(arr) for arr in self.src_arrs]
      jax.block_until_ready(self.src_arrs)

      gc.disable()
      start = time.time()
      future = manager.d2h(
          src_offsets_major_dim=offsets,
          dst_offsets_major_dim=offsets,
          copy_sizes_major_dim=sizes,
      )
      future.Await()
      d2h_times.append(time.time() - start)
      gc.enable()
      gc.collect()

      gc.disable()
      start = time.time()
      future = manager.h2d(
          src_offsets_major_dim=offsets,
          dst_offsets_major_dim=offsets,
          copy_sizes_major_dim=sizes,
      )
      future.Await()
      h2d_times.append(time.time() - start)
      gc.enable()
      gc.collect()

    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] KVCacheManager"
        f" d2h avg time: {np.median(d2h_times):.6f} s (median)"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] KVCacheManager"
        f" h2d avg time: {np.median(h2d_times):.6f} s (median)"
    )

    # Benchmark JAX Pinned Host Baseline
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Benchmarking JAX Pinned"
        " Host Baseline..."
    )
    jax_pinned_d2h_times = []
    jax_pinned_h2d_times = []
    for _ in range(num_iterations):
      gc.disable()
      start = time.time()
      self.jax_pinned_dst_arrs = []
      for j in range(num_layers):
        self.jax_pinned_dst_arrs.append(
            jax.device_put(self.src_arrs[j], pinned_host_sharding)
        )
      jax.block_until_ready(self.jax_pinned_dst_arrs)
      jax_pinned_d2h_times.append(time.time() - start)
      gc.enable()
      gc.collect()

      gc.disable()
      start = time.time()
      self.jax_pinned_tpu_dst_arrs = []
      for j in range(num_layers):
        self.jax_pinned_tpu_dst_arrs.append(
            jax.device_put(self.dst_arrs[j], tpu_sharding)
        )
      jax.block_until_ready(self.jax_pinned_tpu_dst_arrs)
      jax_pinned_h2d_times.append(time.time() - start)
      gc.enable()
      gc.collect()

    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] JAX Pinned D2H avg"
        f" time: {np.median(jax_pinned_d2h_times):.6f} s (median)"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] JAX Pinned H2D avg"
        f" time: {np.median(jax_pinned_h2d_times):.6f} s (median)"
    )

    # Benchmark JAX Standard Baseline (non-pinned NumPy)
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] Benchmarking JAX"
        " Standard Baseline..."
    )
    jax_std_d2h_times = []
    jax_std_h2d_times = []
    self.std_host_numpy_arrs = []
    for _ in range(num_layers):
      self.std_host_numpy_arrs.append(
          np.zeros(shape, dtype=np.float32).astype(dtype)
      )
    for _ in range(num_iterations):
      self.src_arrs = [tpu_mutate_fn(arr) for arr in self.src_arrs]
      jax.block_until_ready(self.src_arrs)
      gc.disable()
      start = time.time()
      self.jax_std_dst_arrs = []
      for j in range(num_layers):
        self.jax_std_dst_arrs.append(jax.device_get(self.src_arrs[j]))
      jax_std_d2h_times.append(time.time() - start)
      gc.enable()
      gc.collect()

      gc.disable()
      start = time.time()
      self.jax_std_tpu_dst_arrs = []
      for j in range(num_layers):
        self.jax_std_tpu_dst_arrs.append(
            jax.device_put(self.std_host_numpy_arrs[j], tpu_sharding)
        )
      jax.block_until_ready(self.jax_std_tpu_dst_arrs)
      jax_std_h2d_times.append(time.time() - start)
      gc.enable()
      gc.collect()

    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] JAX Standard D2H avg"
        f" time: {np.median(jax_std_d2h_times):.6f} s (median)"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] JAX Standard H2D avg"
        f" time: {np.median(jax_std_h2d_times):.6f} s (median)"
    )

    # Calculate bandwidth
    element_size = jnp.empty((), dtype=dtype).nbytes
    total_bytes = np.prod(shape) * element_size * num_layers

    mgr_d2h_bw = total_bytes / np.median(d2h_times) / (1024 * 1024 * 1024)
    mgr_h2d_bw = total_bytes / np.median(h2d_times) / (1024 * 1024 * 1024)

    jax_pinned_d2h_bw = (
        total_bytes / np.median(jax_pinned_d2h_times) / (1024 * 1024 * 1024)
    )
    jax_pinned_h2d_bw = (
        total_bytes / np.median(jax_pinned_h2d_times) / (1024 * 1024 * 1024)
    )
    jax_std_d2h_bw = (
        total_bytes / np.median(jax_std_d2h_times) / (1024 * 1024 * 1024)
    )
    jax_std_h2d_bw = (
        total_bytes / np.median(jax_std_h2d_times) / (1024 * 1024 * 1024)
    )

    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] KVCacheManager"
        f" D2H bandwidth: {mgr_d2h_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] KVCacheManager"
        f" H2D bandwidth: {mgr_h2d_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] JAX Pinned D2H"
        f" bandwidth: {jax_pinned_d2h_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] JAX Pinned H2D"
        f" bandwidth: {jax_pinned_h2d_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] JAX Standard D2H"
        f" bandwidth: {jax_std_d2h_bw:.3f} GB/s"
    )
    print(
        f"[{dtype}, {num_layers} layers, shape={shape}] JAX Standard H2D"
        f" bandwidth: {jax_std_h2d_bw:.3f} GB/s"
    )

    log_telemetry(
        "kv_cache_manager_perf_test_large_shape",
        dtype,
        num_layers,
        shape,
        d2h_times,
        h2d_times,
    )


if __name__ == "__main__":
  absltest.main()
