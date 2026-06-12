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

"""Single-node micro-benchmark to determine whether the staging D2H
(`TransferEngine.stage_d2h`, i.e. the per-(layer,shard) `D2hToHostSlot` path)
is SYNCHRONOUS or ASYNCHRONOUS.

Diagnostic: `stage_d2h` issues the device->host copies and returns a future.
  - If the copy is ASYNC  -> the issue (dispatch) call returns immediately and
    the real work shows up in `future.Await()` (issue << await).
  - If the copy is SYNC   -> the issue call BLOCKS until the copy is done, so
    `future.Await()` finds nothing left to wait on (issue >> await ~ 0).

Run (single TPU host, 8 devices):
  cd /mnt/disks/jcgu/code/ullm/raiden/tpu-raiden
  PYTHONPATH=. python one_node_test/mesaure_stage_d2h.py
"""

import time

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from api.jax import transfer_engine


class MeasureStageD2HTest(absltest.TestCase):

    def test_stage_d2h_bandwidth(self):
        try:
            devices = jax.devices("tpu")
        except RuntimeError:
            self.skipTest("No TPU devices found")
            return

        if not devices:
            self.skipTest("No TPU devices found")
            return

        num_devices = len(devices)
        print(f"Found {num_devices} TPU devices")

        num_layers = 64
        shape = (1024, 128, 8, 2, 128)
        dtype = jnp.bfloat16

        # Mesh: shard the size-8 dim (axis 2) across the `model` axis, like the
        # real KV cache.
        device_array = np.array(devices).reshape((1, num_devices))
        mesh = jax.sharding.Mesh(device_array, ("data", "model"))
        spec = jax.sharding.PartitionSpec(None, None, "model")
        tpu_sharding = jax.sharding.NamedSharding(mesh, spec)

        print("Allocating KV cache...")
        # Deterministic NON-ZERO source so the staged bytes can be verified
        # (zeros would be indistinguishable from an un-copied buffer).
        key = jax.random.key(0)
        src_caches = []
        for i in range(num_layers):
            vals = jax.random.uniform(
                jax.random.fold_in(key, i), shape, dtype=dtype)
            src_caches.append(jax.device_put(vals, tpu_sharding))
        jax.block_until_ready(src_caches)
        print("KV cache allocated")

        # TransferEngine setup
        num_blocks = 1024
        engine = transfer_engine.TransferEngine(
            kv_caches=src_caches,
            local_control_port=0,
            max_blocks=num_blocks,
            num_slots=1,
            unsafe_skip_buffer_lock=True,
        )

        block_ids = list(range(num_blocks))

        transfer_blocks = 64
        transfer_block_ids = block_ids[:transfer_blocks]
        # num "spans" = layers x shards (one D2hToHostSlot call each).
        num_spans = num_layers * num_devices

        # Warmup + VERIFY (round-trip). The host staging holds raw device bytes
        # in device-PHYSICAL (tiled) layout, so we can't compare it to logical
        # numpy directly. Instead verify via a round-trip: D2H blocks [0..T)
        # into slot 0, then H2D those staged bytes back into device blocks
        # [T..2T). In LOGICAL space (tiling-agnostic) device[T..2T) must then
        # equal the untouched source device[0..T) -- proving the D2H staged the
        # correct data.
        T = transfer_blocks
        print("Warmup + round-trip verify...")
        future, _kv, host_views, _ = engine.stage_d2h(  # type: ignore
            slot_idx=0, num_blocks=T, block_ids=list(range(T)))
        future.Await()

        self.assertEqual(
            len(host_views), num_spans,
            f"expected {num_spans} host views, got {len(host_views)}")
        # Sanity: the staged host buffers are non-zero (the copy happened).
        sample_spans = [0, num_spans // 2, num_spans - 1]
        self.assertTrue(
            all(any(bytes(host_views[i])) for i in sample_spans),
            "staged data is all-zero -> D2H did not copy")

        # H2D the staged bytes back into a disjoint device range [T..2T).
        engine.commit_h2d(  # type: ignore
            slot_idx=0, num_blocks=T, local_block_ids=list(range(T, 2 * T)))

        sample_layers = sorted({0, num_layers // 2, num_layers - 1})
        mismatches = 0
        for layer in sample_layers:
            arr = np.asarray(src_caches[layer])  # logical (gathers shards)
            if not np.array_equal(arr[T:2 * T], arr[0:T]):
                mismatches += 1
                bad = int(np.sum(arr[T:2 * T] != arr[0:T]))
                print(f"  MISMATCH layer={layer}: {bad} differing elements")
        print(f"Round-trip verify: checked {len(sample_layers)} layers, "
              f"mismatches={mismatches}")
        self.assertEqual(
            mismatches, 0,
            "round-trip D2H->H2D != source (staged data is wrong)")
        print("Verify OK: D2H->H2D round-trip matches source device data")

        print("Measuring...")
        iterations = 5
        dispatch_times = []
        await_times = []
        total_bytes = 0

        for _ in range(iterations):
            start = time.perf_counter()
            future, _, _, total_bytes = engine.stage_d2h(  # type: ignore
                slot_idx=0, num_blocks=transfer_blocks,
                block_ids=transfer_block_ids)
            dispatch_time = time.perf_counter() - start

            await_start = time.perf_counter()
            future.Await()
            await_time = time.perf_counter() - await_start

            dispatch_times.append(dispatch_time)
            await_times.append(await_time)

        avg_dispatch = float(np.mean(dispatch_times))
        avg_await = float(np.mean(await_times))
        avg_total = avg_dispatch + avg_await
        bw = total_bytes / avg_total / (1024 ** 3)

        print("=" * 60)
        print(f"Total Bytes:          {total_bytes} "
              f"({total_bytes / 2**30:.2f} GiB)")
        print(f"Spans (layers*shards): {num_spans}")
        print(f"Avg Issue Time:       {avg_dispatch * 1e3:.3f} ms  "
              f"({avg_dispatch / num_spans * 1e3:.4f} ms/span)")
        print(f"Avg Completion Time:  {avg_await * 1e3:.3f} ms")
        print(f"Avg Total Time:       {avg_total * 1e3:.3f} ms")
        print(f"Bandwidth:            {bw:.3f} GiB/s")

        # Verdict: which phase dominates?
        ratio = avg_dispatch / avg_total if avg_total > 0 else 0.0
        if ratio > 0.8:
            verdict = ("SYNCHRONOUS  (issue blocks; await ~ 0) "
                       "-> D2H is NOT async DMA")
        elif ratio < 0.2:
            verdict = ("ASYNCHRONOUS (issue returns immediately; "
                       "work in await) -> async DMA")
        else:
            verdict = "MIXED"
        print(f"Issue fraction:       {ratio * 100:.1f}% of total")
        print(f"VERDICT:              {verdict}")
        print("=" * 60)


if __name__ == "__main__":
    absltest.main()
