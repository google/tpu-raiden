"""Micro-bench: full vs partial D2H through the SAME fast raw_transfer API.

Goal: determine whether the 100x staging-path slowdown is caused by *partial*
(sub-buffer offset) copies, or by the KVCacheManager D2hTo path specifically.
If partial here is ~as fast as full, partial is NOT the problem.
"""
import time
import jax
import jax.numpy as jnp
import numpy as np
import raw_transfer

NUM_LAYERS = 16
MAJOR = 128          # device buffer has 128 blocks
COPY = 64            # we copy 64 of them
OFFSET = 32          # at a non-zero offset (forces is_partial)
SHAPE = (MAJOR, 128, 8, 2, 128)
DT = jnp.int32
ITERS = 20

devs = jax.devices("tpu")
n = len(devs)
print(f"{n} TPU devices; shape={SHAPE} dtype={DT}")
mesh = jax.make_mesh((1, n), ("data", "model"))
spec = jax.sharding.PartitionSpec(None, None, "model")
dev_sh = jax.sharding.NamedSharding(mesh, spec)
host_sh = jax.sharding.NamedSharding(mesh, spec, memory_kind="pinned_host")


def mk(sharding):
    a = np.arange(int(np.prod(SHAPE)), dtype=np.int32).reshape(SHAPE)
    return jax.device_put(a, sharding)


src = [mk(dev_sh) for _ in range(NUM_LAYERS)]
dst = [mk(host_sh) for _ in range(NUM_LAYERS)]
jax.block_until_ready(src)
per_shard_block = 128 * (8 // n) * 2 * 128 * 4
print(f"per-shard block bytes = {per_shard_block} (4K-aligned: {per_shard_block % 4096 == 0})")


def run(partial):
    disp, awa = [], []
    for _ in range(ITERS):
        t0 = time.time()
        if partial:
            f = raw_transfer.transfer_d2h_batch_async(
                src, dst,
                src_offsets_major_dim=[OFFSET],
                dst_offsets_major_dim=[0],
                copy_sizes_major_dim=[COPY],
            )
        else:
            f = raw_transfer.transfer_d2h_batch_async(src, dst)
        t1 = time.time()
        f.Await()
        t2 = time.time()
        disp.append(t1 - t0)
        awa.append(t2 - t1)
    blocks = MAJOR if not partial else COPY
    gb = NUM_LAYERS * blocks * per_shard_block * n / 1e9
    med_total = np.median([d + a for d, a in zip(disp, awa)])
    print(f"{'PARTIAL' if partial else 'FULL   '}: "
          f"dispatch={np.median(disp)*1e3:.3f}ms await={np.median(awa)*1e3:.3f}ms "
          f"total={med_total*1e3:.3f}ms  bytes={gb:.3f}GB  BW={gb/med_total:.1f} GB/s")


run(False)
run(True)
run(False)
run(True)
