"""Isolation bench: the engine's REAL staging D2H (stage_d2h) vs the fast
raw_transfer path, over identical device arrays, single-threaded, no network.

If stage_d2h is slow here (uncontended), the staging code path is the culprit.
If it's fast here, the production slowdown is concurrency/contention, not the
copy path itself.
"""
import time
import jax
import jax.numpy as jnp
import numpy as np
import raw_transfer
from api.jax.transfer_engine import TransferEngine

NUM_LAYERS = 64
MAJOR = 128          # device buffer blocks
NUM_BLOCKS = 64      # copy 64 blocks (matches real)
MAX_BLOCKS = 72
NUM_SLOTS = 2
SHAPE = (MAJOR, 128, 8, 2, 128)
DT = jnp.int32
ITERS = 10

devs = jax.devices("tpu")
n = len(devs)
print(f"{n} TPU devices; {NUM_LAYERS} layers shape={SHAPE}")
mesh = jax.make_mesh((1, n), ("data", "model"))
spec = jax.sharding.PartitionSpec(None, None, "model")
dev_sh = jax.sharding.NamedSharding(mesh, spec)
host_sh = jax.sharding.NamedSharding(mesh, spec, memory_kind="pinned_host")


def mk(sharding):
    a = np.arange(int(np.prod(SHAPE)), dtype=np.int32).reshape(SHAPE)
    return jax.device_put(a, sharding)


kv_caches = [mk(dev_sh) for _ in range(NUM_LAYERS)]
jax.block_until_ready(kv_caches)

block_bytes = NUM_BLOCKS * (128 * (8 // n) * 2 * 128 * 4)
total_gb = NUM_LAYERS * n * block_bytes / 1e9
print(f"copying {NUM_BLOCKS} blocks/layer; total ≈ {total_gb:.3f} GB")

# ---- Path A: the engine's real staging D2H ----
eng = TransferEngine(kv_caches=kv_caches, local_control_port=0,
                     max_blocks=MAX_BLOCKS, num_slots=NUM_SLOTS,
                     unsafe_skip_buffer_lock=True)
block_ids = list(range(NUM_BLOCKS))
# warm
fut, *_ = eng.stage_d2h(slot_idx=0, num_blocks=NUM_BLOCKS, block_ids=block_ids)
fut.Await()
disp, awa = [], []
for _ in range(ITERS):
    t0 = time.time()
    fut, *_ = eng.stage_d2h(slot_idx=0, num_blocks=NUM_BLOCKS, block_ids=block_ids)
    t1 = time.time()
    fut.Await()
    t2 = time.time()
    disp.append(t1 - t0); awa.append(t2 - t1)
md = np.median([d + a for d, a in zip(disp, awa)])
print(f"STAGE_D2H (engine path): dispatch={np.median(disp)*1e3:.2f}ms "
      f"await={np.median(awa)*1e3:.2f}ms total={md*1e3:.2f}ms BW={total_gb/md:.1f} GB/s")

# ---- Path B: fast raw_transfer of the same arrays (partial, into pinned dst) ----
dst = [mk(host_sh) for _ in range(NUM_LAYERS)]
jax.block_until_ready(dst)
disp, awa = [], []
for _ in range(ITERS):
    t0 = time.time()
    f = raw_transfer.transfer_d2h_batch_async(
        kv_caches, dst,
        src_offsets_major_dim=[0], dst_offsets_major_dim=[0],
        copy_sizes_major_dim=[NUM_BLOCKS])
    t1 = time.time()
    f.Await()
    t2 = time.time()
    disp.append(t1 - t0); awa.append(t2 - t1)
md = np.median([d + a for d, a in zip(disp, awa)])
print(f"FAST raw_transfer    : dispatch={np.median(disp)*1e3:.2f}ms "
      f"await={np.median(awa)*1e3:.2f}ms total={md*1e3:.2f}ms BW={total_gb/md:.1f} GB/s")
