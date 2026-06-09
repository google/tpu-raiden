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

"""Real KV-cache spec from the Qwen3-32B TP=8 fp8 disaggregated run.

These values are taken verbatim from a production disagg run (see
`scripts/tmp/.../prefill.log` line `Init kv-cache`):

    num_total_layers      = 64
    regular_attn_shape    = (num_blocks, (128, 8, 4, 128))
    regular_attn_sharding = NamedSharding(
        mesh=Mesh('data': 1, 'model': 8), spec=P('data', None, 'model'))
    regular_attn_dtype    = float8_e4m3fn
    block_size            = 128
    max_blocks/req        = 72   (max_model_len 9216 / block_size 128)

Each request moves up to `max_blocks` blocks; the steady-state transfer in the
benchmark moves 64 blocks => 64 layers * 64 blocks * (128*8*4*128) * 1 byte
= 2 GiB, matching the production `h2h_bytes`/`h2d_bytes` of 2147483648.
"""

import jax
import jax.numpy as jnp
import numpy as np

# --- Real spec constants (do not change without re-deriving from a run) -------
NUM_LAYERS = 64
# Per-block shape: (block_size, kv_heads_dim(8), 4, head_dim(128)). The full
# per-layer array is (num_blocks,) + PER_BLOCK_SHAPE.
PER_BLOCK_SHAPE = (128, 8, 4, 128)
BLOCK_SIZE = 128
DTYPE = jnp.float8_e4m3fn
# Mesh + partition spec. Real run: mesh ('data': 1, 'model': 8),
# spec P('data', None, 'model') over the 5-d array => the size-8 dim (axis 2)
# is sharded across the 8 'model' devices; everything else replicated.
MESH_AXIS_NAMES = ("data", "model")
PARTITION_SPEC = ("data", None, "model", None, None)

# Bytes moved per layer per block, summed over all shards (the whole array):
BYTES_PER_LAYER_PER_BLOCK = int(np.prod(PER_BLOCK_SHAPE))  # fp8 => 1 byte/elem


def bytes_for(num_blocks: int) -> int:
    """Total KV bytes (all layers, all shards) for `num_blocks` blocks."""
    return NUM_LAYERS * num_blocks * BYTES_PER_LAYER_PER_BLOCK


def build_sharding(num_devices: int = 8) -> jax.sharding.NamedSharding:
    """Build the real NamedSharding over `num_devices` local TPU devices."""
    devices = jax.devices("tpu")
    if len(devices) < num_devices:
        raise AssertionError(
            f"Need {num_devices} TPU devices, found {len(devices)}")
    device_array = np.array(devices[:num_devices]).reshape(1, num_devices)
    mesh = jax.sharding.Mesh(device_array, MESH_AXIS_NAMES)
    spec = jax.sharding.PartitionSpec(*PARTITION_SPEC)
    return jax.sharding.NamedSharding(mesh, spec)


def gen_blocks(seed: int, layer_idx: int,
               block_ids: jax.Array) -> jax.Array:
    """Deterministic fp8 content for the given (layer, block) ids.

    Content is keyed per (seed, layer, block) so the SAME bytes are produced on
    the producer (to fill its source) and on the consumer (to regenerate the
    expected reference for verification) -- without any source data crossing the
    wire. fp8 has no native RNG, so we draw bf16 uniforms and cast.

    Returns an array of shape (len(block_ids),) + PER_BLOCK_SHAPE.
    """
    base = jax.random.fold_in(jax.random.key(seed), layer_idx)

    def one(block_id):
        key = jax.random.fold_in(base, block_id)
        return jax.random.uniform(key, PER_BLOCK_SHAPE, dtype=jnp.bfloat16)

    return jax.vmap(one)(block_ids).astype(DTYPE)


def make_src_caches(seed: int, kv_cache_num_blocks: int, registered_ids,
                    sharding: jax.sharding.NamedSharding) -> list[jax.Array]:
    """Producer source caches: a large `kv_cache_num_blocks`-block pool per layer
    (matching the real KV cache's huge dim_0), with deterministic content written
    only at `registered_ids` (the blocks any request may pull) and zeros elsewhere.
    """
    reg = jnp.asarray(sorted({int(b) for b in registered_ids}), dtype=jnp.int32)
    shape = (kv_cache_num_blocks,) + PER_BLOCK_SHAPE
    caches = []
    for layer in range(NUM_LAYERS):
        pool = jnp.zeros(shape, DTYPE)
        if reg.size > 0:
            pool = pool.at[reg].set(gen_blocks(seed, layer, reg))
        caches.append(jax.device_put(pool, sharding))
    jax.block_until_ready(caches)
    return caches


def make_dst_caches(kv_cache_num_blocks: int,
                    sharding: jax.sharding.NamedSharding) -> list[jax.Array]:
    """Consumer destination caches: zero-filled large pool, one per layer."""
    shape = (kv_cache_num_blocks,) + PER_BLOCK_SHAPE
    caches = [
        jax.device_put(jnp.zeros(shape, DTYPE), sharding)
        for _ in range(NUM_LAYERS)
    ]
    jax.block_until_ready(caches)
    return caches


def expected_blocks_uint8(seed: int, layer_idx: int, block_ids) -> np.ndarray:
    """Regenerated reference bytes (uint8) for (layer, block_ids)."""
    blocks = gen_blocks(seed, layer_idx,
                        jnp.asarray(block_ids, dtype=jnp.int32))
    return np.asarray(jax.lax.bitcast_convert_type(blocks, jnp.uint8))


def read_blocks_uint8(dst_layer: jax.Array, block_ids) -> np.ndarray:
    """Read only the given blocks of a device cache layer to host as uint8."""
    sel = dst_layer[jnp.asarray(block_ids, dtype=jnp.int32)]
    return np.asarray(jax.lax.bitcast_convert_type(sel, jnp.uint8))


def request_block_map(name: str, num_blocks: int, kv_cache_num_blocks: int,
                      request_idx: int):
    """Per-request block ids drawn from the [0, kv_cache_num_blocks) pool.

    Each request occupies a DISTINCT, non-overlapping window of the pool (so no
    two requests register/read/write the same blocks -- mirroring how vLLM
    allocates distinct physical blocks per request). Returns
    (registered_remote, requested_remote, local):

    - registered_remote: blocks the producer makes available (notify_for_read).
    - requested_remote:  per-pull remote_block_ids (== local length).
    - local:             per-pull local_block_ids (where they land).

    Modes (within the request's window at base = request_idx * window):
      identity  contiguous remote == local              (the real path)
      reorder   contiguous remote reversed -> local      (pure permutation)
      sparse    non-contiguous (even) remote, reversed -> compact local
                (window is 2*num_blocks so the gaps fit)
    """
    window = 2 * num_blocks if name == "sparse" else num_blocks
    base = request_idx * window
    if base + window > kv_cache_num_blocks:
        raise ValueError(
            f"request {request_idx} needs pool blocks [{base}, {base + window}) "
            f"but kv_cache_num_blocks={kv_cache_num_blocks}; increase "
            f"--kv-cache-num-blocks to >= {(request_idx + 1) * window}")
    if name == "identity":
        ids = list(range(base, base + num_blocks))
        return ids, ids, ids
    if name == "reorder":
        win = list(range(base, base + num_blocks))
        return win, list(reversed(win)), win
    if name == "sparse":
        reg = list(range(base, base + window))            # whole window available
        picked = [base + 2 * i for i in range(num_blocks)]  # even => non-contiguous
        local = list(range(base, base + num_blocks))
        return reg, list(reversed(picked)), local
    raise ValueError(f"unknown mapping: {name}")
