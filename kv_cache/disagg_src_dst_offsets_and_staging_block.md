# Disagg KV Cache: `src_offsets` / `dst_offsets`, Staging Blocks, and H2H Block-ID Flow

This note explains how `DisaggKVCacheManagerBase` drives block ids through a
disaggregated prefill → decode KV transfer, given that submitted requests carry
**no** `block_ids` — only `dst_offsets` / `sizes` (and `src_offsets` for H2D).

All line references are to `kv_cache/disagg_kv_cache_manager_base.cc` unless
noted otherwise.

---

## 1. Request shape at submission time

A transfer request (`DisaggTransferRequest`, see
`disagg_kv_cache_manager_base.h`) carries:

- `type` — `kPrefillD2H`, `kDecodeH2D`, `kH2HWrite`, `kH2HRead`
- `src_offsets`, `dst_offsets`, `sizes` — **major-dim offsets** (multiples of
  `block_size_`) and copy sizes; the *chunk plan*
- `block_ids` — **empty at submission**; derived/refreshed by the manager
- `peer`, `entity_id`, `pull_mode`, `callback`

A "staging block id" is just a host-buffer block index. The relationship between
a major-dim offset and a block id is always:

```
block_id = offset / block_size_
offset   = block_id * block_size_
```

Multi-block chunks (where `sizes[i]` is a multiple of `block_size_`) are first
expanded into unit-block entries by `ExpandChunksToBlocks(req, block_size_)`
so every downstream stage sees one block per entry (base.cc:75-90).

---

## 2. Prefill D2H: the host slots are caller-chosen, not allocated

On the prefill/producer side there is **no allocator call** for D2H. The caller
supplies `dst_offsets`, and the manager copies straight into them:

- Orchestrator pushes the `kPrefillD2H` request to the local work queue
  (base.cc:298-301).
- `LocalTransferLoop` runs `D2h(req.src_offsets, req.dst_offsets, req.sizes)`
  (base.cc:553) — a direct device→host copy into the caller's offsets.

After the D2H completes, the prefill's **local staging block ids** are *derived*
from the offsets (base.cc:347-350):

```cpp
req.block_ids.clear();
for (int64_t offset : req.dst_offsets)
  req.block_ids.push_back(offset / block_size_);
```

> Contrast: the non-disagg store path `KVCacheManagerBase::D2hAutoAllocate`
> (`kv_cache_manager_base.cc:337`) *does* allocate, calling
> `AllocateBlocks → block_manager_->Allocate(num_blocks, entity_id, lock=true)`
> and synthesizing `dst_offsets = assigned_block_id * block_size_`. In the disagg
> manager, `block_manager_` is used only on the **decode/receiver** side
> (see §4) and for `Unlock` on completion.

---

## 3. H2H PUSH (default): source ids vs. destination ids

H2H involves two distinct sets of block ids.

**Source ids (prefill-local):** after D2H, the request is flipped to
`kH2HWrite` and pushed to the H2H queue (base.cc:375-376) carrying the staging
ids from §2.

**Destination ids (decode-allocated):** the H2H loop sends the staged blocks via
`H2hWriteDirect(peer, req.block_ids, ...)` (base.cc:618). On the receiver, the
transport handles the Push by allocating *its own* host blocks and writing the
incoming bytes there (`transport/block_transport.cc:227-242`):

```cpp
if (header.op == 1) {  // Push
  TF_ASSIGN_OR_RETURN(allocated_ids,
      delegate_->AllocateBlocks(header.num_blocks, /*entity_id=*/0));
  WriteExact(client_fd, allocated_ids.data(), ...);   // returned to sender
  // ... write into base_host_ptr + dst_id * bytes_per_block
}
```

These receiver-allocated ids are returned to the sender and **overwrite**
`req.block_ids` (base.cc:622).

### Why the overwrite matters

On `kH2hComplete`, the orchestrator refreshes from the completed event before
notifying the decode (base.cc:425-442):

```cpp
req.block_ids = event.request.block_ids;     // receiver-allocated ids
// NOTIFY_COMPLETE:<req_id>:<id0,id1,...>
```

Local staging ids and receiver ids **coincide only when the receiver allocates
sequentially** (e.g. `parallelism == 1`). Under concurrency they differ, so the
ids must be taken from the completed event/peer — otherwise the decode reads the
wrong host blocks. (This was the disagg parallelism corruption bug: NOTIFY had
been sending the local staging ids instead of the receiver-allocated ids.)

---

## 4. Decode H2D: consuming the advertised ids

The decode's `kDecodeH2D` request also arrives with no `src_offsets`. When the
`NOTIFY_COMPLETE` block ids are present, the decode turns them into H2D source
offsets (base.cc:309-313):

```cpp
existing.src_offsets.clear();
for (int bid : existing.block_ids)
  existing.src_offsets.push_back(bid * block_size_);
existing.dst_offsets = req.dst_offsets;   // decode's own device destination
```

When the H2D finishes, the orchestrator `Unlock()`s exactly those block ids on
the decode's `block_manager_`, under `block_manager_mutex_` because the
transport's receiver threads concurrently `Allocate()` (base.cc:391-403).

---

## 5. End-to-end id handoff (PUSH)

```
caller dst_offsets ──/block_size──▶ prefill staging block_ids   (H2H push SOURCE)
                                            │ H2hWriteDirect
receiver AllocateBlocks ───────────▶ decode host block_ids        (H2H push DEST)
                                            │ NOTIFY_COMPLETE (ZMQ)
                                     decode src_offsets = id*block_size  (H2D SOURCE)
                                            │ H2d() then Unlock(block_ids)
```

---

## 6. PULL mode (`kH2HRead`) — the mirror image

- Prefill stages via D2H but does **not** transfer; it advertises its staging
  ids in `NOTIFY_READY` and keeps the request pending so the staging buffer is
  not reused before the decode pulls (base.cc:351-371).
- The decode issues `H2hReadDirect(peer, remote_ids, ...)`, which allocates the
  **local** decode blocks the pulled data lands in and returns those local ids
  (base.cc:628-632; `transport/block_transport.cc:399-408`).
- On the decode's H2D completion in pull mode, the decode sends `PULL_COMPLETE`
  back so the prefill can release its staging buffer (base.cc:381-390).

---

## Key references

| Concern | Location |
| --- | --- |
| Chunk → unit-block expansion | `disagg_kv_cache_manager_base.cc:75-90` |
| Prefill D2H dispatch | `:298-301`, `:553` |
| Staging id derivation (`offset/block_size`) | `:347-350` |
| H2H push, dest id overwrite | `:615-623` |
| Receiver-id refresh + NOTIFY_COMPLETE | `:425-442` |
| Receiver allocation on Push | `transport/block_transport.cc:227-242` |
| Decode H2D src from advertised ids | `:309-313` |
| Unlock on H2D complete | `:391-403` |
| PULL: NOTIFY_READY / staging hold | `:351-371` |
| PULL: H2hReadDirect local alloc | `:628-632`, `transport/block_transport.cc:399-408` |
| Auto-allocate (non-disagg store path) | `kv_cache_manager_base.cc:337-388` |
