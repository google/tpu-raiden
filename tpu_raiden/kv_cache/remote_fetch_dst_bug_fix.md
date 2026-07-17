# Remote-fetch dst host-block bug — root cause + engine fix

Found 2026-07-06 while running the two-node **global prefix-cache** test
(`scripts/two_node_prefix_cache/`), fixed the same day.

## The bug
In the *global* path (consumer discovers a block via `lookup(enable_global=True)`), the
remote fetch crashed the transfer with:
```
raw_buffer_transport.cc:494] ProcessSingleRequest failed:
  OUT_OF_RANGE: Chunk out of bounds. Chunk ptr: … (host_base − 65536), Host size: 4194304
```
i.e. the **destination host block offset was −1**.

## Root cause
The consumer has no local host block for a freshly-discovered remote block, so the
global-lookup fallback returns a REMOTE `RaidenBlockID` with **`host_block_id = -1`**
(`kv_cache_store.cc:146`, ignoring the registry's stored `block_id`). `FetchRemote`
(`kv_cache_store.cc:351`) then used that `-1` as the **local destination** host block id,
and the receiving worker wrote at `host_base + (-1)*block_bytes` → out of bounds. The
completion path also keyed everything by `block_id`, which cannot work when the id is a
`-1` placeholder that must change.

## The fix — receiver auto-allocates the dst, propagates the id back (keyed by hash)
Design (per the intended flow: *the block transport allocates a host block for the H2H
transfer, and the allocated id is propagated to the KVCacheStore to update
`RaidenBlockID.host_block_id`; the completion is keyed by block hash, not block id*):

| # | File | Change |
|---|------|--------|
| 1 | `transport/block_transport.cc` `HandleIncomingPush` | any received `dst == -1` → `AllocateBlocks(1)` on the receiver; then `UpdatePlanDstBlocks(uuid, allocated)` so `GetBlockChunks` can resolve the (previously `-1`) schedule entries. |
| 2 | `transport/block_transport.h` + `kv_cache/kv_cache_manager_base.{h,cc}` | new `BlockTransportDelegate::UpdatePlanDstBlocks(uuid, dst_ids)`; base impl rewrites the k-th placeholder (`-1`) `dst_block_id` in each shard's schedule to the k-th worker-allocated id. |
| 3 | `rpc/raiden_service.proto` | `ControlRequest.transfer_uuid = 13` (so the completion carries the fetch uuid). |
| 4 | `kv_cache/kv_cache_listener.cc` | `TRANSFER_COMPLETED` now sets `transfer_uuid`. |
| 5 | `kv_cache/raiden_controller_embedded.{h,cc}` | `pending_fetches_` is now `uuid → ORDERED hashes`; `block_completion_counts_` keyed by **hash**; on completion map `completed_block_ids[i] → hashes[i]`, emit `FetchCompletionItem{hash, host_block_id = worker-allocated}`. |
| 6 | `kv_cache/kv_cache_store.{h,cc}` | fetch tracked by `hash_to_fetch_` (was `block_to_fetch_` by id); `CompletionPollerLoop` **sets** `block->host_block_id` from the completion (dropped the old `== host_block_id` check). |

Net path: consumer inserts REMOTE `host_block_id=-1` → producer pushes with dst `-1` →
**receiver auto-allocates** a real host block, patches its plan, writes there →
`OnBlocksReceived(allocated)` → listener `TRANSFER_COMPLETED(uuid, allocated)` → controller
re-keys `uuid+order → hash` → store sets `RaidenBlockID.host_block_id = allocated`,
`REMOTE→HOST`. A later `Load` reads that host block.

## Verification
- Two-node prefix cache test **PASS with NO worker workaround** — the consumer now passes
  the discovered REMOTE `RaidenBlockID` (`host_block_id=-1`) straight through, and the
  engine auto-allocates. `RAW-BYTE exact for all 896 (layer, live-block)` via
  Save→FetchRemote→Load.
- No regression: `kv_cache_store_test.cc` **27/27**, `api/jax/kv_cache_store_test.py`
  **17/17** (these exercise the fixed-dst path, which still works — completion carries the
  fixed dst as the "allocated" id).

## Known limitation
For a *single* NUMA listener the auto-allocated id is unambiguous. With **>1 listener**,
each sub-manager would `AllocateBlocks` independently and could pick **different** host
block ids for the same logical block, while `RaidenBlockID` holds one `host_block_id`.
The two-node test uses `num_listeners=1`; multi-listener auto-alloc needs a coordinated
allocation (or a store-side reservation) — noted, not yet addressed.

## Where else this is recorded
- Memory: `global-prefix-cache-remote-fetch.md` ("ENGINE FIX: remote-fetch dst
  auto-allocation").
- `scripts/two_node_prefix_cache/README.md` (§ the fix, updated from the earlier
  worker-workaround description).

### The multi-listener flake — and how to fix it

> **FIXED (2026-07-06):** implemented mitigation 2 below in the engine —
> `KVCacheListener`'s ctor now **retries `bind()` on `EADDRINUSE`** (100 × 100 ms) and,
> on give-up, **fails gracefully** (closes the fd, sets `stopping_` so
> `is_active() == false`, returns) instead of `LOG(FATAL)`; `listen()` handles failure the
> same way (`kv_cache_listener.cc`, added `<chrono>`). So one port collision now fails only
> that test, never the whole suite, and the retry lets a concurrently-freed port succeed.
> **Verified: 4/4 clean back-to-back JAX suite runs** (was aborting ~half the time).