# tpu-raiden JAX APIs used by the KV host-offload connector

This maps every **tpu-raiden JAX API** called by
`tpu-inference/tpu_inference/offload/raiden_offload_connector.py` to its
**C++ implementation** and its **functionality**.

The connector uses two independent tpu-raiden subsystems:

- **`KVCacheStore`** — a *logical* LRU directory (`block_hash → RaidenId` locators),
  owned by the connector **scheduler**. Pure bookkeeping; no data movement.
- **`KVCacheManager`** — the *physical* transfer engine + host-RAM block pool,
  owned by the connector **worker**. Does the device↔host DMA and owns the
  `LogicalBlockManager` chunk pool.

Call chain for every API: `Python wrapper` (`tpu_raiden/api/jax/*.py`) →
`nanobind binding` (`tpu_raiden/frameworks/jax/tpu_raiden_jax_module.cc`) →
`C++ impl`.

## API → C++ implementation → functionality

| Component | JAX API (as called in connector) | nanobind binding | C++ implementation | Functionality |
|---|---|---|---|---|
| **KVCacheStore** (logical directory) | `KVCacheStore(capacity=num_cpu_chunks)` | `tpu_raiden_jax_module.cc:407` `nb::class_<KVCacheStoreWrapper>` init | `KVCacheStore(size_t)` (`kv_cache/kv_cache_store.cc`), backed by `LruCache` (`kv_cache/lru_cache.h`) | Create the logical LRU directory of capacity `num_cpu_chunks` chunks. |
| KVCacheStore | `kv_store.lookup(block_hashes)` | `:410` `"lookup"` | `KVCacheStore::Lookup` (`kv_cache_store.cc`) → `LruCache` lookup | Find the cached prefix: returns `[(hash, [RaidenId])]` for matched blocks, **halting at the first miss**. Drives the cache-hit / load decision. |
| KVCacheStore | `kv_store.insert([h], [[RaidenId]], on_host=True)` | `:434` `"insert"` | `KVCacheStore::Insert` (`kv_cache_store.cc`) → `LruCache` put + LRU eviction | Register a just-saved chunk (`hash → locator`). Returns `(all_inserted, evicted[])`; the evicted list lets the scheduler reclaim the evicted physical chunks. |
| KVCacheStore | `kv_store.pin(matched_hashes)` | `:469` `"pin"` | `KVCacheStore::Pin` (`kv_cache_store.cc:88`) → `LruCache::Pin` (`lru_cache.h`, `pin_count++`) | Protect matched hashes from LRU eviction while a request is loading/using them. **Refcounted**. |
| KVCacheStore | `kv_store.release(raiden_hashes)` | `:477` `"release"` | `KVCacheStore::Release` (`kv_cache_store.cc:101`) → `LruCache::Unpin` (`lru_cache.h`, `pin_count--`) | Drop the pin when the request finishes, making the entry evictable again. |
| **RaidenId** (locator value) | `RaidenId(job_name, job_replica_id, data_name, data_replica_idx)` | `:397` `nb::class_<RaidenId>` (+ `def_rw` fields) | `struct RaidenId` (`kv_cache/kv_cache_store.h:38`) | Value type identifying a stored KV chunk: which job/replica/data stream + the physical `data_replica_idx` (host chunk id). |
| **KVCacheManager** (physical engine) | `KVCacheManager(kv_caches=, local_control_port=0, host_blocks_to_allocate=num_cpu_chunks, unsafe_skip_buffer_lock=True)` | `:106` `nb::init<nb::list, optional<int>, optional<int>, bool, int>` | jax `KVCacheManager` ctor (`frameworks/jax/kv_cache_manager.cc`) → `InitSubManagers` → `KVCacheManagerWithTransfer` (`core/kv_cache_manager_with_transfer.cc`), which allocates the host pool via `LogicalBlockManager` (`kv_cache/logical_block_manager.cc`) | Build the transfer engine: bind device KV buffers and allocate the host-RAM staging/block pool (`host_blocks_to_allocate` blocks). |
| KVCacheManager | `raiden_manager.d2h_auto_allocate(src_blocks)` | `:153` `"d2h_auto_allocate"` | jax `KVCacheManager::D2hAutoAllocate` (`kv_cache_manager.cc:542`) → `KVCacheManagerBase::D2hAutoAllocate` (`kv_cache/kv_cache_manager_base.cc:570`) → `AllocateBlocks` = `LogicalBlockManager::Allocate(n, lock=true)` + `DispatchD2hChunks` (async PjRt device→host copy) | **Save path.** Allocate+lock host chunks and async-copy the device KV blocks into them. Returns `(allocated_chunk_ids, future)`. |
| KVCacheManager | `raiden_manager.h2d(local_src_chunks, local_dst_blocks)` | `:123` `"h2d"` | jax `KVCacheManager::H2d` → `KVCacheManagerBase::H2d` (`kv_cache_manager_base.cc:286`) → `DispatchH2dChunks` (async PjRt host→device copy) | **Load path.** Copy host chunks back into device KV blocks. Returns a `future`. |
| KVCacheManager | `raiden_manager.unlock_blocks(block_ids)` | `:269` `"unlock_blocks"` | jax `KVCacheManager::UnlockBlocks` (`kv_cache_manager.cc`) → per sub-manager `host_block_manager()->Unlock` = `LogicalBlockManager::Unlock` (`logical_block_manager.cc`, `is_locked=false`) | Release the lock on physical host chunks (after the store evicts them, or on a save failure) so `Allocate` can reuse them. |
| **RaidenFuture** (async handle) | `future.Await()` | `:84` `"Await"` | `RaidenFuture::Await` (`core/raiden_future.h:39`) → underlying `raiden::PjRtCopyFuture` | Block until the async D2H/H2D copy completes (used to confirm a save landed before recording it). |

## `RaidenId` — one type, three layers

`RaidenId` is the store's **value type**: the LRU maps
`block hash → std::vector<RaidenId>` (`kv_cache_store.h:100`), so each cached block
records the locator(s) of where its replica/shard lives —
`{job_name, job_replica_id, data_name, data_replica_idx}`. In the offload connector,
`data_replica_idx` is the **physical host chunk id**.

It appears at three layers, which are the **same logical type, not duplicates**:

| Layer | Where | Role |
|---|---|---|
| C++ struct | `kv_cache/kv_cache_store.h:38` | the real data type stored in the LRU; produced/consumed by `Lookup`/`Insert`/`Delete` |
| nanobind binding | `tpu_raiden_jax_module.cc:397` (`nb::class_<…RaidenId>`, `def_rw`) | exposes that struct to Python as `_impl.RaidenId` (read/write fields) |
| Python wrapper | `api/jax/kv_cache_store.py:21` (`class RaidenId`) | thin wrapper holding `self._impl`; `KVCacheStore.lookup` wraps raw ids, `.insert` unwraps `s._impl` |

The Python wrapper is thin (property forwarding + `__repr__`) and largely **redundant**
with the already-bound `_impl.RaidenId` — a candidate for consolidation (re-export
`_impl.RaidenId` directly). Separately, the connector defines its own `RaidenLocator`
dataclass; that one has a real reason to exist — it must be **Ray/pickle-serializable**
(nanobind `RaidenId` objects are not), so locators cross the scheduler↔worker boundary as
`RaidenLocator` and convert to `RaidenId` only at the store call via `to_raiden_id()`.

## `KVCacheStore` vs `KVCacheStoreInternal`

Two different C++ stores exist; the offload connector uses **`KVCacheStore`** (the logical
one). They are NOT interchangeable:

| | `KVCacheStore` (`kv_cache_store.h`) | `KVCacheStoreInternal` (`kv_cache_store_internal.h`) |
|---|---|---|
| Role | **Logical directory only** (metadata) | **All-in-one**: directory + physical pool + transfer |
| Data structure | `LRUCache<string hash, vector<RaidenId>>` | own `lru_list_`/`cache_map_` of `CacheEntry` (block ids + host buffers + insert future) |
| Owns physical blocks? | No | Yes — `LogicalBlockManager host_block_manager_` |
| Does transfers? | No — caller drives D2H/H2D separately | Yes — `LookupAndFetch` (lookup + H2D/remote pull) and `Insert` (D2H + register) in one call; is a `BlockTransportDelegate` with a `BlockTransport server_` |
| Distributed? | Single-node | Multi-node — `GlobalRegistryClient` + transport for cross-node fetch (`LookupAndFetchRemote`) |
| Hash type | `std::string` | `uint64_t` |
| Key API | `Lookup` / `Insert` / `Delete` / `Pin` / `Release` | `LookupAndFetch(…, KVCacheManagerBase&, …)` / `Insert(…, KVCacheManagerBase&, …)` |
| Python-exposed? | **Yes** (`_impl.KVCacheStore` + `api/jax` wrapper) | **No** — C++ only |

In short: `KVCacheStore` is the thin, Python-facing directory the connector orchestrates
around — it drives `KVCacheManager` transfers itself and keeps the two in sync manually
(insert after save, stage `pending_unlocks` on eviction). `KVCacheStoreInternal` folds the
directory, the host block pool, local+remote transport, and a global registry into one
self-contained C++ engine (lookup-and-fetch / insert-with-D2H) for the
distributed/global-registry path — it is **not** used by the offload connector.

## `kv_cache/` component overview

The full set of components in `tpu_raiden/kv_cache/`, by layer:

### Building blocks (data structures)

| File | Role |
|---|---|
| `lru_cache.h` | Generic header-only `LRUCache<Key, Value>` with **pin/unpin** support — pinned entries move to a separate `pinned_list_` and are never LRU-evicted (`pin_count` refcounted). The primitive under the logical directory. |
| `logical_block_manager.{h,cc}` | `LogicalBlockManager` — the **physical host-block allocator/tracker**. Manages block ids `0..N-1` over the host pool: `Allocate(n, lock)`, `Unlock`, `AccessBlock`, plus **LRU eviction of *unlocked* blocks** when full. Tracks which blocks are free/allocated/locked; does not hold the bytes. |

### Logical directory (metadata only)

| File | Role |
|---|---|
| `kv_cache_store.{h,cc}` | `KVCacheStore` — the **logical LRU directory**: `LRUCache<block_hash(string) → vector<RaidenId>>`. `Lookup`/`Insert`/`Delete`/`Pin`/`Release`; defines `struct RaidenId`. Python-exposed; used by the offload connector scheduler. No host RAM, no transfers. |

### Transfer engine

| File | Role |
|---|---|
| `kv_cache_manager_base.{h,cc}` | `KVCacheManagerBase` — the **core DMA/transfer engine**. Owns the device `PjRtBuffer`s + the host pool (via a `HostBufferAllocator` + `LogicalBlockManager`) and implements `H2d`, `D2h`, `D2hAutoAllocate`, `H2hRead/Write`, `AllocateBlocks`, `DispatchD2h/H2dChunks`. The jax `KVCacheManager` (via `KVCacheManagerWithTransfer`) and `KVCacheStoreInternal` both build on it. |

### Integrated / distributed store

| File | Role |
|---|---|
| `kv_cache_store_internal.{h,cc}` | `KVCacheStoreInternal` — the **all-in-one** store folding directory + physical pool + local/remote transport (`BlockTransportDelegate` + `BlockTransport server_`) + a `GlobalRegistryClient` into one C++ engine. `LookupAndFetch` (lookup + H2D/remote pull) and `Insert` (D2H + register). C++-only, `uint64` hashes. The distributed path — not used by the offload connector. |

### Distributed control plane (cross-node)

| File | Role |
|---|---|
| `global_registry/` | A **gRPC distributed directory service** (`GlobalRegistryService`: `Register`/`Lookup`/`Unregister`) mapping KV block hashes → `KVBlockMetadata` (which node/endpoint holds each block). `global_registry_client` (used by `KVCacheStoreInternal` to find remote blocks) + `global_registry_server` (+ standalone `_main`). The cross-node directory. |
| `kv_cache_listener.{h,cc}` | `KVCacheListener` — the **server-side accept loop**. Runs a listener thread on a port, accepts incoming block-transfer connections and dispatches them to its `KVCacheManagerBase* engine`. The producer/server endpoint remote consumers connect to for H2H pulls. |
| `kv_cache_controller.py` | `KVCacheWorkerRpcClient` — a **Python RPC client for KV-cache resharding/coordination** (extends `raiden_controller.WorkerRpcClient`, uses the KV-cache service proto + `TransferPlan`). Control-plane glue coordinating transfer plans across workers. |

**Two ways to use it:** the **split design** (offload connector) — `KVCacheStore`
(logical) orchestrated separately around `KVCacheManager` (= `KVCacheManagerBase`
transfer engine), kept in sync in Python; vs the **integrated design** —
`KVCacheStoreInternal` = directory + pool + transport + registry in one, for
multi-node distributed KV sharing. Layering: `lru_cache` / `logical_block_manager`
(primitives) → `kv_cache_store` (logical) + `kv_cache_manager_base` (physical) →
`kv_cache_store_internal` (integrated) → `global_registry` + `kv_cache_listener` +
`kv_cache_controller` (cross-node control/transport).

## `KVCacheManagerBase` vs `KVCacheManagerWithTransfer`

Class hierarchy: `RaidenManagerBase` → `KVCacheManagerBase`
(`kv_cache/kv_cache_manager_base.h`) → `KVCacheManagerWithTransfer`
(`core/kv_cache_manager_with_transfer.h`). The split is **data plane vs control
plane**.

| | `KVCacheManagerBase` | `KVCacheManagerWithTransfer` (extends Base) |
|---|---|---|
| Role | **Data plane** — "how to move bytes" | **Control plane** — "how to coordinate a distributed KV pull between two hosts" (built on Base's copies) |
| Owns | device KV buffers + host pool (`LogicalBlockManager`) | everything Base has, plus the control/transport state below |
| Key API | `H2d`, `D2h`, `D2hAutoAllocate`, `H2hRead`/`H2hWrite`, `H2hReadExplicit`, `DispatchD2h/H2dChunks`, `AllocateBlocks` — raw copies, stateless per request | TCP **control server** (`StartControlServer`, `local_control_port_`, `ProcessPullStream`); **pull/push protocol** (`NotifyForRead`, `StartRead`, `CompleteReadRaw`, `StartPushInternal`, `get_local_endpoints`); **staging slot pool** (`Slot`, `InitializeSlotPool`, `AcquireSlot`, `ReleaseSlotLocked`); in-flight tracking (`send_entries_`, `StagingReadinessState`, `d2h_layer_futures`) |
| Request hooks | virtual `RegisterActivePlan` / `RegisterRecv` / `OnBlocksReceived` / `WaitForPendingWork` — **default no-ops** | **overrides** those hooks with the real transfer lifecycle |
| Used by | `KVCacheStoreInternal` (as `manager&`); the layer the **offload connector** exercises | the **disagg** engine (cross-host prefill→decode pull) |

**How they connect:** the jax `KVCacheManager`'s `sub_managers_` are always
`KVCacheManagerWithTransfer` instances, so the same objects serve both paths — but
callers use different layers:
- **Offload connector** → only **Base** methods (`d2h_auto_allocate`, `h2d`,
  `unlock_blocks`); no control server / pull protocol.
- **Disagg connector** → the **WithTransfer** methods (`notify_for_read`,
  `start_read`, `complete_read`).

## Component relationships (who uses what)

### Who owns / uses `LogicalBlockManager`

The host-pool bookkeeper is owned by exactly **two** classes; the logical
`KVCacheStore` does **not** use it (it uses `LRUCache` instead).

| Owner | How it uses `LogicalBlockManager` |
|---|---|
| **`KVCacheManagerBase`** (transfer engine) | `host_block_manager_` (`kv_cache_manager_base.h:240`; ctor `.cc:117`,`.cc:219`), public accessor `host_block_manager()` (`:208`). Reached by: `AllocateBlocks` → `Allocate(n, lock=true)` (D2hAutoAllocate save path + disagg slot-pool init) and `KVCacheManager::UnlockBlocks` → `host_block_manager()->Unlock` (`frameworks/jax/kv_cache_manager.cc:733`). So the **offload connector** (via jax `KVCacheManager`) and the **disagg path** (`KVCacheManagerWithTransfer`) reach it through the base. |
| **`KVCacheStoreInternal`** (integrated store) | its own `host_block_manager_` (`kv_cache_store_internal.h:116`; ctor `.cc:71`,`.cc:101`) for tracking blocks of received/remote data. |

### Does `KVCacheStoreInternal` use `KVCacheManagerBase` for transfers? — **Yes**

`KVCacheStoreInternal` owns **no** transfer engine; it receives a
`KVCacheManagerBase& manager` **per call** (`LookupAndFetch(..., manager, ...)`,
`Insert(..., manager, ...)`) and **delegates all data movement to it**:

| Path | Calls on `manager` (KVCacheManagerBase) |
|---|---|
| **Insert (save)** | `manager.SetExternalHostPointers(...)` → `manager.D2hAutoAllocate(...)` (device→host) (`kv_cache_store_internal.cc:284,290`) |
| **LookupAndFetch — local hit** | `manager.SetExternalHostPointers(...)` → `manager.H2d(...)` (host→device) (`:210-211`) |
| **LookupAndFetchRemote** | `registry_client_->Lookup(...)` to locate the owning node, then `manager.H2hReadExplicit(peer, ...)` to pull from it, then `manager.H2d(...)` into device (`:368,426,470-473`) |

`KVCacheStoreInternal`'s *own* pieces are the **directory/orchestration** (`lru_list_`
+ `cache_map_`, `host_block_manager_`), the **`GlobalRegistryClient`** (cross-node
`Register`/`Lookup`), and its **`BlockTransport server_`** (the endpoint other nodes
pull *from*, `:76`). The byte movement itself is always `KVCacheManagerBase`. So
`kv_cache_manager_base` is the transfer engine under **both** designs — the split
offload connector (via jax `KVCacheManager`) and the integrated
`KVCacheStoreInternal`.

## Host Buffer Management

For a detailed explanation of host DRAM buffer management, allocation paths, `LogicalBlockManager`, staging slots, and the dual-allocation mismatch between the controller and the worker, see [host_buffer_management.md](file:///google/src/cloud/jcgu/raiden_controller/google3/third_party/tpu_raiden/tpu_raiden/kv_cache/host_buffer_management.md).


## Notes

- **Two id spaces.** `RaidenId.data_replica_idx` (returned by `d2h_auto_allocate`
  and stored via `insert`) is the *physical host chunk id* in the
  `LogicalBlockManager` pool. It is the same integer the connector later passes to
  `h2d` (as a source chunk) and to `unlock_blocks`.
- **Locking is implicit.** There is no `lock` API; `d2h_auto_allocate` always
  allocates chunks **locked** (`Allocate(.., lock=true)`), and `unlock_blocks` is
  the only way to release them. See `current_work/UNLOCK_BLOCKS_NOTES.md` for the
  lock/unlock lifecycle.
- **Consistency requirement.** The `KVCacheStore` capacity (scheduler) and the
  `KVCacheManager` `host_blocks_to_allocate` pool (worker) are both sized from
  `TPU_OFFLOAD_NUM_CPU_CHUNKS` and must match.
- **Line numbers** are for the tpu-raiden tree at the time of writing; verify
  against the current source if it has moved.

