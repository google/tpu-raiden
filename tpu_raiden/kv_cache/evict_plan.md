# Implementation Plan: KV Cache Evict (DRAM Cleanup)

This document outlines the design and step-by-step implementation plan for adding an `Evict` API to `KVCacheStore` to free host DRAM blocks by unlocking them in `LogicalBlockManager` and updating the store directory and global registry.

Unlike `Save` or `Load`, `Evict` does not execute any data transfer in the data plane. It only updates metadata and unlocks host blocks.

---

## 🏗️ Architectural Overview

The `Evict` feature allows JAX/Torch schedulers to release host DRAM blocks when they are no longer needed (e.g., when the cache is full and we want to reclaim DRAM).

It updates the `KVCacheStore` directory, unregisters the host locations from the `GlobalRegistry`, and notifies the local workers (`KVCacheListener`) to unlock the corresponding physical blocks in `LogicalBlockManager` so they can be reused.

---

## 📝 Workflow (The Lifecycle of an Evict)

1.  **Trigger**: User calls `KVCacheStore::Evict(block_hashes)`.
    *   Input: `list[bytes]` of block hashes to evict.
    *   Output: `dict[bytes, EvictFuture]` mapping hashes to their completion futures.

2.  **Local Validation & State Transition (Pre-evict)**:
    *   For each block hash, `KVCacheStore` locks the cache directory and checks:
        *   Block must be present in the LRU cache.
        *   Block status must be `HOST` or `HOST_AND_HBM`.
        *   Block must NOT be pinned (`pin_count == 0`).
    *   If valid, the store updates the status in the LRU cache:
        *   `HOST` -> `INIT` (effectively evicted from both HBM and Host).
        *   `HOST_AND_HBM` -> `HBM` (evicted from Host, remains in HBM).
    *   This immediate status update ensures that subsequent local lookups will not attempt to use these host blocks.
    *   If invalid (not found, wrong status, or pinned), the store immediately marks the corresponding `EvictFuture` as failed.

3.  **Global Registry Unregistration**:
    *   For valid blocks, `KVCacheStore` sends an asynchronous `Unregister` request to the `GlobalRegistryClient` to remove the host location registration for these hashes.

4.  **Work Enqueue**:
    *   `KVCacheStore` retrieves the `host_block_id`s for each shard from the block's `BlockSliceList`.
    *   It wraps the `block_hashes` and their corresponding `host_block_id`s into an `EvictRequest` and pushes it to the **`EvictWorkQueue`**.

5.  **Controller Dispatch**:
    *   `RaidenControllerEmbedded` polls `EvictWorkQueue`.
    *   It groups the host block IDs by local worker peer and sends a `COMMAND_EXECUTE_EVICT` control RPC containing the `EvictRequest` payload to each local **`KVCacheListener`**.

6.  **Listener Execution (Sync Unlock)**:
    *   `KVCacheListener` receives `COMMAND_EXECUTE_EVICT`.
    *   It extracts the `host_block_id`s for its shards and calls **`KVCacheManagerBase::UnlockBlocks(block_ids)`**.
    *   Since `Unlock` is a fast metadata operation in `LogicalBlockManager` (no data transfer), it executes synchronously.
    *   The listener immediately replies with a `COMMAND_EVICT_COMPLETED` RPC back to the controller.

7.  **Completion Enqueue**:
    *   When the controller receives completion signals from all workers for a specific `evict_id`, it pushes an event to the **`EvictCompletionQueue`**.

8.  **Final Directory Cleanup**:
    *   `KVCacheStore::EvictCompletionPollerLoop` pulls from `EvictCompletionQueue`.
    *   For blocks that transitioned to `INIT` (were `HOST` only), it deletes the block hash entry from the LRU cache entirely.
    *   For blocks that transitioned to `HBM` (were `HOST_AND_HBM`), it keeps them in the cache but clears their `host_block_id` (sets it to -1) in the `RaidenBlockID`.
    *   It resolves the pending `EvictFuture` as completed.

---

## 📊 Component Interaction Diagram

```mermaid
sequenceDiagram
    participant Store as KVCacheStore
    participant Reg as Global Registry
    participant WQ as EvictWorkQueue
    participant CQ as EvictCompletionQueue
    participant Ctrl as Embedded Controller
    participant Listener as KVCacheListener
    participant LBM as LogicalBlockManager (Worker)

    Note over Store, CQ: Same Process (Control Plane)
    Note over Listener, LBM: Worker Process (Data Plane)

    Store->>Store: 1. Verify status (HOST/HOST_AND_HBM) & pin_count == 0
    Store->>Store: 2. Update LRU (HOST->INIT, HOST_AND_HBM->HBM)
    Store->>Reg: 3. Async Unregister(hashes)
    Store->>WQ: 4. Push EvictRequest(hashes, host_block_ids)
    
    Ctrl->>WQ: 5. Pull EvictRequest
    Ctrl->>Listener: 6. COMMAND_EXECUTE_EVICT(host_block_ids)
    
    Listener->>LBM: 7. UnlockBlocks(host_block_ids)
    LBM-->>Listener: 8. OK (Sync)
    Listener->>Ctrl: 9. COMMAND_EVICT_COMPLETED
    
    Ctrl->>CQ: 10. Push EvictCompletion
    Store->>CQ: 11. Pull EvictCompletion
    Store->>Store: 12. If INIT: delete from LRU; If HBM: clear host_block_id
    Store-->>Store: 13. Resolve EvictFuture
```

---

## 📅 Implementation Phasing

### Phase 1: Protobuf & Structure Updates
*   Update `raiden_service.proto`:
    *   Add `COMMAND_EXECUTE_EVICT` and `COMMAND_EVICT_COMPLETED` to `ControlRequest::Command`.
    *   Define `ShardEvictEntryProto`, `ShardEvictScheduleProto`, and `EvictRequest` messages.
*   Add `EvictWorkQueue` and `EvictCompletionQueue` to `kv_cache_store.h`.

### Phase 2: KVCacheStore API Implementation
*   Define `EvictState` and `EvictFuture` in `kv_cache_store.h`.
*   Implement `KVCacheStore::Evict` to validate blocks, update LRU status, call `RegistryClient::Unregister`, and enqueue to `EvictWorkQueue`.
*   Implement `KVCacheStore::PollEvictStatus`.
*   Implement `KVCacheStore::EvictCompletionPollerLoop` to process completions and perform final LRU cleanup.

### Phase 3: Controller Dispatch & Listener Execution
*   Update `RaidenControllerEmbedded` to handle `EvictRequest` and dispatch `COMMAND_EXECUTE_EVICT`.
*   Update `KVCacheListener` to handle `COMMAND_EXECUTE_EVICT`, call `KVCacheManagerBase::UnlockBlocks`, and report completion.

### Phase 4: Python Bindings & Verification
*   Expose `Evict` and `PollEvictStatus` in nanobind JAX module.
*   Expose `EvictFuture` and update Python wrapper in `api/jax/kv_cache_store.py`.
*   Add C++ and Python unit tests to verify the eviction flow (status transitions, unregistration, and block reuse).
