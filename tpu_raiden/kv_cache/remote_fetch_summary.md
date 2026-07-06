# Remote KV Cache Fetching: Architecture & Implementation Summary

This document summarizes the design decisions and implementation details for enabling remote KV cache fetching using an embedded C++ control plane in TPU Raiden.

---

## 🏗️ Architectural Overview

We have migrated the control plane logic from external Python orchestration processes into embedded C++ background threads within the TPU Workers. This eliminates external RPC overhead during transfer negotiation and simplifies deployment.

```
+-------------------------------------------------------------+
|                     RaidenOrchestrator                      |
|             (Global Directory / Registry Service)           |
+------------------------------+------------------------------+
                               ^
         Resolve Address       | Register Local Existence
         (Custom TCP Socket)   | (Custom TCP Socket)
                               v
+------------------------------+------------------------------+
|                         TPU Worker                          |
|                                                             |
|  +------------------------+      Push      +-------------+  |
|  |      KVCacheStore      +--------------->|  WorkQueue  |  |
|  |     (Storage/LRU)      |<--------------+-------------+  |
|  +-----------+------------+      Pull      +------+------+  |
|              |                                    |         |
|              | In-Memory                          | Pull    |
|              v                                    v         |
|  +-----------+------------+               +-------+------+  |
|  |      JAX/Python        |               |   Embedded   |  |
|  |    (Client Space)      |               |  Controller  |  |
|  +-----------+------------+               +-------+------+  |
|              |                                    |         |
|              | Manual H2D                         | Connect |
|              v                                    v         |
|  +-----------+------------+ H2H Write (TCP) +-----+------+  |
|  |    KVCacheManager      |<--------------->|  Remote Peer  |  |
|  |    (Data/Transfer)     |                 |  Controller  |  |
|  +------------------------+                 +--------------+  |
+-------------------------------------------------------------+
```

---

## 📅 Phased Implementations

### 1. Global Lookup Fallback (`cl/942174831`)
*   **Fallback Logic**: Enabled `KVCacheStore::Lookup` to automatically query a centralized `GlobalRegistry` when local cache misses occur. If the registry contains the requested block hash, it returns the remote `RaidenId` of the worker owning the block.
*   **Capacity Guarding**: Truncates the number of block hashes requested in a single lookup sequence based on the available and evictable capacity of the local store, preventing buffer overruns.
*   **Bindings**: Exposed fallback options to JAX and PyTorch wrappers.

### 2. `RaidenBlockID` & Status Lifecycle (`cl/942536892`)
*   **Lifecycle Struct**: Replaced raw `RaidenId` keys in the cache tables with `RaidenBlockID`, which tracks:
    *   `RaidenId`: Owner identifier.
    *   `host_block_id`: Staging memory offset index.
    *   `BlockStatus`: Transitions through `INIT` -> `REMOTE` -> `HOST` -> `HBM`.
*   **Block Reservation**: Added `InsertAndPin` to reserve space and transition blocks to `REMOTE` status during peer discovery.
*   **Rollback/Eviction**: Added `ReleaseAndDelete` to discard unused blocks and clean up failed allocations during errors.

### 3. Asynchronous Remote Fetching & Embedded Control (`cl/942711303`)
*   **Embedded Controller**: Created `RaidenControllerEmbedded` running as a background thread inside each worker. It handles unit registration, peer discovery via `RaidenOrchestrator`, and peer-to-peer transfer negotiation.
*   **Custom TCP Protocol**: All control plane messages (Controller <-> Orchestrator, Controller <-> Controller) reuse a lightweight Custom TCP protocol (4-byte length prefix + serialized Protobuf).
*   **H2H Transfer Design (Option 2)**: Aligned the remote fetch pipeline to **only** execute Host-to-Host (H2H) network writes to local DRAM (`MEMORY_TYPE_DRAM`). The downstream Host-to-Device (H2D) copy is explicitly decoupled and triggered manually by JAX/Python. This avoids complex multi-layer progress tracking inside the transport and eliminates potential shutdown hangs.
*   **Batching Optimization**: Optimized remote fetch requests by grouping multiple blocks destined for the same remote worker into a single negotiated batch transfer.
*   **Python (JAX) Exposure**: Exposed `FetchRemote`, `FetchFuture`, and `PollFetchRemoteStatus` via `nanobind` to support non-blocking fetches in Python pipelines.
