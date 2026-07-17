# Host Buffer Management in TPU Raiden

This document explains how host DRAM buffers are managed, allocated, and used for staging KV cache blocks during transfers in TPU Raiden. It also details the current "dual-allocation" design where the controller's logical view and the worker's physical execution operate on different memory spaces.

## 1. Architecture Overview

Host memory management in Raiden operates at two levels:
1.  **Logical Block Management (`LogicalBlockManager`)**: Tracks which blocks (indices `0..N-1`) are free, allocated, locked, or eligible for LRU eviction. It does not hold the actual memory pointers.
2.  **Physical Staging Slots (`KVCacheManagerWithTransfer`)**: A pool of pre-allocated memory slots used as intermediate buffers for network (H2H) and device (D2H/H2D) transfers.

There is a split between the memory allocated via the gRPC control plane (`CreateBuffers`) and the memory physically used by the transfer engine (`KVCacheManagerBase`).

---

## 2. Physical Memory Allocation Paths

### Path A: Worker Local Pool (Host Staging Pool)
This is the memory actually used for executing transfers.
*   **Trigger**: Allocated during the construction of `KVCacheManagerBase` (on the worker).
*   **Allocator**: 
    *   In JAX production, it uses a `HostBufferAllocator` which allocates **pinned (DMA-mapped) memory** using PJRT APIs.
    *   If no allocator is provided, it falls back to standard page-aligned memory allocation via `posix_memalign`.
*   **Code Reference**: [kv_cache_manager_base.cc:165-205](file:///google/src/cloud/jcgu/raiden_controller/google3/third_party/tpu_raiden/tpu_raiden/kv_cache/kv_cache_manager_base.cc#L165-L205)

### Path B: gRPC Controller Pool (`CreateBuffers`)
This memory is allocated via the control plane RPC but is currently unused during transfers.
*   **Trigger**: The central `RaidenController` calls the `CreateBuffers` RPC during construction.
*   **Allocator**: 
    *   In the JAX integration, the gRPC server is started with `host_allocator = nullptr` ([kv_cache_manager.cc:843](file:///google/src/cloud/jcgu/raiden_controller/google3/third_party/tpu_raiden/tpu_raiden/frameworks/jax/kv_cache_manager.cc#L843)).
    *   `WorkerServiceImpl` falls back to `MallocHostMemoryAllocator` ([worker_service_impl.cc:43-44](file:///google/src/cloud/jcgu/raiden_controller/google3/third_party/tpu_raiden/tpu_raiden/core/controller/worker_service_impl.cc#L43-L44)), which allocates **unpinned memory** via `posix_memalign` ([host_memory_allocator.cc:183](file:///google/src/cloud/jcgu/raiden_controller/google3/third_party/tpu_raiden/tpu_raiden/core/host_memory_allocator.cc#L183)).
*   **Storage**: The allocated buffers are stored in `WorkerServiceImpl::buffers_` map ([worker_service_impl.cc:67](file:///google/src/cloud/jcgu/raiden_controller/google3/third_party/tpu_raiden/tpu_raiden/core/controller/worker_service_impl.cc#L67)).

---

## 3. Logical Block Management (`LogicalBlockManager`)

The `LogicalBlockManager` (`third_party/tpu_raiden/tpu_raiden/kv_cache/logical_block_manager.h`) manages block indices `0..N-1`:
*   **Allocation**: Allocates `n` blocks. If insufficient blocks are available, it attempts to evict unlocked blocks.
*   **Eviction**: Uses a Least Recently Used (LRU) policy based on an access counter updated via `AccessBlock`.
*   **Two Instances**:
    *   **Controller-Side**: `RaidenController` has its own `LogicalBlockManager` to track the buffers it created via `CreateBuffers`.
    *   **Worker-Side**: `KVCacheManagerBase` creates a `LogicalBlockManager` (`host_block_manager_`) tracking the flat physical host pool. It is utilized in two distinct ways depending on the operation:
        1.  **Dynamic Access (Direct Fetch)**: When executing direct device-to-host copies via `D2hAutoAllocate`, `KVCacheManagerBase` dynamically calls `host_block_manager_->Allocate(..., lock=true)`. Using `lock=true` guarantees the payload won't be silently evicted before being pushed over the network. If the staging pool is full, the manager evicts *older unlocked* blocks.
        2.  **Static Slot Pooling (H2H Network Transfers)**: To ensure stability during network transfers, `KVCacheManagerWithTransfer` bypasses dynamic logic. At initialization (`InitializeSlotPool`), it aggressively calls `Allocate(..., lock=true)` in a loop, pinning all blocks permanently into `Slot` objects, rendering LRU inert.

### 3.1. Locking and Unlocking Mechanics

In the `LogicalBlockManager`, blocks are not locked or unlocked automatically. These states are completely dictated by explicit caller arguments:
*   **When is a block LOCKED?**: Only when the caller explicitly invokes `Allocate(num_blocks, lock=true)`. 
    *   In *Static Mode* (`InitializeSlotPool`), this is done exactly once at startup. 
    *   In *Dynamic Mode* (`D2hAutoAllocate`), this is done at the time of buffer creation to protect the payload.
*   **When is a block UNLOCKED?**: Only when the caller explicitly uses `Unlock(std::vector<int> block_ids)`.
    *   Because `D2hAutoAllocate` passes `lock=true`, **consumers must explicitly unlock** the assigned blocks once the memory sequence (e.g., Network H2H Push) finishes, otherwise memory will leak.
    *   In Static Mode, they are only unlocked during `~KVCacheManagerWithTransfer`.

### 3.2. Bypassing LogicalBlockManager for Custom Orchestration

**Question**: *If I want to bypass LogicalBlockManager (worker side), can I just call `D2H`, and the `H2H` without using `D2hAutoAllocate`?*
**Answer**: **Yes, absolutely.**

The core execution layer methods—`D2h`, `H2d`, `H2hRead`, `H2hWrite` (and their `Direct` counterparts)—are completely independent of the `LogicalBlockManager`. They are "dumb" arithmetic executors:
1.  They accept explicit raw integers (`dst_offsets`, `src_block_ids`) and trust them blindly.
2.  They **never** query or mutate the state of `host_block_manager_`.
3.  They strictly translate the integer offsets into DMA arithmetic mappings: `h_base + (logical_block_id * block_byte_size)`.

If an external orchestrator or test framework manually manages block topologies, it can safely instruct the Worker to do `D2h(..., my_device_offsets, my_host_offsets, ...)` and then run a network `H2hWrite(...)`. This takes full control of the raw physical pool while entirely sidestepping all lock/unlock/eviction routines.

---

## 4. Flat Physical Pools vs. Logical Staging Slots

There is a critical distinction between how memory is physically laid out in hardware, and how it is logically grouped by the software orchestrating transfers.

### A. The Hardware View (Flat Physical Pool)
At the bottom level, `KVCacheManagerBase` knows nothing about "slots", "sequences", or "staging". It simply allocates a **single, gigantic contiguous byte buffer** covering the entire host memory needed for a shard:
*   **Total Capacity**: The orchestrator passes down a scalar called `host_blocks_to_allocate` (which equals `slots * max_blocks`). 
*   **Contiguous DMA**: The base class multiplies this block total by `bytes_per_block`, asks the `HostBufferAllocator` for pinned memory exactly once, and retains a single base pointer `h_base`.
*   **Flat Arithmetic**: Whenever raw `D2H`, `H2D`, or `H2H` DMA copies happen, `KVCacheManagerBase` resolves the target pointer using simple flat offset arithmetic: `h_base + (logical_block_id * block_byte_size)`.

### B. The Logical View (Staging Slots)
`KVCacheManagerWithTransfer` sits on top of this flat pool and organizes the flat indices (`0...N`) into higher-order network **Slots**.
For Host-to-Host (H2H) transfers, it logically groups the host memory:
*   A `Slot` holds exactly `max_blocks` flat physical block IDs.
*   **Initialization**: When `KVCacheManagerWithTransfer` spins up, it calculates `Total Blocks = num_slots * max_blocks` and pushes that up to the base class. Then in `InitializeSlotPool`, it polls the base class's `LogicalBlockManager` to slice those flat IDs into bundles (slots), pushing them onto a private `free_slots_` queue.
*   **Acquire/Release**: High-level network transfers ask for an available bundle by calling `AcquireSlotLocked`. They unpack the bundle into flat `block_ids` and pass those directly down to the flat arithmetic layer for DMA. 

---

## 5. The Dual-Allocation Mismatch

Currently, the memory spaces managed by the controller and the worker are disconnected:

1.  **Redundant Allocation**: `CreateBuffers` allocates heap memory and registers it in `WorkerServiceImpl::buffers_`.
2.  **Bypassing `buffers_` during Transfer**: When `TransferBuffers` RPC is called, it only passes offsets (block indices) and delegates to the transfer manager:
    *   **Code Reference**: [worker_service_impl.cc:154](file:///google/src/cloud/jcgu/raiden_controller/google3/third_party/tpu_raiden/tpu_raiden/core/controller/worker_service_impl.cc#L154)
    ```cpp
    future_or = transfer_manager_.D2h(src_offsets, dst_offsets, copy_sizes);
    ```
    The `transfer_manager_` (which is `KVCacheManagerWithTransfer`) executes the copy within its own pre-allocated **Host Staging Pool** (pinned memory) via `HostSpan` ([kv_cache_manager_base.cc:924](file:///google/src/cloud/jcgu/raiden_controller/google3/third_party/tpu_raiden/tpu_raiden/kv_cache/kv_cache_manager_base.cc#L924)), completely ignoring `WorkerServiceImpl::buffers_`.

### Summary of the Mismatch
The controller's `LogicalBlockManager` manages handles to the unused gRPC-allocated buffers. The worker executes transfers using its local staging pool. Because the RPC interfaces use raw offsets (which match the logical block IDs), they align in index space, but operate on different physical memory.

---

## 6. Proposing Consolidation (Using Controller's Buffers)

To make `KVCacheManager` use the memory allocated by the controller (`buffers_`), the following changes are required:

1.  **gRPC Protocol Update**: Update `TransferBuffers` RPC to accept `BufferHandle`s instead of raw offsets.
2.  **Pointer Resolution**: `WorkerServiceImpl` must look up the handles in `buffers_` to resolve them to raw host pointers.
3.  **Thread-Safe Explicit Transfer APIs**:
    *   `KVCacheManagerBase` needs `D2h` and `H2d` overloads that accept explicit host pointers (e.g., `D2hExplicit(..., std::vector<uint8_t*> explicit_dst_ptrs)`), similar to `H2hReadExplicit`.
    *   This avoids using `SetExternalHostPointers` which modifies global state and is not thread-safe.
4.  **Memory Pinning**: Enable `XlaHostMemoryAllocator` for the gRPC server so `CreateBuffers` allocates pinned memory, ensuring high-speed DMA transfers.

---

## 7. E2E Buffer Allocation Example (`test_parallel_pull`)

To understand exactly when and how the host buffers are allocated during an end-to-end H2H transmission (such as `test_parallel_pull`), we must distinguish between **Physical Allocation** (acquiring raw memory from the OS/TPU Runtime) and **Logical Allocation** (claiming the right to use that memory for a specific transfer).

Here is the step-by-step cycle API trace for `test_parallel_pull`:

### Step 1: Physical Buffer Pre-allocation (Constructor)
*   **Action**: Python `producer = KVCacheManager(...)` and `consumer = KVCacheManager(...)` are invoked.
*   **APIs Called**:
    1.  The FFI constructs `KVCacheManagerWithTransfer`.
    2.  This calls the base constructor `KVCacheManagerBase(host_blocks_to_allocate = num_slots * max_blocks)`.
    3.  `KVCacheManagerBase` calls `host_allocator(alloc_size, target_dev)` to acquire one massive chunk of pinned DMA memory covering the total required size (`host_blocks_to_allocate * bytes_per_block`).
*   **Takeaway**: Physical host staging memory is **allocated exactly once, statically, upfront**. No physical allocation (`malloc`, `posix_memalign`) happens during the actual transfers.

### Step 2: Logical Block Allocation (Producer)
*   **Action**: Python `producer.register_read(req_id, uuid, [0, 1])`
*   **APIs Called**:
    1.  C++ `KVCacheManagerWithTransfer::RegisterRead` (or equivalents depending on binding) is hit.
    2.  Calls `AcquireSlotLocked()`.
    3.  `AcquireSlotLocked()` pops a single available `Slot` structure off the `free_slots_` queue. This `Slot` internally holds a sequence of logical block IDs for the flat physical pool.
*   **Takeaway**: The Producer doesn't allocate memory here; it just claims ownership of a piece of the flat pool built in Step 1, flagging those specific logical `block_ids` as ready to be read by remote clients.

### Step 3: Logical Allocation & Network Transfer (Consumer)
*   **Action**: Python `consumer.start_read(req_id, uuid, remote_endpoint, remote_block_ids=[0, 1])`
*   **APIs Called**:
    1.  C++ `KVCacheManagerWithTransfer::StartRead` is hit.
    2.  The Consumer must allocate a target slot to receive the data. It calls its own `AcquireSlotLocked()`, claiming `max_blocks` logical block indices from its own local `free_slots_` queue.
    3.  A `CopySpec` plan is built, mapping local block targets.
    4.  The network call is triggered: `BlockTransport::Pull`.
*   **Network Path**:
    *   The Consumer sends the TCP `Pull` with the `block_hash`.
    *   The Producer matches the hash, retrieves the logical slot reserved in Step 2, and streams the physical bytes reading directly via flat offset arithmetic `h_base + (offset * block_size)`.
    *   The Consumer intercepts the stream via its autonomous port listener and writes incoming bytes directly into its own flat physical pool `h_base + (local_offset * block_size)` (the space we claimed at the start of this step).
