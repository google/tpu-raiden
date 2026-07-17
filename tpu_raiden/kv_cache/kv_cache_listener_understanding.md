# KVCacheListener Overview

## Role and Purpose
`KVCacheListener` acts as the **Data Plane Worker** (or RPC server) for the `KVCacheManager` in the Raiden architecture. It is responsible for executing the actual data movement operations (Push/Receive) across the network.

While `KVCacheStore` manages the logical directory of blocks and `RaidenController` handles the negotiation and control plane logic, `KVCacheListener` drives the execution engine (`KVCacheManager`) to move the bits.

## Key Interactions

### 1. With `KVCacheManager` (Data Plane)
-   **Execution**: It directly invokes methods on `KVCacheManagerBase` (or its derived classes like `KVCacheManagerWithTransfer`) to start transfers.
-   **Callbacks**: It registers callbacks with the `KVCacheManager` to be notified when specific operations (like receiving blocks) complete.

### 2. With `RaidenController` (Control Plane)
-   **Commands**: It receives commands from the local `RaidenController` (e.g., `RaidenControllerEmbedded`) to initiate transfers.
-   **Notifications**: It notifies the local Controller when transfers are completed so the Controller can report back to the Orchestrator.

## Core Operations

### Handling `COMMAND_START_TRANSFER`
When `KVCacheListener` receives this command (usually from the local `RaidenController` after negotiation):
-   **As Sender**: It calls `engine_->PushKVCacheResharded(start_req)` to initiate the push of data to the remote peer.
-   **As Receiver**: It calls `engine_->RegisterActivePlan(...)` to prepare the local buffers to receive incoming data.

### Handling Completion
-   When `KVCacheManager` finishes receiving blocks, it triggers the `BlocksReceivedCallback` registered by `KVCacheListener`.
-   `KVCacheListener` then sends a `COMMAND_TRANSFER_COMPLETED` RPC back to the local `RaidenController` (`local_control_port`), passing the list of completed block IDs.

## Importance for E2E Testing
In End-to-End tests (like `QueueFlowEmbedded` in C++), `KVCacheListener` is essential to simulate the full workflow:
1.  **Sender Node**: Needs a listener to receive the `START_TRANSFER` command and trigger the Push.
2.  **Receiver Node**: Needs a listener to register the active plan and handle completion callbacks.

Without `KVCacheListener`, the control plane might negotiate successfully, but the data plane will never execute the transfers because the internal RPC triggers will be missed.

## Integration with Frameworks (Torch vs JAX)

### 1. PyTorch (`torch`) Integration
In the PyTorch integration, `KVCacheListener` is **embedded directly within the `KVCacheManager`**, making the manager a self-contained transfer node.

-   **Ownership**: `KVCacheManager` owns the listener via `std::unique_ptr<KVCacheListener> listener_`.
-   **Lifecycle**: The constructor takes an optional `listener_port`. If provided, it instantiates and starts the listener automatically.
-   **Python Exposure**: Python users do not see `KVCacheListener`. They interact only with `KVCacheManager`, checking properties like `is_listener_active` or `listener_port` exposed via `nanobind`.

### 2. JAX Integration (Chosen Design: Per-Sub-Manager Listener)
In JAX, `KVCacheManager` acts as a wrapper around multiple `KVCacheManagerWithTransfer` sub-managers (e.g., one per NUMA node).

-   **Design Choice**: Each sub-manager will have its own dedicated `KVCacheListener`.
-   **Implementation**:
    -   Add `std::vector<std::unique_ptr<KVCacheListener>> listeners_` to JAX `KVCacheManager`.
    -   In `InitSubManagers`, instantiate and start a listener for each sub-manager.
    -   Listeners will use consecutive ports starting from a base `listener_port` (e.g., `base_port`, `base_port + 1`, etc.).

## Revisions to `RaidenControllerEmbedded` (C++)
To support multiple listeners per node, `RaidenControllerEmbedded` has been revised:

1.  **Multiple Worker Connections**: Instead of a single `local_worker_port_`, it queries endpoints from all listeners (consecutive ports starting from `local_worker_port_`). [COMPLETED]
2.  **Broadcast/Dispatch**: When processing a `FetchRemote` request, it broadcasts the `START_TRANSFER` command to ALL relevant local listeners. [COMPLETED]
3.  **Completion Aggregation**:
    -   Currently, it waits for `COMMAND_TRANSFER_COMPLETED` and marks the block as done.
    -   With multiple listeners, a single logical fetch operation might involve transfers driven by different listeners (if shards are split across sub-managers).
    -   The controller tracks completion signals from *each* listener involved in the fetch. [COMPLETED]
    -   An operation is only marked fully complete in the `FetchCompletionQueue` when `num_listeners_` have reported completion for that specific block. [COMPLETED]

## Conclusion
Bundling `KVCacheListener` inside the framework-specific `KVCacheManager` simplifies the Python API. In JAX, extending this to support multiple listeners per node (one per NUMA domain) ensures efficient data movement while requiring the Embedded Controller to act as a coordinator across these local data plane workers.


## Bug Fixes &amp; Learnings during Integration

### 1. Callback Bypassing in `KVCacheManagerWithTransfer`
- **Issue**: Transfers were completing (H2H read complete logs appeared), but the `KVCacheListener` was never notified, leading to timeouts in E2E tests.
- **Root Cause**: `KVCacheManagerWithTransfer::OnBlocksReceived` was overriding the base class method but calling `RaidenManagerBase::OnBlocksReceived` instead of `KVCacheManagerBase::OnBlocksReceived` for the fallback path. The listener callback was registered in `KVCacheManagerBase`, so it was bypassed.
- **Fix**: Changed the call to `kv_cache::KVCacheManagerBase::OnBlocksReceived(block_ids, uuid)` in `KVCacheManagerWithTransfer::OnBlocksReceived`.

### 2. Controller Port Routing for Listeners
- **Issue**: In multi-process or complex test setups, listeners might not know which controller port to report completion to if ephemeral ports are used.
- **Fix**: Explicitly pass `listener_controller_port` from the top-level `KVCacheManager` down to listeners, allowing explicit routing of completion signals even when ports are dynamically allocated.

## RaidenControllerEmbedded Revisions (C++) Implementation Details

The revisions to `RaidenControllerEmbedded` (C++) were implemented as follows:

1.  **Auto-discovery of listener capacities**: In `Start()`, the controller queries each listener using `COMMAND_GET_ENDPOINTS` and records the number of shards handled by each listener in `listener_shard_counts_`.
2.  **Broadcast and Shard Translation**: In `ConnectionWorker` handling `COMMAND_NEGOTIATE_FETCH` (Sender side):
    *   Instead of sending `COMMAND_START_TRANSFER` to only the first listener, the controller now iterates over all registered listeners.
    *   For each listener, it translates the **global** shard indices in the schedule to **local** shard indices (0-based for that listener).
    *   It only sends the command to a listener if that listener has relevant shards in the schedule.
3.  **Completion Tracking**: In `ConnectionWorker` handling `COMMAND_TRANSFER_COMPLETED` (Receiver side), the controller tracks how many listeners have reported completion for a given `block_id` using `block_completion_counts_`. It only marks the fetch as fully complete when all expected listeners have reported.

### Verification

Added `QueueFlowMultiListenerEndToEnd` to [kv_cache_store_test.cc](file:///google/src/cloud/jcgu/global_kv_index/google3/third_party/tpu_raiden/tpu_raiden/kv_cache/kv_cache_store_test.cc) verifying remote fetch flow where both src and dst have 2 listeners handling 1 shard each. Test passed successfully.

