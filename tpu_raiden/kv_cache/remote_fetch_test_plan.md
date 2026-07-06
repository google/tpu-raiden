# Plan: Remote Fetch Testing

## 🎯 Goals
Verify the standalone correctness and interactions of `RaidenOrchestrator`, `RaidenControllerEmbedded`, and the internal queues of `KVCacheStore`.

---

## 🛠️ Components to Test

### 1. **Unit Tests (Separation of Concerns)**
*   **`ThreadSafeQueue`**: Verify FIFO order, blocking/non-blocking behavior, and cancellation (`Stop()`).
*   **`RaidenOrchestrator`**: Verify registration, resolution of controllers, and basic error handling without `RaidenControllerEmbedded`.
*   **`RaidenControllerEmbedded`**: Verify internal state transitions, queue polling (using mocked peers or simulated actions), and socket handling.

### 2. **Interaction: Controller ↔️ Orchestrator**
*   Spawn a local `RaidenOrchestrator` on a free port.
*   Instantiate one or more `RaidenControllerEmbedded` instances.
*   Verify registration of controllers in the orchestrator.
*   Verify resolution of peers via the orchestrator.

### 3. **Interaction: Controller ↔️ KVCacheStore (Queues)**
*   Simulate `KVCacheStore` pushing work to `FetchWorkQueue`.
*   Verify `RaidenControllerEmbedded` polls it.
*   Verify `RaidenControllerEmbedded` pushes completion to `FetchCompletionQueue`.
*   Simulate `KVCacheStore` popping from `FetchCompletionQueue`.

---

## 📋 Implementation Plan

### Step 1: Create `raiden_remote_fetch_test.cc`
This file will contain all the tests mentioned above. We will use standard GoogleTest.

### Step 2: Implement Unit Tests
*   Test `ThreadSafeQueue<int>` for basic thread safety and blocking.
*   Test `RaidenOrchestrator` via raw sockets to verify protocol adherence.

### Step 3: Implement Interaction Tests
*   Test full Controller ↔️ Orchestrator handshake.
*   Test Queue flow Controller ↔️ Store.

### Step 4: Add to BUILD
Add the new test target to `third_party/tpu_raiden/tpu_raiden/kv_cache/BUILD`.

---

## ❓ Open Questions / Considerations
*   **Ports**: We need to pick unused ports dynamically to avoid collisions. We can use `net_util::PickUnusedPortOrDie()` if available, or just start from a high base port.
*   **Timeout**: Tests involving blocking operations (like `Pop`) must have timeouts to avoid hanging indefinitely.
