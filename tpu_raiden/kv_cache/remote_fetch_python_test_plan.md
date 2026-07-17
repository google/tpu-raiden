# Plan: Implement E2E Remote Fetch Tests in Python (JAX)

## Objective
Reimplement the following E2E tests from `kv_cache_store_test.cc` in `kv_cache_store_test.py`:
1.  `QueueFlowEmbedded` -> `test_queue_flow_embedded` [STATUS: **PASSED** (CPU, GF, GL)]
2.  `QueueFlowEndToEnd` -> `test_queue_flow_end_to_end` [STATUS: **PASSED** (CPU, GF, GL)]
3.  `QueueFlowMultiListenerEndToEnd` -> `test_queue_flow_multi_listener_end_to_end` [STATUS: **PASSED** (CPU, GF, GL)]
4.  `QueueFlowMultiRemoteEndToEnd` -> `test_queue_flow_multi_remote_end_to_end` [STATUS: **PASSED** (CPU, GF, GL)]

## Strategy & Adaptations for JAX/Python
-   **Buffers**: Use JAX arrays instead of raw memory.
-   **Sharding**: Must handle JAX sharding. To test multiple shards on CPU, we might need to enable simulated CPU devices via `XLA_FLAGS="--xla_force_host_platform_device_count=X"`.
-   **Managers**: Use `KVCacheManager` Python wrapper.
-   **Stores**: Use `KVCacheStore` Python wrapper.
-   **Verification**: Verify data by reading back from JAX arrays if possible, or relying on status upgrades and lack of crashes.

## Test Details

### 1. `test_queue_flow_embedded`
-   **Goal**: Verify high-level Fetch API with embedded controller, multiple blocks, layers, and shards.
-   **Setup**: 2 nodes (Sender, Receiver), 2 layers, 2 shards (if possible, else 1 shard with multiple layers/blocks).
-   **Flow**: Insert HOST blocks in Sender, REMOTE placeholders in Receiver. FetchRemote. Verify.

### 2. `test_queue_flow_end_to_end`
-   **Goal**: Verify complete flow including Global Registry.
-   **Setup**: 2 nodes, Global Registry.
-   **Flow**: Register Sender blocks in Global Registry. Receiver does Lookup(enable_global=True), InsertAndLock, FetchRemote. Verify.

### 3. `test_queue_flow_multi_listener_end_to_end`
-   **Goal**: Verify multi-listener support.
-   **Setup**: 2 nodes, each with multiple listeners (sub-managers).
-   **Challenge**: Need to ensure Python `KVCacheManager` correctly instantiates multiple sub-managers and listeners when multiple shards are present.
-   **Fixes**: On TPU, partitioned JAX device mesh to give each sub-manager a disjoint subset of physical devices. Configured `bytes_per_block` in `RemoteFetchConfig` to the local block size (global size / local shards per listener) to resolve resharded push out-of-bounds overruns.

### 4. `test_queue_flow_multi_remote_end_to_end`
-   **Goal**: Verify fetching from multiple distinct remote nodes.
-   **Setup**: 3 nodes (Sender 1, Sender 2, Receiver).
-   **Flow**: Receiver fetches blocks some from Sender 1, some from Sender 2.

## Pre-requisites & Verification
-   Verify JAX CPU multi-device setup.
-   Run tests using `blaze test` with `--nocheck_visibility` if needed due to existing visibility issues.
