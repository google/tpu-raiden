# Understanding Host-to-Host (H2H) Transfers in TPU Raiden

This document details how data is explicitly shared between discrete workers, focusing on the mechanics of the active TCP Push network layer during a remote fetch operation.

## 1. The H2H Push API

In the current implementation of TPU Raiden, when a remote fetch occurs, data is transferred directly from one worker slice's physical RAM to another worker slice's physical RAM over a TCP socket. 

This is accomplished using the `H2hWrite` API provided by `KVCacheManagerBase`:

```cpp
  virtual absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
  H2hWrite(const std::vector<std::string>& peers,
           const std::vector<int>& src_block_ids,
           const std::vector<int>& dst_block_ids = {}, 
           uint64_t uuid = 0,
           int layer_idx = -1);
```

**Context**: In a typical remote fetch sequence, the **Source Worker** is instructed via a gRPC control-plane command (`TransferRemote`) to send a block to the Destination Worker. The Source Worker will invoke `H2hWrite`, providing:
1. `peers`: The TCP network address of the Destination Worker's Data Plane endpoint (`raiden_transfer_endpoint`).
2. `src_block_ids`: The logical block IDs corresponding to the data dwelling in the Source Worker's Host Staging Pool.
3. `dst_block_ids`: The pre-allocated, logical block IDs inside the Destination Worker's Host Staging Pool where the data should land.

---

## 2. Receiver Autonomy (The Passive Listener)

A fundamental principle of the TPU Raiden block transport system is that the receiver during an `H2hWrite` (Push model transfer) **does not need to explicitly call any API** (like `H2hRead`). 

Because TPU Raiden uses a high-performance active transport layer (`tpu_raiden::transport::BlockTransport`), it handles receiving data autonomously in the background. The receiver's `KVCacheStore` is essentially passive during the physical network transport.

### The Receiver Lifecycle:
1. **Background Listener Initialization**: 
   When the Receiver Worker is instantiated, `KVCacheManagerBase::InitializeSlotPool` spins up `BlockTransport`. `BlockTransport` triggers a background thread (`ListenerLoop`) that listens continuously for incoming raw TCP connections.
   
2. **Network Streaming**:
   When the Source Worker calls `H2hWrite()`, it opens a socket connection to the Receiver Worker and streams active `Push` packets. These packets combine the actual byte payload along with the targeted `dst_block_id` in the header.

3. **Autonomous Pointer Resolution**:
   The independent background thread on the Receiver Worker intercepts these packets via `HandleIncomingPush(...)`. 
   It reads the incoming `dst_block_id` and relays a request up to its delegate (the `KVCacheManagerBase`):
   ```cpp
   virtual uint8_t* GetBlockHostPointer(size_t layer_idx, size_t shard_idx, int block_id) {
     return GetHostPointer(layer_idx, shard_idx) + block_id * bytes_per_block();
   }
   ```
   **`KVCacheManagerBase::GetHostPointer()`** intercepts this `dst_block_id` (a 0-indexed logical offset) and translates it directly into the physical pointer belonging to its own private, pinned **Host Staging Pool** (backed by `HostBufferAllocator`).

4. **Direct Write Completion**:
   The `BlockTransport` layer then performs a raw DMA network-write straight into that physical pointer. 
   Once the payload finishes streaming, the transfer is complete. 

### Conclusion
During remote fetch, the physical memory allocation and layout mapping are determined and managed exclusively by the destination worker's `KVCacheManager`. The Source Worker and Central Controller only deal in logical offsets (`dst_block_ids`), trusting the receiving worker's background processes to handle the physical pointer mapping dynamically upon data arrival.
