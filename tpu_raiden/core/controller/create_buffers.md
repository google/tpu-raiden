# Raiden Worker Buffer Pre-Allocation (`CreateBuffers`)

This document details the design, implementation, and memory allocation behavior of the `CreateBuffers` RPC in TPU Raiden.

## 1. Overview
The `CreateBuffers` RPC is part of the `WorkerService` gRPC interface. Its primary purpose is to allow the central `RaidenController` to pre-allocate physical host memory buffers on remote worker nodes upfront during initialization. 

By establishing these allocations before runtime, the controller can manage memory logically in-memory using a `LogicalBlockManager` with zero network overhead during the actual serving loop.

---

## 2. gRPC Protocol Definition
The RPC is defined in `third_party/tpu_raiden/tpu_raiden/proto/worker_service.proto`:

```protobuf
service WorkerService {
  // Allocates a list of sharded buffers on the transfer worker.
  rpc CreateBuffers(CreateBuffersRequest) returns (CreateBuffersResponse);
  ...
}

message CreateBufferSpec {
  int32 num_shards = 1;      // Number of partitions/shards for this buffer.
  int64 size_bytes = 2;      // Size of each shard allocation in bytes.
}

message CreateBuffersRequest {
  tpu_raiden.rpc.RaidenIdProto unit = 1; // Target work unit (job/replica).
  repeated CreateBufferSpec buffers = 2;  // List of buffer specs to allocate.
}

message CreateBuffersResponse {
  bool success = 1;
  string message = 2;
  repeated BufferProto buffers = 3;       // List of created sharded buffers with handles.
}

message BufferProto {
  repeated BufferHandleProto buffer_handles = 1; // Opaque handles for each shard.
}
```

---

## 3. Controller-Side Flow
In the C++ control plane implementation (`RaidenController`), `CreateBuffers` is invoked automatically during the controller's construction:

1. The worker boots up and registers its IP/port with the controller service via `RegisterWorker`.
2. The controller constructs a `RaidenController` instance for that worker.
3. The `RaidenController` constructor prepares a `CreateBuffersRequest` containing specs for all required blocks (`num_blocks`).
4. It calls `CreateBuffers` via `WorkerServiceClient` to allocate the buffers on the worker.
5. The returned opaque `BufferHandle`s are mapped to logical block IDs (`handle_to_block_id_`) for subsequent logical allocation management.

---

## 4. Worker-Side Implementation
The worker-side gRPC implementation is in `WorkerServiceImpl::CreateBuffers` (`third_party/tpu_raiden/tpu_raiden/core/controller/worker_service_impl.cc`):

1. **Locks Mutex**: Acquires a lock on `mutex_` for thread safety.
2. **Validation**: Verifies that `num_shards` and `size_bytes` are positive.
3. **Allocation Loop**: For each buffer spec and for each shard:
   * Calls the configured `HostMemoryAllocator` (`allocator_->Allocate(size_bytes)`) to allocate host memory.
   * If allocation fails, aborts and returns `success = false`.
   * Generates a unique `BufferHandle` (monotonically increasing integer).
   * Registers the allocated buffer in the internal `buffers_` map.
   * Adds the handle to the response.
4. **Returns Response**: Sets `success = true` and returns the handles.

---

## 5. Memory Allocation Caveats & Dual-Allocation

There is a critical distinction between the memory allocated via `CreateBuffers` and the memory actually used during JAX serving transfers.

### The Host Staging Pool
During worker startup, the `KVCacheManagerBase` (enclosed by `KVCacheManagerWithTransfer`) allocates a large **Host Staging Pool** of memory locally in its constructor:
* If a `host_allocator` is passed, it uses it (e.g., PJRT pinned memory in JAX).
* Otherwise, it falls back to allocating local page-aligned memory via `posix_memalign`.

This pool is divided into "slots" and is used directly by the hardware transfer engine.

### The gRPC Allocator Fallback
When the JAX `KVCacheManager` starts the gRPC `WorkerServiceServer`, it passes `nullptr` for the `host_allocator`:
```cpp
controller::WorkerServiceServer::GetInstance().StartServer(
    /*host_allocator=*/nullptr, KVManagerHolder(numa_manager_.get()), grpc_port);
```
Consequently, `WorkerServiceImpl` defaults to using `MallocHostMemoryAllocator` (which allocates unpinned memory via standard `malloc`).

### Resulting Behavior
1. **Redundant Allocation**: If the controller calls `CreateBuffers` on a JAX worker, `WorkerServiceImpl` will allocate additional host memory using `malloc` and register them in its `buffers_` map.
2. **Transfer Execution**: When `TransferBuffers` is called, it only receives offsets (e.g. block indices) and delegates execution to `transfer_manager_` (the local JAX manager). The JAX manager executes the copy within its own pre-allocated **Host Staging Pool** (pinned memory), completely bypassing the `buffers_` map in `WorkerServiceImpl`.
3. **Implication**: In the JAX serving path, the memory allocated via `CreateBuffers` remains unused, and the actual transfer relies entirely on the local pool allocated by the worker at startup.
