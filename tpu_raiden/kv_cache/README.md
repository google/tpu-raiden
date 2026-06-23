# High-Performance KVCacheManager for Disaggregated Serving & Offloading

## 1. Project Vision & Overview

**`KVCacheManager`** is an advanced multi-chip layout synchronization and
memory orchestration framework designed to serve as the foundational storage
back-end for high-performance Large Language Model (LLM) inference
architectures.

Our ultimate roadmap aims to enable **disaggregated serving topologies** and
high-throughput **Key-Value (KV) cache offloading** to support ultra-long
context handling inside state-of-the-art production serving engines such as
**vLLM** and **SGLang** with exceptional bare-metal hardware utilization.

By pre-extracting physical accelerator buffer handles (`PjRtBuffer`), hardware
layout major-stride dimensions (`slice_byte_size_`), and zero-copy storage
memory holds (`raiden::BufferHoldAndAlias`), `KVCacheManager` completely
bypasses Python iteration overheads during transfer execution loops. It maps
fine-grained sub-region chunk operations directly to highly parallelized
background Direct Memory Access (DMA) device streams natively exposed to
Python orchestration scripts via non-blocking tuple lambda futures.

---

## 2. Completed Milestones & Functionality (Done)

### Core Layout Abstraction & Centralization

* **Zero-Overhead Extraction Engine**: Unified extraction of
  `PjRtCApiBuffer` and `CommonPjRtBuffer` descriptor structs, physical byte
  sizes, and C-API dynamic downcasting abstractions.
* **Loop Dispatch Minimization**: Refactored linear partial array boundary
  bounds checks and chunk aggregation math into centralized private dispatch
  macros (`GetMajorSliceByteSize`, `DispatchD2hChunks`), eliminating
  duplicate buffer lifetime handling code entirely across core modules.

### Dynamic Host Memory Column Block Allocation

* **`LogicalBlockManager` Integration**: Precomputes total available host
  memory slice capacities based on customizable runtime `block_size`
  configuration parameters.
* **Strided Transfer Execution (`d2h_auto_allocate`)**: Allocates free
  host memory block indices dynamically upon transfer execution requests,
  computes absolute physical byte offsets locally (`assigned_block_id *
  block_size_ * slice_byte_size_`), and automatically locks contiguous
  sub-regions without requiring manual destination layout strides. Binds
  tuple arrays containing assigned block IDs and future handles directly via
  nanobind.

### Host Memory Internal Buffer Allocation

* **Internal Host Memory Allocation**: Added an optional `host_blocks_to_allocate` parameter to `KVCacheManager` constructors. When specified, `KVCacheManager` bypasses Python host array extraction entirely and dynamically allocates 64-byte cache-line aligned host memory buffers (`posix_memalign`) internally for each shard. This eliminates the significant latency (often >20 minutes) associated with allocating large host JAX arrays in Python during engine initialization.

### Unsafe Buffer Lock Skipping

* **Exclusive Lock Skipping (`unsafe_skip_buffer_lock`)**: Added an optional `unsafe_skip_buffer_lock` parameter to `KVCacheManager` constructors. When set to `True`, it bypasses exclusive hold/lock acquisition on destination buffers during copies. This is highly optimized for disaggregated serving scenarios where the caller guarantees that no other concurrent threads will modify the buffers during the transfer loop, eliminating synchronization lock overhead entirely.
* **Concurrent Pipeline Verification**: Implemented a multi-threaded unit test (`test_concurrent_transfer_skip_buffer_lock`) simulating concurrent LLM serving. Thread 1 repeatedly executes `h2d`/`d2h` on blocks 0:4, while Thread 2 concurrently copies unique data via `h2d` to blocks 4:8. This verifies that skipping locks allows flawless parallel transfers on different blocks without deadlocks, data races, or corruption.

### POSIX TCP Transport Engine Framework

* **Standalone Binary Transport Prototyping**: Designed custom socket transfer
  pipelines (`tpu_raiden::transport::SocketTransport`) inheriting from virtual
  `mlcl::Transport` adapter baselines. Implemented robust server listening
  loops wrapped within bounding 50ms non-blocking `poll` constraints to
  guarantee immediate listener thread termination joins upon sandbox container
  destruction.

### Block-Level Distributed Server Integration (Active Stage)

* **Embedded Socket Routing Management**: Extends `KVCacheManager` optional
  constructors to spin up dedicated background TCP socket server loops
  encapsulated cleanly inside private nested structs (`BlockTransportServer`).
* **Dynamic Ephemeral Ports**: Binds peer servers to dynamic kernel ephemeral
  ports automatically (passing port `0`), extracting actual assigned ports via
  `getsockname` to resolve strict network isolation limitations inside shared
  verification sandboxes gracefully. Exposed directly via native
  `local_port()` accessors.
* **Pure-Client Decoupled Streams**: Refactored `h2h_write` and
  `h2h_read` to build lightweight independent connection
  streams on the fly via standalone thread-safe `getaddrinfo` host resolution
  loops (`ConnectToPeer`), allowing clients to push or pull data transparently
  without starting background listening server threads.
* **Transparent Symmetric Tensor Exchanges**: Uses custom block pointer packet
  structures (`BlockPacketHeader`). Receiving worker streams inspect
  pre-extracted internal attention layers (`layers_`) to compute target byte
  offset locations locally on the fly, bypassing raw virtual base addresses
  entirely. Implements automated column assignments during push streams and
  local allocations during pulling symmetrically.

---

## 3. Immediate Execution Roadmap (TODO List)

* **Official MLCL Engine Migration**: Replace temporary POSIX TCP socket
  streaming loops with the official production-ready MLCL transport library
  adapter layer once upstream infrastructure matures completely.
* **Overlapped Execution Stream Polling**: Refactor asynchronous memory future
  polling logic to exploit customized parallel CUDA/TPU stream synchronization
  triggers, ensuring background tensor transfers overlap computational
  attention mechanisms flawlessly.
* **Granular Host Memory Pinning**: Implement highly optimized asynchronous
  page-locked (pinned) host memory buffer pooling allocations to maximize
  cross-host multi-lane PCIe/Network memory copy bandwidth efficiency.

---

## 4. Long-Term Strategic Horizon (Stretch Goals)

* **Serving Engine Integration**: Expose native end-to-end API bindings
  supporting continuous background Key-Value cache offloading and elastic
  block swapping inside production runtime engines (**vLLM** and **SGLang**).
* **Cross-Cluster Auto-Scaling Storage**: Enable automated cross-host
  distributed inference topologies capable of live tensor mapping and
  transparent cache slice migration across remote disaggregated memory
  accelerator nodes.
