# Synthesized Master Plan: Zero-Block FFI & Autotuned Network Architecture

-   **Date**: 2026-06-17
-   **Target**: Saturation of 400 Gbps dual-NIC line-rate (320+ Gbps E2E) on GKE
    TPU v7x (Ironwood) from a single JAX process.
-   **Status**: **APPROVED & UNDER IMPLEMENTATION (BATCH 1 & 2 ACTIVE)**
-   **RCA Backed**: Solves (1) the 6M pps network interrupt storm on CPU 72, (2)
    the disabled TCP autotuning buffer cap, (3) the 512+ thread
    over-subscription, and (4) the driver-level spinlocks inside FFI
    `BlockHostUntilDone()`.

--------------------------------------------------------------------------------

## 1. Stack Partitioning & Task Breakdown

```
                                 [Jujutsu stacked commits]

    CL 4 (rzvnvytk) ──► Experimental Deployment (deploy_and_run.sh, setup_pbr.sh: MTU 9000, RSS/RFS, sysctl)
           ▲
           │ (Jujutsu Auto-Rebase)
           │
    CL 3 (yzpwstsv) ──► JAX FFI & Python Wrapper (Zero-Block FFI, JAX stream extraction, GIL release guards)
           ▲
           │ (Jujutsu Auto-Rebase)
           │
    CL 2 (wtnypyql) ──► NUMA Thread Pinning & Core Affinity (C++ Core - Left Untouched)
           ▲
           │ (Jujutsu Auto-Rebase)
           │
    CL 1 (soorqtqr) ──► C++ Core Transport & Base Managers (TCP autotuning, pool downsizing, multi-device pinning)
```

--------------------------------------------------------------------------------

## 2. Implementation Specifications

### 📦 BATCH 1: CL 1 (`soorqtqr`) — Core C++ Transport & Allocator

*   **Goal**: Re-enable TCP autotuning, eliminate thread over-subscription, and
    standardize memory pinning.

#### Tasks:

1.  **Re-enable TCP Autotuning (`raw_buffer_transport.cc`)**:
    *   **Remove** the `SO_SNDBUF` and `SO_RCVBUF` `setsockopt` lines (lines
        327-330 and 533-536).
    *   This allows the Linux kernel to autotune socket buffers dynamically up
        to 128MB.
2.  **Downsize Thread Pools (`kv_cache_manager_base.cc`)**:

    *   Locate the initialization of `dma_pool_`, `push_pool_`, and
        `pull_pool_`.
    *   Change the thread pool sizing from `parallelism` (which creates 192
        threads per manager) to a maximum of 8 threads:

        ```cpp
        size_t pool_size = std::min<size_t>(8, parallelism);
        ```

    *   This reduces the total manager thread count from 384 to 48, drastically
        lowering context-switching overhead.

3.  **Standardize Pinned Staging Buffers (`kv_cache_manager_ffi.cc` &
    `kv_cache_manager_base.cc`)**:

    *   Refactor the worker-side manager initialization to utilize a valid
        `XlaHostMemoryAllocator` instance instead of `nullptr`, ensuring all
        worker-side host buffers are properly pinned and registered with all
        local TPU devices at startup.

--------------------------------------------------------------------------------

### 📦 BATCH 2: CL 3 (`yzpwstsv`) — JAX FFI & Python Wrapper

*   **Goal**: Transition FFI from a host-blocking synchronous model to a
    hardware-synchronized Asynchronous Zero-Block FFI.

#### Tasks:

1.  **Extract JAX Stream Pointer (Python - `kv_cache_manager.py`)**:

    *   In `h2d()` and `d2h()` Python entry points, extract the raw C++ pointers
        of JAX's active compute streams using JAX's internal APIs:

        ```python
        tpu_devices = jax.devices('tpu')
        stream_ptrs = np.array([d.get_stream_for_external_ready_events() for d in tpu_devices], dtype=np.int64)
        ```

    *   Pass these stream pointers sharded across the JAX mesh into the C++ FFI
        custom calls.

2.  **Zero-Block FFI Execution (C++ - `kv_cache_manager_ffi.cc`)**:

    *   In `TriggerRaidenH2dImpl` and `TriggerRaidenD2hImpl`, accept the stream
        pointer buffer from Python.
    *   Cast the pointer back to `stream_executor::Stream*`.
    *   Enqueue the PCIe DMA copy **directly on JAX's compute stream** instead
        of our custom stream.
    *   **Delete `stream->BlockHostUntilDone()` entirely!** This allows the FFI
        custom call to return **instantly** to JAX, letting the TPU hardware
        scheduler handle the synchronization, completely eliminating the 100%
        `%sys` driver spinlock on CPU 72!

3.  **Asynchronous Host Callback for D2H**:

    *   For D2H transfers, instead of blocking the host thread, register a host
        callback on the JAX stream:

        ```cpp
        jax_stream->ThenDoHostCallback([manager, uuid]() {
          manager->SignalD2hComplete(uuid);
        });
        ```

    *   This signals the background transport thread asynchronously when the TPU
        finishes writing.

4.  **Surgically Release the GIL (`tpu_raiden_jax_module.cc`)**:

    *   Add `nb::call_guard<nb::gil_scoped_release>()` to all missing JAX
        bindings (`h2h_write`, `h2h_read`, `d2h_auto_allocate`, `complete_read`)
        to ensure the Python thread never holds the GIL during FFI setup.

--------------------------------------------------------------------------------

### 📦 BATCH 3: CL 4 (`rzvnvytk`) — Experimental Deployment & Tuning

*   **Goal**: Configure jumbo frames, distribute interrupts across cores, and
    tune kernel TCP limits.

#### Tasks:

1.  **Jumbo Frames on `eth0` (`deploy_and_run.sh` / YAMLs)**:
    *   Configure GKE pod interfaces and physical NIC `eth0` to use **MTU 9000**
        (matching `eth1`), reducing `eth0` packet rate by 6x.
2.  **NIC Interrupt Distribution & GRO (`setup_pbr.sh`)**:
    *   Enable Receive Side Scaling (RSS) with multiple queues and Receive Flow
        Steering (RFS) using `ethtool`.
    *   Distribute the interrupt affinities of `eth0`/`eth1` queues across all
        CPU cores of their respective NUMA nodes (avoiding cores running
        application threads).
    *   Ensure GRO/LRO are active.
3.  **Tune Kernel TCP Max Buffers (`setup_pbr.sh`)**:

    *   Add `sysctl` net-tuning parameters to allow TCP windows to scale
        dynamically up to 128MB:

        ```bash
        sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728"
        sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"
        sysctl -w net.core.rmem_max=134217728
        sysctl -w net.core.wmem_max=134217728
        ```

--------------------------------------------------------------------------------

## 3. Verification Sequence

1.  **Batch 1 (CL 1)**: Apply C++ autotuning & pool downsizing -> Format ->
    Compile -> Amend.
2.  **Batch 2 (CL 3)**: Apply Python stream extraction & Zero-Block FFI C++
    casting -> Format -> Test (`blaze test //third_party/tpu_raiden/...`) ->
    Amend.
3.  **Batch 3 (CL 4)**: Apply MTU 9000 & RSS/RFS tuning -> Run E2E GKE
    performance sweep.
