# Single-Node PCIe DMA Microbenchmark (Raw Transfer)

This guide describes how to compile and execute the single-node PCIe Direct Memory Access (DMA) microbenchmark for **TPU Raiden**. 

This benchmark measures the raw, hardware-level Host-to-Device (H2D) and Device-to-Host (D2H) bandwidth of the TPU Raiden engine, isolating physical hardware throughput from Python or high-level framework (e.g., JAX, PyTorch) runtime overhead.

---

## Hardware Architecture & Transfer Mechanics

On a single Cloud TPU VM, data transfers between Host CPU memory and the TPU High Bandwidth Memory (HBM) occur over the high-speed PCIe bus. 

* **Host-to-Device (H2D)**: Data is copied from host-allocated pinned memory buffers (allocated via Raiden's host memory allocator) directly into the TPU's device memory.
* **Device-to-Host (D2H)**: Data is streamed directly from the TPU's local memory back into pinned host memory.

### Physical Compatibility
While this guide focuses on **Google Cloud TPU7x** VMs, other TPU generations should also be supported.

---

## Compilation

We provide a bootstrap script (`build.sh`) at the repository root that automatically manages the Bazel build environment and compiles all necessary dependencies.

To bootstrap the build environment and compile the core packages, run:
```bash
./build.sh jax
```

To execute the benchmark, we use the `bazel run` command, which automatically compiles any stale targets before launching the execution.

---

## Execution Walkthrough

### 1. Configure the TPU Driver Path (Dynamic Loading)
The benchmark dynamically loads the physical TPU driver (`libtpu.so`) at runtime. You must export the `TPU_LIBRARY_PATH` environment variable to point to the location of the driver on your system:

```bash
# On standard Google Cloud TPU VMs
export TPU_LIBRARY_PATH="/lib/libtpu.so"
```

### 2. Run the Benchmark
Execute the benchmark using the standard `bazel run` command. Use `--` to separate Bazel arguments from the benchmark's runtime flags:

```bash
bazel run -c opt //tpu_raiden/core:raw_transfer_perf_test -- \
  --num_tpus=1 \
  --num_layers=64 \
  --num_blocks=16
```

*Note: The flags shown above (e.g., `--num_tpus=1`, `--num_layers=64`, `--num_blocks=16`) are the codebase default values. Configuring them is optional, but they can be adjusted to scale the benchmark intensity:*
- `--num_layers`: The number of simulated transformer layers (scales the number of transfer operations).
- `--num_blocks`: The number of memory blocks allocated per layer (scales the buffer memory size).
- `--num_tpus`: The number of TPU devices to target.
