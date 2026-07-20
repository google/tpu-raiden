# TPU Raiden

> [!IMPORTANT]
> **TPU Raiden is currently under active development and is not yet recommended for general use.**
> If you are interested in adopting this library, please reach out to the owners first to discuss compatibility, or proceed at your own risk.

## Latest Known Good (LKG) Revision
Due to fast-paced active development, the `main` branch may occasionally contain temporary unstable changes. We publish the latest verified stable commit hash in the [`lkg.version`](lkg.version) file at the root of the repository.

To check out the latest verified stable revision that passes all E2E functional and performance test criteria:
```bash
git checkout $(cat lkg.version)
```

## Prerequisites

You will need a python environment to run the JAX or torch code. Our code has been verified with python3.12. So the following should set you up properly:

```bash
cd
python3.12 -m venv .venv312
source .venv312/bin/activate
```

### Installing Bazel
To compile the `tpu_raiden` C++ extension binaries, you will need **Bazel 7.7.0**.

**Option 1: Install Bazel 7.7.0 directly (Linux amd64)**
```bash
sudo wget -O /usr/local/bin/bazel https://github.com/bazelbuild/bazel/releases/download/7.7.0/bazel-7.7.0-linux-x86_64
sudo chmod +x /usr/local/bin/bazel
```

**Option 2: Install via Bazelisk (npm)**
Bazelisk is a wrapper that will automatically read the `.bazelversion` file in the project and download the correct version (7.7.0).
```bash
npm install -g @bazel/bazelisk
```

Verify the installation:
```bash
bazel --version
```

### Installing Patchelf (Required for PyTorch)
To compile and link the PyTorch C++ extension (`_tpu_raiden_torch.so`), you MUST install `patchelf`:
```bash
sudo apt-get install -y patchelf
```
*Why this is necessary:* PyTorch's compiled extension requires `patchelf` to inject a `NEEDED` link on `libpywrap_torch_tpu_common.so` at build time. This ensures TPU backend symbols resolve locally during import without triggering fatal duplicate XLA allocator registration crashes.

### TPUVM Development Notes
* **Disk Space**: Remote Bazel builds on standard TPUVMs can exhaust disk space in `/tmp`. Always point Bazel output to a directory that has enough disk space left.:
  ```bash
  export BAZEL_OUTPUT_BASE=$YOUR_TMP_DIR_WITH_ENOUGH_SPACE
  ```
* **PyTorch Wheel Compatibility**: Ensure your environment aligns with `torch_tpu`'s pinned C++ ABI expectations (e.g., `torch==2.11.0+cpu`).

## Installing or Building `tpu_raiden`

### Option 1: Direct installation from Google Artifact Registry (Googlers only)

> [!NOTE]
> The pre-built `tpu_raiden` wheel will be available on PyPI to public shortly.

If you are a Googler, you can install the pre-built `tpu_raiden` wheel directly from our Google Artifact Registry.

1. Install the Artifact Registry keyring helper to enable authenticated pip downloads:
   ```bash
   pip install keyrings.google-artifactregistry-auth
   ```
2. Install the framework-specific wheel:
   * **For JAX version:**
     ```bash
     pip install tpu-raiden-jax --extra-index-url https://us-python.pkg.dev/cloud-tpu-inference-test/tpu-raiden/simple/
     ```
   * **For PyTorch version:**
     Torch specific wheel will be published soon.

### Option 2: Building from source

We provide a script to handle the build process and compile extension binaries locally. You can scope compilation to specific frameworks:

```bash
./build.sh [jax|torch|both]
```

**What this script does:**
1. Navigates to the workspace directory.
2. Compiles the selected extension modules (`_tpu_raiden_jax.so` and/or `_tpu_raiden_torch.so`) using Bazel.
3. For PyTorch builds, executes `patchelf --add-needed` on the generated shared library.
4. Installs necessary Python dependencies listed in `requirements.txt`.
5. Copies compiled `.so` extension binaries directly into their respective framework source packages.

## Testing `tpu_raiden`

These are the core functional unit tests designed to verify the correctness of the foundational components and APIs. Once the build is complete, you can run the test suite across JAX and PyTorch:

```bash
./run_tests.sh [jax|torch|both]
```

**What this script does:**
1. Sets up `PYTHONPATH` so Python can locate the compiled `bazel-bin` and framework wrapper modules.
2. Executes the selected unit test suites across JAX and/or PyTorch directly via `python`.

## Playing with Raiden

If you'd like to try out Raiden and see it in action, please refer to the [`examples/`](examples/) directory. This folder contains a collection of hands-on scripts designed for users to interact with the library, including various testing scripts and performance microbenchmark scripts that demonstrate Raiden's capabilities.

For detailed instructions on how to run these examples and interpret their outputs, please check out the [Examples README](examples/README.md).

## Persistent Shared Memory Cache (TPU/Host Buffer Persistence)

TPU Raiden supports allocating host memory staging buffers in POSIX Shared Memory (`/dev/shm`). This allows preserving the KV cache in DRAM when the model serving process terminates (e.g., during serving binary updates), preventing cold starts on process restarts.

### 1. Enabling Shared Memory
To enable shared memory, set the following environment variables before starting the model serving process:

```bash
# Enable shared memory by specifying a base namespace key
export RAIDEN_SHM_KEY="raiden_cache"

# Specify a unique identifier of the current model config for validation safety
export RAIDEN_SHM_MODEL_UID="llama_70b_v1_config_hash"

# [Optional] Set a server name if running multiple serving instances on the same host
export RAIDEN_SHM_SERVER_NAME="server_8000"
```

When these variables are active, Raiden will automatically check for compatible shared memory segments:
- **Cold Boot (First run)**: Raiden creates `/dev/shm/raiden_cache_<server_name>_dev_<local_dev_id>` files, initializes layout validation metadata headers, and sets up mappings.
- **Warm Boot (Restarts)**: Raiden automatically re-attaches to the existing shared memory files, verifies that the model configuration (`RAIDEN_SHM_MODEL_UID` and caching dimensions) matches, and re-registers the pages with the TPU DMA engine without re-allocation.

### 2. Running Multiple Servers on the Same Host
If you are running multiple model servers on the same TPU VM, you can avoid namespace collisions by specifying a unique `RAIDEN_SHM_SERVER_NAME` for each server instance (e.g. `server_8000` and `server_8008`). If specified, Raiden automatically namespaces the file paths as `/dev/shm/<base_key>_<server_name>_dev_<dev_id>`.

### 3. Disabling Shared Memory
To disable shared memory and fall back to standard anonymous private memory allocations, simply unset the environment variables:

```bash
unset RAIDEN_SHM_KEY
unset RAIDEN_SHM_MODEL_UID
unset RAIDEN_SHM_SERVER_NAME
```

### 3. Manual Cleanup & Memory Reclamation
Because POSIX shared memory files survive process termination, you may need to clean them up manually to free up host DRAM on the TPUVM.

To see currently allocated Raiden shared memory files:
```bash
ls -la /dev/shm/ | grep raiden_cache
```

To reclaim memory, unlink/delete the shared memory files:
```bash
rm -f /dev/shm/raiden_cache_*
```
*(Note: unlinking deletes the filenames immediately, and the physical host DRAM pages are freed by the kernel as soon as all active serving processes detach or exit).*\n# Test BAP
