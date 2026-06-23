# TPU Raiden

> [!IMPORTANT]
> **TPU Raiden is currently under active development and is not yet recommended for general use.**
> If you are interested in adopting this library, please reach out to the owners first to discuss compatibility, or proceed at your own risk.

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

## Building `tpu_raiden`

We provide a script to handle the build process and install required dependencies. You can scope compilation to specific frameworks:

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

Once the build is complete, you can run the test suite across JAX and PyTorch:

```bash
./run_tests.sh [jax|torch|both]
```

**What this script does:**
1. Sets up `PYTHONPATH` so Python can locate the compiled `bazel-bin` and framework wrapper modules.
2. Executes the selected unit test suites across JAX and/or PyTorch directly via `python`.