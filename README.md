# TPU Raiden

> [!IMPORTANT]
> **TPU Raiden is currently under active development and is not yet recommended for general use.**
> If you are interested in adopting this library, please reach out to the owners first to discuss compatibility, or proceed at your own risk.

## Prerequisites

### Installing Bazel
To compile the `raw_transfer` binaries, you will need **Bazel 7.7.0**.

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

## Building `raw_transfer`

We provide a script to handle the build process and install the required Python dependencies. Run the following command from the repository root:

```bash
./build_raw_transfer.sh
```

**What this script does:**
1. Navigates to the workspace directory.
2. Compiles the `//raw_transfer:raw_transfer_binaries` target using Bazel (it uses `--disk_cache` to speed up subsequent builds).
3. Installs the necessary Python dependencies listed in `requirements.txt`.
4. Artifacts will be available in the `bazel-bin/raw_transfer/` directory.

## Testing `raw_transfer`

Once the build is complete, you can run the tests to verify the installation and check performance:

```bash
./test_raw_transfer.sh
```

**What this script does:**
1. Sets up the `PYTHONPATH` so Python can locate the compiled `bazel-bin` artifacts.
2. Executes `test_import.py` to ensure the C++ extensions load correctly, saving the output to `import.log`.
3. Executes `test_raw_transfer_perf.py` to benchmark performance, saving the output to `perf_test.log`.