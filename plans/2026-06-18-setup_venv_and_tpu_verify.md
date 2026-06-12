# Integration, Refactoring, and TPU VM Verification Plan for `tpu_raiden`

This document details the step-by-step implementation and verification plan to
transition the `tpu_raiden` virtual environment bootstrapping from fragile
shell-based hacks to a robust, Bazel-run Python tool, clean up the Kokoro CI
presubmit runner, and execute physical TPU VM integration tests.

## 1. Pre-Flight Check & VCS Hygiene Setup

To prevent committing the compiled C++ extension binary (`_tpu_raiden_torch.so`)
and to ensure a clean base, the following pre-flight steps must be executed.

### Step 1.1: Workspace Sync and Rebase

1.  **Sync to HEAD**: Sync the Jujutsu workspace to fully integrate the
    submitted CL 934048334.

    ```bash
    /usr/bin/jj piper sync --cl 934048334
    ```

2.  **Conflict Audit**: Verify that the rebase completed with zero merge
    conflicts.

    ```bash
    /usr/bin/jj status --no-pager --color=never
    ```

    *If conflicts are present, they must be resolved before proceeding.*

### Step 1.2: VCS Ignored Patterns Audit

1.  **Create/Update `.gitignore`**: Create or update the `.gitignore` file at
    the root of `third_party/tpu_raiden/` to explicitly ignore compiled shared
    libraries so that Jujutsu does not automatically track them during
    compilation. *Target File*:
    `/google/src/cloud/amylin/fix-guitar-ci-runner/google3/third_party/tpu_raiden/.gitignore`
    *Content*:

    ```
    # Ignore compiled C++ shared libraries and build artifacts
    *.so
    *.so.*
    *.abi3.so
    ```

2.  **VCS Untrack**: Untrack any accidentally tracked binary files in the
    working copy:

    ```bash
    /usr/bin/jj file untrack third_party/tpu_raiden/tpu_raiden/frameworks/torch/_tpu_raiden_torch.so
    ```

3.  **Verify Gitignore**: Verify that `_tpu_raiden_torch.so` is no longer shown
    as a tracked addition in `jj status`.

--------------------------------------------------------------------------------

## 2. Step-by-Step Implementation

### Task 2.1: Implement the Native Virtualenv Setup Tool (To be executed by Implementer)

-   **Target Files**:
    -   `/google/src/cloud/amylin/fix-guitar-ci-runner/google3/third_party/tpu_raiden/tools/setup_venv.py`
    -   `/google/src/cloud/amylin/fix-guitar-ci-runner/google3/third_party/tpu_raiden/tools/BUILD`
-   **Verification Target**: `//third_party/tpu_raiden/tools:setup_venv`

1.  **Create Directory**: Create
    `/google/src/cloud/amylin/fix-guitar-ci-runner/google3/third_party/tpu_raiden/tools/`.
2.  **Implement `setup_venv.py`**:

    -   Write a robust, pure Python script that:

        -   Reads `sys.executable` to resolve the path of the hermetic Python
            interpreter.
        -   Parses command-line arguments:
        -   `--venv_dir`: Absolute path where the virtualenv should be created.
        -   `--wheels_dir`: Absolute path to the directory containing
            precompiled wheels.
        -   Programmatically creates the virtualenv using the standard `venv`
            library:

        ```python
        import venv
        venv.create(args.venv_dir, with_pip=True)
        ```

        -   Programmatically upgrades `pip` and installs all wheels (`*.whl`)
            found in the `--wheels_dir` using `subprocess.run` with the
            virtualenv's Python/Pip executable:

        ```python
        # E.g., [venv_dir]/bin/pip install --upgrade pip
        # E.g., [venv_dir]/bin/pip install --find-links=[wheels_dir] [wheels_dir]/*.whl
        ```

3.  **Define BUILD Target**:

    -   Create
        `/google/src/cloud/amylin/fix-guitar-ci-runner/google3/third_party/tpu_raiden/tools/BUILD`.
    -   Define the `setup_venv` target as a standard `py_binary`:

        ```bazel
        load("//third_party/bazel_rules/rules_python/python:py_binary.bzl", "py_binary")

        package(
            default_visibility = ["//third_party/tpu_raiden:internal"],
        )

        py_binary(
            name = "setup_venv",
            srcs = ["setup_venv.py"],
            srcs_version = "PY3",
        )
        ```

4.  **Delete Obsolete Bootstrapper**:

    -   Completely delete
        `/google/src/cloud/amylin/fix-guitar-ci-runner/google3/third_party/tpu_raiden/bootstrap_python.py`.
    -   Remove the `bootstrap_python` target from
        `/google/src/cloud/amylin/fix-guitar-ci-runner/google3/third_party/tpu_raiden/BUILD`.

### Task 2.2: Clean up CI Runner Script (To be executed by Implementer)

-   **Target File**:
    `/google/src/cloud/amylin/fix-guitar-ci-runner/google3/third_party/tpu_raiden/kokoro/gcp_ubuntu/presubmit.sh`

1.  **Remove Fragile Bootstrap Code**:
    -   Remove the fragile `cquery`, `find`, and dummy file writing hacks (lines
        65-87 in the original script).
    -   Remove the manual virtualenv creation and wheel installation commands
        (lines 88-96 in the original script).
2.  **Integrate Setup Tool**:

    -   Replace the removed block with a clean invocation of `bazel run
        //tools:setup_venv`:

        ```bash
        echo "=== 5. Setting up Hermetic Python Virtual Environment ==="
        HERMETIC_VENV="${WORK_DIR}/venv_hermetic"
        "${BAZEL_BIN}" "${BAZEL_STARTUP_FLAGS[@]}" run \
          --override_module=torch_tpu=${DUMMY_TORCH_TPU_MODULE} \
          //tools:setup_venv -- --venv_dir="${HERMETIC_VENV}" --wheels_dir="${WHEELS_DIR}"

        source "${HERMETIC_VENV}/bin/activate"

        # Export PYTHON_BIN_PATH for the custom repository rule
        export PYTHON_BIN_PATH="${HERMETIC_VENV}/bin/python"
        echo "Exported PYTHON_BIN_PATH=${PYTHON_BIN_PATH}"
        ```

--------------------------------------------------------------------------------

## 3. Local CPU Emulation Verification

To guarantee a 100% green baseline on CPU before migrating to the physical TPU
VM, execute the following verification suite locally.

1.  **Format & Clean**:
    -   Run `jj fix` to format all modified files.
    -   Run `build_cleaner` on `third_party/tpu_raiden/tools` and
        `third_party/tpu_raiden`.
2.  **CPU Build**:

    -   Build all targets in the package to verify compilation:

        ```bash
        blaze build //third_party/tpu_raiden/... --noshow_progress --show_result=0 --curses=no --color=no --keep_going
        ```

3.  **CPU Unit Tests**:

    -   Execute CPU-bound unit tests:

        ```bash
        blaze test //third_party/tpu_raiden/kv_cache:logical_block_manager_test --noshow_progress --show_result=0 --curses=no --color=no --keep_going
        ```

    -   Verify that all local tests pass cleanly.

--------------------------------------------------------------------------------

## 4. Physical TPU VM Verification (Both JAX and Torch)

Once the CPU baseline is verified, perform a physical deployment and validation
run on the remote TPU VM.

### Remote VM Details

-   **VM Name**: `ai-ninja-tw-tpu7x-8-4`
-   **IP Address**: `136.116.96.118`
-   **User**: `amylin`

### Step 4.1: Sync to TPU VM (To be executed by Implementer)

1.  Sync all local workspace changes to the remote TPU VM using `gcloud compute
    tpus tpu-vm scp` or the project's sync script.
2.  Verify that the synced directory on the VM contains the new `tools/` folder
    and the modified `presubmit.sh`.

### Step 4.2: Compilation & Extension Verification (To be executed by Implementer)

1.  **SSH to TPU VM**: Connect to the remote VM.
2.  **Execute Setup Tool**: Run the setup tool using the VM's Bazel to create
    the virtualenv and install the physical JAX/Torch wheels.

    ```bash
    bazel run //tools:setup_venv -- --venv_dir="${HOME}/venv_tpu" --wheels_dir="${HOME}/wheels"
    ```

3.  **Compile Extensions**:

    -   Activate the virtualenv:

        ```bash
        source "${HOME}/venv_tpu/bin/activate"
        ```

    -   Compile the C++ extensions for both JAX and Torch on the physical
        hardware:

        ```bash
        ./build.sh jax
        ./build.sh torch
        ```

4.  **VCS Hygiene Check**:

    -   Run `jj status` or `git status` on the VM to verify that the newly
        compiled binary `_tpu_raiden_torch.so` is **not** tracked by the VCS
        (due to the `.gitignore` rules).

### Step 4.3: Physical Integration Tests (To be executed by Implementer)

1.  **JAX Physical Integration Tests**:

    -   Run the JAX physical integration tests on the TPU:

        ```bash
        python3 tpu_raiden/frameworks/jax/kv_cache_manager_test.py
        ```

2.  **Torch Physical Integration Tests**:

    -   Run the Torch physical integration tests on the TPU:

        ```bash
        python3 tpu_raiden/frameworks/torch/kv_cache_manager_test.py
        ```

3.  **Confirm Success**: Ensure both JAX and Torch integration tests complete
    successfully with 0 failures on physical TPU hardware.

--------------------------------------------------------------------------------

## 5. Final Codebase Integration & CL Submission

1.  **Re-verify VCS Cleanliness**:

    -   Confirm that no binary `.so` files are staged or tracked in the local
        workspace:

        ```bash
        /usr/bin/jj status --no-pager --color=never
        ```

    -   Verify that the diff only contains the intended text changes:

        ```bash
        /usr/bin/jj diff --git --no-pager --color=never
        ```

2.  **Upload to Critique**:

    -   Upload the changes to create/update the CL:

        ```bash
        /usr/bin/jj piper upload
        ```

3.  **Presubmit Validation**:

    -   Run presubmits on the uploaded CL:

        ```bash
        /usr/bin/jj piper presubmit --eager
        ```
