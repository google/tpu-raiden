# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import os
import sys

# Force RTLD_GLOBAL resolution for C++ shared extensions to bridge global symbol pointers!
sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)


# Force CPU compilation generic instruction set targeting to prevent SIGILL crashes on remote worker nodes!
os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=4 --xla_cpu_max_isa=AVX2 "
    + os.environ.get("XLA_FLAGS", "")
)

from absl.testing import absltest
import jax

# Completely disable JAX dynamic compilation cache to prevent SIGILL CPU instruction mismatches on heterogeneous nodes!
jax.config.update("jax_enable_compilation_cache", False)


from api.jax import mc_jax_ffi_test_main as client


class RaidenFfiCorrectnessTest(absltest.TestCase):

  def test_ffi_sharded_copy_correctness(self):
    platform = jax.devices()[0].platform
    devices = jax.devices(platform)
    num_devices = len(devices)
    self.assertIn(num_devices, [4, 8])
    print(
        f"[{platform.upper()} Test] Initializing {num_devices} local {platform}"
        f" devices: {devices}"
    )

    # Setup local sharded JAX mesh
    mesh, tpu_sharding = client.setup_distributed_mesh(devices)

    # Run identical correctness test logic!
    success = client.run_correctness_test(
        mesh=mesh,
        tpu_sharding=tpu_sharding,
        num_processes=1,  # Single local host process
        num_local_devices=num_devices,
        process_id=0,
        global_blocks=64,  # Scaled down cache size for fast E2E tests
        layer_count=2,
    )

    self.assertTrue(
        success, "E2E sharded copy closed-loop verification failed!"
    )


if __name__ == "__main__":
  absltest.main()
