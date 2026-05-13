# Copyright 2026 Google LLC
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

import sys
import os
import jax
import jax.numpy as jnp
import numpy as np

# Point to the bazel-bin directory containing the compiled .so files
# Assuming this script is in raw_transfer_test/ and bazel-bin is in raw_transfer_lib/
# Use PYTHONPATH environment variable to load the library instead of modifying sys.path
# e.g., PYTHONPATH=$(pwd)/bazel-bin python test_import.py


# Import the standalone module directly
import raw_transfer

def main():
    print(f"Testing raw_transfer APIs from compiled standalone library at {raw_transfer.__file__}...")
    
    shape = (128, 128)
    dtype = jnp.float32
    
    try:
        tpu_devices = jax.devices("tpu")
        if not tpu_devices:
            print("No TPU devices found, skipping test.")
            return
    except Exception as e:
        print(f"Error finding TPU: {e}")
        return

    # Create sharding using a mesh to simulate the test behavior
    devices = np.array(tpu_devices[:1])
    mesh = jax.sharding.Mesh(devices, ("x",))
    spec = jax.sharding.PartitionSpec() # replicated
    
    tpu_sharding = jax.sharding.NamedSharding(mesh, spec)
    pinned_host_sharding = jax.sharding.NamedSharding(mesh, spec, memory_kind="pinned_host")
    
    # 1. Create a simple array on TPU
    src_arr = jax.device_put(jnp.ones(shape, dtype=dtype), tpu_sharding)
    print(f"Source array created on: {src_arr.devices()}")

    # 2. Create a destination array on the CPU (pinned_host)
    def _create_zeros():
      return jnp.zeros(shape, dtype=dtype)
    alloc_zeros_cpu = jax.jit(_create_zeros, out_shardings=pinned_host_sharding)
    dst_arr = alloc_zeros_cpu()
    
    print(f"Destination array created on: {dst_arr.devices()}")
    
    try:
        # 3. Test async D2H
        print("\nRunning transfer_d2h_async...")
        future = raw_transfer.transfer_d2h_async(src_arr, dst_arr)
        future.Await()
        print("D2H transfer complete.")
        np.testing.assert_array_equal(np.asarray(dst_arr), np.asarray(src_arr))
        
        # 4. Test async H2D
        print("\nRunning transfer_h2d_async...")
        alloc_zeros_tpu = jax.jit(_create_zeros, out_shardings=tpu_sharding)
        device_dst = alloc_zeros_tpu()
        future = raw_transfer.transfer_h2d_async(dst_arr, device_dst)
        future.Await()
        print("H2D transfer complete.")
        np.testing.assert_array_equal(np.asarray(device_dst), np.asarray(src_arr))
        
        # 5. Test batch APIs
        print("\nRunning transfer_d2h_batch_async...")
        src_arrs = [src_arr, src_arr]
        dst_arrs = [alloc_zeros_cpu(), alloc_zeros_cpu()]
        futures = raw_transfer.transfer_d2h_batch_async(src_arrs, dst_arrs)
        futures.Await()
        print("Batch D2H transfer complete.")
        
        # 6. Test naive batch APIs
        print("\nRunning transfer_d2h_batch_async_naive...")
        dst_arrs_naive = [alloc_zeros_cpu(), alloc_zeros_cpu()]
        futures_naive = raw_transfer.transfer_d2h_batch_async_naive(src_arrs, dst_arrs_naive)
        futures_naive.Await()
        print("Naive Batch D2H transfer complete.")
        np.testing.assert_array_equal(np.asarray(dst_arrs_naive[0]), np.asarray(src_arr))

        print("\nRunning transfer_h2d_batch_async_naive...")
        device_dst_naive = [alloc_zeros_tpu(), alloc_zeros_tpu()]
        futures_h2d_naive = raw_transfer.transfer_h2d_batch_async_naive(dst_arrs_naive, device_dst_naive)
        futures_h2d_naive.Await()
        print("Naive Batch H2D transfer complete.")
        np.testing.assert_array_equal(np.asarray(device_dst_naive[0]), np.asarray(src_arr))
        
        print("\nAll API tests passed successfully!")
        
    except Exception as e:
        print(f"Error during transfer: {e}")

if __name__ == "__main__":
    main()
