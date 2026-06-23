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

"""Integration tests for JAX WeightSynchronizer Python API."""

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from absl.testing import absltest  # pylint: disable=g-import-not-at-top
import jax
import jax.numpy as jnp
import socket
import numpy as np

from tpu_raiden.api.jax import weight_synchronizer
from tpu_raiden.weight_sync import weight_synchronizer_service_pb2

WeightSynchronizer = weight_synchronizer.WeightSynchronizer


class WeightSynchronizerIntegrationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    try:
      self.devices = jax.devices("tpu")
    except RuntimeError:
      self.devices = jax.devices("cpu")
    self.mesh = jax.sharding.Mesh(np.array(self.devices), ("data",))
    self.sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec("data")
    )
    self.shape = (8, 128)
    self.dtype = jnp.float32

  def test_push_synchronization(self):
    src_arrs = [
        jax.device_put(
            jnp.ones(self.shape, dtype=self.dtype) * 5.0, self.sharding
        )
    ]
    dst1_arrs = [
        jax.device_put(jnp.zeros(self.shape, dtype=self.dtype), self.sharding)
    ]
    dst2_arrs = [
        jax.device_put(jnp.zeros(self.shape, dtype=self.dtype), self.sharding)
    ]

    for arr in src_arrs:
      arr.block_until_ready()
    for arr in dst1_arrs:
      arr.block_until_ready()
    for arr in dst2_arrs:
      arr.block_until_ready()

    ws_source = WeightSynchronizer(
        jax_arrays=src_arrs,
        local_port=0,
        unsafe_skip_buffer_lock=True,
        control_port=0,
    )
    ws_dest1 = WeightSynchronizer(
        jax_arrays=dst1_arrs, local_port=0, unsafe_skip_buffer_lock=True
    )
    ws_dest2 = WeightSynchronizer(
        jax_arrays=dst2_arrs, local_port=0, unsafe_skip_buffer_lock=True
    )

    req = weight_synchronizer_service_pb2.ControlRequest(
        command=weight_synchronizer_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
        peers=[
            f"127.0.0.1:{ws_dest1.local_port}",
            f"127.0.0.1:{ws_dest2.local_port}",
        ],
    )
    payload = req.SerializeToString()

    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM, 0)
    sock.connect(("::1", ws_source.control_port))
    sock.sendall(len(payload).to_bytes(4, "big") + payload)

    resp_len = int.from_bytes(sock.recv(4), "big")
    resp_bytes = sock.recv(resp_len)
    resp = weight_synchronizer_service_pb2.ControlResponse()
    resp.ParseFromString(resp_bytes)
    assert resp.success
    sock.close()

    for arr in dst1_arrs:
      np.testing.assert_array_equal(np.asarray(arr), 5.0)
    for arr in dst2_arrs:
      np.testing.assert_array_equal(np.asarray(arr), 5.0)

  def test_pull_synchronization(self):
    src_arrs = [
        jax.device_put(
            jnp.ones(self.shape, dtype=self.dtype) * 10.0, self.sharding
        )
    ]
    dst1_arrs = [
        jax.device_put(jnp.zeros(self.shape, dtype=self.dtype), self.sharding)
    ]
    dst2_arrs = [
        jax.device_put(jnp.zeros(self.shape, dtype=self.dtype), self.sharding)
    ]

    for arr in src_arrs:
      arr.block_until_ready()
    for arr in dst1_arrs:
      arr.block_until_ready()
    for arr in dst2_arrs:
      arr.block_until_ready()

    ws_source = WeightSynchronizer(
        jax_arrays=src_arrs,
        local_port=0,
        unsafe_skip_buffer_lock=True,
        control_port=0,
    )
    ws_dest1 = WeightSynchronizer(
        jax_arrays=dst1_arrs, local_port=0, unsafe_skip_buffer_lock=True
    )
    ws_dest2 = WeightSynchronizer(
        jax_arrays=dst2_arrs, local_port=0, unsafe_skip_buffer_lock=True
    )

    # Self-push to populate ws_source's host buffer with current device weights
    req = weight_synchronizer_service_pb2.ControlRequest(
        command=weight_synchronizer_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
        peers=[f"127.0.0.1:{ws_source.local_port}"],
    )
    payload = req.SerializeToString()

    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM, 0)
    sock.connect(("::1", ws_source.control_port))
    sock.sendall(len(payload).to_bytes(4, "big") + payload)

    resp_len = int.from_bytes(sock.recv(4), "big")
    resp_bytes = sock.recv(resp_len)
    resp = weight_synchronizer_service_pb2.ControlResponse()
    resp.ParseFromString(resp_bytes)
    assert resp.success
    sock.close()

    ws_dest1.pull_weights(f"127.0.0.1:{ws_source.local_port}")
    ws_dest2.pull_weights(f"127.0.0.1:{ws_source.local_port}")

    for arr in dst1_arrs:
      np.testing.assert_array_equal(np.asarray(arr), 10.0)
    for arr in dst2_arrs:
      np.testing.assert_array_equal(np.asarray(arr), 10.0)


if __name__ == "__main__":
  absltest.main()
