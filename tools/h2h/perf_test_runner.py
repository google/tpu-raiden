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

"""Dynamic performance test runner tool for raw memory offloading collectives."""

import os
import sys
import time
from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import numpy as np
from rpc import coordination_helper
from tpu_raiden.api.jax import kv_cache_manager

_ROLE = flags.DEFINE_string(
    'role', None, 'Role of the task: sender or receiver.'
)
_PEER = flags.DEFINE_string(
    'peer', None, 'Dynamic gRPC address of the peer (for receiver).'
)
_GRPC_PORT = flags.DEFINE_integer(
    'grpc_port', 50051, 'Pre-agreed static gRPC coordination port.'
)
_NUM_BLOCKS = flags.DEFINE_integer(
    'num_blocks', 512, 'Number of cache blocks to allocate.'
)
_BLOCK_SIZE = flags.DEFINE_integer('block_size', 8, 'Size of cache blocks.')
_NUM_LAYERS = flags.DEFINE_integer(
    'num_layers', 8, 'Number of transformer layers.'
)
_PARALLELISM = flags.DEFINE_integer(
    'parallelism', 1, 'Number of parallel TCP streams for H2H.'
)
_PEER_IPS = flags.DEFINE_string(
    'peer_ips', '', 'Comma-separated list of peer IPs for multi-NIC transfers.'
)
_LOCAL_IPS = flags.DEFINE_string(
    'local_ips',
    '',
    'Comma-separated list of local IPs for multi-NIC transfers.',
)


def setup_shardings(devices):
  num_devices = len(devices)
  axis_shapes = (1, num_devices)
  axis_names = ('data', 'model')
  devices_array = np.array(devices).reshape(axis_shapes)
  mesh = jax.sharding.Mesh(devices_array, axis_names)
  spec = jax.sharding.PartitionSpec(None, None, 'model')

  tpu_sharding = jax.sharding.NamedSharding(mesh, spec)
  host_sharding = jax.sharding.NamedSharding(
      mesh, spec, memory_kind='pinned_host'
  )
  return tpu_sharding, host_sharding


def format_ipv6_endpoint(address_str: str) -> str:
  """Encloses raw IPv6 address in square brackets [] for gRPC/sockets."""
  if ':' in address_str and not address_str.startswith('['):
    parts = address_str.rsplit(':', 1)
    if (
        len(parts) == 2 and '.' not in parts[0]
    ):  # Check if it is an IPv6 hex address
      return f'[{parts[0]}]:{parts[1]}'
  return address_str


def get_peer_grpc_bns_path(peer_arg: str) -> str:
  """Constructs the FQDN google:///bns/... path dynamically."""
  if 'google://' in peer_arg:
    return peer_arg

  author = os.environ.get('XM_EXPERIMENT_AUTHOR')
  xid = os.environ.get('XM_XID')
  wid = os.environ.get('XM_WID', '1')
  chubby_cell = os.environ.get('LOCKSERVICE_LOCALITY', 'ok')
  cell = os.environ.get('BORG_CELL', chubby_cell)

  if not author or not xid:
    # Fallback for local workstation testing
    return peer_arg

  peer_role = peer_arg.split(':')[0]
  peer_port_name = 'grpc'  # Pre-allocated port name registered in BNS

  bns_path = f'/bns/{chubby_cell}/borg/{cell}/bns/{author}/{author}_group_{xid}.{wid}.{peer_role}/0:{peer_port_name}'
  resolved_peer_bns = f'google:///{bns_path}'
  print(f'Dynamic peer BNS coordinate resolved: {resolved_peer_bns}')
  return resolved_peer_bns


def populate_deterministic_cache(
    num_blocks: int, num_layers: int, shape: tuple[int, ...], sharding
) -> list[jax.Array]:
  """Populates unique deterministic sequence data across layers on Sender."""
  arrs = []
  for layer_idx in range(num_layers):
    # Generate unique float32 data sequence based on layer index
    base = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) + float(
        layer_idx * 1000.0
    )
    arrs.append(jax.device_put(base, sharding))
  jax.block_until_ready(arrs)
  return arrs


def verify_deterministic_cache(
    num_blocks: int,
    num_layers: int,
    shape: tuple[int, ...],
    dst_tpu_arrs: list[jax.Array],
) -> bool:
  """Verifies Receiver's pulled arrays contain exact sequence from Sender."""
  print(
      'Verifying data consistency across all sharded cache layers'
      ' (optimized)...'
  )
  for layer_idx in range(num_layers):
    actual = dst_tpu_arrs[layer_idx]
    expected = jax.device_put(
        jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
        + float(layer_idx * 1000.0),
        actual.sharding,
    )
    correct = jnp.all(actual == expected)
    if not bool(correct):
      print(f'Verification FAILED on Layer {layer_idx}!')

      flat_actual = np.array(actual).flatten()
      flat_expected = np.array(expected).flatten()
      mismatches = flat_actual != flat_expected
      num_mismatches = np.sum(mismatches)
      print(
          f'  Mismatched elements: {num_mismatches} / {flat_actual.size}'
          f' ({num_mismatches / flat_actual.size * 100:.4f}%)'
      )
      mismatch_indices = np.where(mismatches)[0]
      print(f'  First 10 mismatch indices: {mismatch_indices[:10]}')
      for idx in mismatch_indices[:10]:
        print(
            f'    Index {idx}: actual={flat_actual[idx]},'
            f' expected={flat_expected[idx]}'
        )
      return False
  print('Data consistency verified successfully! 0% corruption.')
  return True


def main(_):
  if not _ROLE.value:
    raise ValueError('--role must be specified')

  devices = jax.devices('tpu')
  if not devices:
    raise RuntimeError('No TPU devices found.')
  # Sort JAX devices by physical NUMA node to force contiguous sharding
  sorted_devices = sorted(
      devices, key=lambda d: d.numa_node if hasattr(d, 'numa_node') else -1
  )
  print(
      f'Initialized JAX. Local TPU chips available: {len(sorted_devices)}'
      ' (sorted by NUMA)'
  )

  tpu_sharding, _ = setup_shardings(sorted_devices)

  # Physical sharding shape matching kv_cache_manager layouts
  # Layout: (num_blocks, head_count, block_size, head_dim)
  # Shape: (_NUM_BLOCKS, 32, _BLOCK_SIZE, 8, 128)
  cache_shape = (_NUM_BLOCKS.value, 32, _BLOCK_SIZE.value, 8, 128)

  if _ROLE.value == 'sender':
    print('Starting H2H Sender process...')

    # Populate unique cache data in TPU memory
    tpu_src_arrs = populate_deterministic_cache(
        _NUM_BLOCKS.value, _NUM_LAYERS.value, cache_shape, tpu_sharding
    )

    # Start background coordination server on Borg-allocated named port
    grpc_port_env = os.environ.get('BORG_PORT_GRPC')
    if grpc_port_env:
      grpc_port = int(grpc_port_env)
      print(f'Borg-allocated named port grpc detected: {grpc_port}')
    else:
      grpc_port = _GRPC_PORT.value
      print(
          'WARNING: BORG_PORT_GRPC not found. Falling back to flag default:'
          f' {grpc_port}'
      )

    coordination_server = coordination_helper.CoordinationServer(port=grpc_port)
    bound_grpc_port = coordination_server.start()
    print(f'Coordination gRPC server started on port: {bound_grpc_port}')

    # Resolve local IPs (with fallback to hostname resolution)
    manager = None
    try:
      # Start KVCacheManager using the high-level Python wrapper.
      manager = kv_cache_manager.KVCacheManager(
          kv_caches=tpu_src_arrs,
          local_control_port=0,
          max_blocks=_NUM_BLOCKS.value,
          num_slots=2,
          unsafe_skip_buffer_lock=True,
      )
      # Offload TPU data to internal C++ host buffer
      manager.d2h().Await()

      ips = manager.local_ips
      ports = manager.local_ports
      print(f'KVCacheManager bound to IPs: {ips}, Ports: {ports}')

      # Register the read with a deterministic transfer ID and UUID for coordination
      total_cache_blocks = _NUM_BLOCKS.value
      block_ids = list(range(total_cache_blocks))
      transfer_req_id = 'perf_test_runner_transfer'
      transfer_uuid = 99999
      manager.register_read(transfer_req_id, transfer_uuid, block_ids)
      print(
          f'Registered read for transfer: {transfer_req_id} (UUID:'
          f' {transfer_uuid})'
      )

      # Share dynamic socket server endpoints and block IDs
      endpoints = ','.join(f'{ip}:{port}' for ip, port in zip(ips, ports))
      coordination_server.set_metadata(
          port=0, block_ids=block_ids, host_ip=endpoints
      )
      print(
          f'Metadata published! Endpoints: {endpoints}. Waiting for Receiver...'
      )

      # Block cleanly until peer Receiver signals transfer completion
      try:
        coordination_server.wait_for_shutdown()
        print('Receiver finished! Shutting down Sender coordination server...')
      except KeyboardInterrupt:
        pass
      finally:
        coordination_server.stop()
    finally:
      if manager:
        print('Closing KVCacheManager...')
        manager.close()

  elif _ROLE.value == 'receiver':
    print('Starting H2H Receiver process...')
    # Connect and pull dynamic metadata (coordination via native BNS resolving)
    resolved_bns_peer = get_peer_grpc_bns_path(_PEER.value)
    print(f'Connecting to peer coordination BNS: {resolved_bns_peer}')
    client = coordination_helper.CoordinationClient(
        server_address=resolved_bns_peer
    )

    max_retries = 15
    peer_socket_port = None
    src_block_ids = []
    peer_host_ip = None
    for attempt in range(1, max_retries + 1):
      try:
        peer_socket_port, src_block_ids, peer_host_ip = client.get_metadata()
        break
      except Exception as e:
        print(
            f'Attempt {attempt}/{max_retries} waiting for peer BNS registration'
            f' ({e}). Retrying in 5s...'
        )
        time.sleep(5)

    if peer_socket_port is None:
      raise RuntimeError(
          f'Failed to coordinate with peer {resolved_bns_peer} after'
          f' {max_retries} attempts.'
      )

    print(
        f'Metadata received! Dynamic peer socket port: {peer_socket_port},'
        f' peer host IP: {peer_host_ip}, block count: {len(src_block_ids)}'
    )

    # Setup local empty device arrays
    device_arrs = [
        jax.device_put(jnp.empty(cache_shape, dtype=jnp.float32), tpu_sharding)
        for _ in range(_NUM_LAYERS.value)
    ]
    jax.block_until_ready(device_arrs)

    # Settle dynamic socket endpoint address (format as [IPv6]:port for each)
    def format_endpoints(endpoints_str):
      parts = [ep.strip() for ep in endpoints_str.split(',')]
      formatted = [format_ipv6_endpoint(ep) for ep in parts]
      return ','.join(formatted)

    peer_h2h_address = format_endpoints(peer_host_ip)
    print(f'Resolved Peer dynamic H2H endpoints: {peer_h2h_address}')

    manager = None
    try:
      # Initialize local KVCacheManager using the wrapper
      manager = kv_cache_manager.KVCacheManager(
          kv_caches=device_arrs,
          local_control_port=0,
          max_blocks=_NUM_BLOCKS.value,
          num_slots=2,
          unsafe_skip_buffer_lock=True,
      )

      def get_rx_bytes(iface):
        try:
          with open(f'/sys/class/net/{iface}/statistics/rx_bytes', 'r') as f:
            return int(f.read().strip())
        except Exception:
          return 0

      # Measure E2E transfer performance
      print('Executing H2H Read E2E offloading transfer...')
      rx_eth0_start = get_rx_bytes('eth0')
      rx_eth1_start = get_rx_bytes('eth1')
      start_time = time.perf_counter()

      # Use start_read directly with matching deterministic IDs and dual-NIC multi-endpoints
      transfer_req_id = 'perf_test_runner_transfer'
      transfer_uuid = 99999

      future = manager.start_read(
          req_id=transfer_req_id,
          uuid=transfer_uuid,
          remote_endpoint=peer_h2h_address,
          remote_block_ids=src_block_ids,
          local_block_ids=src_block_ids,
          parallelism=_PARALLELISM.value,
      )

      # Await completion
      future.Await()

      end_time = time.perf_counter()
      rx_eth0_end = get_rx_bytes('eth0')
      rx_eth1_end = get_rx_bytes('eth1')

      elapsed_time = end_time - start_time

      # Calculate metrics
      block_byte_size = np.prod(cache_shape[1:]) * 4
      global_bytes = _NUM_LAYERS.value * len(src_block_ids) * block_byte_size
      local_bytes = global_bytes / jax.process_count()

      total_megabytes = local_bytes / (1024 * 1024)
      bandwidth_gbps = (local_bytes * 8) / (elapsed_time * 1e9)

      bytes_eth0 = rx_eth0_end - rx_eth0_start
      bytes_eth1 = rx_eth1_end - rx_eth1_start
      bw_eth0 = (bytes_eth0 * 8) / (elapsed_time * 1e9)
      bw_eth1 = (bytes_eth1 * 8) / (elapsed_time * 1e9)

      print('\n--- H2H Performance Test Results ---')
      print(f'Data Volume Transferred: {total_megabytes:.2f} MB')
      print(f'Elapsed Time (TCP H2H + Copy): {elapsed_time:.4f} seconds')
      print(f'Effective Bandwidth: {bandwidth_gbps:.3f} Gbps')
      print('--- NIC Bandwidth Breakdown ---')
      print(f'  eth0 (standard): {bw_eth0:.3f} Gbps')
      print(f'  eth1 (jumbo   ): {bw_eth1:.3f} Gbps')
      print(f'  Sum Measured   : {bw_eth0 + bw_eth1:.3f} Gbps')
      print('------------------------------------\n')

      # Flush host cache data onto TPU device memory
      print('Flushing H2H host blocks to TPU local device memory...')
      manager.h2d().Await()

      # Validate data consistency
      success = verify_deterministic_cache(
          _NUM_BLOCKS.value, _NUM_LAYERS.value, cache_shape, device_arrs
      )
      if not success:
        # Signal Sender to exit even on failure to avoid dangling resources
        client.shutdown()
        sys.exit(1)

      # Signal successful transfer complete to unblock Sender
      print('Signalling completion to peer Sender...')
      client.shutdown()
      print('E2E performance test runner completed successfully!')
    finally:
      if manager:
        print('Closing KVCacheManager...')
        manager.close()

  else:
    raise ValueError(f'Unknown role: {_ROLE.value}')


if __name__ == '__main__':
  # Bypasses supervisor-injected flags on physical TPU nodes
  app.run(main, flags_parser=lambda args: flags.FLAGS(args, known_only=True))
