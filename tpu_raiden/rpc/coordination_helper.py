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

"""gRPC helper for coordination of Raiden metadata."""

import concurrent.futures
import threading
from typing import Any, Dict, List, NamedTuple, Tuple
import grpc
from tpu_raiden.rpc import coordination_pb2
from tpu_raiden.rpc import coordination_pb2_grpc


class TransferMetadata(NamedTuple):
  endpoints: List[Dict[str, Any]]
  transfer_uuid: int
  transfer_req_id: str
  block_ids: List[int]


class CoordinationServicer(coordination_pb2_grpc.CoordinationServiceServicer):
  """gRPC servicer to hold transfer metadata dynamically."""

  def __init__(self):
    self._metadata_event = threading.Event()
    self._shutdown_event = threading.Event()
    self._endpoints = []
    self._transfer_uuid = 0
    self._transfer_req_id = ''
    self._block_ids = []
    self._barrier = None
    self._entries = {}
    self._lock = threading.Lock()

  def set_metadata(
      self,
      endpoints: List[coordination_pb2.EndpointDescriptor],
      transfer_uuid: int,
      transfer_req_id: str,
      block_ids: List[int],
  ):
    self._endpoints = endpoints
    self._transfer_uuid = transfer_uuid
    self._transfer_req_id = transfer_req_id
    self._block_ids = block_ids
    self._metadata_event.set()

  def GetTransferMetadata(self, request, context):
    # Wait for the server runner to set the actual dynamic Port first.
    self._metadata_event.wait()
    return coordination_pb2.MetadataResponse(
        endpoints=self._endpoints,
        transfer_uuid=self._transfer_uuid,
        transfer_req_id=self._transfer_req_id,
        block_ids=self._block_ids,
    )

  def Shutdown(self, request, context):
    self._shutdown_event.set()
    return coordination_pb2.ShutdownResponse(success=True)

  def wait_for_shutdown(self):
    """Blocks until Shutdown RPC is called."""
    self._shutdown_event.wait()

  def CollectReplicaInfo(self, request, context):
    with self._lock:
      if self._barrier is None:
        self._barrier = threading.Barrier(request.expected_count)

      self._entries[request.device_id] = list(request.info)

    try:
      self._barrier.wait()
    except threading.BrokenBarrierError:
      context.abort(grpc.StatusCode.INTERNAL, 'Barrier broken')

    with self._lock:
      sorted_entries = sorted(self._entries.items())
      response_entries = [
          coordination_pb2.ReplicaInfoEntry(device_id=k, info=v)
          for k, v in sorted_entries
      ]
      return coordination_pb2.CollectReplicaInfoResponse(
          entries=response_entries
      )


class CoordinationServer:
  """Background gRPC server to coordinate metadata."""

  def __init__(self, port: int = 0):
    self._port = port
    self._server = None
    self._servicer = CoordinationServicer()

  def start(self) -> int:
    """Starts the gRPC server in the background. Returns bound port."""
    self._server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=2)
    )
    coordination_pb2_grpc.add_CoordinationServiceServicer_to_server(
        self._servicer, self._server
    )
    # Bind to ephemeral port securely using LOAS2 credentials in Prod
    bound_port = self._server.add_insecure_port(f'[::]:{self._port}')
    self._server.start()
    return bound_port

  def set_metadata(
      self,
      endpoints: List[Dict[str, Any]],
      transfer_uuid: int,
      transfer_req_id: str,
      block_ids: List[int],
  ):
    """Sets the transfer metadata on the servicer."""
    proto_endpoints = []
    for ep in endpoints:
      proto_ep = coordination_pb2.EndpointDescriptor(
          endpoint=ep['endpoint'],
          shards=ep['shards'],
      )
      proto_endpoints.append(proto_ep)
    self._servicer.set_metadata(
        endpoints=proto_endpoints,
        transfer_uuid=transfer_uuid,
        transfer_req_id=transfer_req_id,
        block_ids=block_ids,
    )

  def wait_for_shutdown(self):
    """Blocks until peer Receiver calls Shutdown RPC."""
    self._servicer.wait_for_shutdown()

  def stop(self):
    if self._server:
      self._server.stop(0)


class CoordinationClient:
  """gRPC client to query metadata from peer."""

  def __init__(self, server_address: str):
    self._server_address = server_address

  def _get_channel(self) -> grpc.Channel:
    """Returns a channel to the peer server."""
    channel = grpc.insecure_channel(self._server_address)
    return channel

  def get_metadata(self) -> TransferMetadata:
    """Queries peer server and returns TransferMetadata."""
    with self._get_channel() as channel:
      stub = coordination_pb2_grpc.CoordinationServiceStub(channel)
      request = coordination_pb2.MetadataRequest(client_id='receiver')
      # Call RPC blocking until server is ready and replies.
      response = stub.GetTransferMetadata(request)
      endpoints = []
      for ep in response.endpoints:
        endpoints.append({
            'endpoint': ep.endpoint,
            'shards': list(ep.shards),
        })
      return TransferMetadata(
          endpoints=endpoints,
          transfer_uuid=response.transfer_uuid,
          transfer_req_id=response.transfer_req_id,
          block_ids=list(response.block_ids),
      )

  def collect_replica_info(
      self, device_id: int, expected_count: int, info: List[int]
  ) -> List[Tuple[int, List[int]]]:
    """Collects metadata from all replicas.

    Args:
      device_id: The ID of the calling device.
      expected_count: The total number of devices participating.
      info: The metadata to share from this device.

    Returns:
      A list of tuples containing (device_id, info) from all devices.
    """
    with self._get_channel() as channel:
      stub = coordination_pb2_grpc.CoordinationServiceStub(channel)
      request = coordination_pb2.CollectReplicaInfoRequest(
          device_id=device_id, expected_count=expected_count, info=info
      )
      response = stub.CollectReplicaInfo(request)
      return [(e.device_id, list(e.info)) for e in response.entries]

  def shutdown(self):
    """Signals peer CoordinationServer to shut down and exit."""
    with self._get_channel() as channel:
      stub = coordination_pb2_grpc.CoordinationServiceStub(channel)
      request = coordination_pb2.ShutdownRequest(client_id='receiver')
      stub.Shutdown(request)
