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

import concurrent.futures
import threading
from typing import List, Tuple
import grpc
from rpc import coordination_pb2
from rpc import coordination_pb2_grpc


class CoordinationServicer(coordination_pb2_grpc.CoordinationServiceServicer):
  """gRPC servicer to hold transfer metadata dynamically."""

  def __init__(self):
    self._metadata_event = threading.Event()
    self._shutdown_event = threading.Event()
    self._port = 0
    self._block_ids = []
    self._host_ip = ''

  def set_metadata(self, port: int, block_ids: List[int], host_ip: str):
    self._port = port
    self._block_ids = block_ids
    self._host_ip = host_ip
    self._metadata_event.set()

  def GetTransferMetadata(self, request, context):
    # Wait for the server runner to set the actual dynamic Port first.
    self._metadata_event.wait()
    return coordination_pb2.MetadataResponse(
        port=self._port, block_ids=self._block_ids, host_ip=self._host_ip
    )

  def Shutdown(self, request, context):
    self._shutdown_event.set()
    return coordination_pb2.ShutdownResponse(success=True)


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

  def set_metadata(self, port: int, block_ids: List[int], host_ip: str):
    self._servicer.set_metadata(port, block_ids, host_ip)

  def wait_for_shutdown(self):
    """Blocks until peer Receiver calls Shutdown RPC."""
    self._servicer._shutdown_event.wait()

  def stop(self):
    if self._server:
      self._server.stop(0)


class CoordinationClient:
  """gRPC client to query metadata from peer."""

  def __init__(self, server_address: str):
    self._server_address = server_address

  def get_metadata(self) -> Tuple[int, List[int], str]:
    """Queries peer server and returns (port, block_ids, host_ip) tuple."""
      with grpc.insecure_channel(self._server_address) as channel:
      stub = coordination_pb2_grpc.CoordinationServiceStub(channel)
      request = coordination_pb2.MetadataRequest(client_id='receiver')
      # Call RPC blocking until server is ready and replies.
      response = stub.GetTransferMetadata(request)
      return response.port, list(response.block_ids), response.host_ip

  def shutdown(self):
    """Signals peer CoordinationServer to shut down and exit."""
      with grpc.insecure_channel(self._server_address) as channel:
      stub = coordination_pb2_grpc.CoordinationServiceStub(channel)
      request = coordination_pb2.ShutdownRequest(client_id='receiver')
      stub.Shutdown(request)
