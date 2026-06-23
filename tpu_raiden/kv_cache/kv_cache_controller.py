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

# Copyright 2026 The TPU Raiden Authors. All Rights Reserved.
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
"""Controller RPC client for KV Cache resharding."""

from typing import Optional

from tpu_raiden.rpc import raiden_controller
from tpu_raiden.rpc import raiden_service_pb2
from tpu_raiden.rpc.raiden_controller import RaidenId
from tpu_raiden.rpc.raiden_controller import TransferPlan


class KVCacheWorkerRpcClient(raiden_controller.WorkerRpcClient):
  """Worker RPC client using KV Cache service proto."""

  def __init__(
      self,
      endpoint_addresses: Optional[dict[RaidenId, str]] = None,
      resolve_timeout: float = 300.0,
      name_resolver: Optional[raiden_controller.NameResolver] = None,
  ):
    super().__init__(
        endpoint_addresses=endpoint_addresses,
        resolve_timeout=resolve_timeout,
        name_resolver=name_resolver,
        proto_module=raiden_service_pb2,
    )
