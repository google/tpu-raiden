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
from absl.testing import absltest
from tpu_raiden.rpc import coordination_helper


class CoordinationHelperTest(absltest.TestCase):

  def test_collect_replica_info(self):
    server = coordination_helper.CoordinationServer()
    port = server.start()
    server_address = f'localhost:{port}'

    client1 = coordination_helper.CoordinationClient(server_address)
    client2 = coordination_helper.CoordinationClient(server_address)

    def call_client(client, device_id, expected_count, info):
      return client.collect_replica_info(device_id, expected_count, info)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
      future1 = executor.submit(call_client, client1, 0, 2, [1, 2, 3])
      future2 = executor.submit(call_client, client2, 1, 2, [4, 5, 6])

      res1 = future1.result(timeout=5)
      res2 = future2.result(timeout=5)

    self.assertEqual(res1, res2)
    self.assertLen(res1, 2)
    self.assertEqual(res1[0], (0, [1, 2, 3]))
    self.assertEqual(res1[1], (1, [4, 5, 6]))

    server.stop()

  def test_metadata_coordination(self):
    server = coordination_helper.CoordinationServer()
    port = server.start()
    server_address = f'localhost:{port}'

    client = coordination_helper.CoordinationClient(server_address)

    endpoints = [
        {'endpoint': '10.0.0.1:8888', 'shards': [0, 1]},
        {'endpoint': '10.0.0.2:8888', 'shards': [2, 3]},
    ]
    transfer_uuid = 123456789
    transfer_req_id = 'test-request-id'
    block_ids = [10, 20, 30]

    server.set_metadata(
        endpoints=endpoints,
        transfer_uuid=transfer_uuid,
        transfer_req_id=transfer_req_id,
        block_ids=block_ids,
    )

    metadata = client.get_metadata()

    self.assertEqual(metadata.endpoints, endpoints)
    self.assertEqual(metadata.transfer_uuid, transfer_uuid)
    self.assertEqual(metadata.transfer_req_id, transfer_req_id)
    self.assertEqual(metadata.block_ids, block_ids)

    server.stop()


if __name__ == '__main__':
  absltest.main()
