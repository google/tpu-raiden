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


from absl.testing import absltest

from tpu_raiden.api.jax import kv_cache_store


class KVCacheStoreTest(absltest.TestCase):

  def test_basic_tests(self):
    controller = kv_cache_store.KVCacheStore(capacity=20)
    self.assertEqual(controller.capacity(), 20)

    hashes = [6001, 6002]
    slices = [
        [kv_cache_store.RaidenId("inference_server", "0", "kv_cache", 0)],
        [kv_cache_store.RaidenId("inference_server", "1", "kv_cache", 0)],
    ]

    # 1. Insert
    self.assertTrue(controller.insert(hashes, slices, True))
    self.assertFalse(controller.insert(hashes, slices, True))  # Already exists

    # 2. Lookup with a partial miss at the end
    hashes_with_miss = [6001, 6002, 6003]
    lookup_res = controller.lookup(hashes_with_miss)
    self.assertLen(lookup_res, 2)
    self.assertEqual(lookup_res[0][0], 6001)
    self.assertLen(lookup_res[0][1], 1)
    self.assertEqual(lookup_res[0][1][0].job_name, "inference_server")
    self.assertEqual(lookup_res[0][1][0].job_replica_id, "0")

    # Lookup with an early miss
    hashes_early_miss = [6001, 6003, 6002]
    lookup_res_early = controller.lookup(hashes_early_miss)
    self.assertLen(lookup_res_early, 1)
    self.assertEqual(lookup_res_early[0][0], 6001)

    # 3. Delete
    controller.delete(hashes, slices)
    self.assertTrue(controller.insert(hashes, slices, True))  # Succesful again

  def test_pin_and_release(self):
    controller = kv_cache_store.KVCacheStore(capacity=2)

    hashes = [7001, 7002]
    slices = [
        [kv_cache_store.RaidenId("inference_server", "0", "kv_cache", 0)],
        [kv_cache_store.RaidenId("inference_server", "1", "kv_cache", 0)],
    ]

    self.assertTrue(controller.insert(hashes, slices, True))

    # Pin both
    self.assertTrue(controller.pin(hashes))

    # Inserting a third element should fail to evict because both items are
    # pinned.
    hash_3 = [7003]
    slice_3 = [
        [kv_cache_store.RaidenId("inference_server", "2", "kv_cache", 0)]
    ]
    controller.insert(hash_3, slice_3, True)

    # Release 7001
    controller.release([7001])

    # Now inserting a fourth element (7004) should successfully evict 7001
    hash_4 = [7004]
    slice_4 = [
        [kv_cache_store.RaidenId("inference_server", "3", "kv_cache", 0)]
    ]
    controller.insert(hash_4, slice_4, True)

    self.assertEmpty(controller.lookup([7001, 7002]))
    self.assertLen(controller.lookup([7002]), 1)

  def test_partial_pin_rollback(self):
    controller = kv_cache_store.KVCacheStore(capacity=2)

    hashes = [8001, 8002]
    slices = [
        [kv_cache_store.RaidenId("inference_server", "0", "kv_cache", 0)],
        [kv_cache_store.RaidenId("inference_server", "1", "kv_cache", 0)],
    ]
    self.assertTrue(controller.insert(hashes, slices, True))

    # Attempt to pin a sequence with a missing hash (8003).
    self.assertFalse(controller.pin([8001, 8002, 8003]))

    # Now inserting two new items (8004, 8005) should successfully evict 8001
    # and 8002 because their pins were completely rolled back!
    self.assertTrue(
        controller.insert(
            [8004, 8005],
            [
                [
                    kv_cache_store.RaidenId(
                        "inference_server", "2", "kv_cache", 0
                    )
                ],
                [
                    kv_cache_store.RaidenId(
                        "inference_server", "3", "kv_cache", 0
                    )
                ],
            ],
            True,
        )
    )

    self.assertEmpty(controller.lookup([8001, 8002]))
    self.assertLen(controller.lookup([8004, 8005]), 2)


if __name__ == "__main__":
  absltest.main()
