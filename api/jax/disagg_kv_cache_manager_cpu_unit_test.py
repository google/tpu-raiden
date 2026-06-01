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

"""CPU-only unit tests for the nanobind bindings of api/jax/_kv_cache_manager.

These tests cover the parts of the disagg KV cache manager surface that do
NOT require constructing a real manager (which needs working PJRT buffers):

  * DisaggTransferRequest struct shape (default values, every field round-trip,
    Python callback storage).
  * DisaggTransferRequestType enum values and distinctness.
  * PjRtCopyFuture class export.

Lifecycle / properties / peer registry / SubmitRequest preconditions for the
C++ base class are covered by the C++ gunit target
//kv_cache:disagg_kv_cache_manager_base_test, which uses a mock subclass to
bypass PJRT entirely.

The full Python E2E push flow is covered by disagg_kv_cache_manager_test.py
with --device_type=cpu.
"""

from absl.testing import absltest
from absl.testing import parameterized

from api.jax import _kv_cache_manager as ext


# -----------------------------------------------------------------------------
# DisaggTransferRequest struct
# -----------------------------------------------------------------------------
class DisaggTransferRequestBindingTest(parameterized.TestCase):

  def test_default_construction(self):
    req = ext.DisaggTransferRequest()
    self.assertEqual(req.uuid, 0)
    self.assertEqual(req.req_id, 0)
    self.assertEqual(list(req.src_offsets), [])
    self.assertEqual(list(req.dst_offsets), [])
    self.assertEqual(list(req.sizes), [])
    self.assertEqual(list(req.block_ids), [])
    self.assertEqual(req.peer, "")
    self.assertEqual(req.entity_id, 0)

  def test_uuid_round_trip_large(self):
    req = ext.DisaggTransferRequest()
    req.uuid = 2**63 + 1  # uuid is uint64
    self.assertEqual(req.uuid, 2**63 + 1)

  def test_req_id_round_trip(self):
    req = ext.DisaggTransferRequest()
    req.req_id = 12345
    self.assertEqual(req.req_id, 12345)

  @parameterized.named_parameters(
      ("prefill_d2h", "PREFILL_D2H"),
      ("decode_h2d", "DECODE_H2D"),
      ("h2h_read", "H2H_READ"),
  )
  def test_type_field_accepts_each_enum(self, name):
    req = ext.DisaggTransferRequest()
    val = getattr(ext.DisaggTransferRequestType, name)
    req.type = val
    self.assertEqual(req.type, val)

  def test_vector_fields_round_trip(self):
    req = ext.DisaggTransferRequest()
    req.src_offsets = [0, 2, 4]
    req.dst_offsets = [1, 3, 5]
    req.sizes = [2, 2, 2]
    req.block_ids = [10, 11, 12, 13, 14, 15]
    self.assertEqual(list(req.src_offsets), [0, 2, 4])
    self.assertEqual(list(req.dst_offsets), [1, 3, 5])
    self.assertEqual(list(req.sizes), [2, 2, 2])
    self.assertEqual(list(req.block_ids), [10, 11, 12, 13, 14, 15])

  def test_peer_and_entity_id_round_trip(self):
    req = ext.DisaggTransferRequest()
    req.peer = "decode_engine_3"
    req.entity_id = 99
    self.assertEqual(req.peer, "decode_engine_3")
    self.assertEqual(req.entity_id, 99)

  def test_callback_is_storable_and_callable(self):
    seen = []
    req = ext.DisaggTransferRequest()
    req.callback = lambda status: seen.append(status)
    req.callback(None)
    self.assertEqual(seen, [None])

  def test_callback_can_be_reassigned(self):
    req = ext.DisaggTransferRequest()
    a, b = [], []
    req.callback = lambda s: a.append(s)
    req.callback = lambda s: b.append(s)
    req.callback("done")
    self.assertEqual(a, [])
    self.assertEqual(b, ["done"])


# -----------------------------------------------------------------------------
# DisaggTransferRequestType enum
# -----------------------------------------------------------------------------
class DisaggTransferRequestTypeTest(absltest.TestCase):

  def test_all_members_present(self):
    members = {m for m in dir(ext.DisaggTransferRequestType) if m.isupper()}
    self.assertEqual(members, {"PREFILL_D2H", "DECODE_H2D", "H2H_READ"})

  def test_members_are_distinct(self):
    values = {
        ext.DisaggTransferRequestType.PREFILL_D2H,
        ext.DisaggTransferRequestType.DECODE_H2D,
        ext.DisaggTransferRequestType.H2H_READ,
    }
    self.assertLen(values, 3)

  def test_member_equality(self):
    self.assertEqual(
        ext.DisaggTransferRequestType.PREFILL_D2H,
        ext.DisaggTransferRequestType.PREFILL_D2H,
    )
    self.assertNotEqual(
        ext.DisaggTransferRequestType.PREFILL_D2H,
        ext.DisaggTransferRequestType.DECODE_H2D,
    )


# -----------------------------------------------------------------------------
# PjRtCopyFuture class export
# -----------------------------------------------------------------------------
class PjRtCopyFutureBindingTest(absltest.TestCase):

  def test_class_exposed(self):
    self.assertTrue(hasattr(ext, "PjRtCopyFuture"))

  def test_methods_bound(self):
    cls = ext.PjRtCopyFuture
    for method in ("Await", "IsReady"):
      self.assertTrue(
          hasattr(cls, method), f"PjRtCopyFuture missing method {method}"
      )


if __name__ == "__main__":
  absltest.main()
