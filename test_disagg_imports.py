"""Smoke test: verify DisaggKVCacheManager imports + bindings are wired up.

Run with PYTHONPATH set to the workspace root so the `api.jax` package
resolves to api/jax/_kv_cache_manager.so. E.g.:

    PYTHONPATH=. python test_disagg_imports.py
"""

import unittest


class DisaggImportTest(unittest.TestCase):

    def test_nanobind_extension_exposes_symbols(self):
        from api.jax import _kv_cache_manager as ext

        for sym in (
            "KVCacheManager",
            "DisaggKVCacheManager",
            "DisaggTransferRequest",
            "DisaggTransferRequestType",
        ):
            self.assertTrue(
                hasattr(ext, sym),
                f"_kv_cache_manager.so is missing symbol {sym!r}",
            )

    def test_python_wrappers_import(self):
        from api.jax.kv_cache_manager import KVCacheManager
        from api.jax.disagg_kv_cache_manager import (
            DisaggKVCacheManager,
            DisaggTransferRequest,
            DisaggTransferRequestType,
        )
        self.assertTrue(callable(KVCacheManager))
        self.assertTrue(callable(DisaggKVCacheManager))
        self.assertIsNotNone(DisaggTransferRequest)
        self.assertIsNotNone(DisaggTransferRequestType)

    def test_request_type_enum_members(self):
        from api.jax import _kv_cache_manager as ext

        expected_members = {
            "PREFILL_D2H",
            "DECODE_H2D",
            "H2H_WRITE",
            "H2H_READ",
        }
        actual = {m for m in dir(ext.DisaggTransferRequestType) if m.isupper()}
        missing = expected_members - actual
        self.assertFalse(
            missing,
            f"DisaggTransferRequestType missing members: {missing}. "
            f"Got: {sorted(actual)}",
        )

    def test_request_struct_is_constructible(self):
        from api.jax import _kv_cache_manager as ext

        req = ext.DisaggTransferRequest()
        req.request_id = 42
        req.type = ext.DisaggTransferRequestType.PREFILL_D2H
        req.src_offsets = [0, 1]
        req.dst_offsets = [0, 1]
        req.sizes = [128, 128]
        req.peer = "decode"
        req.block_ids = [0, 1]
        req.entity_id = 7
        req.callback = lambda status: None

        self.assertEqual(req.request_id, 42)
        self.assertEqual(req.type, ext.DisaggTransferRequestType.PREFILL_D2H)
        self.assertEqual(list(req.src_offsets), [0, 1])
        self.assertEqual(req.peer, "decode")


if __name__ == "__main__":
    unittest.main(verbosity=2)
