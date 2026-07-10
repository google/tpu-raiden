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
"""Stage 2 acceptance tests for Qwen3.5 hybrid KV admission.

These tests intentionally describe the end-state API from
RESHARD_STAGE2_HYBRID_BLOCK.md. They are expected to fail until Stage 2 adds
Qwen3.5 admission specs, hybrid layout derivation, and the manager admission
surface.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Iterable, Mapping

from absl.testing import absltest

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tpu_raiden.api.torch import kv_cache_manager


NUM_LAYERS = 60
FA_LAYER_INDICES = tuple(range(3, NUM_LAYERS, 4))
FA_LAYER = FA_LAYER_INDICES[0]
GDN_LAYER = 0
HOST_BLOCKS = 1

TOPOLOGY_CASES: Mapping[str, Mapping[str, Any]] = {
    "pcp8_prefill": {
        "model_server_role": "kv_producer",
        "pcp_size": 8,
        "tp_size": 1,
        "dp_size": 1,
        "local_kv_heads": 2,
        "physical_fa_head_slots": 2,
        "local_linear_key_heads": 16,
        "local_linear_value_heads": 64,
        "block_tokens": 4096,
        "slot_bytes": 4_268_032,
        "gdn_used_bytes": 4_268_032,
        "fa_live_bytes": 4_194_304,
        "fa_region": {
            "name": "fa_payload",
            "offset_bytes": 0,
            "stride_bytes": 1024,
            "unit_bytes": 512,
            "num_units": 4096,
            "units_per_stride": 2,
        },
        "gdn_regions": {
            "gdn_conv_q": {
                "offset_bytes": 0,
                "stride_bytes": 24_576,
                "unit_bytes": 256,
                "num_units": 3,
                "units_per_stride": 16,
            },
            "gdn_conv_k": {
                "offset_bytes": 4096,
                "stride_bytes": 24_576,
                "unit_bytes": 256,
                "num_units": 3,
                "units_per_stride": 16,
            },
            "gdn_conv_v": {
                "offset_bytes": 8192,
                "stride_bytes": 24_576,
                "unit_bytes": 256,
                "num_units": 3,
                "units_per_stride": 64,
            },
            "gdn_ssm": {
                "offset_bytes": 73_728,
                "stride_bytes": 65_536,
                "unit_bytes": 65_536,
                "num_units": 64,
                "units_per_stride": 1,
            },
        },
    },
    "tp2dp4_decode": {
        "model_server_role": "kv_consumer",
        "pcp_size": 1,
        "tp_size": 2,
        "dp_size": 4,
        "local_kv_heads": 1,
        "physical_fa_head_slots": 2,
        "local_linear_key_heads": 8,
        "local_linear_value_heads": 32,
        "block_tokens": 2048,
        "slot_bytes": 2_134_016,
        "gdn_used_bytes": 2_134_016,
        "fa_live_bytes": 1_048_576,
        "fa_region": {
            "name": "fa_payload",
            "offset_bytes": 0,
            "stride_bytes": 1024,
            "unit_bytes": 512,
            "num_units": 2048,
            "units_per_stride": 1,
        },
        "gdn_regions": {
            "gdn_conv_q": {
                "offset_bytes": 0,
                "stride_bytes": 12_288,
                "unit_bytes": 256,
                "num_units": 3,
                "units_per_stride": 8,
            },
            "gdn_conv_k": {
                "offset_bytes": 2048,
                "stride_bytes": 12_288,
                "unit_bytes": 256,
                "num_units": 3,
                "units_per_stride": 8,
            },
            "gdn_conv_v": {
                "offset_bytes": 4096,
                "stride_bytes": 12_288,
                "unit_bytes": 256,
                "num_units": 3,
                "units_per_stride": 32,
            },
            "gdn_ssm": {
                "offset_bytes": 36_864,
                "stride_bytes": 65_536,
                "unit_bytes": 65_536,
                "num_units": 32,
                "units_per_stride": 1,
            },
        },
    },
}


def _field(obj: Any, name: str) -> Any:
  if isinstance(obj, Mapping):
    return obj[name]
  return getattr(obj, name)


def _regions_by_name(obj: Any) -> dict[str, Any]:
  regions = _field(obj, "regions")
  return {_field(region, "name"): region for region in regions}


def _assert_region(test: absltest.TestCase, region: Any,
                   expected: Mapping[str, int | str]) -> None:
  for key, value in expected.items():
    test.assertEqual(_field(region, key), value, key)


def _kind_text(obj: Any) -> str:
  kind = _field(obj, "kind")
  if hasattr(kind, "name"):
    kind = kind.name
  return str(kind).lower()


def _assert_kind_contains(test: absltest.TestCase, obj: Any,
                          needles: Iterable[str]) -> None:
  kind = _kind_text(obj)
  test.assertTrue(
      any(needle in kind for needle in needles),
      f"kind {kind!r} did not contain any of {tuple(needles)!r}",
  )


def _region_live_bytes(region: Any) -> int:
  return (_field(region, "unit_bytes") * _field(region, "num_units") *
          _field(region, "units_per_stride"))


class ReshardStage2HybridBlockAcceptanceTest(absltest.TestCase):

  def test_qwen35_admission_specs_cover_required_server_topologies(self) -> None:
    from tpu_raiden.api.torch import hybrid_layout

    for topology, expected in TOPOLOGY_CASES.items():
      with self.subTest(topology=topology):
        spec = hybrid_layout.qwen35_397b_admission_spec(topology)
        self.assertEqual(_field(spec, "topology"), topology)
        self.assertEqual(_field(spec, "model_server_role"),
                         expected["model_server_role"])
        self.assertEqual(_field(spec, "num_layers"), NUM_LAYERS)
        self.assertEqual(tuple(_field(spec, "fa_layer_indices")),
                         FA_LAYER_INDICES)
        self.assertEqual(_field(spec, "pcp_size"), expected["pcp_size"])
        self.assertEqual(_field(spec, "tp_size"), expected["tp_size"])
        self.assertEqual(_field(spec, "dp_size"), expected["dp_size"])
        self.assertEqual(_field(spec, "local_kv_heads"),
                         expected["local_kv_heads"])
        self.assertEqual(_field(spec, "physical_fa_head_slots"),
                         expected["physical_fa_head_slots"])
        self.assertEqual(_field(spec, "local_linear_key_heads"),
                         expected["local_linear_key_heads"])
        self.assertEqual(_field(spec, "local_linear_value_heads"),
                         expected["local_linear_value_heads"])
        self.assertEqual(_field(spec, "block_tokens"),
                         expected["block_tokens"])
        self.assertEqual(_field(spec, "slot_bytes"), expected["slot_bytes"])
        self.assertEqual(_field(spec, "gdn_used_bytes"),
                         expected["gdn_used_bytes"])
        self.assertLen(_field(spec, "layouts"), NUM_LAYERS)

    with self.assertRaisesRegex(Exception, "topology|unknown|unsupported"):
      hybrid_layout.qwen35_397b_admission_spec("tp4_unsupported")

  def test_qwen35_layouts_match_model_server_physical_geometry(self) -> None:
    from tpu_raiden.api.torch import hybrid_layout

    for topology, expected in TOPOLOGY_CASES.items():
      with self.subTest(topology=topology):
        spec = hybrid_layout.qwen35_397b_admission_spec(topology)
        layouts = list(_field(spec, "layouts"))

        fa_layout = layouts[FA_LAYER]
        self.assertEqual(_field(fa_layout, "slot_bytes"),
                         expected["slot_bytes"])
        _assert_kind_contains(self, fa_layout, ("full", "fa"))
        fa_region = _regions_by_name(fa_layout)["fa_payload"]
        _assert_region(self, fa_region, expected["fa_region"])
        self.assertEqual(_region_live_bytes(fa_region),
                         expected["fa_live_bytes"])
        self.assertLessEqual(_region_live_bytes(fa_region),
                             expected["slot_bytes"])

        gdn_layout = layouts[GDN_LAYER]
        self.assertEqual(_field(gdn_layout, "slot_bytes"),
                         expected["slot_bytes"])
        _assert_kind_contains(self, gdn_layout, ("mamba", "gdn"))
        gdn_regions = _regions_by_name(gdn_layout)
        for name, region_expected in expected["gdn_regions"].items():
          _assert_region(self, gdn_regions[name], region_expected)
        self.assertEqual(
            sum(_region_live_bytes(gdn_regions[name])
                for name in expected["gdn_regions"]),
            expected["gdn_used_bytes"],
        )
        self.assertLessEqual(expected["gdn_used_bytes"],
                             expected["slot_bytes"])

  def test_qwen35_35b_tp4_layout_matches_reference_offsets(self) -> None:
    from tpu_raiden.api.torch import hybrid_layout

    cfg = hybrid_layout.ModelKVConfig(
        num_layers=40,
        fa_layer_indices=tuple(range(3, 40, 4)),
        num_kv_heads=2,
        head_dim=256,
        kv_itemsize=1,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        conv_kernel=4,
        physical_fa_head_slots=2,
    )
    block_tokens = 1056
    slot_bytes = hybrid_layout.derive_slot_bytes(cfg, block_tokens)
    self.assertEqual(slot_bytes, 1_081_344)
    self.assertEqual(hybrid_layout.derive_gdn_used_bytes(cfg), 536_576)

    layouts = hybrid_layout.build_layer_layouts(cfg, block_tokens)
    fa_layout = layouts[3]
    self.assertEqual(_field(fa_layout, "slot_bytes"), slot_bytes)
    _assert_region(
        self,
        _regions_by_name(fa_layout)["fa_payload"],
        {
            "name": "fa_payload",
            "offset_bytes": 0,
            "stride_bytes": 1024,
            "unit_bytes": 512,
            "num_units": 1056,
            "units_per_stride": 2,
        },
    )

    gdn_regions = _regions_by_name(layouts[0])
    expected_regions = {
        "gdn_conv_q": {
            "name": "gdn_conv_q",
            "offset_bytes": 0,
            "stride_bytes": 4096,
            "unit_bytes": 256,
            "num_units": 3,
            "units_per_stride": 4,
        },
        "gdn_conv_k": {
            "name": "gdn_conv_k",
            "offset_bytes": 1024,
            "stride_bytes": 4096,
            "unit_bytes": 256,
            "num_units": 3,
            "units_per_stride": 4,
        },
        "gdn_conv_v": {
            "name": "gdn_conv_v",
            "offset_bytes": 2048,
            "stride_bytes": 4096,
            "unit_bytes": 256,
            "num_units": 3,
            "units_per_stride": 8,
        },
        "gdn_ssm": {
            "name": "gdn_ssm",
            "offset_bytes": 12_288,
            "stride_bytes": 65_536,
            "unit_bytes": 65_536,
            "num_units": 8,
            "units_per_stride": 1,
        },
    }
    for name, expected in expected_regions.items():
      _assert_region(self, gdn_regions[name], expected)
    self.assertEqual(
        sum(_region_live_bytes(gdn_regions[name])
            for name in expected_regions),
        536_576,
    )

  def test_host_only_manager_admits_qwen35_kv_cache(self) -> None:
    from tpu_raiden.api.torch import hybrid_layout

    for topology, expected in TOPOLOGY_CASES.items():
      with self.subTest(topology=topology):
        spec = hybrid_layout.qwen35_397b_admission_spec(topology)
        manager = kv_cache_manager.KVCacheManager.create_host_only(
            num_layers=NUM_LAYERS,
            num_shards=1,
            slice_byte_size=_field(spec, "slot_bytes"),
            node_id=0,
            local_port=0,
            host_blocks=HOST_BLOCKS,
            parallelism=4,
        )
        # pylint: disable=protected-access
        self.assertTrue(hasattr(manager._impl, "set_block_layouts_native"))
        self.assertEqual(
            manager._impl.layer_block_byte_size(FA_LAYER),
            expected["slot_bytes"],
        )
        # pylint: enable=protected-access

        summary = manager.admit_qwen35_kv_cache(spec)
        self.assertTrue(_field(summary, "admitted"))
        self.assertEqual(_field(summary, "topology"), topology)
        self.assertEqual(_field(summary, "model_server_role"),
                         expected["model_server_role"])
        self.assertEqual(_field(summary, "num_layers"), NUM_LAYERS)
        self.assertEqual(_field(summary, "fa_layers"), len(FA_LAYER_INDICES))
        self.assertEqual(_field(summary, "gdn_layers"),
                         NUM_LAYERS - len(FA_LAYER_INDICES))
        self.assertEqual(_field(summary, "block_tokens"),
                         expected["block_tokens"])
        self.assertEqual(_field(summary, "slot_bytes"),
                         expected["slot_bytes"])
        self.assertEqual(_field(summary, "gdn_used_bytes"),
                         expected["gdn_used_bytes"])
        self.assertEqual(manager.admission_summary(), summary)
        self.assertEqual(manager.fa_layer_indices(), list(FA_LAYER_INDICES))

        fa_ref = manager.get_block_ref(FA_LAYER, 0)
        self.assertIsInstance(_field(fa_ref, "ptr"), int)
        self.assertNotEqual(_field(fa_ref, "ptr"), 0)
        self.assertEqual(_field(fa_ref, "slot_bytes"),
                         expected["slot_bytes"])
        _assert_kind_contains(self, fa_ref, ("full", "fa"))
        _assert_region(self, _regions_by_name(fa_ref)["fa_payload"],
                       expected["fa_region"])

        gdn_ref = manager.get_block_ref(GDN_LAYER, 0)
        self.assertIsInstance(_field(gdn_ref, "ptr"), int)
        self.assertNotEqual(_field(gdn_ref, "ptr"), 0)
        self.assertEqual(_field(gdn_ref, "slot_bytes"),
                         expected["slot_bytes"])
        _assert_kind_contains(self, gdn_ref, ("mamba", "gdn"))
        gdn_regions = _regions_by_name(gdn_ref)
        for name, region_expected in expected["gdn_regions"].items():
          _assert_region(self, gdn_regions[name], region_expected)

        with self.assertRaisesRegex(Exception, "block|range|bound"):
          manager.get_block_ref(FA_LAYER, HOST_BLOCKS)

  def test_torch_backed_constructor_admits_mocked_vllm_runner_kv_cache(
      self) -> None:
    from tpu_raiden.api.torch import hybrid_layout

    spec = hybrid_layout.qwen35_397b_admission_spec("pcp8_prefill")
    runner = types.SimpleNamespace(
        kv_caches=[object() for _ in range(NUM_LAYERS)])

    class FakeNativeKVCacheManager:

      def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.num_layers = len(kwargs["kv_caches"])
        self.slice_byte_size = _field(spec, "slot_bytes")
        self.native_layouts = None

      def layer_block_byte_size(self, layer_idx):
        if layer_idx < 0 or layer_idx >= self.num_layers:
          return -1
        return _field(spec, "slot_bytes")

      def set_block_layouts_native(self, layouts):
        self.native_layouts = list(layouts)

      def layer_indices_of_kind_native(self, kind_id):
        return [
            layer_idx
            for layer_idx, layout in enumerate(self.native_layouts or [])
            if layout[0] == kind_id
        ]

      def get_hybrid_block_ref_native(self, layer_idx, shard_idx, block_id):
        layout = self.native_layouts[layer_idx]
        return {
            "ptr": 4096 + layer_idx,
            "slot_bytes": layout[1],
            "kind": "full_attention" if layout[0] == 0 else "mamba_state",
            "layer_idx": layer_idx,
            "shard_idx": shard_idx,
            "block_id": block_id,
            "regions": [{
                "name": region[0],
                "offset_bytes": region[1],
                "stride_bytes": region[2],
                "unit_bytes": region[3],
                "num_units": region[4],
                "units_per_stride": region[5],
            } for region in layout[2]],
        }

    old_torch_impl = kv_cache_manager._TORCH_IMPL  # pylint: disable=protected-access
    kv_cache_manager._TORCH_IMPL = types.SimpleNamespace(  # pylint: disable=protected-access
        KVCacheManager=FakeNativeKVCacheManager)
    try:
      manager = kv_cache_manager.KVCacheManager(
          kv_caches=runner.kv_caches,
          node_id=3,
          local_control_port=-1,
          max_blocks=1,
          num_slots=1,
          timeout_s=5.0,
          unsafe_skip_buffer_lock=True,
          parallelism=2,
      )
      summary = manager.admit_qwen35_kv_cache(spec)
    finally:
      kv_cache_manager._TORCH_IMPL = old_torch_impl  # pylint: disable=protected-access

    self.assertTrue(_field(summary, "admitted"))
    self.assertEqual(_field(summary, "topology"), "pcp8_prefill")
    self.assertEqual(_field(summary, "num_layers"), NUM_LAYERS)
    self.assertEqual(_field(summary, "slot_bytes"), _field(spec, "slot_bytes"))
    self.assertEqual(manager.fa_layer_indices(), list(FA_LAYER_INDICES))
    self.assertEqual(manager._impl.kwargs["kv_caches"], runner.kv_caches)  # pylint: disable=protected-access
    self.assertEqual(len(manager._impl.native_layouts), NUM_LAYERS)  # pylint: disable=protected-access

  def test_torch_backed_constructor_admits_compact_gdn_layer_sizes(
      self) -> None:
    from tpu_raiden.api.torch import hybrid_layout

    spec = hybrid_layout.qwen35_397b_admission_spec("pcp8_prefill")

    class FakeNativeKVCacheManager:

      def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.num_layers = len(kwargs["kv_caches"])
        self.slice_byte_size = _field(spec, "slot_bytes")
        self.native_layouts = None

      def layer_block_byte_size(self, layer_idx):
        if layer_idx < 0 or layer_idx >= self.num_layers:
          return -1
        if layer_idx in FA_LAYER_INDICES:
          return _field(spec, "slot_bytes")
        return _field(spec, "gdn_used_bytes")

      def set_block_layouts_native(self, layouts):
        self.native_layouts = list(layouts)

      def layer_indices_of_kind_native(self, kind_id):
        return [
            layer_idx
            for layer_idx, layout in enumerate(self.native_layouts or [])
            if layout[0] == kind_id
        ]

    old_torch_impl = kv_cache_manager._TORCH_IMPL  # pylint: disable=protected-access
    kv_cache_manager._TORCH_IMPL = types.SimpleNamespace(  # pylint: disable=protected-access
        KVCacheManager=FakeNativeKVCacheManager)
    try:
      manager = kv_cache_manager.KVCacheManager(
          kv_caches=[object() for _ in range(NUM_LAYERS)],
          node_id=3,
          local_control_port=-1,
          max_blocks=1,
          num_slots=1,
          timeout_s=5.0,
          unsafe_skip_buffer_lock=True,
          parallelism=2,
      )
      summary = manager.admit_qwen35_kv_cache(spec)
    finally:
      kv_cache_manager._TORCH_IMPL = old_torch_impl  # pylint: disable=protected-access

    self.assertTrue(_field(summary, "admitted"))
    self.assertEqual(_field(summary, "slot_bytes"), _field(spec, "slot_bytes"))
    self.assertEqual(_field(summary, "gdn_used_bytes"),
                     _field(spec, "gdn_used_bytes"))
    self.assertEqual(manager.fa_layer_indices(), list(FA_LAYER_INDICES))
    self.assertEqual(len(manager._impl.native_layouts), NUM_LAYERS)  # pylint: disable=protected-access

  def test_admission_rejects_legacy_compact_gdn_slot_size(self) -> None:
    from tpu_raiden.api.torch import hybrid_layout

    spec = hybrid_layout.qwen35_397b_admission_spec("pcp8_prefill")
    manager = kv_cache_manager.KVCacheManager.create_host_only(
        num_layers=NUM_LAYERS,
        num_shards=1,
        slice_byte_size=1_067_008,
        node_id=0,
        local_port=0,
        host_blocks=HOST_BLOCKS,
        parallelism=4,
    )

    with self.assertRaisesRegex(Exception, "slot|byte|size|layout"):
      manager.admit_qwen35_kv_cache(spec)


if __name__ == "__main__":
  absltest.main()
