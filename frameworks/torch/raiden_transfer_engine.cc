// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/shared_ptr.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"
#include "core/transfer_engine_base.h"
#include "frameworks/torch/kv_cache_manager.h"
#include "frameworks/torch/torch_nanobind_utils.h"
#include "kv_cache/kv_cache_manager_base.h"

namespace nb = nanobind;

namespace tpu_raiden::kv_cache {
namespace {

using TensorList = std::vector<at::Tensor>;

std::vector<std::vector<at::Tensor>> SingleShardLayers(
    const TensorList& kv_caches) {
  std::vector<std::vector<at::Tensor>> layers;
  layers.reserve(kv_caches.size());
  for (const auto& kv_cache : kv_caches) {
    layers.push_back({kv_cache});
  }
  return layers;
}

static nb::list HostSpanMemoryViews(
    const std::vector<KVCacheHostSpan>& host_spans) {
  nb::list views;
  for (const KVCacheHostSpan& span : host_spans) {
    if (span.nbytes >
        static_cast<size_t>(std::numeric_limits<ssize_t>::max())) {
      throw std::overflow_error("host span is too large for Python view");
    }
    PyObject* mv =
        PyMemoryView_FromMemory(reinterpret_cast<char*>(span.ptr),
                                static_cast<ssize_t>(span.nbytes), PyBUF_WRITE);
    if (mv == nullptr) {
      throw std::runtime_error("Failed to create Python memoryview");
    }
    views.append(nb::steal<nb::object>(mv));
  }
  return views;
}

static nb::dict CopySpecToDict(const CopySpec& spec) {
  nb::dict out;
  out["src_offsets"] = spec.src_offsets;
  out["dst_offsets"] = spec.dst_offsets;
  out["sizes"] = spec.sizes;
  return out;
}

static nb::dict CopyPlanToDict(const CopyPlan& plan) {
  nb::dict out;
  out["num_blocks"] = plan.num_blocks;
  out["requested_remote_block_ids"] = plan.requested_remote_block_ids;
  out["requested_local_block_ids"] = plan.requested_local_block_ids;
  out["producer_remote_block_ids"] = plan.producer_remote_block_ids;
  out["h2d_local_block_ids"] = plan.h2d_local_block_ids;
  out["host_dst_to_src"] = plan.host_dst_to_src;
  out["requires_host_reorder"] = plan.RequiresHostReorder();
  out["d2h_copy"] = CopySpecToDict(plan.d2h_copy);
  out["h2d_copy"] = CopySpecToDict(plan.h2d_copy);
  return out;
}

class RaidenTransferEngine : public TransferEngineBase {
 public:
  RaidenTransferEngine(const TensorList& kv_caches, int64_t tp_rank,
                       int64_t local_control_port, int64_t max_blocks,
                       int64_t num_slots, double timeout_s,
                       bool unsafe_skip_buffer_lock)
      : TransferEngineBase(
            CreateKvCacheManager(kv_caches, num_slots, max_blocks,
                                 unsafe_skip_buffer_lock),
            tp_rank, local_control_port, max_blocks, num_slots, timeout_s,
            unsafe_skip_buffer_lock),
        kv_caches_(kv_caches) {}

  ~RaidenTransferEngine() override = default;

  std::vector<int64_t> RegisterKvCache(const TensorList& kv_caches) {
    kv_caches_ = kv_caches;
    if (num_slots_ <= 0) {
      throw std::invalid_argument("num_slots must be configured first");
    }
    if (max_blocks_ >
        std::numeric_limits<int>::max() / std::max<int64_t>(num_slots_, 1)) {
      throw std::invalid_argument("host staging block count exceeds int range");
    }

    bool has_tpu = false;
    for (const auto& t : kv_caches_) {
      if (t.device().type() == c10::DeviceType::PrivateUse1) {
        has_tpu = true;
        break;
      }
    }

    if (has_tpu) {
      const int host_blocks_to_allocate =
          static_cast<int>(num_slots_ * max_blocks_);
      kv_transfer_ = std::make_unique<tpu_raiden::torch::KVCacheManager>(
          SingleShardLayers(kv_caches_), /*block_size=*/1,
          /*local_port=*/std::nullopt, host_blocks_to_allocate,
          /*external_host_ptrs=*/std::nullopt, unsafe_skip_buffer_lock_);
      auto status =
          kv_transfer_->ConfigureHostStagingSlots(num_slots_, max_blocks_);
      if (!status.ok()) {
        throw std::runtime_error("Failed to configure KVCacheManager: " +
                                 std::string(status.message()));
      }
    } else {
      kv_transfer_ = nullptr;
    }
    InitializeSlotPool(num_slots_);

    std::vector<int64_t> region_ids;
    region_ids.reserve(kv_caches_.size());
    for (size_t i = 0; i < kv_caches_.size(); ++i) {
      region_ids.push_back(static_cast<int64_t>(i));
    }
    return region_ids;
  }

  void RegisterHostBuffers(nb::object /*host_pool*/, int64_t tp_rank) {
    tp_rank_ = tp_rank;
  }

  const TensorList& kv_caches() const { return kv_caches_; }

 private:
  static std::unique_ptr<kv_cache::KVCacheManagerBase> CreateKvCacheManager(
      const TensorList& kv_caches, int64_t num_slots, int64_t max_blocks,
      bool unsafe_skip_buffer_lock) {
    bool has_tpu = false;
    for (const auto& t : kv_caches) {
      if (t.device().type() == c10::DeviceType::PrivateUse1) {
        has_tpu = true;
        break;
      }
    }
    if (has_tpu) {
      const int host_blocks_to_allocate =
          static_cast<int>(num_slots * max_blocks);
      auto kv_transfer = std::make_unique<tpu_raiden::torch::KVCacheManager>(
          SingleShardLayers(kv_caches), /*block_size=*/1,
          /*local_port=*/std::nullopt, host_blocks_to_allocate,
          /*external_host_ptrs=*/std::nullopt, unsafe_skip_buffer_lock);
      auto status =
          kv_transfer->ConfigureHostStagingSlots(num_slots, max_blocks);
      if (!status.ok()) {
        throw std::runtime_error("Failed to configure KVCacheManager: " +
                                 std::string(status.message()));
      }
      return kv_transfer;
    }
    return nullptr;
  }

  TensorList kv_caches_;
};

void AwaitAll(nb::object futures) {
  if (nb::isinstance<TransferFuture>(futures)) {
    auto future = nb::cast<std::shared_ptr<TransferFuture>>(futures);
    future->Await();
    return;
  }
  for (nb::handle item : futures) {
    auto future = nb::cast<std::shared_ptr<TransferFuture>>(item);
    future->Await();
  }
}

bool IsReady(nb::object futures) {
  if (nb::isinstance<TransferFuture>(futures)) {
    return nb::cast<std::shared_ptr<TransferFuture>>(futures)->IsReady();
  }
  for (nb::handle item : futures) {
    if (!nb::cast<std::shared_ptr<TransferFuture>>(item)->IsReady()) {
      return false;
    }
  }
  return true;
}

}  // namespace

NB_MODULE(_raiden_transfer_engine, m) {
  nb::class_<TransferFuture>(m, "RaidenTransferFuture")
      .def("Await", &TransferFuture::Await)
      .def("wait", &TransferFuture::Await)
      .def("IsReady", &TransferFuture::IsReady)
      .def("is_ready", &TransferFuture::IsReady);

  nb::class_<RaidenTransferEngine>(m, "RaidenTransferEngine")
      .def(nb::init<const TensorList&, int64_t, int64_t, int64_t, int64_t,
                    double, bool>(),
           nb::arg("kv_caches"), nb::arg("tp_rank"),
           nb::arg("local_control_port"), nb::arg("max_blocks"),
           nb::arg("num_slots"), nb::arg("timeout_s") = 120.0,
           nb::arg("unsafe_skip_buffer_lock") = true)
      .def_prop_ro("uses_prepared_tpu_buffers",
                   &RaidenTransferEngine::UsesPreparedTpuBuffers)
      .def_prop_ro("local_control_port",
                   &RaidenTransferEngine::local_control_port)
      .def_prop_ro("local_data_port", &RaidenTransferEngine::local_data_port)
      .def("register_kv_cache", &RaidenTransferEngine::RegisterKvCache,
           nb::arg("kv_caches"))
      .def("register_host_buffers", &RaidenTransferEngine::RegisterHostBuffers,
           nb::arg("host_pool"), nb::arg("tp_rank"))
      .def("register_send", &RaidenTransferEngine::RegisterSend,
           nb::arg("req_id"), nb::arg("uuid"), nb::arg("block_ids"))
      .def("submit_load", &RaidenTransferEngine::SubmitLoad, nb::arg("req_id"),
           nb::arg("uuid"), nb::arg("remote_endpoint"),
           nb::arg("remote_block_ids"), nb::arg("local_block_ids"))
      .def(
          "stage_d2h",
          [](RaidenTransferEngine& self, int64_t slot_idx, int64_t num_blocks,
             const std::vector<int64_t>& block_ids) {
            auto result = self.IssueD2H(slot_idx, num_blocks, block_ids);
            return nb::make_tuple(result.future, self.kv_caches(),
                                  HostSpanMemoryViews(result.host_spans),
                                  result.total_bytes);
          },
          nb::kw_only(), nb::arg("slot_idx"), nb::arg("num_blocks"),
          nb::arg("block_ids"))
      .def(
          "stage_d2h_sync",
          [](RaidenTransferEngine& self, int64_t slot_idx, int64_t num_blocks,
             const std::vector<int64_t>& block_ids) {
            auto result = self.IssueD2H(slot_idx, num_blocks, block_ids);
            result.future->Await();
          },
          nb::kw_only(), nb::arg("slot_idx"), nb::arg("num_blocks"),
          nb::arg("block_ids"))
      .def(
          "commit_h2d",
          [](RaidenTransferEngine& self, int64_t slot_idx, int64_t num_blocks,
             const std::vector<int64_t>& local_block_ids) {
            auto res = self.CommitH2DRaw(slot_idx, num_blocks, local_block_ids);
            return nb::make_tuple(res.duration_issue_ms, res.duration_wait_ms,
                                  res.duration_total_ms, res.total_bytes);
          },
          nb::kw_only(), nb::arg("slot_idx"), nb::arg("num_blocks"),
          nb::arg("local_block_ids"))
      .def(
          "rank_layer_views",
          [](RaidenTransferEngine& self, int64_t slot_idx, int64_t rank,
             int64_t num_blocks) {
            if (rank != self.tp_rank()) {
              throw std::invalid_argument(
                  "Raiden internal slots are per-rank only");
            }
            return HostSpanMemoryViews(self.LayerSpans(slot_idx, num_blocks));
          },
          nb::arg("slot_idx"), nb::arg("rank"), nb::arg("num_blocks"))
      .def(
          "unpack_rank_layers",
          [](RaidenTransferEngine& self, int64_t slot_idx, int64_t rank,
             int64_t num_blocks, nb::object layer_buffers) {
            if (rank != self.tp_rank()) {
              throw std::invalid_argument(
                  "Raiden internal slots are per-rank only");
            }
            std::vector<KVCacheHostSpan> spans =
                self.LayerSpans(slot_idx, num_blocks);
            size_t idx = 0;
            for (nb::handle item : layer_buffers) {
              if (idx >= spans.size()) {
                throw std::invalid_argument("too many layer buffers");
              }
              nb::ndarray<> source = nb::cast<nb::ndarray<>>(item);
              const size_t source_bytes = source.nbytes();
              if (source_bytes != spans[idx].nbytes) {
                throw std::invalid_argument("layer buffer size mismatch");
              }
              std::memcpy(spans[idx].ptr, source.data(), spans[idx].nbytes);
              ++idx;
            }
            if (idx != spans.size()) {
              throw std::invalid_argument("too few layer buffers");
            }
          },
          nb::arg("slot_idx"), nb::arg("rank"), nb::arg("num_blocks"),
          nb::arg("layer_buffers"))
      .def("submit_d2h", &RaidenTransferEngine::SubmitD2H, nb::kw_only(),
           nb::arg("slot_idx"), nb::arg("num_blocks"), nb::arg("block_ids"))
      .def("submit_h2d", &RaidenTransferEngine::SubmitH2D, nb::kw_only(),
           nb::arg("slot_idx"), nb::arg("num_blocks"),
           nb::arg("local_block_ids"))
      .def("poll_finished",
           [](RaidenTransferEngine& self) {
             auto [done_sending, done_recving, failed_recving] =
                 self.PollFinishedRaw();
             return nb::make_tuple(done_sending, done_recving, failed_recving);
           })
      .def("poll_transfer_ops", &RaidenTransferEngine::PollTransferOps)
      .def("wait_transfer", &RaidenTransferEngine::WaitTransfer,
           nb::arg("op_id"))
      .def("_count_copy_segments_for_testing",
           &RaidenTransferEngine::CountCopySegmentsForTesting,
           nb::arg("block_ids"))
      .def(
          "_send_copy_plan_for_testing",
          [](RaidenTransferEngine& self,
             const std::vector<int64_t>& block_ids) {
            return CopyPlanToDict(
                self.BuildProducerCopyPlanForTesting(block_ids));
          },
          nb::arg("block_ids"))
      .def(
          "_load_copy_plan_for_testing",
          [](RaidenTransferEngine& self,
             const std::vector<int64_t>& remote_block_ids,
             const std::vector<int64_t>& local_block_ids) {
            return CopyPlanToDict(self.BuildLoadCopyPlanForTesting(
                remote_block_ids, local_block_ids));
          },
          nb::arg("remote_block_ids"), nb::arg("local_block_ids"));

  m.def("await_all", &AwaitAll, nb::arg("futures"));
  m.def("is_ready", &IsReady, nb::arg("futures"));
}

}  // namespace tpu_raiden::kv_cache
