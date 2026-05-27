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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "api/torch/kv_cache_manager.h"
#include "kv_cache/kv_cache_manager_base.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "raiden_lib/raw_transfer/raw_transfer_core.h"
#include "torch/extension.h"  // IWYU pragma: keep

namespace py = pybind11;

namespace tpu_raiden::kv_cache {
namespace {

using TensorList = std::vector<at::Tensor>;

[[noreturn]] void ThrowStatus(const std::string& context,
                              const absl::Status& status) {
  throw std::runtime_error(context + ": " + std::string(status.message()));
}

template <typename T>
T ValueOrThrow(const std::string& context, absl::StatusOr<T> value_or) {
  if (!value_or.ok()) {
    ThrowStatus(context, value_or.status());
  }
  return std::move(value_or).value();
}

void ValidateCpuTensor(const at::Tensor& tensor, const char* role) {
  if (!tensor.device().is_cpu()) {
    throw std::invalid_argument(std::string(role) + " must be a CPU tensor");
  }
  if (!tensor.is_contiguous()) {
    throw std::invalid_argument(std::string(role) + " must be contiguous");
  }
}

std::vector<std::vector<at::Tensor>> SingleShardLayers(
    const TensorList& kv_caches) {
  std::vector<std::vector<at::Tensor>> layers;
  layers.reserve(kv_caches.size());
  for (const auto& kv_cache : kv_caches) {
    layers.push_back({kv_cache});
  }
  return layers;
}

class RaidenTransferFuture {
 public:
  RaidenTransferFuture() = default;

  void Add(std::shared_ptr<raiden::PjRtCopyFuture> future) {
    futures_.push_back(std::move(future));
  }

  void Add(raiden::PjRtCopyFuture future) {
    futures_.push_back(
        std::make_shared<raiden::PjRtCopyFuture>(std::move(future)));
  }

  void AddAll(const std::shared_ptr<RaidenTransferFuture>& other) {
    futures_.insert(futures_.end(), other->futures_.begin(),
                    other->futures_.end());
  }

  void Await() {
    for (const auto& future : futures_) {
      if (future) {
        future->Await();
      }
    }
    futures_.clear();
  }

  bool IsReady() const {
    for (const auto& future : futures_) {
      if (future && !future->IsReady()) {
        return false;
      }
    }
    return true;
  }

 private:
  std::vector<std::shared_ptr<raiden::PjRtCopyFuture>> futures_;
};

struct RaidenStageResult {
  std::shared_ptr<RaidenTransferFuture> future;
  TensorList host_views;
  int64_t total_bytes = 0;
  int64_t copy_segments = 0;
};

class RaidenTransferEngine {
 public:
  RaidenTransferEngine(const TensorList& kv_caches, int64_t tp_rank,
                       int64_t local_control_port, int64_t max_blocks,
                       int64_t num_slots, double timeout_s,
                       bool unsafe_skip_buffer_lock)
      : tp_rank_(tp_rank),
        local_control_port_(static_cast<int>(local_control_port)),
        local_data_port_(static_cast<int>(local_control_port) + 1),
        max_blocks_(max_blocks),
        timeout_s_(timeout_s),
        unsafe_skip_buffer_lock_(unsafe_skip_buffer_lock) {
    if (max_blocks_ <= 0) {
      throw std::invalid_argument("max_blocks must be positive");
    }
    if (num_slots <= 0) {
      throw std::invalid_argument("num_slots must be positive");
    }
    RegisterKvCache(kv_caches);
    AllocateHostSlots(num_slots);
  }

  ~RaidenTransferEngine() = default;

  std::vector<int64_t> RegisterKvCache(const TensorList& kv_caches) {
    kv_caches_ = kv_caches;
    kv_transfer_ = std::make_unique<tpu_raiden::torch::KVCacheManager>(
        SingleShardLayers(kv_caches_), /*block_size=*/1,
        /*local_port=*/std::nullopt, /*host_blocks_to_allocate=*/std::nullopt,
        /*external_host_ptrs=*/std::nullopt, unsafe_skip_buffer_lock_);

    std::vector<int64_t> region_ids;
    region_ids.reserve(kv_caches_.size());
    for (size_t i = 0; i < kv_caches_.size(); ++i) {
      region_ids.push_back(static_cast<int64_t>(i));
    }
    return region_ids;
  }

  void RegisterHostBuffers(py::object /*host_pool*/, int64_t tp_rank) {
    tp_rank_ = tp_rank;
  }

  bool UsesPreparedTpuBuffers() const { return kv_transfer_ != nullptr; }

  py::tuple StageD2H(int64_t slot_idx, int64_t num_blocks,
                     const std::vector<int64_t>& block_ids) {
    RaidenStageResult result = IssueD2H(slot_idx, num_blocks, block_ids);
    return py::make_tuple(result.future, kv_caches_, result.host_views,
                          result.total_bytes);
  }

  void StageD2HSync(int64_t slot_idx, int64_t num_blocks,
                    const std::vector<int64_t>& block_ids) {
    RaidenStageResult result = IssueD2H(slot_idx, num_blocks, block_ids);
    result.future->Await();
  }

  py::tuple CommitH2D(int64_t slot_idx, int64_t num_blocks,
                      const std::vector<int64_t>& local_block_ids) {
    if (num_blocks != static_cast<int64_t>(local_block_ids.size())) {
      throw std::invalid_argument("num_blocks must match len(local_block_ids)");
    }
    auto t0 = std::chrono::steady_clock::now();
    RaidenStageResult result = IssueH2D(slot_idx, num_blocks, local_block_ids);
    auto t_issued = std::chrono::steady_clock::now();
    result.future->Await();
    auto t_done = std::chrono::steady_clock::now();
    return py::make_tuple(DurationMs(t0, t_issued),
                          DurationMs(t_issued, t_done), DurationMs(t0, t_done),
                          result.total_bytes);
  }

  py::object RankLayerViews(int64_t slot_idx, int64_t rank,
                            int64_t num_blocks) {
    if (rank != tp_rank_) {
      throw std::invalid_argument("Raiden internal slots are per-rank only");
    }
    py::list views;
    for (const auto& view : LayerViews(slot_idx, num_blocks)) {
      views.append(view);
    }
    return views;
  }

  void UnpackRankLayers(int64_t slot_idx, int64_t rank, int64_t num_blocks,
                        py::object layer_buffers) {
    if (rank != tp_rank_) {
      throw std::invalid_argument("Raiden internal slots are per-rank only");
    }
    TensorList views = LayerViews(slot_idx, num_blocks);
    size_t idx = 0;
    for (py::handle item : layer_buffers) {
      if (idx >= views.size()) {
        throw std::invalid_argument("too many layer buffers");
      }
      py::buffer source = py::reinterpret_borrow<py::buffer>(item);
      py::buffer_info info = source.request();
      if (static_cast<int64_t>(info.size * info.itemsize) !=
          views[idx].nbytes()) {
        throw std::invalid_argument("layer buffer size mismatch");
      }
      std::memcpy(views[idx].data_ptr(), info.ptr, views[idx].nbytes());
      ++idx;
    }
    if (idx != views.size()) {
      throw std::invalid_argument("too few layer buffers");
    }
  }

  int64_t SubmitD2H(int64_t slot_idx, int64_t num_blocks,
                    const std::vector<int64_t>& block_ids) {
    PendingOperation op;
    op.future = IssueD2H(slot_idx, num_blocks, block_ids).future;
    return StorePending(std::move(op));
  }

  int64_t SubmitH2D(int64_t slot_idx, int64_t num_blocks,
                    const std::vector<int64_t>& local_block_ids) {
    PendingOperation op;
    op.future = IssueH2D(slot_idx, num_blocks, local_block_ids).future;
    return StorePending(std::move(op));
  }

  int64_t RegisterSend(const std::string& req_id, uint64_t uuid,
                       const std::vector<int64_t>& block_ids) {
    if (block_ids.empty()) {
      return 0;
    }

    auto entry = std::make_shared<SendEntry>();
    entry->req_id = req_id;
    entry->uuid = uuid;
    entry->registered_num_blocks = static_cast<int64_t>(block_ids.size());
    entry->registered_block_ids = block_ids;
    for (int64_t block_id : block_ids) {
      entry->registered_block_set.insert(block_id);
    }

    std::lock_guard<std::mutex> lock(mu_);
    send_entries_[uuid] = std::move(entry);
    return static_cast<int64_t>(uuid);
  }

  int64_t SubmitLoad(const std::string& req_id, uint64_t uuid,
                     const std::string& remote_endpoint,
                     const std::vector<int64_t>& remote_block_ids,
                     const std::vector<int64_t>& local_block_ids) {
    (void)req_id;
    (void)uuid;
    (void)remote_endpoint;
    CopyPlan load_plan = BuildLoadCopyPlan(remote_block_ids, local_block_ids);
    const int64_t op_id = next_op_id_++;
    if (load_plan.num_blocks == 0) {
      return op_id;
    }
    throw std::runtime_error(
        "Raiden cross-host load is not implemented in this checkpoint");
  }

  py::tuple PollFinished() {
    std::vector<std::string> done_sending;
    std::vector<std::string> done_recving;
    std::vector<std::string> failed_recving;
    {
      std::lock_guard<std::mutex> lock(mu_);
      done_sending.assign(done_sending_.begin(), done_sending_.end());
      done_recving.assign(done_recving_.begin(), done_recving_.end());
      failed_recving.assign(failed_recving_.begin(), failed_recving_.end());
      done_sending_.clear();
      done_recving_.clear();
      failed_recving_.clear();
    }
    return py::make_tuple(done_sending, done_recving, failed_recving);
  }

  std::vector<int64_t> PollTransferOps() {
    std::vector<int64_t> done;
    for (auto it = pending_.begin(); it != pending_.end();) {
      if (!it->second.future || it->second.future->IsReady()) {
        if (it->second.future) {
          it->second.future->Await();
        }
        done.push_back(it->first);
        it = pending_.erase(it);
      } else {
        ++it;
      }
    }
    return done;
  }

  void WaitTransfer(int64_t op_id) {
    auto it = pending_.find(op_id);
    if (it == pending_.end()) {
      throw std::invalid_argument("unknown Raiden transfer op id");
    }
    if (it->second.future) {
      it->second.future->Await();
    }
    pending_.erase(it);
  }

  int local_control_port() const { return local_control_port_; }
  int local_data_port() const { return local_data_port_; }

  int64_t CountCopySegmentsForTesting(
      const std::vector<int64_t>& block_ids) const {
    return static_cast<int64_t>(
        Offsets(block_ids, /*source_is_compact=*/false).sizes.size());
  }

  py::dict SendCopyPlanForTesting(const std::vector<int64_t>& block_ids) const {
    return CopyPlanToDict(BuildProducerCopyPlan(block_ids));
  }

  py::dict LoadCopyPlanForTesting(
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids) const {
    return CopyPlanToDict(BuildLoadCopyPlan(remote_block_ids, local_block_ids));
  }

 private:
  struct CopySpec {
    std::vector<int64_t> src_offsets;
    std::vector<int64_t> dst_offsets;
    std::vector<int64_t> sizes;
  };

  static KVCacheCopySpec ToKVCacheCopySpec(const CopySpec& spec) {
    return {.src_offsets = spec.src_offsets,
            .dst_offsets = spec.dst_offsets,
            .sizes = spec.sizes};
  }

  struct CopyPlan {
    int64_t num_blocks = 0;
    std::vector<int64_t> requested_remote_block_ids;
    std::vector<int64_t> requested_local_block_ids;
    std::vector<int64_t> producer_remote_block_ids;
    std::vector<int64_t> h2d_local_block_ids;
    std::vector<size_t> host_dst_to_src;
    CopySpec d2h_copy;
    CopySpec h2d_copy;

    bool RequiresHostReorder() const { return !host_dst_to_src.empty(); }
  };

  struct PendingOperation {
    std::shared_ptr<RaidenTransferFuture> future;
  };

  struct HostSlot {
    TensorList layers;
  };

  struct SendEntry {
    std::string req_id;
    uint64_t uuid = 0;
    int64_t registered_num_blocks = 0;
    std::vector<int64_t> registered_block_ids;
    std::set<int64_t> registered_block_set;
  };

  static double DurationMs(std::chrono::steady_clock::time_point start,
                           std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
  }

  static CopySpec Offsets(const std::vector<int64_t>& block_ids,
                          bool source_is_compact) {
    const int64_t n = static_cast<int64_t>(block_ids.size());
    CopySpec spec;
    spec.src_offsets.reserve(block_ids.size());
    spec.dst_offsets.reserve(block_ids.size());
    spec.sizes.reserve(block_ids.size());
    for (int64_t start = 0; start < n;) {
      int64_t end = start + 1;
      while (end < n && block_ids[end] == block_ids[end - 1] + 1) {
        ++end;
      }
      const int64_t run_size = end - start;
      if (source_is_compact) {
        spec.src_offsets.push_back(start);
        spec.dst_offsets.push_back(block_ids[start]);
      } else {
        spec.src_offsets.push_back(block_ids[start]);
        spec.dst_offsets.push_back(start);
      }
      spec.sizes.push_back(run_size);
      start = end;
    }
    return spec;
  }

  static py::dict CopySpecToDict(const CopySpec& spec) {
    py::dict out;
    out["src_offsets"] = spec.src_offsets;
    out["dst_offsets"] = spec.dst_offsets;
    out["sizes"] = spec.sizes;
    return out;
  }

  static py::dict CopyPlanToDict(const CopyPlan& plan) {
    py::dict out;
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

  static std::vector<int64_t> CanonicalSendBlockIds(
      const std::vector<int64_t>& block_ids) {
    std::vector<int64_t> ordered = block_ids;
    std::stable_sort(ordered.begin(), ordered.end());
    return ordered;
  }

  static CopyPlan BuildProducerCopyPlan(
      const std::vector<int64_t>& block_ids) {
    CopyPlan plan;
    plan.num_blocks = static_cast<int64_t>(block_ids.size());
    plan.requested_remote_block_ids = block_ids;
    plan.producer_remote_block_ids = CanonicalSendBlockIds(block_ids);
    plan.d2h_copy =
        Offsets(plan.producer_remote_block_ids, /*source_is_compact=*/false);
    return plan;
  }

  static CopyPlan BuildLoadCopyPlan(
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids) {
    if (remote_block_ids.size() != local_block_ids.size()) {
      throw std::invalid_argument(
          "remote_block_ids and local_block_ids must have same length");
    }
    CopyPlan plan;
    plan.num_blocks = static_cast<int64_t>(remote_block_ids.size());
    plan.requested_remote_block_ids = remote_block_ids;
    plan.requested_local_block_ids = local_block_ids;
    if (remote_block_ids.empty()) {
      return plan;
    }
    std::vector<size_t> remote_order(remote_block_ids.size());
    std::vector<size_t> local_order(local_block_ids.size());
    for (size_t i = 0; i < remote_order.size(); ++i) {
      remote_order[i] = i;
      local_order[i] = i;
    }
    std::stable_sort(remote_order.begin(), remote_order.end(),
                     [&](size_t a, size_t b) {
                       return remote_block_ids[a] < remote_block_ids[b];
                     });
    std::stable_sort(local_order.begin(), local_order.end(),
                     [&](size_t a, size_t b) {
                       return local_block_ids[a] < local_block_ids[b];
                     });

    plan.producer_remote_block_ids.reserve(remote_order.size());
    plan.h2d_local_block_ids.reserve(local_order.size());
    plan.host_dst_to_src.reserve(local_order.size());
    std::vector<size_t> source_pos_by_original_idx(remote_order.size());
    for (size_t source_pos = 0; source_pos < remote_order.size();
         ++source_pos) {
      const size_t original_idx = remote_order[source_pos];
      plan.producer_remote_block_ids.push_back(remote_block_ids[original_idx]);
      source_pos_by_original_idx[original_idx] = source_pos;
    }
    bool identity_reorder = true;
    for (size_t dst_pos = 0; dst_pos < local_order.size(); ++dst_pos) {
      const size_t original_idx = local_order[dst_pos];
      const size_t src_pos = source_pos_by_original_idx[original_idx];
      plan.h2d_local_block_ids.push_back(local_block_ids[original_idx]);
      plan.host_dst_to_src.push_back(src_pos);
      identity_reorder = identity_reorder && (src_pos == dst_pos);
    }
    if (identity_reorder) {
      plan.host_dst_to_src.clear();
    }
    plan.h2d_copy =
        Offsets(plan.h2d_local_block_ids, /*source_is_compact=*/true);
    return plan;
  }

  TensorList LayerViews(int64_t slot_idx, int64_t num_blocks) {
    if (slot_idx < 0 || slot_idx >= static_cast<int64_t>(host_slots_.size())) {
      throw std::out_of_range("slot_idx out of range");
    }
    if (num_blocks < 0 || num_blocks > max_blocks_) {
      throw std::out_of_range("num_blocks out of range");
    }
    TensorList host_views;
    host_views.reserve(kv_caches_.size());
    for (size_t layer_idx = 0; layer_idx < host_slots_[slot_idx].layers.size();
         ++layer_idx) {
      at::Tensor tensor =
          host_slots_[slot_idx].layers[layer_idx].narrow(0, 0, num_blocks);
      ValidateCpuTensor(tensor, "Host staging view");
      host_views.push_back(std::move(tensor));
    }
    return host_views;
  }

  RaidenStageResult IssueD2H(int64_t slot_idx, int64_t num_blocks,
                             const std::vector<int64_t>& block_ids) {
    if (num_blocks != static_cast<int64_t>(block_ids.size())) {
      throw std::invalid_argument("num_blocks must match len(block_ids)");
    }
    CopySpec copy_spec = Offsets(block_ids, /*source_is_compact=*/false);
    KVCacheCopySpec transfer_spec = ToKVCacheCopySpec(copy_spec);
    TensorList host_views = LayerViews(slot_idx, num_blocks);
    auto future = std::make_shared<RaidenTransferFuture>();
    int64_t total_bytes = 0;
    for (size_t i = 0; i < kv_transfer_->num_layers(); ++i) {
      total_bytes += static_cast<int64_t>(host_views[i].nbytes());
      future->Add(ValueOrThrow("Failed to issue D2H transfer",
                               kv_transfer_->D2hTo(
                                   i, host_views[i].data_ptr(),
                                   host_views[i].nbytes(), transfer_spec)));
    }
    return {.future = std::move(future),
            .host_views = std::move(host_views),
            .total_bytes = total_bytes,
            .copy_segments = static_cast<int64_t>(copy_spec.sizes.size())};
  }

  RaidenStageResult IssueH2D(int64_t slot_idx, int64_t num_blocks,
                             const std::vector<int64_t>& local_block_ids) {
    if (num_blocks != static_cast<int64_t>(local_block_ids.size())) {
      throw std::invalid_argument(
          "num_blocks must match len(local_block_ids)");
    }
    CopySpec copy_spec = Offsets(local_block_ids, /*source_is_compact=*/true);
    KVCacheCopySpec transfer_spec = ToKVCacheCopySpec(copy_spec);
    TensorList host_views = LayerViews(slot_idx, num_blocks);
    auto future = std::make_shared<RaidenTransferFuture>();
    int64_t total_bytes = 0;
    for (size_t i = 0; i < kv_transfer_->num_layers(); ++i) {
      total_bytes += static_cast<int64_t>(host_views[i].nbytes());
      future->Add(ValueOrThrow("Failed to issue H2D transfer",
                               kv_transfer_->H2dFrom(
                                   i, host_views[i].data_ptr(),
                                   host_views[i].nbytes(), transfer_spec)));
    }
    return {.future = std::move(future),
            .host_views = std::move(host_views),
            .total_bytes = total_bytes,
            .copy_segments = static_cast<int64_t>(copy_spec.sizes.size())};
  }

  int64_t StorePending(PendingOperation op) {
    const int64_t op_id = next_op_id_++;
    pending_[op_id] = std::move(op);
    return op_id;
  }

  void AllocateHostSlots(int64_t num_slots) {
    if (kv_caches_.empty()) return;
    host_slots_.clear();
    free_slots_.clear();
    host_slots_.reserve(num_slots);
    for (int64_t slot = 0; slot < num_slots; ++slot) {
      HostSlot host_slot;
      host_slot.layers.reserve(kv_caches_.size());
      for (const auto& kv : kv_caches_) {
        std::vector<int64_t> shape;
        shape.reserve(kv.dim());
        shape.push_back(max_blocks_);
        for (int64_t d = 1; d < kv.dim(); ++d) {
          shape.push_back(kv.size(d));
        }
        at::Tensor host = at::empty(
            shape,
            kv.options().device(c10::Device(c10::kCPU)).pinned_memory(true));
        ValidateCpuTensor(host, "Host staging allocation");
        host_slot.layers.push_back(std::move(host));
      }
      host_slots_.push_back(std::move(host_slot));
      free_slots_.push_back(slot);
    }
  }

  TensorList kv_caches_;
  std::unique_ptr<tpu_raiden::torch::KVCacheManager> kv_transfer_;
  int64_t tp_rank_ = 0;
  int local_control_port_ = 0;
  int local_data_port_ = 0;
  int64_t max_blocks_ = 0;
  double timeout_s_ = 120.0;
  bool unsafe_skip_buffer_lock_ = true;
  int64_t next_op_id_ = 1;
  std::map<int64_t, PendingOperation> pending_;
  std::vector<HostSlot> host_slots_;
  std::deque<int64_t> free_slots_;
  std::map<uint64_t, std::shared_ptr<SendEntry>> send_entries_;
  std::set<std::string> done_sending_;
  std::set<std::string> done_recving_;
  std::set<std::string> failed_recving_;
  std::mutex mu_;
};

void AwaitAll(py::object futures) {
  if (py::isinstance<RaidenTransferFuture>(futures)) {
    auto future = futures.cast<std::shared_ptr<RaidenTransferFuture>>();
    future->Await();
    return;
  }
  for (py::handle item : futures) {
    auto future = item.cast<std::shared_ptr<RaidenTransferFuture>>();
    future->Await();
  }
}

bool IsReady(py::object futures) {
  if (py::isinstance<RaidenTransferFuture>(futures)) {
    return futures.cast<std::shared_ptr<RaidenTransferFuture>>()->IsReady();
  }
  for (py::handle item : futures) {
    if (!item.cast<std::shared_ptr<RaidenTransferFuture>>()->IsReady()) {
      return false;
    }
  }
  return true;
}

}  // namespace

PYBIND11_MODULE(_raiden_transfer_engine, m) {
  py::class_<RaidenTransferFuture, std::shared_ptr<RaidenTransferFuture>>(
      m, "RaidenTransferFuture")
      .def("Await", &RaidenTransferFuture::Await)
      .def("wait", &RaidenTransferFuture::Await)
      .def("IsReady", &RaidenTransferFuture::IsReady)
      .def("is_ready", &RaidenTransferFuture::IsReady);

  py::class_<RaidenTransferEngine>(m, "RaidenTransferEngine")
      .def(py::init<const TensorList&, int64_t, int64_t, int64_t, int64_t,
                    double, bool>(),
           py::arg("kv_caches"), py::arg("tp_rank"),
           py::arg("local_control_port"), py::arg("max_blocks"),
           py::arg("num_slots"), py::arg("timeout_s") = 120.0,
           py::arg("unsafe_skip_buffer_lock") = true)
      .def_property_readonly("uses_prepared_tpu_buffers",
                             &RaidenTransferEngine::UsesPreparedTpuBuffers)
      .def_property_readonly("local_control_port",
                             &RaidenTransferEngine::local_control_port)
      .def_property_readonly("local_data_port",
                             &RaidenTransferEngine::local_data_port)
      .def("register_kv_cache", &RaidenTransferEngine::RegisterKvCache,
           py::arg("kv_caches"))
      .def("register_host_buffers", &RaidenTransferEngine::RegisterHostBuffers,
           py::arg("host_pool"), py::arg("tp_rank"))
      .def("register_send", &RaidenTransferEngine::RegisterSend,
           py::arg("req_id"), py::arg("uuid"), py::arg("block_ids"))
      .def("submit_load", &RaidenTransferEngine::SubmitLoad, py::arg("req_id"),
           py::arg("uuid"), py::arg("remote_endpoint"),
           py::arg("remote_block_ids"), py::arg("local_block_ids"))
      .def("stage_d2h", &RaidenTransferEngine::StageD2H, py::kw_only(),
           py::arg("slot_idx"), py::arg("num_blocks"), py::arg("block_ids"))
      .def("stage_d2h_sync", &RaidenTransferEngine::StageD2HSync,
           py::kw_only(), py::arg("slot_idx"), py::arg("num_blocks"),
           py::arg("block_ids"))
      .def("commit_h2d", &RaidenTransferEngine::CommitH2D, py::kw_only(),
           py::arg("slot_idx"), py::arg("num_blocks"),
           py::arg("local_block_ids"))
      .def("rank_layer_views", &RaidenTransferEngine::RankLayerViews,
           py::arg("slot_idx"), py::arg("rank"), py::arg("num_blocks"))
      .def("unpack_rank_layers", &RaidenTransferEngine::UnpackRankLayers,
           py::arg("slot_idx"), py::arg("rank"), py::arg("num_blocks"),
           py::arg("layer_buffers"))
      .def("submit_d2h", &RaidenTransferEngine::SubmitD2H, py::kw_only(),
           py::arg("slot_idx"), py::arg("num_blocks"), py::arg("block_ids"))
      .def("submit_h2d", &RaidenTransferEngine::SubmitH2D, py::kw_only(),
           py::arg("slot_idx"), py::arg("num_blocks"),
           py::arg("local_block_ids"))
      .def("poll_finished", &RaidenTransferEngine::PollFinished)
      .def("poll_transfer_ops", &RaidenTransferEngine::PollTransferOps)
      .def("wait_transfer", &RaidenTransferEngine::WaitTransfer,
           py::arg("op_id"))
      .def("_count_copy_segments_for_testing",
           &RaidenTransferEngine::CountCopySegmentsForTesting,
           py::arg("block_ids"))
      .def("_send_copy_plan_for_testing",
           &RaidenTransferEngine::SendCopyPlanForTesting,
           py::arg("block_ids"))
      .def("_load_copy_plan_for_testing",
           &RaidenTransferEngine::LoadCopyPlanForTesting,
           py::arg("remote_block_ids"), py::arg("local_block_ids"));

  m.def("await_all", &AwaitAll, py::arg("futures"));
  m.def("is_ready", &IsReady, py::arg("futures"));
}

}  // namespace tpu_raiden::kv_cache
