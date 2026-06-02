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

// Copyright 2026 Google LLC
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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_TRANSFER_ENGINE_BASE_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_TRANSFER_ENGINE_BASE_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/raw_transfer_core.h"
#include "kv_cache/kv_cache_manager_base.h"

namespace tpu_raiden {

class TransferFuture {
 public:
  TransferFuture() = default;

  void Add(raiden::PjRtCopyFuture future) {
    futures_.push_back(std::move(future));
  }

  void AddAll(const std::shared_ptr<TransferFuture>& other) {
    futures_.insert(futures_.end(), other->futures_.begin(),
                    other->futures_.end());
  }

  void Await() {
    for (auto& future : futures_) {
      if (future.IsValid()) {
        absl::Status status = future.Await().status();
        if (!status.ok()) {
          throw std::runtime_error("Async transfer failed: " +
                                   std::string(status.message()));
        }
      }
    }
    futures_.clear();
  }

  bool IsReady() const {
    for (const auto& future : futures_) {
      if (future.IsValid() && !future.IsReady()) {
        return false;
      }
    }
    return true;
  }

 private:
  std::vector<raiden::PjRtCopyFuture> futures_;
};

struct CopySpec {
  std::vector<int64_t> src_offsets;
  std::vector<int64_t> dst_offsets;
  std::vector<int64_t> sizes;
};

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

struct StageResult {
  std::shared_ptr<TransferFuture> future;
  std::vector<kv_cache::KVCacheHostSpan> host_spans;
  int64_t total_bytes = 0;
  int64_t copy_segments = 0;
};

struct CommitResult {
  double duration_issue_ms;
  double duration_wait_ms;
  double duration_total_ms;
  int64_t total_bytes;
};

class TransferEngineBase {
 public:
  TransferEngineBase(std::unique_ptr<kv_cache::KVCacheManagerBase> kv_transfer,
                     int64_t tp_rank, int64_t local_control_port,
                     int64_t max_blocks, int64_t num_slots, double timeout_s,
                     bool unsafe_skip_buffer_lock);

  virtual ~TransferEngineBase();

  bool UsesPreparedTpuBuffers() const { return kv_transfer_ != nullptr; }

  int64_t SubmitD2H(int64_t slot_idx, int64_t num_blocks,
                    const std::vector<int64_t>& block_ids);

  int64_t SubmitH2D(int64_t slot_idx, int64_t num_blocks,
                    const std::vector<int64_t>& local_block_ids);

  int64_t RegisterSend(const std::string& req_id, uint64_t uuid,
                       const std::vector<int64_t>& block_ids);

  int64_t SubmitLoad(const std::string& req_id, uint64_t uuid,
                     const std::string& remote_endpoint,
                     const std::vector<int64_t>& remote_block_ids,
                     const std::vector<int64_t>& local_block_ids);

  std::vector<int64_t> PollTransferOps();

  void WaitTransfer(int64_t op_id);

  std::tuple<std::vector<std::string>, std::vector<std::string>,
             std::vector<std::string>>
  PollFinishedRaw();

  // Helpers for subclasses / nanobindings
  virtual StageResult IssueD2H(int64_t slot_idx, int64_t num_blocks,
                               const std::vector<int64_t>& block_ids);

  virtual StageResult IssueH2D(int64_t slot_idx, int64_t num_blocks,
                               const std::vector<int64_t>& local_block_ids);

  virtual CommitResult CommitH2DRaw(
      int64_t slot_idx, int64_t num_blocks,
      const std::vector<int64_t>& local_block_ids);

  virtual std::vector<kv_cache::KVCacheHostSpan> LayerSpans(int64_t slot_idx,
                                                            int64_t num_blocks);

  int local_control_port() const { return local_control_port_; }
  int local_data_port() const { return local_data_port_; }
  int64_t tp_rank() const { return tp_rank_; }

  // Testing helpers
  int64_t CountCopySegmentsForTesting(
      const std::vector<int64_t>& block_ids) const;
  CopyPlan BuildProducerCopyPlanForTesting(
      const std::vector<int64_t>& block_ids) const;
  CopyPlan BuildLoadCopyPlanForTesting(
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids) const;

 protected:
  struct PendingOperation {
    std::shared_ptr<TransferFuture> future;
    std::shared_ptr<std::promise<void>> load_promise;
  };

  struct SendEntry {
    std::string req_id;
    uint64_t uuid = 0;
    int64_t slot_idx = -1;
    int64_t num_blocks = 0;
    int64_t registered_num_blocks = 0;
    int64_t total_bytes = 0;
    int64_t copy_segments = 0;
    std::vector<int64_t> registered_block_ids;
    std::set<int64_t> registered_block_set;
    std::chrono::steady_clock::time_point register_start;
    std::chrono::steady_clock::time_point d2h_issue_start;
    std::chrono::steady_clock::time_point d2h_issue_done;
    std::chrono::steady_clock::time_point d2h_done;
    bool stage_done = false;
    bool failed = false;
    bool pull_started = false;
    bool slot_released = false;
    std::chrono::steady_clock::time_point deadline;
  };

  struct PullBlockDescriptor {
    uint64_t remote_block_base = 0;
    uint64_t num_blocks = 0;
  };

  struct alignas(8) ControlRequestHeader {
    uint32_t magic = 0x52414944;  // "RAID"
    uint32_t op = 0;
    uint64_t uuid = 0;
    uint64_t num_blocks = 0;
  };

  struct alignas(8) ControlResponseHeader {
    uint32_t magic = 0x44494152;  // "DIAR"
    int32_t status = 0;
    uint32_t num_layers = 0;
    uint32_t data_port = 0;
    uint64_t message_len = 0;
  };

  static constexpr uint32_t kControlMagic = 0x52414944;
  static constexpr uint32_t kResponseMagic = 0x44494152;
  static constexpr uint32_t kOpAck = 2;
  static constexpr uint32_t kOpPullStream = 3;

  std::string EndpointWithPort(const std::string& endpoint, int port) const;
  ControlResponseHeader ReadControlResponseHeader(int fd);
  void AckSend(uint64_t uuid);
  void ConfigureDataPortFromKvTransfer();
  uint64_t StagingBlockBase(int64_t slot_idx) const;
  std::vector<int> ContiguousBlockIds(uint64_t base, uint64_t count) const;
  static void ValidateRequestedBlocks(
      const SendEntry& entry, const std::vector<int64_t>& requested_block_ids);

  void InitializeSlotPool(int64_t num_slots);
  int64_t AcquireSlot();
  int64_t AcquireSlotLocked();
  void ReleaseSlotLocked(int64_t slot_idx);
  void ReleaseEntrySlotLocked(const std::shared_ptr<SendEntry>& entry);

  void StartControlServer();
  void StopControlServer();
  void ControlServerLoop();
  void HandleControlConnection(int fd);
  void ProcessPullStream(int fd, const ControlRequestHeader& req);
  void AckRemote(const std::string& remote_endpoint, uint64_t uuid);

  int64_t StorePending(PendingOperation op);
  std::chrono::steady_clock::time_point DeadlineFromNow() const;

  static CopySpec Offsets(const std::vector<int64_t>& block_ids,
                          bool source_is_compact);
  static kv_cache::KVCacheCopySpec ToKVCacheCopySpec(const CopySpec& spec);

  std::unique_ptr<kv_cache::KVCacheManagerBase> kv_transfer_;
  int64_t tp_rank_ = 0;
  int local_control_port_ = 0;
  int local_data_port_ = 0;
  int64_t max_blocks_ = 0;
  int64_t num_slots_ = 0;
  double timeout_s_ = 120.0;
  bool unsafe_skip_buffer_lock_ = true;

  int64_t next_op_id_ = 1;
  std::map<int64_t, PendingOperation> pending_;
  std::deque<int64_t> free_slots_;
  std::map<uint64_t, std::shared_ptr<SendEntry>> send_entries_;
  std::set<uint64_t> pending_acks_;
  std::set<std::string> done_sending_;
  std::set<std::string> done_recving_;
  std::set<std::string> failed_recving_;

  std::mutex mu_;
  std::condition_variable cv_;
  int control_fd_ = -1;
  std::atomic<bool> stopping_{false};
  std::thread control_thread_;
  std::mutex control_workers_mu_;
  std::vector<std::thread> control_workers_;
  std::mutex worker_threads_mu_;
  std::vector<std::thread> worker_threads_;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_TRANSFER_ENGINE_BASE_H_
