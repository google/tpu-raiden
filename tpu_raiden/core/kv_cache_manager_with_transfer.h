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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_KV_CACHE_MANAGER_WITH_TRANSFER_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_KV_CACHE_MANAGER_WITH_TRANSFER_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tpu_raiden/core/host_memory_allocator.h"
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/kv_cache/kv_cache_manager_base.h"

namespace tpu_raiden {

struct EndpointDescriptor {
  std::string endpoint;
  std::vector<int64_t> shards;
};

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
        absl::Status status = future.Await();
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
  std::vector<int64_t> h2d_host_block_ids;
  std::vector<int64_t> transport_host_block_ids;
  std::vector<size_t> host_dst_to_src;
  CopySpec d2h_copy;
  CopySpec h2d_copy;

  bool RequiresHostReorder() const { return !host_dst_to_src.empty(); }
};

struct StageResult {
  // TransferFuture is shared because it is exported to Python via nanobind.
  // Python garbage collection and C++ pending operations list share ownership
  // of the future.
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

struct PendingCopy {
  int64_t host_block_id;
  int64_t chip_block_id;
};

class KVCacheManagerWithTransfer : public kv_cache::KVCacheManagerBase {
 public:
  struct Slot {
    int64_t slot_idx;
    std::vector<int> block_ids;
  };

  KVCacheManagerWithTransfer(
      const std::vector<std::vector<xla::PjRtBuffer*>>& layer_buffers,
      std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
      bool unsafe_skip_buffer_lock, int parallelism,
      HostBufferAllocator host_allocator, int64_t node_id = 0,
      int64_t local_control_port = -1, int64_t max_blocks = 0,
      int64_t num_slots = 0, double timeout_s = 120.0);

  KVCacheManagerWithTransfer(
      const std::vector<std::vector<xla::PjRtBuffer*>>& layer_buffers,
      size_t slice_byte_size, const std::vector<int64_t>& dimensions,
      size_t physical_size, std::optional<int> local_port,
      std::optional<int> host_blocks_to_allocate, bool unsafe_skip_buffer_lock,
      int parallelism, HostBufferAllocator host_allocator, int64_t node_id = 0,
      int64_t local_control_port = -1, int64_t max_blocks = 0,
      int64_t num_slots = 0, double timeout_s = 120.0,
      std::optional<int> assigned_numa_node = std::nullopt);

  // Metadata-based constructor for FFI / CPU-only testing
  KVCacheManagerWithTransfer(
      size_t num_layers, size_t num_shards, size_t slice_byte_size,
      std::optional<int> local_port, std::optional<int> host_blocks_to_allocate,
      int parallelism = 1, int64_t node_id = 0, int64_t local_control_port = -1,
      int64_t max_blocks = 0, int64_t num_slots = 0, double timeout_s = 120.0);

  virtual ~KVCacheManagerWithTransfer();

  virtual int64_t NotifyForRead(const std::string& req_id, uint64_t uuid,
                                const std::vector<int64_t>& block_ids);

  virtual void StartRead(
      const std::string& req_id, uint64_t uuid,
      const std::string& remote_endpoint,
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids, int parallelism = 1,
      std::optional<std::vector<int64_t>> local_host_block_ids = std::nullopt);

  virtual void StartRead(
      const std::string& req_id, uint64_t uuid,
      const std::vector<std::string>& remote_endpoints,
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids, int parallelism = 1,
      std::optional<std::vector<int64_t>> local_host_block_ids = std::nullopt);

  virtual std::tuple<std::vector<std::string>, std::vector<std::string>,
                     std::vector<std::string>>
  CompleteReadRaw();

  absl::Status RegisterActivePlan(uint64_t uuid,
                                  const rpc::StartTransferRequest& request,
                                  bool is_sender) override;

  absl::Status OnBlocksReceived(const std::vector<int>& block_ids,
                                uint64_t uuid = 0) override;

  virtual std::vector<EndpointDescriptor> get_local_endpoints() const;

  virtual void StartRead(
      const std::string& req_id, uint64_t uuid,
      const std::vector<EndpointDescriptor>& remote_descriptors,
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids, int parallelism = 1,
      std::optional<std::vector<int64_t>> local_host_block_ids = std::nullopt);

  virtual int local_control_port() const { return local_control_port_; }
  virtual int64_t node_id() const { return node_id_; }

 protected:
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

  struct StagingLayerReady {
    bool done = false;
    absl::Status status = absl::OkStatus();
  };

  struct StagingReadinessState {
    int64_t slot_idx = -1;
    int64_t num_blocks = 0;
    size_t num_layers = 0;
    size_t num_shards = 0;
    std::mutex mu;
    std::condition_variable cv;
    std::vector<StagingLayerReady> layers;
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
    uint32_t consumer_data_port = 0;
    uint8_t consumer_ip[16] = {0};
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

  std::vector<int> ContiguousBlockIds(uint64_t base, uint64_t count) const;
  static void ValidateRequestedBlocks(
      const SendEntry& entry, const std::vector<int64_t>& requested_block_ids);

  absl::Status InitializeSlotPool(int64_t num_slots);
  Slot AcquireSlot();
  Slot AcquireSlotLocked();
  void ReleaseSlotLocked(int64_t slot_idx);
  void ReleaseEntrySlotLocked(const std::shared_ptr<SendEntry>& entry);

  void StartControlServer();
  void StopControlServer();
  void ControlServerLoop();
  void HandleControlConnection(int fd);
  void ProcessPullStream(int fd, const ControlRequestHeader& req);
  void AckRemote(const std::string& remote_endpoint, uint64_t uuid);
  absl::Status WaitForStagingBlockRead(size_t layer_idx, size_t shard_idx,
                                       int block_id);
  std::shared_ptr<StagingReadinessState> CreateStagingReadiness(
      int64_t slot_idx, int64_t num_blocks);
  void MarkStagingLayerReady(
      const std::shared_ptr<StagingReadinessState>& state, size_t layer_idx,
      size_t shard_idx, absl::Status status);
  void RemoveStagingReadinessLocked(int64_t slot_idx);

  using H2dIssueFuture =
      std::shared_future<absl::StatusOr<raiden::PjRtCopyFuture>>;

  absl::Status WaitForPendingWork() override;

  struct RecvEntry {
    std::string req_id;
    int64_t slot_idx = -1;  // host staging slot to release on completion
    CopySpec h2d_copy;
    std::vector<int64_t> chip_block_ids;
    absl::flat_hash_map<int64_t, int64_t> host_to_chip;
    std::map<std::pair<size_t, size_t>, std::vector<PendingCopy>>
        pending_h2d_copies;
    std::vector<H2dIssueFuture> h2d_dispatch_futures;
    int32_t total_blocks = 0;
    int32_t num_completed_blocks = 0;
    std::vector<int> accumulated_host_block_ids;
    std::chrono::steady_clock::time_point deadline;
  };
  absl::flat_hash_map<uint64_t, RecvEntry> active_recv_entries_;

  void StartPushInternal(uint64_t uuid, const std::string& remote_data_endpoint,
                         const std::vector<int64_t>& src_block_ids,
                         const std::vector<int64_t>& dst_block_ids);

  std::chrono::steady_clock::time_point DeadlineFromNow() const;

  static CopySpec Offsets(const std::vector<int64_t>& block_ids,
                          bool source_is_compact);
  static kv_cache::KVCacheCopySpec ToKVCacheCopySpec(const CopySpec& spec);

  int64_t node_id_ = 0;
  int local_control_port_ = 0;
  int local_data_port_ = 0;
  int64_t max_blocks_ = 0;
  int64_t num_slots_ = 0;
  double timeout_s_ = 120.0;
  bool unsafe_skip_buffer_lock_ = true;

  std::deque<Slot> free_slots_;
  std::vector<Slot> all_slots_;
  // SendEntry is shared across threads: created/timed-out/cleaned-up on the
  // main thread, but accessed asynchronously in control worker threads
  // handling pull connections.
  std::map<uint64_t, std::shared_ptr<SendEntry>> send_entries_;
  std::set<uint64_t> pending_acks_;
  std::set<std::string> done_sending_;
  std::set<std::string> done_recving_;
  std::set<std::string> failed_recving_;
  // StagingReadinessState is shared because it is captured by value in the
  // async PjRt copy callbacks (e.g. OnReady). A shared_ptr is required here
  // to ensure the state stays alive even if the entry is removed from this
  // map (e.g. on timeout, cancellation, or slot release) before the async
  // callback runs.
  std::map<int64_t, std::shared_ptr<StagingReadinessState>> staging_readiness_;
  std::map<int64_t, std::shared_ptr<StagingReadinessState>>
      active_producer_blocks_;
  std::mutex mu_;
  std::condition_variable cv_;
  int control_fd_ = -1;
  std::atomic<bool> stopping_{false};
  std::thread control_thread_;

 private:
  std::optional<int> GetLocalTpuNumaNode(xla::PjRtBuffer* buf) const;

  StageResult IssueH2D(int64_t slot_idx, int64_t num_blocks,
                       const std::vector<int64_t>& local_block_ids);

  std::vector<kv_cache::KVCacheHostSpan> LayerSpans(int64_t slot_idx,
                                                    int64_t num_blocks);
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_KV_CACHE_MANAGER_WITH_TRANSFER_H_
