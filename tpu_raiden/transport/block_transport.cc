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

#include "tpu_raiden/transport/block_transport.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>  // NOLINT
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_raiden/core/status_macros.h"
#include "tpu_raiden/transport/lib/raw_buffer_transport.h"
#include "tpu_raiden/transport/lib/socket/util.h"

namespace tpu_raiden {
namespace transport {

namespace {

using lib::ReadExact;
using lib::ReadVExact;
using lib::WriteExact;
using lib::WriteVExact;

constexpr uint8_t kUseBlockChunksFlag = 0x80;

#ifndef UIO_MAXIOV
#define UIO_MAXIOV 1024
#endif

#ifndef IOV_MAX
#define IOV_MAX UIO_MAXIOV
#endif

std::vector<struct iovec> ToIovec(const std::vector<BlockChunk>& chunks) {
  std::vector<struct iovec> iov;
  iov.reserve(chunks.size());
  for (const auto& chunk : chunks) {
    iov.push_back({.iov_base = chunk.ptr, .iov_len = chunk.size});
  }
  return iov;
}

absl::Status ValidateChunks(BlockTransportDelegate* delegate, size_t l,
                            size_t sh, const std::vector<BlockChunk>& chunks) {
  uint8_t* base = delegate->GetBlockArrayHostPointer(l, sh);
  // Some legacy delegates intentionally expose only scattered per-block
  // pointers and no flat array base. Preserve that contract; pool-aware
  // managers always expose their exact pool span here.
  if (base != nullptr) {
    const size_t host_size = delegate->GetBlockArrayHostSize(l, sh);
    const uintptr_t base_addr = reinterpret_cast<uintptr_t>(base);
    for (const auto& chunk : chunks) {
      const uintptr_t chunk_addr = reinterpret_cast<uintptr_t>(chunk.ptr);
      if (chunk_addr < base_addr) {
        return absl::OutOfRangeError(absl::StrCat(
            "Chunk out of bounds. Chunk ptr: ", chunk_addr,
            ", size: ", chunk.size, ", Block array base: ", base_addr,
            ", Block array size: ", host_size));
      }
      const size_t offset = chunk_addr - base_addr;
      if (offset > host_size || chunk.size > host_size - offset) {
        return absl::OutOfRangeError(absl::StrCat(
            "Chunk out of bounds. Chunk ptr: ", chunk_addr,
            ", size: ", chunk.size, ", Block array base: ", base_addr,
            ", Block array size: ", host_size));
      }
    }
  }
  const BlockChunkRegionValidationMode region_mode =
      delegate->block_chunk_region_validation_mode();
  if (region_mode != BlockChunkRegionValidationMode::kDisabled) {
    absl::Status status = delegate->ValidateBlockChunksInRegions(l, sh, chunks);
    if (!status.ok()) {
      if (region_mode == BlockChunkRegionValidationMode::kFail) {
        return status;
      }
      LOG(WARNING) << "Block chunk region validation warning: "
                   << status.ToString();
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<MajorOrder> ParseMajorOrder(uint8_t value) {
  uint8_t order_val = value & ~kUseBlockChunksFlag;
  switch (order_val) {
    case static_cast<uint8_t>(MajorOrder::kLayerMajor):
      return MajorOrder::kLayerMajor;
    case static_cast<uint8_t>(MajorOrder::kBlockMajor):
      return MajorOrder::kBlockMajor;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unknown block transport major order: ", value));
  }
}

template <typename Fn>
absl::Status ForEachPayload(MajorOrder major_order,
                            const std::vector<int>& layer_ids,
                            size_t num_shards, size_t num_blocks, Fn fn) {
  switch (static_cast<int>(major_order)) {
    case static_cast<int>(MajorOrder::kLayerMajor):
      for (int l : layer_ids) {
        for (size_t sh = 0; sh < num_shards; ++sh) {
          for (size_t k = 0; k < num_blocks; ++k) {
            RETURN_IF_ERROR(fn(l, sh, k));
          }
        }
      }
      return absl::OkStatus();
    case static_cast<int>(MajorOrder::kBlockMajor):
      for (size_t k = 0; k < num_blocks; ++k) {
        for (int l : layer_ids) {
          for (size_t sh = 0; sh < num_shards; ++sh) {
            RETURN_IF_ERROR(fn(l, sh, k));
          }
        }
      }
      return absl::OkStatus();
  }
  return absl::InvalidArgumentError("Unknown block transport major order");
}

}  // namespace

BlockTransport::BlockTransport(BlockTransportDelegate* delegate, int local_port,
                               const std::vector<std::string>& local_ips,
                               int parallelism)
    : RawBufferTransport(delegate, local_port, local_ips),
      block_delegate_(delegate),
      parallelism_(parallelism) {
  socket_workers_.reserve(parallelism_);
  for (int i = 0; i < parallelism_; ++i) {
    socket_workers_.push_back(
        std::thread(&BlockTransport::SocketWorkerLoop, this));
  }
}

BlockTransport::~BlockTransport() {
  {
    absl::MutexLock lock(scheduler_mu_);
    scheduler_stopping_ = true;
  }
  scheduler_cv_.SignalAll();
  for (auto& t : socket_workers_) {
    if (t.joinable()) t.join();
  }
}

void BlockTransport::SocketWorkerLoop() {
  while (!scheduler_stopping_) {
    std::unique_ptr<WriteTask> task;
    {
      absl::MutexLock lock(scheduler_mu_);
      while (true) {
        if (scheduler_stopping_) return;
        task = SelectNextTask();
        if (task) break;
        scheduler_cv_.Wait(&scheduler_mu_);
      }
    }

    if (task) {
      task->run();
      {
        absl::MutexLock lock(scheduler_mu_);
        auto it = peer_queues_.find(task->peer);
        if (it != peer_queues_.end()) {
          it->second.active_streams--;
        }
      }
      scheduler_cv_.SignalAll();
    }
  }
}

std::unique_ptr<BlockTransport::WriteTask> BlockTransport::SelectNextTask() {
  if (active_peers_.empty()) return nullptr;

  size_t start_idx = rr_index_;
  do {
    const std::string& peer = active_peers_[rr_index_];
    rr_index_ = (rr_index_ + 1) % active_peers_.size();

    auto& q = peer_queues_[peer];
    if (!q.tasks.empty() && q.active_streams < parallelism_) {
      q.active_streams++;
      auto task = std::move(q.tasks.front());
      q.tasks.pop_front();
      return task;
    }
  } while (rr_index_ != start_idx);

  return nullptr;
}

absl::Status BlockTransport::HandleCustomRequest(int client_fd,
                                                 const PacketHeader& header) {
  LOG(INFO) << "HandleCustomRequest (H2H read start): client_fd=" << client_fd
            << ", op=" << static_cast<int>(header.op)
            << ", uuid=" << header.uuid
            << ", numa=" << block_delegate_->node_id();

  if (header.op == 1 || header.op == 6) {
    return HandleIncomingPush(client_fd, header);
  } else if (header.op == 2) {
    return HandleIncomingPull(client_fd, header);
  } else {
    return absl::UnimplementedError(
        absl::StrCat("Unsupported block transport op: ", header.op));
  }
}

absl::Status BlockTransport::HandleIncomingPush(int client_fd,
                                                const PacketHeader& header) {
  ASSIGN_OR_RETURN(MajorOrder major_order, ParseMajorOrder(header.flags));
  std::vector<int> target_layers;
  if (header.local_id == 0xFFFFFFFF) {
    target_layers.resize(block_delegate_->num_block_arrays());
    std::iota(target_layers.begin(), target_layers.end(), 0);
  } else {
    if (header.local_id >= block_delegate_->num_block_arrays()) {
      return absl::OutOfRangeError(
          absl::StrCat("push block-array index ", header.local_id,
                       " out of range: ", block_delegate_->num_block_arrays()));
    }
    target_layers = {static_cast<int>(header.local_id)};
  }

  // Resolve the expectation source before the explicit-destination handshake
  // or any payload write. A uuid whose registered receive plan carries pool
  // fields is plan-declared: it addresses exactly one declared pool and its
  // completion gate is the plan's global push count. nullopt keeps the
  // header-declared (legacy) contract.
  std::optional<PoolPushProgressSpec> pool_progress_spec;
  for (int target_layer : target_layers) {
    ASSIGN_OR_RETURN(
        std::optional<PoolPushProgressSpec> candidate,
        block_delegate_->GetPoolPushProgressSpec(target_layer, header.uuid));
    if (!candidate.has_value()) {
      if (pool_progress_spec.has_value()) {
        return absl::InvalidArgumentError(
            "push mixes pool-keyed and legacy block arrays");
      }
      continue;
    }
    if (target_layers.size() != 1) {
      return absl::InvalidArgumentError(
          "pool-keyed push must address exactly one transfer pool");
    }
    if (candidate->expected_pushes == 0 || candidate->expected_pools == 0) {
      return absl::InvalidArgumentError(
          "pool-keyed push progress counts must be positive");
    }
    pool_progress_spec = *candidate;
  }

  std::vector<int> allocated_ids;

  std::vector<int> src_block_ids;
  if (header.op == 1) {
    ASSIGN_OR_RETURN(allocated_ids, block_delegate_->AllocateBlocks(
                                        header.count_or_size, header.uuid));
    RETURN_IF_ERROR(WriteExact(client_fd, allocated_ids.data(),
                               header.count_or_size * sizeof(int)));
  } else {
    allocated_ids.resize(header.count_or_size, 0);
    RETURN_IF_ERROR(ReadExact(client_fd, allocated_ids.data(),
                              header.count_or_size * sizeof(int)));
    src_block_ids.resize(header.count_or_size, 0);
    RETURN_IF_ERROR(ReadExact(client_fd, src_block_ids.data(),
                              header.count_or_size * sizeof(int)));
    uint8_t ack = 1;
    RETURN_IF_ERROR(WriteExact(client_fd, &ack, 1));
  }

  RETURN_IF_ERROR(ForEachPayload(
      major_order, target_layers, block_delegate_->num_shards(),
      header.count_or_size, [&](size_t l, size_t sh, size_t k) -> absl::Status {
        ABSL_DCHECK_LT(k, allocated_ids.size());
        const int dst_id = allocated_ids[k];
        uint32_t sender_size = 0;
        RETURN_IF_ERROR(
            ReadExact(client_fd, &sender_size, sizeof(sender_size)));

        const int64_t block_id_val = dst_id;
        int64_t src_bid = -1;
        if (!src_block_ids.empty()) {
          ABSL_DCHECK_LT(k, src_block_ids.size());
          src_bid = src_block_ids[k];
        }
        std::vector<BlockChunk> chunks = block_delegate_->GetBlockChunks(
            l, sh, absl::MakeConstSpan(&block_id_val, 1), sender_size,
            header.uuid, static_cast<int64_t>(header.remote_id),
            /*peer=*/"", src_bid);
        if (chunks.empty()) {
          return absl::NotFoundError(
              absl::StrCat("No transfer chunks found for block ", dst_id,
                           " and uuid ", header.uuid));
        }
        RETURN_IF_ERROR(ValidateChunks(block_delegate_, l, sh, chunks));

        uint32_t expected_size = 0;
        for (const auto& chunk : chunks) {
          expected_size += chunk.size;
        }
        if (sender_size != expected_size) {
          return absl::InternalError(absl::StrCat(
              "Block transfer size mismatch! Sender offered: ", sender_size,
              " bytes, but Receiver expected: ", expected_size,
              " bytes for Block ID: ", dst_id));
        }

        RETURN_IF_ERROR(ReadVExact(client_fd, ToIovec(chunks)));
        return absl::OkStatus();
      }));

  // Unified receive accounting: one progress map and one increment path for
  // both contracts; only the expectation source and the completion callback
  // differ. Plan-declared mode counts every sender's streams against the
  // plan's global per-pool total and retires the uuid when all declared pools
  // complete; header-declared (legacy) mode counts this push's parallelism
  // from header.reserved and retires the uuid when all constructor layers
  // complete.
  const int l =
      (header.local_id == 0xFFFFFFFF) ? 0 : static_cast<int>(header.local_id);
  const bool plan_declared = pool_progress_spec.has_value();
  const size_t expected_chunks =
      plan_declared ? pool_progress_spec->expected_pushes : header.reserved;
  bool trigger_completion = false;
  {
    absl::MutexLock lock(progress_mu_);
    auto& progress = layer_progress_[{header.uuid, l}];
    progress.completed_chunks++;
    if (plan_declared && progress.completed_chunks > expected_chunks) {
      return absl::AlreadyExistsError(
          absl::StrCat("pool ", l, " received more than ", expected_chunks,
                       " push streams for UUID ", header.uuid));
    }
    if (progress.completed_chunks == expected_chunks &&
        !progress.on_layer_received_called) {
      progress.on_layer_received_called = true;
      trigger_completion = true;
      if (plan_declared) {
        size_t completed_pools = 0;
        for (const auto& [key, pool_progress] : layer_progress_) {
          if (key.first == header.uuid &&
              pool_progress.on_layer_received_called) {
            ++completed_pools;
          }
        }
        if (completed_pools == pool_progress_spec->expected_pools) {
          for (auto it = layer_progress_.begin();
               it != layer_progress_.end();) {
            if (it->first.first == header.uuid) {
              auto erase_it = it++;
              layer_progress_.erase(erase_it);
            } else {
              ++it;
            }
          }
        }
      } else {
        bool all_layers_called = true;
        for (size_t layer = 0; layer < block_delegate_->num_layers(); ++layer) {
          auto it = layer_progress_.find({header.uuid, layer});
          if (it == layer_progress_.end() ||
              !it->second.on_layer_received_called) {
            all_layers_called = false;
            break;
          }
        }
        if (all_layers_called) {
          for (size_t layer = 0; layer < block_delegate_->num_layers();
               ++layer) {
            layer_progress_.erase({header.uuid, layer});
          }
        }
      }
    }
  }

  if (trigger_completion) {
    if (plan_declared) {
      RETURN_IF_ERROR(block_delegate_->OnPoolReceived(l, header.uuid));
    } else {
      RETURN_IF_ERROR(block_delegate_->OnLayerReceived(l, header.uuid));
    }
  }

  LOG(INFO) << "HandleCustomRequest (H2H read complete): client_fd="
            << client_fd << ", uuid=" << header.uuid
            << ", numa=" << block_delegate_->node_id();
  RETURN_IF_ERROR(
      block_delegate_->OnBlocksReceived(allocated_ids, header.uuid));
  uint8_t ack = 1;
  RETURN_IF_ERROR(WriteExact(client_fd, &ack, 1));
  return absl::OkStatus();
}

absl::Status BlockTransport::HandleIncomingPull(int client_fd,
                                                const PacketHeader& header) {
  ASSIGN_OR_RETURN(MajorOrder major_order, ParseMajorOrder(header.flags));
  if (block_delegate_->shard_factor() == 0) {
    return absl::InvalidArgumentError("shard_factor must be positive");
  }
  if (header.count_or_size % block_delegate_->shard_factor() != 0) {
    return absl::InvalidArgumentError(
        "Requested remote block count is not divisible by shard_factor");
  }
  PacketHeader resp_header = {};
  resp_header.op = 2;
  resp_header.flags = header.flags;
  resp_header.remote_id = header.local_id;
  resp_header.local_id = 0;
  resp_header.count_or_size = header.count_or_size;

  RETURN_IF_ERROR(WriteExact(client_fd, &resp_header, sizeof(resp_header)));

  size_t local_blocks = header.count_or_size / block_delegate_->shard_factor();
  if (header.remote_id >
          static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      local_blocks > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      local_blocks > static_cast<size_t>(std::numeric_limits<int>::max()) -
                         static_cast<size_t>(header.remote_id)) {
    return absl::OutOfRangeError("Requested block range exceeds int range");
  }

  auto state = std::make_shared<SendStreamState>();
  state->client_fd = client_fd;
  state->uuid = header.uuid;
  state->remote_id = static_cast<int>(header.remote_id);
  state->count_or_size = header.count_or_size;
  state->major_order = major_order;
  state->current_step = 0;
  state->total_steps = block_delegate_->num_block_arrays() *
                       block_delegate_->num_shards() * local_blocks;

  {
    absl::MutexLock lock(active_sends_mu_);
    active_sends_[header.uuid] = state;
  }

  TriggerNextSendStep(state);
  return absl::OkStatus();
}

void BlockTransport::TriggerNextSendStep(
    std::shared_ptr<SendStreamState> state) {
  if (state->current_step >= state->total_steps) {
    {
      absl::MutexLock lock(active_sends_mu_);
      active_sends_.erase(state->uuid);
    }
    return;
  }

  size_t l, sh, k;
  ResolveStepCoordinates(state, &l, &sh, &k);
  int block_id = state->remote_id + k;

  block_delegate_->RegisterBlockReadinessCallback(
      l, sh, block_id, state->uuid,
      [this, state, l, sh, block_id](absl::Status status) {
        if (!status.ok()) {
          LOG(ERROR) << "Pull response failed at step " << state->current_step
                     << " for uuid " << state->uuid
                     << ", status: " << status.ToString();
          shutdown(state->client_fd, SHUT_RDWR);
          close(state->client_fd);
          absl::MutexLock lock(active_sends_mu_);
          active_sends_.erase(state->uuid);
          return;
        }

        block_delegate_->ScheduleAsyncTask([this, state, l, sh, block_id]() {
          const int64_t block_id_val = block_id;
          std::vector<BlockChunk> chunks = block_delegate_->GetBlockChunks(
              l, sh, absl::MakeConstSpan(&block_id_val, 1),
              block_delegate_->block_bytes(l), state->uuid);
          if (chunks.empty()) {
            LOG(ERROR) << "No transfer chunks found for block " << block_id
                       << " and uuid " << state->uuid;
            shutdown(state->client_fd, SHUT_RDWR);
            close(state->client_fd);
            absl::MutexLock lock(active_sends_mu_);
            active_sends_.erase(state->uuid);
            return;
          }
          absl::Status s = ValidateChunks(block_delegate_, l, sh, chunks);
          if (!s.ok()) {
            LOG(ERROR) << "Chunks validation failed: " << s.ToString();
            shutdown(state->client_fd, SHUT_RDWR);
            close(state->client_fd);
            absl::MutexLock lock(active_sends_mu_);
            active_sends_.erase(state->uuid);
            return;
          }

          uint32_t total_size = GetChunksTotalSize(chunks);
          s = WriteExact(state->client_fd, &total_size, sizeof(total_size));
          if (!s.ok()) {
            LOG(ERROR) << "Write size failed: " << s.ToString();
            shutdown(state->client_fd, SHUT_RDWR);
            close(state->client_fd);
            absl::MutexLock lock(active_sends_mu_);
            active_sends_.erase(state->uuid);
            return;
          }
          s = WriteVExact(state->client_fd, ToIovec(chunks));
          if (!s.ok()) {
            LOG(ERROR) << "Write payload failed: " << s.ToString();
            shutdown(state->client_fd, SHUT_RDWR);
            close(state->client_fd);
            absl::MutexLock lock(active_sends_mu_);
            active_sends_.erase(state->uuid);
            return;
          }

          state->current_step++;
          TriggerNextSendStep(state);
        });
      });
}

void BlockTransport::ResolveStepCoordinates(
    const std::shared_ptr<SendStreamState>& state, size_t* layer, size_t* shard,
    size_t* block_idx) {
  size_t L = block_delegate_->num_block_arrays();
  size_t Sh = block_delegate_->num_shards();
  size_t K = state->count_or_size / block_delegate_->shard_factor();
  size_t s = state->current_step;

  if (state->major_order == MajorOrder::kLayerMajor) {
    *block_idx = s % K;
    *shard = (s / K) % Sh;
    *layer = s / (K * Sh);
  } else {
    *shard = s % Sh;
    *layer = (s / Sh) % L;
    *block_idx = s / (Sh * L);
  }
}

uint32_t BlockTransport::GetChunksTotalSize(
    const std::vector<BlockChunk>& chunks) {
  uint32_t total = 0;
  for (const auto& chunk : chunks) {
    total += chunk.size;
  }
  return total;
}

absl::StatusOr<std::vector<int>> BlockTransport::SyncPush(
    const std::vector<std::string>& peers,
    const std::vector<int>& src_block_ids,
    const std::vector<int>& dst_block_ids, int parallelism,
    MajorOrder major_order, uint64_t uuid, int layer_idx) {
  auto promise =
      std::make_shared<std::promise<absl::StatusOr<std::vector<int>>>>();
  auto future = promise->get_future();
  AsyncPush(peers, src_block_ids, dst_block_ids, parallelism, major_order, uuid,
            layer_idx, [promise](absl::StatusOr<std::vector<int>> res) {
              promise->set_value(std::move(res));
            });
  return future.get();
}

void BlockTransport::AsyncPush(
    const std::vector<std::string>& peers,
    const std::vector<int>& src_block_ids,
    const std::vector<int>& dst_block_ids, int parallelism,
    MajorOrder major_order, uint64_t uuid, int layer_idx,
    std::function<void(absl::StatusOr<std::vector<int>>)> on_complete) {
  size_t num_blocks = src_block_ids.size();
  if (num_blocks == 0) {
    on_complete(absl::InvalidArgumentError("Block list cannot be empty"));
    return;
  }
  if (peers.empty()) {
    on_complete(absl::InvalidArgumentError("Peer list cannot be empty"));
    return;
  }

  int P = parallelism;
  if (P <= 0) {
    on_complete(absl::InvalidArgumentError("parallelism must be positive"));
    return;
  }
  if (static_cast<int>(num_blocks) < P) P = num_blocks;

  auto shared_src_block_ids = std::make_shared<std::vector<int>>(src_block_ids);
  auto shared_dst_block_ids = std::make_shared<std::vector<int>>(dst_block_ids);
  auto allocated_ids = std::make_shared<std::vector<int>>(num_blocks, 0);
  auto statuses =
      std::make_shared<std::vector<absl::Status>>(P, absl::OkStatus());
  auto remaining_workers = std::make_shared<std::atomic<int>>(P);

  const size_t base_blocks_per_stream = num_blocks / P;
  const size_t remainder = num_blocks % P;
  size_t block_offset = 0;

  for (int i = 0; i < P; ++i) {
    const size_t block_count =
        base_blocks_per_stream + (static_cast<size_t>(i) < remainder ? 1 : 0);
    std::string local_ip =
        local_ips_.empty() ? "" : local_ips_[i % local_ips_.size()];
    std::string remote_peer = peers[i % peers.size()];

    auto task_run = [this, i, remote_peer, local_ip, block_offset, block_count,
                     shared_src_block_ids, shared_dst_block_ids, allocated_ids,
                     statuses, remaining_workers, major_order, uuid, layer_idx,
                     P, on_complete]() {
      H2hWriteWorker(i, remote_peer, local_ip, block_offset, block_count,
                     *shared_src_block_ids, *shared_dst_block_ids,
                     *allocated_ids, *statuses, major_order, uuid, layer_idx,
                     P);

      if (remaining_workers->fetch_sub(1) == 1) {
        absl::Status final_status = absl::OkStatus();
        for (const auto& s : *statuses) {
          if (!s.ok()) {
            final_status = s;
            break;
          }
        }
        if (!final_status.ok()) {
          on_complete(final_status);
        } else {
          on_complete(*allocated_ids);
        }
      }
    };

    auto task = std::make_unique<WriteTask>();
    task->uuid = uuid;
    task->layer_idx = layer_idx;
    task->stream_idx = i;
    task->peer = remote_peer;
    task->run = std::move(task_run);

    {
      absl::MutexLock lock(scheduler_mu_);
      auto& pq = peer_queues_[task->peer];
      pq.tasks.push_back(std::move(task));
      if (std::find(active_peers_.begin(), active_peers_.end(), remote_peer) ==
          active_peers_.end()) {
        active_peers_.push_back(remote_peer);
      }
    }
    scheduler_cv_.SignalAll();
    block_offset += block_count;
  }
}

absl::StatusOr<std::vector<int>> BlockTransport::SyncPull(
    const std::vector<std::string>& peers,
    const std::vector<int>& src_block_ids,
    const std::vector<int>& local_block_ids,
    const std::vector<uint8_t*>& explicit_dst_ptrs, int parallelism,
    MajorOrder major_order, BlockReceivedCallback on_block_received,
    uint64_t uuid) {
  size_t num_blocks = src_block_ids.size();
  if (num_blocks == 0) {
    return absl::InvalidArgumentError("Block list cannot be empty");
  }
  if (peers.empty()) {
    return absl::InvalidArgumentError("Peer list cannot be empty");
  }

  if (block_delegate_->shard_factor() == 0) {
    return absl::InvalidArgumentError("shard_factor must be positive");
  }
  if (num_blocks % block_delegate_->shard_factor() != 0) {
    return absl::InvalidArgumentError(
        "Block count must be divisible by shard_factor");
  }
  size_t local_blocks = num_blocks / block_delegate_->shard_factor();
  if (local_blocks == 0) {
    return absl::InvalidArgumentError("Local block list cannot be empty");
  }
  if (!explicit_dst_ptrs.empty() &&
      explicit_dst_ptrs.size() !=
          block_delegate_->num_block_arrays() * block_delegate_->num_shards()) {
    return absl::InvalidArgumentError("explicit_dst_ptrs size mismatch");
  }

  std::vector<int> allocated_ids;
  if (!local_block_ids.empty()) {
    if (local_block_ids.size() != local_blocks) {
      return absl::InvalidArgumentError("local_block_ids size mismatch");
    }
    allocated_ids = local_block_ids;
  } else {
    ASSIGN_OR_RETURN(allocated_ids,
                     block_delegate_->AllocateBlocks(local_blocks));
  }

  int P = parallelism;
  if (P <= 0) {
    return absl::InvalidArgumentError("parallelism must be positive");
  }
  if (static_cast<int>(local_blocks) < P) P = local_blocks;

  std::vector<std::thread> threads;
  std::vector<absl::Status> statuses(P, absl::OkStatus());

  threads.reserve(P);
  const size_t base_blocks_per_stream = local_blocks / P;
  const size_t remainder = local_blocks % P;
  size_t local_block_offset = 0;
  size_t remote_block_offset = 0;

  for (int i = 0; i < P; ++i) {
    const size_t local_block_count =
        base_blocks_per_stream + (static_cast<size_t>(i) < remainder ? 1 : 0);
    const size_t remote_block_count =
        local_block_count * block_delegate_->shard_factor();

    std::string local_ip =
        local_ips_.empty() ? "" : local_ips_[i % local_ips_.size()];
    std::string remote_peer = peers[i % peers.size()];

    threads.push_back(std::thread(
        &BlockTransport::H2hReadWorker, this, i, remote_peer, local_ip,
        local_block_offset, local_block_count, remote_block_offset,
        remote_block_count, std::ref(src_block_ids), std::ref(allocated_ids),
        std::ref(explicit_dst_ptrs), std::ref(statuses), major_order,
        on_block_received, uuid));

    local_block_offset += local_block_count;
    remote_block_offset += remote_block_count;
  }

  for (auto& t : threads) {
    if (t.joinable()) t.join();
  }

  for (int i = 0; i < P; ++i) {
    if (!statuses[i].ok()) return statuses[i];
  }

  return allocated_ids;
}

void BlockTransport::H2hWriteWorker(int stream_idx, absl::string_view peer,
                                    absl::string_view local_ip,
                                    size_t block_offset, size_t block_count,
                                    const std::vector<int>& src_block_ids,
                                    const std::vector<int>& dst_block_ids,
                                    std::vector<int>& allocated_ids,
                                    std::vector<absl::Status>& statuses,
                                    MajorOrder major_order, uint64_t uuid,
                                    int layer_idx, int parallelism) {
  auto status_or_fd = BorrowConnection(peer, local_ip);
  if (!status_or_fd.ok()) {
    statuses[stream_idx] = status_or_fd.status();
    return;
  }

  const int fd = status_or_fd.value();
  bool ok_to_pool = false;
  auto fd_cleaner = absl::MakeCleanup(
      [&] { ReturnConnection(ok_to_pool, fd, peer, local_ip); });

  PacketHeader header = {};
  header.op = dst_block_ids.empty() ? 1 : 6;
  header.flags = static_cast<uint8_t>(major_order);
  header.buffer_id = 0;
  header.remote_id = static_cast<uint32_t>(block_delegate_->node_id());
  header.local_id =
      (layer_idx == -1) ? 0xFFFFFFFF : static_cast<uint32_t>(layer_idx);
  header.uuid = uuid;
  header.reserved = static_cast<uint16_t>(parallelism);
  if (block_count > std::numeric_limits<uint32_t>::max()) {
    statuses[stream_idx] = absl::OutOfRangeError("Block count exceeds uint32");
    return;
  }
  header.count_or_size = static_cast<uint32_t>(block_count);

  absl::Status s = WriteExact(fd, &header, sizeof(header));
  if (!s.ok()) {
    statuses[stream_idx] = s;
    return;
  }

  if (header.op == 6) {
    ABSL_DCHECK_LE(block_offset + block_count, dst_block_ids.size());
    s = WriteExact(fd, &dst_block_ids[block_offset], block_count * sizeof(int));
    if (!s.ok()) {
      statuses[stream_idx] = s;
      return;
    }
    s = WriteExact(fd, &src_block_ids[block_offset], block_count * sizeof(int));
    if (!s.ok()) {
      statuses[stream_idx] = s;
      return;
    }
    uint8_t ack = 0;
    s = ReadExact(fd, &ack, 1);
    if (!s.ok() || ack != 1) {
      statuses[stream_idx] =
          absl::InternalError("Explicit push destination handshake failed");
      return;
    }
    for (size_t k = 0; k < block_count; ++k) {
      allocated_ids[block_offset + k] = dst_block_ids[block_offset + k];
    }
  } else {
    std::vector<int> stream_allocated_ids(block_count, 0);
    s = ReadExact(fd, stream_allocated_ids.data(), block_count * sizeof(int));
    if (!s.ok()) {
      statuses[stream_idx] = s;
      return;
    }

    for (size_t k = 0; k < block_count; ++k) {
      ABSL_DCHECK_LT(block_offset + k, allocated_ids.size());
      ABSL_DCHECK_LT(k, stream_allocated_ids.size());
      allocated_ids[block_offset + k] = stream_allocated_ids[k];
    }
  }

  std::vector<int> target_layers;
  if (layer_idx == -1) {
    target_layers.resize(block_delegate_->num_block_arrays());
    std::iota(target_layers.begin(), target_layers.end(), 0);
  } else {
    target_layers = {layer_idx};
  }

  s = ForEachPayload(
      major_order, target_layers, block_delegate_->num_shards(), block_count,
      [&](size_t l, size_t sh, size_t k) -> absl::Status {
        ABSL_DCHECK_LT(block_offset + k, src_block_ids.size());
        const int src_id = src_block_ids[block_offset + k];

        const int64_t block_id_val = src_id;
        const int64_t dst_id_val =
            block_offset + k < dst_block_ids.size()
                ? static_cast<int64_t>(dst_block_ids[block_offset + k])
                : -1;
        std::vector<BlockChunk> chunks = block_delegate_->GetBlockChunks(
            l, sh, absl::MakeConstSpan(&block_id_val, 1),
            block_delegate_->block_bytes(l), uuid, -1, peer,
            /*src_block_id=*/-1, /*dst_block_id=*/dst_id_val);
        if (chunks.empty()) {
          return absl::NotFoundError(
              absl::StrCat("No transfer chunks found for block ", src_id,
                           " and uuid ", uuid));
        }
        RETURN_IF_ERROR(ValidateChunks(block_delegate_, l, sh, chunks));

        uint32_t total_size = 0;
        for (const auto& chunk : chunks) {
          total_size += chunk.size;
        }

        RETURN_IF_ERROR(WriteExact(fd, &total_size, sizeof(total_size)));
        return WriteVExact(fd, ToIovec(chunks));
      });
  if (!s.ok()) {
    statuses[stream_idx] = s;
    return;
  }

  uint8_t ack = 0;
  s = ReadExact(fd, &ack, 1);
  if (!s.ok() || ack != 1) {
    statuses[stream_idx] = absl::InternalError("Push verification failed");
    return;
  }

  ok_to_pool = true;
}

void BlockTransport::ForgetPushProgress(uint64_t uuid) {
  absl::MutexLock lock(progress_mu_);
  for (auto it = layer_progress_.begin(); it != layer_progress_.end();) {
    if (it->first.first == uuid) {
      // absl::flat_hash_map::erase(iterator) returns void.
      auto erase_it = it++;
      layer_progress_.erase(erase_it);
    } else {
      ++it;
    }
  }
}

void BlockTransport::H2hReadWorker(
    int stream_idx, absl::string_view peer, absl::string_view local_ip,
    size_t local_block_offset, size_t local_block_count,
    size_t remote_block_offset, size_t remote_block_count,
    const std::vector<int>& src_block_ids,
    const std::vector<int>& allocated_ids,
    const std::vector<uint8_t*>& explicit_dst_ptrs,
    std::vector<absl::Status>& statuses, MajorOrder major_order,
    BlockReceivedCallback on_block_received, uint64_t uuid) {
  auto status_or_fd = BorrowConnection(peer, local_ip);
  if (!status_or_fd.ok()) {
    statuses[stream_idx] = status_or_fd.status();
    return;
  }

  const int fd = status_or_fd.value();
  bool ok_to_pool = false;
  auto fd_cleaner = absl::MakeCleanup(
      [&] { ReturnConnection(ok_to_pool, fd, peer, local_ip); });

  size_t SF = block_delegate_->shard_factor();

  if (remote_block_offset > src_block_ids.size() ||
      remote_block_count > src_block_ids.size() - remote_block_offset) {
    statuses[stream_idx] =
        absl::OutOfRangeError("Remote block range exceeds source block list");
    return;
  }

  struct PullChunk {
    size_t local_start_idx;
    size_t local_count;
    int base_remote_id;
    size_t remote_count;
  };
  std::vector<PullChunk> chunks;

  if (local_block_count > 0) {
    size_t curr_local_start = 0;
    size_t curr_local_count = 1;
    ABSL_DCHECK_LT(remote_block_offset, src_block_ids.size());
    int curr_base_remote_id = src_block_ids[remote_block_offset];

    for (size_t k = 1; k < local_block_count; ++k) {
      ABSL_DCHECK_LT(remote_block_offset + k * SF - 1, src_block_ids.size());
      int prev_last_remote = src_block_ids[remote_block_offset + k * SF - 1];
      ABSL_DCHECK_LT(remote_block_offset + k * SF, src_block_ids.size());
      int curr_first_remote = src_block_ids[remote_block_offset + k * SF];

      if (curr_first_remote == prev_last_remote + 1) {
        curr_local_count++;
      } else {
        chunks.push_back({curr_local_start, curr_local_count,
                          curr_base_remote_id, curr_local_count * SF});
        curr_local_start = k;
        curr_local_count = 1;
        curr_base_remote_id = curr_first_remote;
      }
    }
    chunks.push_back({curr_local_start, curr_local_count, curr_base_remote_id,
                      curr_local_count * SF});
  }

  for (const auto& chunk : chunks) {
    PacketHeader header = {};
    header.op = 2;  // Pull request
    header.flags = static_cast<uint8_t>(major_order);
    int remote_read_block_id =
        block_delegate_->GetRemoteReadBlockId(chunk.base_remote_id, 0);
    if (remote_read_block_id < 0 ||
        static_cast<uint64_t>(remote_read_block_id) >
            std::numeric_limits<uint32_t>::max() ||
        chunk.remote_count > std::numeric_limits<uint32_t>::max()) {
      statuses[stream_idx] =
          absl::OutOfRangeError("Remote block range exceeds transport header");
      return;
    }
    header.remote_id = static_cast<uint32_t>(remote_read_block_id);
    header.count_or_size = static_cast<uint32_t>(chunk.remote_count);
    header.uuid = uuid;

    absl::Status s = WriteExact(fd, &header, sizeof(header));
    if (!s.ok()) {
      statuses[stream_idx] = s;
      return;
    }

    PacketHeader resp_header = {};
    s = ReadExact(fd, &resp_header, sizeof(resp_header));
    if (!s.ok()) {
      statuses[stream_idx] = s;
      return;
    }
    if (resp_header.op != 2 ||
        resp_header.count_or_size != chunk.remote_count) {
      statuses[stream_idx] =
          absl::InternalError("Unexpected block pull response header");
      return;
    }
    if (resp_header.flags != static_cast<uint8_t>(major_order)) {
      statuses[stream_idx] =
          absl::InternalError("Unexpected block pull response major order");
      return;
    }

    std::vector<int> target_layers(block_delegate_->num_block_arrays());
    std::iota(target_layers.begin(), target_layers.end(), 0);
    s = ForEachPayload(
        major_order, target_layers, block_delegate_->num_shards(),
        chunk.local_count, [&](size_t l, size_t sh, size_t k) -> absl::Status {
          uint8_t* base_host_ptr = nullptr;
          if (!explicit_dst_ptrs.empty()) {
            base_host_ptr =
                explicit_dst_ptrs[l * block_delegate_->num_shards() + sh];
          } else {
            base_host_ptr = block_delegate_->GetBlockArrayHostPointer(l, sh);
          }
          if (base_host_ptr == nullptr) {
            return absl::FailedPreconditionError(
                "Destination host pointer is null");
          }

          ABSL_DCHECK_LT(local_block_offset + chunk.local_start_idx + k,
                         allocated_ids.size());
          const int dst_id =
              allocated_ids[local_block_offset + chunk.local_start_idx + k];
          if (dst_id < 0) {
            return absl::InvalidArgumentError(
                "Destination block id is negative");
          }

          const int64_t block_id_val = dst_id;
          std::vector<BlockChunk> chunks = block_delegate_->GetBlockChunks(
              l, sh, absl::MakeConstSpan(&block_id_val, 1),
              block_delegate_->block_bytes(l), uuid);
          if (chunks.empty()) {
            return absl::NotFoundError(
                absl::StrCat("No transfer chunks found for block ", dst_id,
                             " and uuid ", uuid));
          }
          RETURN_IF_ERROR(ValidateChunks(block_delegate_, l, sh, chunks));

          if (!explicit_dst_ptrs.empty()) {
            uint8_t* default_base =
                block_delegate_->GetBlockArrayHostPointer(l, sh);
            if (default_base == nullptr) {
              return absl::FailedPreconditionError(
                  "explicit destination pointers require a flat block array "
                  "base");
            }
            uint8_t* explicit_base = base_host_ptr;
            for (auto& chunk : chunks) {
              size_t offset = chunk.ptr - default_base;
              chunk.ptr = explicit_base + offset;
            }
          }

          uint32_t expected_size = 0;
          for (const auto& chunk : chunks) {
            expected_size += chunk.size;
          }

          uint32_t sender_size = 0;
          RETURN_IF_ERROR(ReadExact(fd, &sender_size, sizeof(sender_size)));

          if (sender_size != expected_size) {
            return absl::InternalError(absl::StrCat(
                "Block transfer size mismatch! Sender offered: ", sender_size,
                " bytes, but Receiver expected: ", expected_size,
                " bytes for Block ID: ", dst_id));
          }

          RETURN_IF_ERROR(ReadVExact(fd, ToIovec(chunks)));

          if (on_block_received != nullptr) {
            RETURN_IF_ERROR(on_block_received(l, sh, dst_id, expected_size));
          }
          return absl::OkStatus();
        });
    if (!s.ok()) {
      statuses[stream_idx] = s;
      return;
    }
  }
  ok_to_pool = true;
}

}  // namespace transport
}  // namespace tpu_raiden
