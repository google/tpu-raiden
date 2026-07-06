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

#include "tpu_raiden/kv_cache/kv_cache_store.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "grpcpp/grpcpp.h"
#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/security/credentials.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_client.h"
#include "tpu_raiden/kv_cache/lru_cache.h"
#include "tpu_raiden/kv_cache/raiden_controller_embedded.h"

namespace tpu_raiden {
namespace kv_cache {

namespace {
RaidenId FromProto(const ::tpu_raiden::rpc::RaidenIdProto& proto) {
  return RaidenId{
      .job_name = proto.job_name(),
      .job_replica_id = proto.job_replica_id(),
      .data_name = proto.data_name(),
      .data_replica_idx = proto.data_replica_idx(),
  };
}
}  // namespace

KVCacheStore::KVCacheStore(size_t capacity, std::string global_registry_address,
                           RaidenId raiden_id,
                           std::optional<RemoteFetchConfig> remote_config)
    : lru_cache_(capacity),
      raiden_id_(std::move(raiden_id)),
      config_(remote_config) {
  if (!global_registry_address.empty()) {
    auto channel = grpc::CreateChannel(global_registry_address,
                                       grpc::InsecureChannelCredentials());
    registry_client_ =
        std::make_shared<global_registry::GlobalRegistryClient>(channel);
  }

  if (remote_config.has_value()) {
    controller_ = std::make_unique<RaidenControllerEmbedded>(
        this, remote_config->controller_port,
        remote_config->orchestrator_address, remote_config->local_worker_port,
        std::vector<std::string>{}, remote_config->bytes_per_block,
        remote_config->num_shards, remote_config->num_listeners);

    absl::Status status = controller_->Start();
    if (!status.ok()) {
      LOG(ERROR) << "Failed to start RaidenControllerEmbedded: " << status;
    }

    completion_poller_thread_ =
        std::thread(&KVCacheStore::CompletionPollerLoop, this);
    load_completion_poller_thread_ =
        std::thread(&KVCacheStore::LoadCompletionPollerLoop, this);
    save_completion_poller_thread_ =
        std::thread(&KVCacheStore::SaveCompletionPollerLoop, this);
    evict_completion_poller_thread_ =
        std::thread(&KVCacheStore::EvictCompletionPollerLoop, this);
  }
}

KVCacheStore::~KVCacheStore() {
  if (controller_) {
    controller_->Stop();
  }
  completion_queue_.Stop();
  load_completion_queue_.Stop();
  save_completion_queue_.Stop();
  evict_completion_queue_.Stop();
  if (completion_poller_thread_.joinable()) {
    completion_poller_thread_.join();
  }
  if (load_completion_poller_thread_.joinable()) {
    load_completion_poller_thread_.join();
  }
  if (save_completion_poller_thread_.joinable()) {
    save_completion_poller_thread_.join();
  }
  if (evict_completion_poller_thread_.joinable()) {
    evict_completion_poller_thread_.join();
  }
}

absl::StatusOr<BlockSliceList> KVCacheStore::Lookup(
    const std::vector<std::string>& block_hashes, bool enable_global) {
  BlockSliceList results;

  size_t local_hits = 0;
  size_t limit = 0;
  {
    absl::MutexLock lock(mutex_);
    limit = std::min(block_hashes.size(), lru_cache_.available_space());
    results.reserve(limit);
    for (size_t i = 0; i < limit; ++i) {
      const std::string& hash = block_hashes[i];
      RaidenBlockID* peeked = lru_cache_.Peek(hash);
      if (!peeked) {
        break;
      }
      if (peeked->status == BlockStatus::HOST ||
          peeked->status == BlockStatus::HOST_AND_HBM) {
        RaidenBlockID* existing = lru_cache_.Get(hash);
        results.push_back(std::make_pair(hash, *existing));
      } else {
        results.push_back(std::make_pair(hash, *peeked));
      }
      local_hits++;
    }
  }

  if (enable_global && local_hits < limit && registry_client_) {
    std::vector<std::string> remaining_hashes(block_hashes.begin() + local_hits,
                                              block_hashes.begin() + limit);
    auto global_results_or = registry_client_->Lookup(remaining_hashes);
    if (global_results_or.ok()) {
      const auto& global_results = global_results_or.value();
      for (size_t i = 0; i < global_results.size(); ++i) {
        const auto& metadata = global_results[i];
        RaidenId remote_id = FromProto(metadata.raiden_id());
        results.push_back(
            std::make_pair(remaining_hashes[i],
                           RaidenBlockID(remote_id, -1, BlockStatus::REMOTE)));
      }
    } else {
      LOG(WARNING) << "Global registry lookup failed: "
                   << global_results_or.status().message();
    }
  }

  return results;
}

std::pair<bool, BlockSliceList> KVCacheStore::Insert(
    const std::vector<std::string>& block_hashes,
    const std::vector<RaidenBlockID>& slices, bool /*on_host*/) {
  ABSL_CHECK_EQ(block_hashes.size(), slices.size());
  absl::MutexLock lock(mutex_);
  BlockSliceList evicted_entries;
  bool all_inserted = true;

  for (size_t i = 0; i < block_hashes.size(); ++i) {
    const std::string& hash = block_hashes[i];
    if (lru_cache_.Contains(hash)) {
      // NOTE(jcgu): This is technically true, as the key already exists in the
      // cache. However, for the purpose of this insert, we treat it as if it
      // was inserted.
      all_inserted = false;
      continue;
    }
    std::optional<std::pair<std::string, RaidenBlockID>> evicted;
    evicted = lru_cache_.Put(hash, slices[i]);
    if (evicted.has_value()) {
      evicted_entries.push_back(std::move(*evicted));
    }
  }

  return std::make_pair(all_inserted, std::move(evicted_entries));
}

// Categorizes block_hashes into existing (local) and new (remote) hashes.
// 1. Pins all existing block hashes.
// 2. Inserts new block hashes into the LRU cache if space permits.
// 3. Pins the newly inserted block hashes, with full rollback on failure.
std::pair<bool, BlockSliceList> KVCacheStore::InsertAndPin(
    const std::vector<std::string>& block_hashes,
    const std::vector<RaidenBlockID>& slices, bool /*on_host*/) {
  ABSL_CHECK_EQ(block_hashes.size(), slices.size());
  absl::MutexLock lock(mutex_);
  BlockSliceList evicted_entries;

  std::vector<size_t> existing_indices;
  std::vector<size_t> new_indices;
  for (size_t i = 0; i < block_hashes.size(); ++i) {
    if (lru_cache_.Contains(block_hashes[i])) {
      existing_indices.push_back(i);
    } else {
      new_indices.push_back(i);
    }
  }

  // 1. Pin all existing block_hashes
  for (size_t idx = 0; idx < existing_indices.size(); ++idx) {
    size_t i = existing_indices[idx];
    if (!lru_cache_.Pin(block_hashes[i])) {
      for (size_t j = 0; j < idx; ++j) {
        lru_cache_.Unpin(block_hashes[existing_indices[j]]);
      }
      return std::make_pair(false, std::move(evicted_entries));
    }
  }

  // 2. Check if free space in lru_cache can hold all new block_hashes
  if (lru_cache_.available_space() < new_indices.size()) {
    for (size_t i : existing_indices) {
      lru_cache_.Unpin(block_hashes[i]);
    }
    return std::make_pair(false, std::move(evicted_entries));
  }

  // Insert all new block hashes into the lru cache list
  for (size_t i : new_indices) {
    const std::string& hash = block_hashes[i];
    std::optional<std::pair<std::string, RaidenBlockID>> evicted;
    evicted = lru_cache_.Put(hash, slices[i]);
    if (evicted.has_value()) {
      evicted_entries.push_back(std::move(*evicted));
    }
  }

  // 3. Pin the newly inserted block_hashes in the lru cache list
  for (size_t idx = 0; idx < new_indices.size(); ++idx) {
    size_t i = new_indices[idx];
    if (!lru_cache_.Pin(block_hashes[i])) {
      for (size_t j = 0; j < idx; ++j) {
        lru_cache_.Unpin(block_hashes[new_indices[j]]);
      }
      for (size_t j : existing_indices) {
        lru_cache_.Unpin(block_hashes[j]);
      }
      for (size_t j : new_indices) {
        lru_cache_.Erase(block_hashes[j]);
      }
      for (auto it = evicted_entries.rbegin(); it != evicted_entries.rend();
           ++it) {
        lru_cache_.PutBack(it->first, std::move(it->second));
      }
      return std::make_pair(false, BlockSliceList{});
    }
  }

  return std::make_pair(true, std::move(evicted_entries));
}

// Reverts an InsertAndPin operation.
// 1. Unpins all block hashes in the LRU cache.
// 2. Deletes non-local block hashes whose pin count reaches 0.
// 3. Restores evicted entries to the back of the LRU cache in reverse order for
//    each deleted non-local block. Returns the number of deleted non-local
//    blocks and the remaining unrestored evicted entries.
std::pair<size_t, BlockSliceList> KVCacheStore::ReleaseAndDelete(
    const std::vector<std::string>& block_hashes,
    BlockSliceList pending_evict_entries) {
  absl::MutexLock lock(mutex_);
  size_t deleted_non_local_blocks = 0;
  for (const std::string& hash : block_hashes) {
    lru_cache_.Unpin(hash);
    auto* val = lru_cache_.Peek(hash);
    if (val != nullptr && val->status != BlockStatus::HOST &&
        val->status != BlockStatus::HOST_AND_HBM &&
        lru_cache_.GetPinCount(hash) == 0) {
      lru_cache_.Erase(hash);
      deleted_non_local_blocks++;
    }
  }

  if (deleted_non_local_blocks > pending_evict_entries.size()) {
    LOG(WARNING) << "Number of deleted non-local blocks ("
                 << deleted_non_local_blocks
                 << ") exceeds number of pending evict entries ("
                 << pending_evict_entries.size() << ").";
  }

  size_t to_restore =
      std::min(deleted_non_local_blocks, pending_evict_entries.size());
  for (size_t i = 0; i < to_restore; ++i) {
    auto& entry = pending_evict_entries.back();
    lru_cache_.PutBack(entry.first, std::move(entry.second));
    pending_evict_entries.pop_back();
  }

  return std::make_pair(deleted_non_local_blocks,
                        std::move(pending_evict_entries));
}

void KVCacheStore::Delete(const std::vector<std::string>& block_hashes,
                          const std::vector<RaidenBlockID>& slices) {
  absl::MutexLock lock(mutex_);
  for (const std::string& hash : block_hashes) {
    lru_cache_.Erase(hash);
  }
}

bool KVCacheStore::Pin(const std::vector<std::string>& block_hashes) {
  absl::MutexLock lock(mutex_);
  for (size_t i = 0; i < block_hashes.size(); ++i) {
    if (!lru_cache_.Pin(block_hashes[i])) {
      for (size_t j = 0; j < i; ++j) {
        lru_cache_.Unpin(block_hashes[j]);
      }
      return false;
    }
  }
  return true;
}

void KVCacheStore::Release(const std::vector<std::string>& block_hashes) {
  absl::MutexLock lock(mutex_);
  for (const std::string& hash : block_hashes) {
    lru_cache_.Unpin(hash);
  }
}

int KVCacheStore::GetPinCount(const std::string& hash) const {
  absl::MutexLock lock(mutex_);
  return lru_cache_.GetPinCount(hash);
}

size_t KVCacheStore::capacity() const {
  absl::MutexLock lock(mutex_);
  return lru_cache_.capacity();
}

absl::flat_hash_map<std::string, KVCacheStore::FetchFuture>
KVCacheStore::FetchRemote(const std::vector<std::string>& block_hashes) {
  absl::flat_hash_map<std::string, FetchFuture> results;
  absl::flat_hash_map<RaidenId, FetchRequestItem, RaidenIdHash> grouped_reqs;

  {
    absl::MutexLock lock(mutex_);
    for (const auto& hash : block_hashes) {
      RaidenBlockID* block = lru_cache_.Get(hash);
      if (block) {
        if (block->status == BlockStatus::REMOTE) {
          auto& item = grouped_reqs[block->raiden_id];
          item.src_raiden_id = block->raiden_id;
          item.block_hashes.push_back(hash);
          item.dst_block_ids.push_back(block->host_block_id);
        }
      }
    }
  }

  FetchRequest full_request;
  for (auto& [raiden_id, item] : grouped_reqs) {
    for (size_t i = 0; i < item.block_hashes.size(); ++i) {
      const auto& hash = item.block_hashes[i];
      int dst_block_id = item.dst_block_ids[i];

      std::shared_ptr<FetchState> state;
      {
        absl::MutexLock lock(fetch_mu_);
        uint64_t fetch_id = next_fetch_id_++;
        state = std::make_shared<FetchState>();
        state->fetch_id = fetch_id;
        state->block_hash = hash;
        state->pending_blocks.insert(dst_block_id);
        block_to_fetch_[dst_block_id] = fetch_id;
        active_fetches_[fetch_id] = state;
      }
      results[hash] = FetchFuture(state);
    }
    full_request.push_back(std::move(item));
  }

  if (!full_request.empty()) {
    PushFetchWork(std::move(full_request));
  }

  return results;
}

void KVCacheStore::CompletionPollerLoop() {
  FetchCompletion completion;
  while (PopFetchCompletion(completion)) {
    std::vector<global_registry::Registration> write_through_regs;
    {
      absl::MutexLock lock(mutex_);
      for (const auto& item : completion) {
        if (item.success) {
          RaidenBlockID* block = lru_cache_.Get(item.block_hash);
          if (block) {
            if (block->status == BlockStatus::REMOTE &&
                block->host_block_id == item.host_block_id) {
              block->status = BlockStatus::HOST;
              block->raiden_id = raiden_id_;
              LOG(INFO) << "Upgraded block " << item.block_hash << " to HOST";
              if (registry_client_) {
                write_through_regs.push_back({
                    .prefix_hash = item.block_hash,
                    .raiden_id = raiden_id_,
                    .block_id = item.host_block_id,
                });
              }
            }
          }
        } else {
          LOG(ERROR) << "Fetch failed for " << item.block_hash << ": "
                     << item.error_message;
        }
      }
    }

    if (!write_through_regs.empty() && registry_client_) {
      std::thread([client = registry_client_,
                   regs = std::move(write_through_regs)]() {
        auto status = client->Register(regs);
        if (!status.ok()) {
          LOG(WARNING) << "Async batch write-through failed: "
                       << status.message();
        } else {
          LOG(INFO) << "Async batch write-through succeeded for " << regs.size()
                    << " blocks";
        }
      }).detach();
    }

    {
      absl::MutexLock lock(fetch_mu_);
      for (const auto& item : completion) {
        auto it = block_to_fetch_.find(item.host_block_id);
        if (it != block_to_fetch_.end()) {
          uint64_t fetch_id = it->second;
          auto state_it = active_fetches_.find(fetch_id);
          if (state_it != active_fetches_.end()) {
            auto& state = state_it->second;
            state->pending_blocks.erase(item.host_block_id);
            if (!item.success) {
              state->failed = true;
              state->error_message += item.error_message + ";";
            }

            if (state->pending_blocks.empty()) {
              if (state->failed) {
                failed_fetches_.push_back(state->block_hash);
              } else {
                done_fetches_.push_back(state->block_hash);
              }
              state->notification.Notify();
              active_fetches_.erase(state_it);
            }
          }
          block_to_fetch_.erase(it);
        }
      }
    }
  }
}

absl::Status KVCacheStore::FetchFuture::Await() {
  if (!state_) {
    return absl::InvalidArgumentError("FetchFuture has no valid state.");
  }
  state_->notification.WaitForNotification();
  if (state_->failed) {
    return absl::InternalError(state_->error_message);
  }
  return absl::OkStatus();
}

bool KVCacheStore::FetchFuture::IsDone() const {
  if (!state_) return true;
  return state_->notification.HasBeenNotified();
}

std::tuple<std::vector<std::string>, std::vector<std::string>,
           std::vector<std::string>>
KVCacheStore::PollFetchRemoteStatus() {
  absl::MutexLock lock(fetch_mu_);
  std::vector<std::string> pending;
  pending.reserve(active_fetches_.size());
  for (const auto& [id, state] : active_fetches_) {
    pending.push_back(state->block_hash);
  }

  std::vector<std::string> done = std::move(done_fetches_);
  std::vector<std::string> failed = std::move(failed_fetches_);

  done_fetches_.clear();
  failed_fetches_.clear();

  return {std::move(done), std::move(failed), std::move(pending)};
}

}  // namespace kv_cache
}  // namespace tpu_raiden

namespace tpu_raiden {
namespace kv_cache {

absl::Status KVCacheStore::LoadFuture::Await() {
  if (!state_) {
    return absl::InvalidArgumentError("LoadFuture has no valid state.");
  }
  state_->notification.WaitForNotification();
  if (state_->failed) {
    return absl::InternalError(state_->error_message);
  }
  return absl::OkStatus();
}

bool KVCacheStore::LoadFuture::IsDone() const {
  if (!state_) return true;
  return state_->notification.HasBeenNotified();
}

absl::flat_hash_map<std::string, KVCacheStore::LoadFuture> KVCacheStore::Load(
    const std::vector<std::string>& block_hashes,
    const std::vector<int>& dst_hbm_block_ids) {
  absl::flat_hash_map<std::string, LoadFuture> results;
  uint64_t load_id = 0;
  {
    absl::MutexLock lock(&load_mu_);
    load_id = next_load_id_++;
  }
  LoadRequest full_request;
  full_request.load_id = load_id;

  std::vector<std::shared_ptr<LoadState>> created_states;

  {
    absl::MutexLock lock(&mutex_);
    for (size_t i = 0; i < block_hashes.size(); ++i) {
      const auto& hash = block_hashes[i];
      if (i >= dst_hbm_block_ids.size()) {
        auto state = std::make_shared<LoadState>();
        state->load_id = load_id;
        state->block_hash = hash;
        state->failed = true;
        state->error_message = "Missing destination HBM block ID";
        state->notification.Notify();
        results[hash] = LoadFuture(state);
        continue;
      }
      int dst_hbm_id = dst_hbm_block_ids[i];

      RaidenBlockID* block = lru_cache_.Get(hash);
      if (!block) {
        auto state = std::make_shared<LoadState>();
        state->load_id = load_id;
        state->block_hash = hash;
        state->failed = true;
        state->error_message = "Block hash not found in local directory";
        state->notification.Notify();
        results[hash] = LoadFuture(state);
        continue;
      }

      if (block->status != BlockStatus::HOST &&
          block->status != BlockStatus::HOST_AND_HBM) {
        auto state = std::make_shared<LoadState>();
        state->load_id = load_id;
        state->block_hash = hash;
        state->failed = true;
        state->error_message =
            absl::StrCat("Block is not on HOST. Current status: ",
                         static_cast<int>(block->status));
        state->notification.Notify();
        results[hash] = LoadFuture(state);
        continue;
      }

      auto state = std::make_shared<LoadState>();
      state->load_id = load_id;
      state->block_hash = hash;
      results[hash] = LoadFuture(state);
      created_states.push_back(state);

      full_request.block_hashes.push_back(hash);
      full_request.src_block_ids.push_back(block->host_block_id);
      full_request.dst_block_ids.push_back(dst_hbm_id);
    }
  }

  if (full_request.block_hashes.empty()) {
    return results;
  }

  {
    absl::MutexLock lock(&load_mu_);
    auto& load_map = active_loads_[load_id];
    for (const auto& state : created_states) {
      load_map[state->block_hash] = state;
    }
  }

  PushLoadWork(std::move(full_request));
  return results;
}

std::tuple<std::vector<std::string>, std::vector<std::string>,
           std::vector<std::string>>
KVCacheStore::PollLoadStatus() {
  absl::MutexLock lock(&load_mu_);
  std::vector<std::string> pending;
  for (const auto& [id, load_map] : active_loads_) {
    for (const auto& [hash, state] : load_map) {
      pending.push_back(hash);
    }
  }

  std::vector<std::string> done = std::move(done_loads_);
  std::vector<std::string> failed = std::move(failed_loads_);

  done_loads_.clear();
  failed_loads_.clear();

  return {std::move(done), std::move(failed), std::move(pending)};
}

void KVCacheStore::LoadCompletionPollerLoop() {
  LoadCompletion completion;
  while (PopLoadCompletion(completion)) {
    uint64_t load_id = completion.load_id;
    {
      absl::MutexLock lock(&mutex_);
      for (const auto& item : completion.items) {
        if (item.success) {
          RaidenBlockID* block = lru_cache_.Get(item.block_hash);
          if (block) {
            if (block->status == BlockStatus::HOST) {
              block->status = BlockStatus::HOST_AND_HBM;
              block->hbm_block_id = item.dst_hbm_block_id;
              LOG(INFO) << "Upgraded block " << item.block_hash
                        << " to HOST_AND_HBM (Host ID: " << block->host_block_id
                        << ", HBM ID: " << item.dst_hbm_block_id << ")";
            }
          }
        } else {
          LOG(ERROR) << "Load failed for " << item.block_hash << ": "
                     << item.error_message;
        }
      }
    }

    {
      absl::MutexLock lock(&load_mu_);
      auto load_it = active_loads_.find(load_id);
      if (load_it != active_loads_.end()) {
        auto& load_map = load_it->second;
        for (const auto& item : completion.items) {
          auto state_it = load_map.find(item.block_hash);
          if (state_it != load_map.end()) {
            auto& state = state_it->second;
            if (!item.success) {
              state->failed = true;
              state->error_message += item.error_message + ";";
            }
            if (state->failed) {
              failed_loads_.push_back(state->block_hash);
            } else {
              done_loads_.push_back(state->block_hash);
            }
            state->notification.Notify();
            load_map.erase(state_it);
          }
        }
        if (load_map.empty()) {
          active_loads_.erase(load_it);
        }
      }
    }
  }
}

absl::Status KVCacheStore::SaveFuture::Await() {
  if (!state_) {
    return absl::InvalidArgumentError("SaveFuture has no valid state.");
  }
  state_->notification.WaitForNotification();
  if (state_->failed) {
    return absl::InternalError(state_->error_message);
  }
  return absl::OkStatus();
}

bool KVCacheStore::SaveFuture::IsDone() const {
  if (!state_) return true;
  return state_->notification.HasBeenNotified();
}

absl::flat_hash_map<std::string, KVCacheStore::SaveFuture> KVCacheStore::Save(
    const std::vector<std::string>& block_hashes,
    const std::vector<int>& src_hbm_block_ids) {
  ABSL_CHECK_EQ(block_hashes.size(), src_hbm_block_ids.size());

  absl::flat_hash_map<std::string, SaveFuture> results;
  std::vector<std::shared_ptr<SaveState>> created_states;
  SaveRequest full_request;

  uint64_t save_id;
  {
    absl::MutexLock lock(&save_mu_);
    save_id = next_save_id_++;
  }
  full_request.save_id = save_id;

  {
    absl::MutexLock lock(mutex_);
    for (size_t i = 0; i < block_hashes.size(); ++i) {
      const auto& hash = block_hashes[i];
      int src_hbm_id = src_hbm_block_ids[i];

      RaidenBlockID* block = lru_cache_.Peek(hash);
      if (block == nullptr) {
        auto state = std::make_shared<SaveState>();
        state->save_id = save_id;
        state->block_hash = hash;
        state->failed = true;
        state->error_message = "Block not found in cache";
        state->notification.Notify();
        results[hash] = SaveFuture(state);
        continue;
      }

      if (block->status != BlockStatus::HBM) {
        auto state = std::make_shared<SaveState>();
        state->save_id = save_id;
        state->block_hash = hash;
        state->failed = true;
        state->error_message = absl::StrCat("Block status is not HBM: ",
                                            static_cast<int>(block->status));
        state->notification.Notify();
        results[hash] = SaveFuture(state);
        continue;
      }

      if (lru_cache_.GetPinCount(hash) == 0) {
        auto state = std::make_shared<SaveState>();
        state->save_id = save_id;
        state->block_hash = hash;
        state->failed = true;
        state->error_message = "Block is not pinned";
        state->notification.Notify();
        results[hash] = SaveFuture(state);
        continue;
      }

      if (block->hbm_block_id != src_hbm_id) {
        auto state = std::make_shared<SaveState>();
        state->save_id = save_id;
        state->block_hash = hash;
        state->failed = true;
        state->error_message =
            absl::StrCat("HBM block ID mismatch. Cache: ", block->hbm_block_id,
                         ", Requested: ", src_hbm_id);
        state->notification.Notify();
        results[hash] = SaveFuture(state);
        continue;
      }

      auto state = std::make_shared<SaveState>();
      state->save_id = save_id;
      state->block_hash = hash;
      results[hash] = SaveFuture(state);
      created_states.push_back(state);

      full_request.block_hashes.push_back(hash);
      full_request.src_block_ids.push_back(src_hbm_id);
    }
  }

  if (full_request.block_hashes.empty()) {
    return results;
  }

  {
    absl::MutexLock lock(&save_mu_);
    auto& save_map = active_saves_[save_id];
    for (const auto& state : created_states) {
      save_map[state->block_hash] = state;
    }
  }

  PushSaveWork(std::move(full_request));
  return results;
}

std::tuple<std::vector<std::string>, std::vector<std::string>,
           std::vector<std::string>>
KVCacheStore::PollSaveStatus() {
  absl::MutexLock lock(&save_mu_);
  std::vector<std::string> pending;
  for (const auto& [id, save_map] : active_saves_) {
    for (const auto& [hash, state] : save_map) {
      pending.push_back(hash);
    }
  }

  std::vector<std::string> done = std::move(done_saves_);
  std::vector<std::string> failed = std::move(failed_saves_);

  done_saves_.clear();
  failed_saves_.clear();

  return {std::move(done), std::move(failed), std::move(pending)};
}

void KVCacheStore::SaveCompletionPollerLoop() {
  SaveCompletion completion;
  while (PopSaveCompletion(completion)) {
    uint64_t save_id = completion.save_id;
    std::vector<global_registry::Registration> write_through_regs;
    {
      absl::MutexLock lock(&mutex_);
      for (const auto& item : completion.items) {
        if (item.success) {
          RaidenBlockID* block = lru_cache_.Get(item.block_hash);
          if (block) {
            if (block->status == BlockStatus::HBM) {
              block->status = BlockStatus::HOST_AND_HBM;
              block->host_block_id = item.dst_host_block_id;
              LOG(INFO) << "Upgraded block " << item.block_hash
                        << " to HOST_AND_HBM (Host ID: "
                        << item.dst_host_block_id
                        << ", HBM ID: " << block->hbm_block_id << ")";
              if (registry_client_) {
                write_through_regs.push_back({
                    .prefix_hash = item.block_hash,
                    .raiden_id = raiden_id_,
                    .block_id = item.dst_host_block_id,
                });
              }
            }
          }
        } else {
          LOG(ERROR) << "Save failed for " << item.block_hash << ": "
                     << item.error_message;
        }
      }
    }

    if (!write_through_regs.empty() && registry_client_) {
      std::thread([client = registry_client_,
                   regs = std::move(write_through_regs)]() {
        auto status = client->Register(regs);
        if (!status.ok()) {
          LOG(WARNING) << "Async batch write-through failed after Save: "
                       << status.message();
        } else {
          LOG(INFO) << "Async batch write-through succeeded after Save for "
                    << regs.size() << " blocks";
        }
      }).detach();
    }

    {
      absl::MutexLock lock(&save_mu_);
      auto save_it = active_saves_.find(save_id);
      if (save_it != active_saves_.end()) {
        auto& save_map = save_it->second;
        for (const auto& item : completion.items) {
          auto state_it = save_map.find(item.block_hash);
          if (state_it != save_map.end()) {
            auto& state = state_it->second;
            if (!item.success) {
              state->failed = true;
              state->error_message += item.error_message + ";";
            }
            if (state->failed) {
              failed_saves_.push_back(state->block_hash);
            } else {
              done_saves_.push_back(state->block_hash);
            }
            state->notification.Notify();
            save_map.erase(state_it);
          }
        }
        if (save_map.empty()) {
          active_saves_.erase(save_it);
        }
      }
    }
  }
}

absl::Status KVCacheStore::EvictFuture::Await() {
  if (!state_) {
    return absl::InvalidArgumentError("Future has no state");
  }
  state_->notification.WaitForNotification();
  if (state_->failed) {
    return absl::InternalError(state_->error_message);
  }
  return absl::OkStatus();
}

bool KVCacheStore::EvictFuture::IsDone() const {
  return state_ && state_->notification.HasBeenNotified();
}

absl::flat_hash_map<std::string, KVCacheStore::EvictFuture> KVCacheStore::Evict(
    const std::vector<std::string>& block_hashes) {
  absl::flat_hash_map<std::string, EvictFuture> results;
  uint64_t evict_id = 0;
  {
    absl::MutexLock lock(&evict_mu_);
    evict_id = next_evict_id_++;
  }
  EvictRequest full_request;
  full_request.evict_id = evict_id;

  std::vector<std::shared_ptr<EvictState>> created_states;
  std::vector<std::string> registry_unregs;

  {
    absl::MutexLock lock(&mutex_);
    for (const auto& hash : block_hashes) {
      RaidenBlockID* block = lru_cache_.Get(hash);
      if (!block) {
        auto state = std::make_shared<EvictState>();
        state->evict_id = evict_id;
        state->block_hash = hash;
        state->failed = true;
        state->error_message = "Block hash not found in local directory";
        state->notification.Notify();
        results[hash] = EvictFuture(state);
        continue;
      }

      if (block->status != BlockStatus::HOST &&
          block->status != BlockStatus::HOST_AND_HBM) {
        auto state = std::make_shared<EvictState>();
        state->evict_id = evict_id;
        state->block_hash = hash;
        state->failed = true;
        state->error_message =
            absl::StrCat("Block is not on HOST. Current status: ",
                         static_cast<int>(block->status));
        state->notification.Notify();
        results[hash] = EvictFuture(state);
        continue;
      }

      if (lru_cache_.GetPinCount(hash) > 0) {
        auto state = std::make_shared<EvictState>();
        state->evict_id = evict_id;
        state->block_hash = hash;
        state->failed = true;
        state->error_message = "Block is pinned and cannot be evicted";
        state->notification.Notify();
        results[hash] = EvictFuture(state);
        continue;
      }

      BlockStatus old_status = block->status;
      if (old_status == BlockStatus::HOST) {
        block->status = BlockStatus::INIT;
      } else if (old_status == BlockStatus::HOST_AND_HBM) {
        block->status = BlockStatus::HBM;
      }

      registry_unregs.push_back(hash);

      auto state = std::make_shared<EvictState>();
      state->evict_id = evict_id;
      state->block_hash = hash;
      results[hash] = EvictFuture(state);
      created_states.push_back(state);

      full_request.block_hashes.push_back(hash);
      full_request.host_block_ids.push_back(block->host_block_id);
    }
  }

  if (full_request.block_hashes.empty()) {
    return results;
  }

  if (registry_client_ && !registry_unregs.empty()) {
    std::thread([client = registry_client_, unregs = std::move(registry_unregs),
                 id = raiden_id_]() {
      auto status = client->Unregister(unregs, id);
      if (!status.ok()) {
        LOG(ERROR)
            << "Failed to unregister evicted blocks from global registry: "
            << status;
      }
    }).detach();
  }

  {
    absl::MutexLock lock(&evict_mu_);
    auto& evict_map = active_evicts_[evict_id];
    for (const auto& state : created_states) {
      evict_map[state->block_hash] = state;
    }
  }

  PushEvictWork(std::move(full_request));
  return results;
}

std::tuple<std::vector<std::string>, std::vector<std::string>,
           std::vector<std::string>>
KVCacheStore::PollEvictStatus() {
  absl::MutexLock lock(&evict_mu_);
  std::vector<std::string> pending;
  for (const auto& [id, evict_map] : active_evicts_) {
    for (const auto& [hash, state] : evict_map) {
      pending.push_back(hash);
    }
  }

  std::vector<std::string> done = std::move(done_evicts_);
  std::vector<std::string> failed = std::move(failed_evicts_);

  done_evicts_.clear();
  failed_evicts_.clear();

  return {std::move(done), std::move(failed), std::move(pending)};
}

void KVCacheStore::EvictCompletionPollerLoop() {
  EvictCompletion completion;
  while (PopEvictCompletion(completion)) {
    uint64_t evict_id = completion.evict_id;

    {
      absl::MutexLock lock(&mutex_);
      for (const auto& item : completion.items) {
        if (item.success) {
          RaidenBlockID* block = lru_cache_.Get(item.block_hash);
          if (block) {
            if (block->status == BlockStatus::INIT) {
              lru_cache_.Erase(item.block_hash);
            } else if (block->status == BlockStatus::HBM) {
              block->host_block_id = -1;
            }
          }
        } else {
          LOG(ERROR) << "Evict failed for " << item.block_hash << ": "
                     << item.error_message;
        }
      }
    }

    {
      absl::MutexLock lock(&evict_mu_);
      auto evict_it = active_evicts_.find(evict_id);
      if (evict_it != active_evicts_.end()) {
        auto& evict_map = evict_it->second;
        for (const auto& item : completion.items) {
          auto state_it = evict_map.find(item.block_hash);
          if (state_it != evict_map.end()) {
            auto& state = state_it->second;
            if (!item.success) {
              state->failed = true;
              state->error_message += item.error_message + ";";
            }
            if (state->failed) {
              failed_evicts_.push_back(state->block_hash);
            } else {
              done_evicts_.push_back(state->block_hash);
            }
            state->notification.Notify();
            evict_map.erase(state_it);
          }
        }
        if (evict_map.empty()) {
          active_evicts_.erase(evict_it);
        }
      }
    }
  }
}

}  // namespace kv_cache
}  // namespace tpu_raiden
