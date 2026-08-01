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

#include "tpu_raiden/kv_cache/host_offload_backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_raiden/core/buffer.h"
#include "tpu_raiden/core/controller/raiden_controller.h"
#include "tpu_raiden/core/status_macros.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_client.h"
#include "tpu_raiden/kv_cache/kv_cache_metadata.h"
#include "tpu_raiden/kv_cache/kv_cache_store_backend.h"
#include "tpu_raiden/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_raiden/kv_cache/kv_cache_store_client.h"
#include "tpu_raiden/kv_cache/kv_cache_store_server.h"
#include "tpu_raiden/kv_cache/lru_cache.h"
#include "tpu_raiden/kv_cache/raiden_id.h"
#include "tpu_raiden/proto/kv_cache_store_service.pb.h"
#include "tpu_raiden/proto/worker_service.pb.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

namespace {

std::vector<::tpu_raiden::proto::RaidenWorkerEndpointsProto>
BuildLocalWorkerEndpoints(controller::RaidenController* ctrl) {
  std::vector<::tpu_raiden::proto::RaidenWorkerEndpointsProto> result;
  if (ctrl == nullptr || ctrl->worker_registry() == nullptr) {
    return result;
  }
  const auto& registered_workers = ctrl->worker_registry()->GetRegisteredWorkers();
  result.reserve(registered_workers.size());
  for (const auto& reg : registered_workers) {
    ::tpu_raiden::proto::RaidenWorkerEndpointsProto group;
    group.set_node_id(reg.node_id);
    group.set_worker_id(reg.worker_id);
    group.mutable_endpoints()->Reserve(reg.raiden_transfer_endpoints.size());
    for (const auto& ep : reg.raiden_transfer_endpoints) {
      auto* ep_proto = group.add_endpoints();
      ep_proto->set_endpoint(ep.endpoint);
      ep_proto->mutable_shards()->Add(ep.shards.begin(), ep.shards.end());
    }
    result.push_back(std::move(group));
  }
  return result;
}

}  // namespace

HostOffloadBackend::HostOffloadBackend(
    size_t capacity, std::optional<KVCacheMetadata> metadata,
    RaidenId raiden_id, controller::RaidenController* raiden_controller,
    std::shared_ptr<global_registry::GlobalRegistryClient> registry_client)
    : lru_cache_(capacity),
      metadata_(std::move(metadata)),
      raiden_id_(std::move(raiden_id)),
      raiden_controller_(raiden_controller),
      registry_client_(std::move(registry_client)),
      server_(KVCacheStoreServer::Create()) {}

HostOffloadBackend::~HostOffloadBackend() {
  if (server_) {
    server_->Shutdown();
  }
}

absl::StatusOr<std::shared_ptr<KVCacheStoreBackend>> HostOffloadBackend::Create(
    const BackendConfig& config) {
  if (config.capacity == 0) {
    return absl::InvalidArgumentError(
        "HostOffloadBackend requires capacity > 0");
  }
  std::shared_ptr<global_registry::GlobalRegistryClient> registry_client;
  if (!config.global_registry_address.empty()) {
    auto channel = grpc::CreateChannel(config.global_registry_address,
                                       grpc::InsecureChannelCredentials());
    registry_client = std::make_shared<
        ::tpu_raiden::kv_cache::global_registry::GlobalRegistryClient>(channel);
  }
  return std::make_shared<HostOffloadBackend>(
      config.capacity, config.metadata, config.raiden_id,
      /*raiden_controller=*/nullptr, std::move(registry_client));
}

absl::StatusOr<std::shared_ptr<KVCacheStoreBackend>> HostOffloadBackend::Create(
    const BackendConfig& config,
    controller::RaidenController* absl_nonnull controller) {
  if (config.capacity == 0) {
    return absl::InvalidArgumentError(
        "HostOffloadBackend requires capacity > 0");
  }
  std::shared_ptr<global_registry::GlobalRegistryClient> registry_client;
  if (!config.global_registry_address.empty()) {
    auto channel = grpc::CreateChannel(config.global_registry_address,
                                       grpc::InsecureChannelCredentials());
    registry_client = std::make_shared<
        ::tpu_raiden::kv_cache::global_registry::GlobalRegistryClient>(channel);
  }
  bool has_registry = (registry_client != nullptr);
  auto backend = std::make_shared<HostOffloadBackend>(
      config.capacity, config.metadata, config.raiden_id, controller,
      std::move(registry_client));
  if (controller != nullptr && has_registry) {
    RETURN_IF_ERROR(backend->StartServer(""));
  }
  return backend;
}

absl::Status HostOffloadBackend::StartServer(absl::string_view server_address) {
  controller::RaidenController* ctrl = nullptr;
  {
    absl::MutexLock lock(mutex_);
    ctrl = raiden_controller_;
  }
  if (ctrl == nullptr) {
    return absl::FailedPreconditionError(
        "Cannot start KVCacheStoreServer without a RaidenController.");
  }

  absl::MutexLock lock(mutex_);
  if (!server_) {
    server_ = KVCacheStoreServer::Create();
  }

  std::string target_host(server_address);
  if (target_host.empty()) {
    std::string ctrl_addr = ctrl->controller_address();
    if (!ctrl_addr.empty()) {
      size_t last_colon = ctrl_addr.rfind(':');
      size_t last_bracket = ctrl_addr.rfind(']');
      if (last_colon != std::string::npos &&
          (last_bracket == std::string::npos || last_colon > last_bracket)) {
        target_host = ctrl_addr.substr(0, last_colon);
      } else {
        target_host = ctrl_addr;
      }
    }
  }

  return server_->StartServer(this, ctrl, target_host);
}

absl::StatusOr<BlockSliceList> HostOffloadBackend::Lookup(
    absl::Span<const std::string> block_hashes, const LookupOptions& options) {
  if (block_hashes.empty()) {
    return BlockSliceList{};
  }

  BlockSliceList results;
  results.reserve(block_hashes.size());

  // Tier 0: Check Local Host LRU Cache
  {
    absl::MutexLock lock(mutex_);
    for (const auto& hash : block_hashes) {
      const RaidenBlockID* existing = lru_cache_.Peek(hash);
      if (!existing) {
        break;  // First miss
      }
      results.push_back(std::make_pair(hash, *existing));
    }
  }

  if (results.size() == block_hashes.size() || options.max_tier == 0) {
    return results;
  }

  // Tier 1: Check Global Registry for remaining missing hashes
  std::shared_ptr<global_registry::GlobalRegistryClient> client;
  RaidenId local_id;
  {
    absl::MutexLock lock(mutex_);
    client = registry_client_;
    local_id = raiden_id_;
  }

  if (!client) {
    return results;  // Return partial local matches
  }

  absl::Span<const std::string> missing_hashes =
      block_hashes.subspan(results.size());
  auto global_res_or = client->Lookup(
      std::vector<std::string>(missing_hashes.begin(), missing_hashes.end()));
  if (!global_res_or.ok()) {
    LOG(WARNING) << "Global registry lookup failed: "
                 << global_res_or.status().message();
    return results;
  }

  const auto& global_results = global_res_or.value();
  size_t num_results = std::min(global_results.size(), missing_hashes.size());
  for (size_t i = 0; i < num_results; ++i) {
    const auto& metadata = global_results[i];
    const auto& proto_id = metadata.raiden_id();
    RaidenId remote_id{
        .job_name = proto_id.job_name(),
        .job_replica_id = proto_id.job_replica_id(),
        .data_name = proto_id.data_name(),
        .data_replica_idx = proto_id.data_replica_idx(),
    };
    BlockStatus status =
        (remote_id == local_id) ? BlockStatus::HOST : BlockStatus::REMOTE;
    results.push_back(
        std::make_pair(missing_hashes[i],
                       RaidenBlockID(remote_id, metadata.block_id(), status)));
  }

  return results;
}

std::pair<bool, BlockSliceList> HostOffloadBackend::Insert(
    absl::Span<const std::string> block_hashes,
    absl::Span<const RaidenBlockID> slices, bool /*on_host*/) {
  BlockSliceList evicted_entries;
  bool all_inserted = true;
  std::vector<global_registry::Registration> registrations;
  std::shared_ptr<global_registry::GlobalRegistryClient> client;
  RaidenId local_id;

  {
    absl::MutexLock lock(mutex_);
    for (size_t i = 0; i < block_hashes.size(); ++i) {
      const std::string& hash = block_hashes[i];
      if (lru_cache_.Contains(hash)) {
        all_inserted = false;
        if (i < slices.size()) {
          if (RaidenBlockID* existing = lru_cache_.PeekMutable(hash)) {
            *existing = slices[i];
            SetMetadataEntry(hash, slices[i]);
          }
        }
        continue;
      }
      ReclaimStaleCandidate(hash);
      std::optional<std::pair<std::string, RaidenBlockID>> evicted;
      if (i < slices.size()) {
        evicted = lru_cache_.Put(hash, slices[i]);
        SetMetadataEntry(hash, slices[i]);
      } else {
        evicted = lru_cache_.Put(hash, RaidenBlockID());
      }
      if (evicted.has_value()) {
        evicted_entries.push_back(std::move(*evicted));
      }
      if (i < slices.size()) {
        registrations.push_back({
            .prefix_hash = hash,
            .raiden_id = raiden_id_,
            .block_id = slices[i].host_block_id,
        });
      }
    }
    client = registry_client_;
    local_id = raiden_id_;
  }

  if (client != nullptr) {
    if (!registrations.empty()) {
      auto status = client->Register(registrations);
      if (!status.ok()) {
        LOG(WARNING) << "Global registry register failed: " << status.message();
      }
    }
    if (!evicted_entries.empty()) {
      std::vector<std::string> evicted_hashes;
      evicted_hashes.reserve(evicted_entries.size());
      for (const auto& [hash, block_id] : evicted_entries) {
        evicted_hashes.push_back(hash);
      }
      auto status = client->Unregister(evicted_hashes, local_id);
      if (!status.ok()) {
        LOG(WARNING) << "Global registry unregister for evicted blocks failed: "
                     << status.message();
      }
    }
  }

  return std::make_pair(all_inserted, std::move(evicted_entries));
}

bool HostOffloadBackend::InsertAndLock(
    absl::Span<const std::string> block_hashes,
    absl::Span<const RaidenBlockID> slices, bool /*on_host*/) {
  absl::MutexLock lock(mutex_);

  std::vector<size_t> existing_indices;
  std::vector<size_t> new_indices;
  for (size_t i = 0; i < block_hashes.size(); ++i) {
    if (lru_cache_.Contains(block_hashes[i])) {
      existing_indices.push_back(i);
    } else {
      new_indices.push_back(i);
    }
  }

  for (size_t idx = 0; idx < existing_indices.size(); ++idx) {
    size_t i = existing_indices[idx];
    if (!lru_cache_.Pin(block_hashes[i])) {
      for (size_t j = 0; j < idx; ++j) {
        lru_cache_.Unpin(block_hashes[existing_indices[j]]);
      }
      return false;
    }
  }

  if (lru_cache_.available_space() < new_indices.size()) {
    for (size_t i : existing_indices) {
      lru_cache_.Unpin(block_hashes[i]);
    }
    return false;
  }

  size_t eviction_count = 0;
  for (auto it = new_indices.rbegin(); it != new_indices.rend(); ++it) {
    size_t i = *it;
    const std::string& hash = block_hashes[i];
    ReclaimStaleCandidate(hash);
    std::optional<std::pair<std::string, RaidenBlockID>> evicted;
    if (i < slices.size()) {
      evicted = lru_cache_.Put(hash, slices[i]);
      SetMetadataEntry(hash, slices[i]);
    } else {
      evicted = lru_cache_.Put(hash, RaidenBlockID());
    }
    if (evicted.has_value()) {
      eviction_count++;
    }
  }

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
        if (const RaidenBlockID* val =
                lru_cache_.PeekIncludingCandidates(block_hashes[j])) {
          ClearMetadataEntry(*val);
        }
        lru_cache_.Erase(block_hashes[j]);
      }
      for (size_t j = 0; j < eviction_count; ++j) {
        lru_cache_.RestoreLastCandidate();
      }
      return false;
    }
  }

  if (eviction_count > 0) {
    pending_eviction_counts_[GetSortedHashes(block_hashes)] = eviction_count;
  }
  return true;
}

size_t HostOffloadBackend::ReleaseAndDelete(
    absl::Span<const std::string> block_hashes) {
  absl::MutexLock lock(mutex_);
  size_t deleted_blocks = 0;

  for (auto it = block_hashes.rbegin(); it != block_hashes.rend(); ++it) {
    lru_cache_.Unpin(*it);
  }

  for (const std::string& hash : block_hashes) {
    auto* val = lru_cache_.Peek(hash);
    if (val != nullptr && lru_cache_.GetPinCount(hash) == 0 &&
        val->status != BlockStatus::HOST &&
        val->status != BlockStatus::HOST_AND_HBM) {
      lru_cache_.Erase(hash);
      deleted_blocks++;
    }
  }

  size_t restoration_count = 0;
  auto it = pending_eviction_counts_.find(GetSortedHashes(block_hashes));
  if (it != pending_eviction_counts_.end()) {
    restoration_count = it->second;
    pending_eviction_counts_.erase(it);
  }

  size_t to_restore = std::min(deleted_blocks, restoration_count);
  for (size_t i = 0; i < to_restore; ++i) {
    lru_cache_.RestoreLastCandidate();
  }

  return deleted_blocks;
}

void HostOffloadBackend::Delete(absl::Span<const std::string> block_hashes,
                                absl::Span<const RaidenBlockID> /*slices*/) {
  std::shared_ptr<global_registry::GlobalRegistryClient> client;
  RaidenId local_id;
  std::vector<std::string> deleted_hashes;

  {
    absl::MutexLock lock(mutex_);
    for (const std::string& hash : block_hashes) {
      if (lru_cache_.GetPinCount(hash) > 0) {
        LOG(WARNING) << "Delete skipped pinned block hash (release it first): "
                     << absl::BytesToHexString(hash);
        continue;
      }
      if (const RaidenBlockID* val = lru_cache_.PeekIncludingCandidates(hash)) {
        ClearMetadataEntry(*val);
      }
      lru_cache_.Erase(hash);
      deleted_hashes.push_back(hash);
    }
    client = registry_client_;
    local_id = raiden_id_;
  }

  if (client != nullptr && !deleted_hashes.empty()) {
    auto status = client->Unregister(deleted_hashes, local_id);
    if (!status.ok()) {
      LOG(WARNING) << "Global registry unregister failed: " << status.message();
    }
  }
}

bool HostOffloadBackend::Pin(absl::Span<const std::string> block_hashes) {
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

void HostOffloadBackend::Release(absl::Span<const std::string> block_hashes) {
  absl::MutexLock lock(mutex_);
  for (auto it = block_hashes.rbegin(); it != block_hashes.rend(); ++it) {
    lru_cache_.Unpin(*it);
  }
  pending_eviction_counts_.erase(GetSortedHashes(block_hashes));
}

int HostOffloadBackend::GetPinCount(const std::string& hash) const {
  absl::MutexLock lock(mutex_);
  return lru_cache_.GetPinCount(hash);
}

size_t HostOffloadBackend::GetCapacity() const {
  absl::MutexLock lock(mutex_);
  return lru_cache_.capacity();
}

size_t HostOffloadBackend::GetSize() const {
  absl::MutexLock lock(mutex_);
  return lru_cache_.size();
}

size_t HostOffloadBackend::GetAvailableSpace() const {
  absl::MutexLock lock(mutex_);
  return lru_cache_.available_space();
}

absl::StatusOr<size_t> HostOffloadBackend::RecoverFromLocalManifest() {
  absl::MutexLock lock(mutex_);
  if (!metadata_.has_value()) {
    return absl::FailedPreconditionError(
        "KVCacheMetadata is required for crash recovery");
  }
  std::vector<KVCacheMetadata::Entry> entries = metadata_->ValidEntries();
  if (entries.empty()) {
    return 0;
  }

  absl::flat_hash_map<absl::string_view, const KVCacheMetadata::Entry*> newest;
  newest.reserve(entries.size());
  for (const KVCacheMetadata::Entry& entry : entries) {
    auto [it, inserted] = newest.try_emplace(entry.hash, &entry);
    if (!inserted && entry.seq > it->second->seq) {
      it->second = &entry;
    }
  }

  std::vector<const KVCacheMetadata::Entry*> recoverable;
  recoverable.reserve(newest.size());
  for (const auto& [hash, entry] : newest) {
    recoverable.push_back(entry);
  }
  std::sort(recoverable.begin(), recoverable.end(),
            [](const KVCacheMetadata::Entry* a,
               const KVCacheMetadata::Entry* b) { return a->seq < b->seq; });

  if (raiden_controller_ != nullptr) {
    std::vector<int> block_ids;
    block_ids.reserve(recoverable.size());
    for (const KVCacheMetadata::Entry* entry : recoverable) {
      block_ids.push_back(entry->block_id);
    }
    absl::Status allocate_status =
        raiden_controller_->AllocateTargetBlockIds(block_ids);
    if (!allocate_status.ok()) {
      return allocate_status;
    }
  }

  uint64_t max_seq = 0;
  for (const KVCacheMetadata::Entry* entry : recoverable) {
    lru_cache_.Put(entry->hash, RaidenBlockID(raiden_id_, entry->block_id,
                                              BlockStatus::HOST));
    max_seq = std::max(max_seq, entry->seq);
  }
  next_metadata_seq_ = max_seq + 1;

  for (const KVCacheMetadata::Entry& entry : entries) {
    if (newest.at(entry.hash)->block_id != entry.block_id) {
      absl::Status status = metadata_->Clear(entry.block_id);
      if (!status.ok()) {
        LOG(WARNING) << "Failed to clear the stale metadata entry for block "
                     << entry.block_id << ": " << status.message();
      }
    }
  }

  return recoverable.size();
}

bool HostOffloadBackend::ValidateAndPinHostBlocks(
    absl::Span<const int> host_block_ids) {
  absl::MutexLock lock(mutex_);
  std::vector<std::string> keys_to_pin;
  keys_to_pin.reserve(host_block_ids.size());
  for (int host_id : host_block_ids) {
    bool found = false;
    for (const auto& [key, it] : lru_cache_.map()) {
      if (it->value.host_block_id == host_id &&
          (it->value.status == BlockStatus::HOST ||
           it->value.status == BlockStatus::HOST_AND_HBM)) {
        keys_to_pin.push_back(key);
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  for (size_t i = 0; i < keys_to_pin.size(); ++i) {
    if (!lru_cache_.Pin(keys_to_pin[i])) {
      for (size_t j = 0; j < i; ++j) {
        lru_cache_.Unpin(keys_to_pin[j]);
      }
      return false;
    }
  }
  return true;
}

std::vector<std::string> HostOffloadBackend::GetEvictableKeys(size_t count) {
  absl::MutexLock lock(mutex_);
  return lru_cache_.GetEvictableKeys(count);
}

std::vector<int> HostOffloadBackend::Evict(
    const std::vector<std::string>& block_hashes) {
  std::vector<int> host_ids_to_deallocate;
  std::shared_ptr<global_registry::GlobalRegistryClient> client;
  RaidenId local_id;
  std::vector<std::string> evicted_hashes;

  {
    absl::MutexLock lock(mutex_);
    for (const std::string& hash : block_hashes) {
      const RaidenBlockID* block = lru_cache_.PeekIncludingCandidates(hash);
      if (block != nullptr && lru_cache_.GetPinCount(hash) == 0 &&
          (block->status == BlockStatus::HOST ||
           block->status == BlockStatus::HOST_AND_HBM)) {
        host_ids_to_deallocate.push_back(block->host_block_id);
        ClearMetadataEntry(*block);
        lru_cache_.Erase(hash);
        evicted_hashes.push_back(hash);
      }
    }
    client = registry_client_;
    local_id = raiden_id_;
  }

  if (client != nullptr && !evicted_hashes.empty()) {
    auto status = client->Unregister(evicted_hashes, local_id);
    if (!status.ok()) {
      LOG(WARNING) << "Global registry unregister failed: " << status.message();
    }
  }

  return host_ids_to_deallocate;
}

std::vector<std::string> HostOffloadBackend::GetEvictCandidateKeys() const {
  absl::MutexLock lock(mutex_);
  return lru_cache_.GetEvictCandidateKeys();
}

void HostOffloadBackend::SetRaidenController(
    controller::RaidenController* controller) {
  absl::MutexLock lock(mutex_);
  raiden_controller_ = controller;
}

absl::StatusOr<std::shared_ptr<KVCacheStoreClient>>
HostOffloadBackend::GetKVCacheStoreClient(const RaidenId& remote_id) {
  controller::RaidenController* ctrl = nullptr;
  {
    absl::MutexLock lock(mutex_);
    auto it = store_clients_.find(remote_id);
    if (it != store_clients_.end()) {
      return it->second;
    }
    ctrl = raiden_controller_;
  }

  if (ctrl == nullptr) {
    return absl::FailedPreconditionError("RaidenController is null");
  }

  tpu_raiden::rpc::RaidenIdProto remote_proto;
  remote_proto.set_job_name(remote_id.job_name);
  remote_proto.set_job_replica_id(remote_id.job_replica_id);
  remote_proto.set_data_name(remote_id.data_name);
  remote_proto.set_data_replica_idx(remote_id.data_replica_idx);

  ASSIGN_OR_RETURN(std::string address,
                   ctrl->ResolvePeerController(remote_proto));
  auto channel =
      grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  auto client = std::make_shared<KVCacheStoreClient>(channel);
  {
    absl::MutexLock lock(mutex_);
    store_clients_[remote_id] = client;
  }
  return client;
}

tsl::Future<> HostOffloadBackend::Load(
    const RaidenId& remote_id, absl::Span<const std::string> block_hashes,
    absl::Span<const int32_t> device_block_ids) {
  if (block_hashes.empty()) {
    return tsl::Future<>(absl::OkStatus());
  }

  if (!device_block_ids.empty() &&
      device_block_ids.size() != block_hashes.size()) {
    return tsl::Future<>(absl::InvalidArgumentError(absl::StrCat(
        "Mismatched device_block_ids count (", device_block_ids.size(),
        ") vs block_hashes count (", block_hashes.size(), ").")));
  }

  controller::RaidenController* ctrl = nullptr;
  {
    absl::MutexLock lock(mutex_);
    ctrl = raiden_controller_;
  }

  if (ctrl == nullptr) {
    return tsl::Future<>(
        absl::FailedPreconditionError("RaidenController is null"));
  }

  auto client_or = GetKVCacheStoreClient(remote_id);
  if (!client_or.ok()) {
    return tsl::Future<>(client_or.status());
  }
  std::shared_ptr<KVCacheStoreClient> client = std::move(client_or.value());

  auto host_blocks_or = ctrl->AllocateBlockIds(block_hashes.size());
  if (!host_blocks_or.ok()) {
    return tsl::Future<>(host_blocks_or.status());
  }
  std::vector<int32_t> dst_host_block_ids(host_blocks_or.value().begin(),
                                          host_blocks_or.value().end());

  auto [load_promise, load_future] = tsl::MakePromise<>();

  tpu_raiden::rpc::RaidenIdProto client_raiden_id = ctrl->unit();
  std::vector<::tpu_raiden::proto::RaidenWorkerEndpointsProto>
      client_worker_endpoints = BuildLocalWorkerEndpoints(ctrl);
  tsl::Future<proto::FetchResponse> fetch_future = client->Fetch(
      block_hashes, device_block_ids, dst_host_block_ids, client_raiden_id,
      client_worker_endpoints);

  fetch_future.OnReady(
      [this, dst_host_block_ids,
       dev_ids_vec = std::vector<int32_t>(device_block_ids.begin(),
                                          device_block_ids.end()),
       load_promise = std::move(load_promise)](
          const absl::StatusOr<proto::FetchResponse>& response_or) mutable {
        controller::RaidenController* ctrl_cb = nullptr;
        {
          absl::MutexLock lock(mutex_);
          ctrl_cb = raiden_controller_;
        }
        if (!response_or.ok()) {
          if (ctrl_cb) {
            (void)ctrl_cb->DeallocateBlockIds(dst_host_block_ids);
          }
          load_promise.Set(response_or.status());
          return;
        }

        const auto& response = response_or.value();
        if (!response.failed_block_hashes().empty()) {
          if (ctrl_cb) {
            (void)ctrl_cb->DeallocateBlockIds(dst_host_block_ids);
          }
          std::string err_msg = response.error_message().empty()
                                    ? "Fetch RPC returned failed blocks"
                                    : response.error_message();
          load_promise.Set(absl::InternalError(err_msg));
          return;
        }

        if (dev_ids_vec.empty()) {
          if (ctrl_cb) {
            (void)ctrl_cb->DeallocateBlockIds(dst_host_block_ids);
          }
          load_promise.Set(absl::OkStatus());
          return;
        }

        std::vector<Buffer> src_buffers;
        src_buffers.reserve(dst_host_block_ids.size());
        for (int id : dst_host_block_ids) {
          src_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                                   rpc::MEMORY_TYPE_DRAM);
        }

        std::vector<Buffer> dst_buffers;
        dst_buffers.reserve(dev_ids_vec.size());
        for (int id : dev_ids_vec) {
          dst_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                                   rpc::MEMORY_TYPE_HBM);
        }

        if (!ctrl_cb) {
          load_promise.Set(
              absl::FailedPreconditionError("RaidenController is null"));
          return;
        }

        tsl::Future<> h2d_future =
            ctrl_cb->TransferBuffers(src_buffers, dst_buffers);

        h2d_future.OnReady(
            [this, dst_host_block_ids, load_promise = std::move(load_promise)](
                absl::Status status) mutable {
              controller::RaidenController* ctrl_h2d = nullptr;
              {
                absl::MutexLock lock(mutex_);
                ctrl_h2d = raiden_controller_;
              }
              if (ctrl_h2d) {
                (void)ctrl_h2d->DeallocateBlockIds(dst_host_block_ids);
              }
              load_promise.Set(status);
            });
      });

  return load_future;
}

void HostOffloadBackend::SetMetadataEntry(absl::string_view hash,
                                          const RaidenBlockID& block) {
  if (!metadata_.has_value()) {
    return;
  }
  if (block.status != BlockStatus::HOST &&
      block.status != BlockStatus::HOST_AND_HBM) {
    return;
  }
  absl::Status status =
      metadata_->Set(block.host_block_id, hash, next_metadata_seq_++);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to set the metadata entry for block "
                 << block.host_block_id << ": " << status.message();
  }
}

void HostOffloadBackend::ClearMetadataEntry(const RaidenBlockID& block) {
  if (!metadata_.has_value()) {
    return;
  }
  if (block.status != BlockStatus::HOST &&
      block.status != BlockStatus::HOST_AND_HBM) {
    return;
  }
  absl::Status status = metadata_->Clear(block.host_block_id);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to clear the metadata entry for block "
                 << block.host_block_id << ": " << status.message();
  }
}

void HostOffloadBackend::ReclaimStaleCandidate(const std::string& hash) {
  // The caller only reaches here when Contains(hash) is false, so any entry
  // still findable under this hash is an eviction candidate.
  const RaidenBlockID* stale = lru_cache_.PeekIncludingCandidates(hash);
  if (stale == nullptr) {
    return;
  }
  ClearMetadataEntry(*stale);
  // HOST / HOST_AND_HBM means host_block_id is a block of the LOCAL
  // allocator (REMOTE entries carry the peer's coordinate instead). The
  // fresh insert replaces the value in place, so this is the last point the
  // old block id is visible -- return it to the allocator here.
  if (raiden_controller_ != nullptr && stale->host_block_id >= 0 &&
      (stale->status == BlockStatus::HOST ||
       stale->status == BlockStatus::HOST_AND_HBM)) {
    absl::Status status =
        raiden_controller_->DeallocateBlockIds({stale->host_block_id});
    if (!status.ok()) {
      LOG(WARNING) << "Failed to deallocate host block "
                   << stale->host_block_id
                   << " of replaced eviction candidate: " << status.message();
    }
  }
}

std::vector<std::string> HostOffloadBackend::GetSortedHashes(
    absl::Span<const std::string> hashes) const {
  std::vector<std::string> sorted(hashes.begin(), hashes.end());
  std::sort(sorted.begin(), sorted.end());
  return sorted;
}
}  // namespace kv_cache
}  // namespace tpu_raiden

REGISTER_KV_CACHE_STORE_BACKEND(
    "HostOffloadBackend",
    static_cast<absl::StatusOr<
        std::shared_ptr<::tpu_raiden::kv_cache::KVCacheStoreBackend>> (*)(
        const ::tpu_raiden::kv_cache::BackendConfig&)>(
        ::tpu_raiden::kv_cache::HostOffloadBackend::Create));
