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

#include "tpu_raiden/kv_cache/global_registry/global_registry_server.h"

#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry.pb.h"
#include "tpu_raiden/kv_cache/raiden_id.h"

namespace tpu_raiden {
namespace kv_cache {
namespace global_registry {
namespace {

RaidenId FromProto(const ::tpu_raiden::rpc::RaidenIdProto& proto) {
  return RaidenId{
      .job_name = proto.job_name(),
      .job_replica_id = proto.job_replica_id(),
      .data_name = proto.data_name(),
      .data_replica_idx = proto.data_replica_idx(),
  };
}

void ToProto(const RaidenId& id, ::tpu_raiden::rpc::RaidenIdProto* proto) {
  proto->set_job_name(id.job_name);
  proto->set_job_replica_id(id.job_replica_id);
  proto->set_data_name(id.data_name);
  proto->set_data_replica_idx(id.data_replica_idx);
}

}  // namespace

GlobalRegistryServiceImpl::GlobalRegistryServiceImpl(
    absl::Duration default_ttl, absl::Duration cleanup_interval,
    int64_t pull_owned_batch_size)
    : default_ttl_(default_ttl),
      cleanup_interval_(cleanup_interval),
      pull_owned_batch_size_(pull_owned_batch_size > 0
                                 ? pull_owned_batch_size
                                 : kDefaultPullOwnedBatchSize) {
  StartCleanupThread();
}

GlobalRegistryServiceImpl::~GlobalRegistryServiceImpl() { StopCleanupThread(); }

void GlobalRegistryServiceImpl::StartCleanupThread() {
  if (cleanup_interval_ > absl::ZeroDuration()) {
    cleanup_thread_ =
        std::thread(&GlobalRegistryServiceImpl::CleanupLoop, this);
  }
}

void GlobalRegistryServiceImpl::StopCleanupThread() {
  shutdown_ = true;
  if (cleanup_thread_.joinable()) {
    cleanup_thread_.join();
  }
}

void GlobalRegistryServiceImpl::CleanupLoop() {
  while (!shutdown_) {
    // Sleep in small increments to respond to shutdown quickly
    absl::Time start = absl::Now();
    while (absl::Now() - start < cleanup_interval_ && !shutdown_) {
      absl::SleepFor(absl::Milliseconds(100));
    }
    if (shutdown_) break;
    CleanupExpiredEntries();
  }
}

grpc::Status GlobalRegistryServiceImpl::Register(grpc::ServerContext* context,
                                                 const RegisterRequest* request,
                                                 RegisterResponse* response) {
  absl::MutexLock lock(mutex_);

  // 1. Fail-fast validation
  for (const auto& entry : request->entries()) {
    if (entry.prefix_hash().empty()) {
      response->set_success(false);
      response->set_error_message("prefix_hash cannot be empty");
      return grpc::Status::OK;
    }
    if (!entry.has_metadata() || !entry.metadata().has_raiden_id() ||
        entry.metadata().raiden_id().job_name().empty()) {
      response->set_success(false);
      response->set_error_message(
          "metadata.raiden_id.job_name cannot be empty");
      return grpc::Status::OK;
    }
  }

  // 2. Apply updates
  absl::Time now = absl::Now();
  for (const auto& entry : request->entries()) {
    const std::string& prefix_hash = entry.prefix_hash();
    const KVBlockMetadata& meta = entry.metadata();
    RaidenId raiden_id = FromProto(meta.raiden_id());

    absl::Duration ttl = entry.ttl_seconds() > 0
                             ? absl::Seconds(entry.ttl_seconds())
                             : default_ttl_;
    absl::Time expire_time = now + ttl;
    int32_t block_id = meta.block_id();

    auto& entries = registry_[prefix_hash];
    bool found = false;
    for (auto& existing : entries) {
      if (existing.raiden_id == raiden_id) {
        existing.block_id = block_id;
        existing.expire_time = expire_time;
        found = true;
        break;
      }
    }
    if (!found) {
      entries.push_back({raiden_id, block_id, expire_time});
      // Updated together with registry_ under the same critical section; both
      // are in-memory only, so a crash discards them together. For replica
      // support, treat owner_index_ as derived data.
      owner_index_[raiden_id].insert(prefix_hash);
    }
  }

  response->set_success(true);
  return grpc::Status::OK;
}

grpc::Status GlobalRegistryServiceImpl::Lookup(grpc::ServerContext* context,
                                               const LookupRequest* request,
                                               LookupResponse* response) {
  absl::MutexLock lock(mutex_);
  absl::Time now = absl::Now();

  for (const std::string& hash : request->prefix_hashes()) {
    auto it = registry_.find(hash);
    if (it == registry_.end()) {
      break;
    }

    const auto& entries = it->second;
    std::vector<RegistryEntry> valid_entries;
    valid_entries.reserve(entries.size());
    for (const auto& entry : entries) {
      if (entry.expire_time > now) {
        valid_entries.push_back(entry);
      }
    }

    if (valid_entries.empty()) {
      break;
    }

    // Round-robin selection
    size_t& idx = round_robin_indices_[hash];
    idx = idx % valid_entries.size();
    const auto& picked = valid_entries[idx];
    idx++;  // Increment for next lookup

    auto* meta = response->add_results();
    ToProto(picked.raiden_id, meta->mutable_raiden_id());
    meta->set_block_id(picked.block_id);
  }

  return grpc::Status::OK;
}

grpc::Status GlobalRegistryServiceImpl::Unregister(
    grpc::ServerContext* context, const UnregisterRequest* request,
    UnregisterResponse* response) {
  absl::MutexLock lock(mutex_);

  if (!request->has_raiden_id() || request->raiden_id().job_name().empty()) {
    response->set_success(false);
    response->set_error_message("raiden_id cannot be empty");
    return grpc::Status::OK;
  }
  RaidenId raiden_id = FromProto(request->raiden_id());

  bool overall_success = true;
  std::vector<std::string> errors;

  for (const std::string& hash : request->prefix_hashes()) {
    if (hash.empty()) continue;

    auto it = registry_.find(hash);
    if (it == registry_.end()) {
      // Already gone, safe to ignore for eviction.
      continue;
    }

    auto& entries = it->second;
    auto orig_size = entries.size();
    entries.erase(std::remove_if(entries.begin(), entries.end(),
                                 [&raiden_id](const RegistryEntry& entry) {
                                   return entry.raiden_id == raiden_id;
                                 }),
                  entries.end());

    if (entries.size() == orig_size) {
      overall_success = false;
      errors.push_back(absl::StrCat(hash, ": raiden_id mismatch or not found"));
    } else {
      EraseFromOwnerIndex(raiden_id, hash);
    }

    if (entries.empty()) {
      registry_.erase(it);
      round_robin_indices_.erase(hash);
    }
  }

  response->set_success(overall_success);
  if (!errors.empty()) {
    std::string error_msg;
    for (const auto& err : errors) {
      if (!error_msg.empty()) error_msg += "; ";
      error_msg += err;
    }
    response->set_error_message(error_msg);
  }

  return grpc::Status::OK;
}

grpc::Status GlobalRegistryServiceImpl::PullOwned(
    grpc::ServerContext* context, const PullOwnedRequest* request,
    grpc::ServerWriter<PullOwnedResponse>* writer) {
  if (!request->has_raiden_id() || request->raiden_id().job_name().empty()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "raiden_id cannot be empty");
  }
  RaidenId raiden_id = FromProto(request->raiden_id());

  // Snapshot the owner's hash set in one short critical section. Every batch
  // below is re-validated against `registry_`, so entries removed after this
  // snapshot are simply omitted from the stream.
  std::vector<std::string> hashes;
  {
    absl::MutexLock lock(mutex_);
    auto it = owner_index_.find(raiden_id);
    if (it == owner_index_.end()) {
      return grpc::Status::OK;
    }
    hashes.assign(it->second.begin(), it->second.end());
  }

  const size_t batch_size = static_cast<size_t>(pull_owned_batch_size_);
  for (size_t start = 0; start < hashes.size(); start += batch_size) {
    const size_t end = std::min(hashes.size(), start + batch_size);
    PullOwnedResponse response;
    {
      absl::MutexLock lock(mutex_);
      absl::Time now = absl::Now();
      for (size_t i = start; i < end; ++i) {
        auto it = registry_.find(hashes[i]);
        if (it == registry_.end()) continue;
        for (const RegistryEntry& entry : it->second) {
          if (!(entry.raiden_id == raiden_id)) continue;
          // Only reachable for this owner's entry, and Register keeps at most
          // one entry per (hash, owner): an expired match means there is
          // nothing to emit for this hash.
          if (entry.expire_time <= now) break;
          PullOwnedEntry* out = response.add_entries();
          out->set_prefix_hash(hashes[i]);
          out->set_block_id(entry.block_id);
          if (entry.expire_time == absl::InfiniteFuture()) {
            out->set_remaining_ttl_seconds(0);
          } else {
            absl::Duration remaining = entry.expire_time - now;
            int64_t seconds = absl::ToInt64Seconds(remaining);
            if (absl::Seconds(seconds) < remaining) ++seconds;
            out->set_remaining_ttl_seconds(std::max<int64_t>(seconds, 1));
          }
          break;  // Register keeps at most one entry per (hash, owner).
        }
      }
    }
    // Write outside the lock: a slow client must not stall the registry.
    if (response.entries_size() > 0 && !writer->Write(response)) {
      return grpc::Status(grpc::StatusCode::CANCELLED,
                          "PullOwned stream closed by client");
    }
  }
  return grpc::Status::OK;
}

size_t GlobalRegistryServiceImpl::GetOwnerIndexSizeForTest(
    const RaidenId& raiden_id) const {
  absl::MutexLock lock(mutex_);
  auto it = owner_index_.find(raiden_id);
  return it == owner_index_.end() ? 0 : it->second.size();
}

void GlobalRegistryServiceImpl::EraseFromOwnerIndex(const RaidenId& raiden_id,
                                                    const std::string& hash) {
  auto it = owner_index_.find(raiden_id);
  if (it == owner_index_.end()) {
    return;
  }
  it->second.erase(hash);
  if (it->second.empty()) {
    owner_index_.erase(it);
  }
}

void GlobalRegistryServiceImpl::CleanupExpiredEntries() {
  absl::MutexLock lock(mutex_);
  absl::Time now = absl::Now();

  std::vector<std::string> keys_to_remove;

  for (auto& [hash, entries] : registry_) {
    for (const auto& entry : entries) {
      if (entry.expire_time <= now) {
        EraseFromOwnerIndex(entry.raiden_id, hash);
      }
    }
    entries.erase(std::remove_if(entries.begin(), entries.end(),
                                 [now](const RegistryEntry& entry) {
                                   return entry.expire_time <= now;
                                 }),
                  entries.end());

    if (entries.empty()) {
      keys_to_remove.push_back(hash);
    }
  }

  for (const auto& key : keys_to_remove) {
    registry_.erase(key);
    round_robin_indices_.erase(key);
  }
}

}  // namespace global_registry
}  // namespace kv_cache
}  // namespace tpu_raiden
