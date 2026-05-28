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

#include "kv_cache/global_registry/global_registry_server.h"

#include <grpcpp/grpcpp.h>

#include <cstdint>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "third_party/grpc/include/grpcpp/server_context.h"
#include "third_party/grpc/include/grpcpp/support/status.h"
#include "kv_cache/global_registry/global_registry.pb.h"

namespace tpu_raiden {
namespace kv_cache {
namespace global_registry {

GlobalRegistryServiceImpl::GlobalRegistryServiceImpl(
    absl::Duration default_ttl, absl::Duration cleanup_interval)
    : default_ttl_(default_ttl), cleanup_interval_(cleanup_interval) {
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
    if (entry.metadata().host_address().empty()) {
      response->set_success(false);
      response->set_error_message("host_address cannot be empty");
      return grpc::Status::OK;
    }
  }

  // 2. Apply updates
  absl::Time now = absl::Now();
  for (const auto& entry : request->entries()) {
    const std::string& prefix_hash = entry.prefix_hash();
    const KVBlockMetadata& meta = entry.metadata();
    const std::string& host = meta.host_address();

    absl::Duration ttl = entry.ttl_seconds() > 0
                             ? absl::Seconds(entry.ttl_seconds())
                             : default_ttl_;
    absl::Time expire_time = now + ttl;

    int32_t block_id = meta.block_id();

    registry_[prefix_hash] = {host, block_id, expire_time};
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
    if (it == registry_.end() || it->second.expire_time <= now) {
      break;
    }

    auto* meta = response->add_results();
    meta->set_host_address(it->second.host_address);
    meta->set_block_id(it->second.block_id);
  }

  return grpc::Status::OK;
}

grpc::Status GlobalRegistryServiceImpl::Unregister(
    grpc::ServerContext* context, const UnregisterRequest* request,
    UnregisterResponse* response) {
  absl::MutexLock lock(mutex_);

  const std::string& host = request->host_address();
  if (host.empty()) {
    response->set_success(false);
    response->set_error_message("host_address cannot be empty");
    return grpc::Status::OK;
  }

  bool overall_success = true;
  std::vector<std::string> errors;

  for (const std::string& hash : request->prefix_hashes()) {
    if (hash.empty()) continue;

    auto it = registry_.find(hash);
    if (it == registry_.end()) {
      // Already gone, safe to ignore for eviction.
      continue;
    }

    if (it->second.host_address == host) {
      registry_.erase(it);
    } else {
      overall_success = false;
      errors.push_back(hash + ": host mismatch");
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

void GlobalRegistryServiceImpl::CleanupExpiredEntries() {
  absl::MutexLock lock(mutex_);
  absl::Time now = absl::Now();

  std::vector<std::string> keys_to_remove;

  for (const auto& [hash, entry] : registry_) {
    if (entry.expire_time <= now) {
      keys_to_remove.push_back(hash);
    }
  }

  for (const auto& key : keys_to_remove) {
    registry_.erase(key);
  }
}

}  // namespace global_registry
}  // namespace kv_cache
}  // namespace tpu_raiden
