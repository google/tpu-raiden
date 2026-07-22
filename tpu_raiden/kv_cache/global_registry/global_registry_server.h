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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_GLOBAL_REGISTRY_GLOBAL_REGISTRY_SERVER_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_GLOBAL_REGISTRY_GLOBAL_REGISTRY_SERVER_H_

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>  // NOLINT

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry.grpc.pb.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry.pb.h"
#include "tpu_raiden/kv_cache/raiden_id.h"

namespace tpu_raiden {
namespace kv_cache {
namespace global_registry {

struct RegistryEntry {
  RaidenId raiden_id;
  int32_t block_id;
  absl::Time expire_time;
};

class GlobalRegistryServiceImpl final : public GlobalRegistryService::Service {
 public:
  // Default maximum number of entries per streamed PullOwned response.
  static constexpr int64_t kDefaultPullOwnedBatchSize = 4096;

  explicit GlobalRegistryServiceImpl(
      absl::Duration default_ttl = absl::Hours(1),
      absl::Duration cleanup_interval = absl::Minutes(5),
      int64_t pull_owned_batch_size = kDefaultPullOwnedBatchSize);

  ~GlobalRegistryServiceImpl() override;

  // Disallow copy and assign
  GlobalRegistryServiceImpl(const GlobalRegistryServiceImpl&) = delete;
  GlobalRegistryServiceImpl& operator=(const GlobalRegistryServiceImpl&) =
      delete;

  grpc::Status Register(grpc::ServerContext* context,
                        const RegisterRequest* request,
                        RegisterResponse* response) override;

  grpc::Status Lookup(grpc::ServerContext* context,
                      const LookupRequest* request,
                      LookupResponse* response) override;

  grpc::Status Unregister(grpc::ServerContext* context,
                          const UnregisterRequest* request,
                          UnregisterResponse* response) override;

  grpc::Status PullOwned(
      grpc::ServerContext* context, const PullOwnedRequest* request,
      grpc::ServerWriter<PullOwnedResponse>* writer) override;

  // Force a cleanup of expired entries.
  void CleanupExpiredEntries();

  // Number of prefix hashes currently indexed for `raiden_id`. Test-only:
  // verifies that Unregister and cleanup shrink the owner index.
  size_t GetOwnerIndexSizeForTest(const RaidenId& raiden_id) const;

 private:
  void StartCleanupThread();
  void StopCleanupThread();
  void CleanupLoop();

  // Removes `hash` from `raiden_id`'s owner-index set, dropping the owner key
  // once its set becomes empty.
  void EraseFromOwnerIndex(const RaidenId& raiden_id, const std::string& hash)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  absl::Duration default_ttl_;
  absl::Duration cleanup_interval_;
  const int64_t pull_owned_batch_size_;

  mutable absl::Mutex mutex_;
  // Key: prefix_hash
  // Value: list of active registration entries
  absl::flat_hash_map<std::string, std::vector<RegistryEntry>> registry_
      ABSL_GUARDED_BY(mutex_);

  // Key: prefix_hash
  // Value: last returned index for round-robin selection
  absl::flat_hash_map<std::string, size_t> round_robin_indices_
      ABSL_GUARDED_BY(mutex_);

  // Secondary index over `registry_` for PullOwned: owner -> prefix hashes
  // that have an entry owned by that RaidenId. Keys only; `registry_` remains
  // the sole source of truth for block IDs and expiration. Maintained
  // together with `registry_` under `mutex_` in Register, Unregister and
  // CleanupExpiredEntries.
  absl::flat_hash_map<RaidenId, absl::flat_hash_set<std::string>, RaidenIdHash>
      owner_index_ ABSL_GUARDED_BY(mutex_);

  std::atomic<bool> shutdown_{false};
  std::thread cleanup_thread_;
};

}  // namespace global_registry
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_GLOBAL_REGISTRY_GLOBAL_REGISTRY_SERVER_H_
