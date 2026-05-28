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
#include <cstdint>
#include <string>
#include <thread>  // NOLINT

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "third_party/grpc/include/grpcpp/server_context.h"
#include "third_party/grpc/include/grpcpp/support/status.h"
#include "kv_cache/global_registry/global_registry.grpc.pb.h"
#include "kv_cache/global_registry/global_registry.pb.h"

namespace tpu_raiden {
namespace kv_cache {
namespace global_registry {

struct RegistryEntry {
  std::string host_address;
  int32_t block_id;
  absl::Time expire_time;
};

class GlobalRegistryServiceImpl final : public GlobalRegistryService::Service {
 public:
  explicit GlobalRegistryServiceImpl(
      absl::Duration default_ttl = absl::Hours(1),
      absl::Duration cleanup_interval = absl::Minutes(5));

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

  // Force a cleanup of expired entries.
  void CleanupExpiredEntries();

 private:
  void StartCleanupThread();
  void StopCleanupThread();
  void CleanupLoop();

  absl::Duration default_ttl_;
  absl::Duration cleanup_interval_;

  absl::Mutex mutex_;
  // Key: prefix_hash
  // Value: single active registration entry
  absl::flat_hash_map<std::string, RegistryEntry> registry_
      ABSL_GUARDED_BY(mutex_);

  std::atomic<bool> shutdown_{false};
  std::thread cleanup_thread_;
};

}  // namespace global_registry
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_GLOBAL_REGISTRY_GLOBAL_REGISTRY_SERVER_H_
