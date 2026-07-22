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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_GLOBAL_REGISTRY_GLOBAL_REGISTRY_CLIENT_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_GLOBAL_REGISTRY_GLOBAL_REGISTRY_CLIENT_H_

#include <grpcpp/grpcpp.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry.grpc.pb.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry.pb.h"
#include "tpu_raiden/kv_cache/raiden_id.h"

namespace tpu_raiden {
namespace kv_cache {
namespace global_registry {

struct Registration {
  std::string prefix_hash;
  RaidenId raiden_id;
  int32_t block_id;
  absl::Duration ttl = absl::ZeroDuration();
};

class GlobalRegistryClient {
 public:
  explicit GlobalRegistryClient(std::shared_ptr<grpc::Channel> channel);

  // Registers a batch of KV cache entries.
  absl::Status Register(const std::vector<Registration>& registrations);

  // Looks up KV cache entries for a batch of prefix hashes.
  // The lookup processes prefix hashes sequentially and stops at the first miss
  // (a hash with no active registrations). All subsequent prefix hashes in the
  // input vector are treated as misses and are omitted from the response.
  // The returned vector is aligned in order with the input `prefix_hashes` (the
  // i-th element of the returned vector corresponds to the i-th input hash).
  // The size of the returned vector will be equal to the number of sequential
  // hits before the first miss.
  absl::StatusOr<std::vector<KVBlockMetadata>> Lookup(
      const std::vector<std::string>& prefix_hashes);

  // Unregisters a batch of KV cache entries for a raiden id.
  absl::Status Unregister(const std::vector<std::string>& prefix_hashes,
                          const RaidenId& raiden_id);

  // A single active registration returned by PullOwned.
  struct PulledEntry {
    std::string prefix_hash;
    int32_t block_id;
    // Seconds until the registration expires as reported by the server;
    // 0 means it never expires.
    int64_t remaining_ttl_seconds;
  };

  // Pulls all active registrations owned by `raiden_id`, draining the
  // server-side stream. Intended for owner restart handling; see the
  // PullOwned RPC documentation for the consistency contract.
  absl::StatusOr<std::vector<PulledEntry>> PullOwned(const RaidenId& raiden_id);

 private:
  std::unique_ptr<GlobalRegistryService::Stub> stub_;
};

// Cumulative hash helper.
// Calculates SHA256 hash of (parent_hash_hex + tokens_binary).
// Returns hex-encoded SHA256 string.
std::string CalculatePrefixHash(const std::vector<int64_t>& tokens,
                                absl::string_view parent_hash = "");

}  // namespace global_registry
}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_GLOBAL_REGISTRY_GLOBAL_REGISTRY_CLIENT_H_
