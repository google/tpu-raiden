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

#include "tpu_raiden/kv_cache/global_registry/global_registry_client.h"

#include <grpcpp/grpcpp.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/support/status.h"
#include <openssl/sha.h>
#include "tpu_raiden/kv_cache/global_registry/global_registry.grpc.pb.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry.pb.h"
#include "tpu_raiden/kv_cache/raiden_id.h"

namespace tpu_raiden {
namespace kv_cache {
namespace global_registry {

namespace {
void ToProto(const RaidenId& id, ::tpu_raiden::rpc::RaidenIdProto* proto) {
  proto->set_job_name(id.job_name);
  proto->set_job_replica_id(id.job_replica_id);
  proto->set_data_name(id.data_name);
  proto->set_data_replica_idx(id.data_replica_idx);
}
}  // namespace

GlobalRegistryClient::GlobalRegistryClient(
    std::shared_ptr<grpc::Channel> channel)
    : stub_(GlobalRegistryService::NewStub(channel)) {}

absl::Status GlobalRegistryClient::Register(
    const std::vector<Registration>& registrations) {
  RegisterRequest request;
  for (const auto& reg : registrations) {
    auto* entry = request.add_entries();
    entry->set_prefix_hash(reg.prefix_hash);
    auto* meta = entry->mutable_metadata();
    ToProto(reg.raiden_id, meta->mutable_raiden_id());
    meta->set_block_id(reg.block_id);
    if (reg.ttl > absl::ZeroDuration()) {
      entry->set_ttl_seconds(absl::ToInt64Seconds(reg.ttl));
    }
  }

  RegisterResponse response;
  grpc::ClientContext context;
  grpc::Status status = stub_->Register(&context, request, &response);

  if (!status.ok()) {
    return absl::InternalError(status.error_message());
  }
  if (!response.success()) {
    return absl::FailedPreconditionError(response.error_message());
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<KVBlockMetadata>> GlobalRegistryClient::Lookup(
    const std::vector<std::string>& prefix_hashes) {
  LookupRequest request;
  request.mutable_prefix_hashes()->Reserve(prefix_hashes.size());
  for (const auto& hash : prefix_hashes) {
    request.add_prefix_hashes(hash);
  }

  LookupResponse response;
  grpc::ClientContext context;
  grpc::Status status = stub_->Lookup(&context, request, &response);

  if (!status.ok()) {
    return absl::InternalError(status.error_message());
  }

  std::vector<KVBlockMetadata> result;
  result.reserve(response.results_size());
  for (const auto& meta : response.results()) {
    result.push_back(meta);
  }
  return result;
}

absl::StatusOr<std::vector<GlobalRegistryClient::PulledEntry>>
GlobalRegistryClient::PullOwned(const RaidenId& raiden_id) {
  PullOwnedRequest request;
  ToProto(raiden_id, request.mutable_raiden_id());

  grpc::ClientContext context;
  std::unique_ptr<grpc::ClientReader<PullOwnedResponse>> reader =
      stub_->PullOwned(&context, request);

  std::vector<PulledEntry> entries;
  PullOwnedResponse response;
  while (reader->Read(&response)) {
    for (const auto& entry : response.entries()) {
      entries.push_back({entry.prefix_hash(), entry.block_id(),
                         entry.remaining_ttl_seconds()});
    }
  }

  grpc::Status status = reader->Finish();
  if (!status.ok()) {
    return absl::InternalError(status.error_message());
  }
  return entries;
}

absl::Status GlobalRegistryClient::Unregister(
    const std::vector<std::string>& prefix_hashes, const RaidenId& raiden_id) {
  UnregisterRequest request;
  request.mutable_prefix_hashes()->Reserve(prefix_hashes.size());
  for (const auto& hash : prefix_hashes) {
    request.add_prefix_hashes(hash);
  }
  ToProto(raiden_id, request.mutable_raiden_id());

  UnregisterResponse response;
  grpc::ClientContext context;
  grpc::Status status = stub_->Unregister(&context, request, &response);

  if (!status.ok()) {
    return absl::InternalError(status.error_message());
  }
  if (!response.success()) {
    return absl::FailedPreconditionError(response.error_message());
  }
  return absl::OkStatus();
}

std::string CalculatePrefixHash(const std::vector<int64_t>& tokens,
                                absl::string_view parent_hash) {
  SHA256_CTX sha256;
  SHA256_Init(&sha256);

  if (!parent_hash.empty()) {
    SHA256_Update(&sha256, parent_hash.data(), parent_hash.size());
  }

  if (!tokens.empty()) {
    SHA256_Update(&sha256, tokens.data(), tokens.size() * sizeof(int64_t));
  }

  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256_Final(hash, &sha256);

  return absl::BytesToHexString(absl::string_view(
      reinterpret_cast<const char*>(hash), SHA256_DIGEST_LENGTH));
}

}  // namespace global_registry
}  // namespace kv_cache
}  // namespace tpu_raiden
