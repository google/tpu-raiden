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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_raiden/core/buffer.h"
#include "tpu_raiden/core/controller/raiden_controller.h"
#include "tpu_raiden/core/numa_thread_pool.h"
#include "tpu_raiden/core/status_macros.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_client.h"
#include "tpu_raiden/kv_cache/host_offload_backend.h"
#include "tpu_raiden/kv_cache/kv_cache_metadata.h"
#include "tpu_raiden/kv_cache/kv_cache_store_backend.h"
#include "tpu_raiden/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_raiden/kv_cache/lru_cache.h"
#include "tpu_raiden/kv_cache/raiden_id.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

// Bind-and-advertise address for this store's RaidenController.
//
// Empty ip preserves the legacy behaviour (RaidenController binds the wildcard
// interface). A port of 0 lets gRPC choose; either way RaidenController splices
// the actually-bound port back into the address it advertises.
std::string ComposeControllerAddress(absl::string_view store_server_ip,
                                     int raiden_controller_port) {
  if (store_server_ip.empty()) {
    return raiden_controller_port > 0
               ? absl::StrCat("[::]:", raiden_controller_port)
               : "";
  }
  return absl::StrCat(store_server_ip, ":", raiden_controller_port);
}

}  // namespace

absl::StatusOr<std::unique_ptr<KVCacheStore>> KVCacheStore::Create(
    absl::Span<const BackendConfig> backend_configs, size_t capacity,
    absl::string_view global_registry_address, RaidenId raiden_id,
    int num_shards, int64_t shard_size_bytes,
    absl::string_view raiden_orchestrator_address,
    absl::string_view store_server_ip, int raiden_controller_port,
    std::optional<KVCacheMetadata> metadata) {
  if (backend_configs.empty()) {
    return absl::InvalidArgumentError("backend_configs must not be empty");
  }

  std::vector<std::shared_ptr<KVCacheStoreBackend>> backends;
  backends.reserve(backend_configs.size());

  for (size_t i = 0; i < backend_configs.size(); ++i) {
    const auto& config = backend_configs[i];
    BackendConfig effective_config = config;

    // Apply capacity and metadata fallbacks ONLY to primary Tier 0 backend
    // (local host DRAM)
    if (i == 0) {
      if (effective_config.capacity == 0 && capacity > 0) {
        effective_config.capacity = capacity;
      }
      if (!effective_config.metadata.has_value() && metadata.has_value()) {
        effective_config.metadata = metadata;
      }
    }

    // Apply global registry address and raiden_id across all backend
    // configurations
    if (effective_config.global_registry_address.empty() &&
        !global_registry_address.empty()) {
      effective_config.global_registry_address =
          std::string(global_registry_address);
    }
    if (effective_config.raiden_id.empty() && !raiden_id.empty()) {
      effective_config.raiden_id = raiden_id;
    }

    ASSIGN_OR_RETURN(
        auto backend,
        KVCacheStoreBackendFactory::Instance().CreateBackend(effective_config));
    backends.push_back(std::move(backend));
  }

  RaidenId effective_raiden_id = !backend_configs[0].raiden_id.empty()
                                     ? backend_configs[0].raiden_id
                                     : raiden_id;

  auto store = absl::WrapUnique(new KVCacheStore(
      std::move(backends), effective_raiden_id, num_shards, shard_size_bytes,
      raiden_orchestrator_address, store_server_ip, raiden_controller_port,
      global_registry_address));

  if (store->raiden_controller_ != nullptr) {
    store->SetRaidenController(store->raiden_controller_.get());
  }

  return store;
}

absl::StatusOr<std::unique_ptr<KVCacheStore>> KVCacheStore::Create(
    const BackendConfig& config, size_t capacity,
    absl::string_view global_registry_address, RaidenId raiden_id,
    int num_shards, int64_t shard_size_bytes,
    absl::string_view raiden_orchestrator_address,
    absl::string_view store_server_ip, int raiden_controller_port,
    std::optional<KVCacheMetadata> metadata) {
  return KVCacheStore::Create(absl::MakeConstSpan(&config, 1), capacity,
                              global_registry_address, raiden_id, num_shards,
                              shard_size_bytes, raiden_orchestrator_address,
                              store_server_ip, raiden_controller_port,
                              std::move(metadata));
}

KVCacheStore::KVCacheStore(
    std::vector<std::shared_ptr<KVCacheStoreBackend>> backends,
    RaidenId raiden_id, int num_shards, int64_t shard_size_bytes,
    absl::string_view raiden_orchestrator_address,
    absl::string_view store_server_ip, int raiden_controller_port,
    absl::string_view global_registry_address)
    : backends_(std::move(backends)),
      raiden_id_(std::move(raiden_id)),
      store_server_ip_(store_server_ip),
      write_through_pool_(std::make_unique<::tpu_raiden::NumaThreadPool>(4)) {
  // Created before SetRaidenController: publishing happens there, and needs it.
  if (!global_registry_address.empty()) {
    auto channel = grpc::CreateChannel(std::string(global_registry_address),
                                       grpc::InsecureChannelCredentials());
    registry_client_ =
        std::make_shared<global_registry::GlobalRegistryClient>(channel);
  }
  if (num_shards > 0) {
    ::tpu_raiden::rpc::RaidenIdProto unit_proto;
    unit_proto.set_job_name(raiden_id_.job_name);
    unit_proto.set_job_replica_id(raiden_id_.job_replica_id);
    unit_proto.set_data_name(raiden_id_.data_name);
    unit_proto.set_data_replica_idx(raiden_id_.data_replica_idx);

    size_t cap = capacity();
    raiden_controller_ =
        std::make_unique<::tpu_raiden::controller::RaidenController>(
            unit_proto, cap, num_shards, shard_size_bytes,
            raiden_orchestrator_address,
            ComposeControllerAddress(store_server_ip, raiden_controller_port));
  }
  if (raiden_controller_) {
    SetRaidenController(raiden_controller_.get());
    RegisterReadRemoteHooks();
    poller_thread_ =
        std::make_unique<std::thread>(&KVCacheStore::PollerLoop, this);
  }
}

KVCacheStore::KVCacheStore(std::shared_ptr<KVCacheStoreBackend> backend,
                           RaidenId raiden_id, int num_shards,
                           int64_t shard_size_bytes,
                           absl::string_view raiden_orchestrator_address,
                           absl::string_view store_server_ip,
                           int raiden_controller_port,
                           absl::string_view global_registry_address)
    : KVCacheStore(
          std::vector<std::shared_ptr<KVCacheStoreBackend>>{std::move(backend)},
          std::move(raiden_id), num_shards, shard_size_bytes,
          raiden_orchestrator_address, store_server_ip, raiden_controller_port,
          global_registry_address) {}

KVCacheStore::KVCacheStore(size_t capacity,
                           absl::string_view global_registry_address,
                           RaidenId raiden_id, int num_shards,
                           int64_t shard_size_bytes,
                           absl::string_view raiden_orchestrator_address,
                           absl::string_view store_server_ip,
                           int raiden_controller_port,
                           std::optional<KVCacheMetadata> metadata)
    : raiden_id_(raiden_id),
      store_server_ip_(store_server_ip),
      write_through_pool_(std::make_unique<::tpu_raiden::NumaThreadPool>(4)) {
  if (!global_registry_address.empty()) {
    auto channel = grpc::CreateChannel(std::string(global_registry_address),
                                       grpc::InsecureChannelCredentials());
    registry_client_ =
        std::make_shared<global_registry::GlobalRegistryClient>(channel);
  }
  if (num_shards > 0) {
    ::tpu_raiden::rpc::RaidenIdProto unit_proto;
    unit_proto.set_job_name(raiden_id_.job_name);
    unit_proto.set_job_replica_id(raiden_id_.job_replica_id);
    unit_proto.set_data_name(raiden_id_.data_name);
    unit_proto.set_data_replica_idx(raiden_id_.data_replica_idx);

    raiden_controller_ =
        std::make_unique<::tpu_raiden::controller::RaidenController>(
            unit_proto, capacity, num_shards, shard_size_bytes,
            raiden_orchestrator_address,
            ComposeControllerAddress(store_server_ip, raiden_controller_port));
  }

  backends_ = {std::make_shared<HostOffloadBackend>(
      capacity, std::move(metadata), raiden_id_, raiden_controller_.get())};

  if (raiden_controller_) {
    SetRaidenController(raiden_controller_.get());
    RegisterReadRemoteHooks();
    poller_thread_ =
        std::make_unique<std::thread>(&KVCacheStore::PollerLoop, this);
  }
}

KVCacheStore::KVCacheStore(
    size_t capacity,
    std::unique_ptr<::tpu_raiden::controller::RaidenController>
        raiden_controller,
    absl::string_view global_registry_address, RaidenId raiden_id,
    std::optional<KVCacheMetadata> metadata,
    absl::string_view store_server_ip)
    : raiden_id_(raiden_id),
      raiden_controller_(std::move(raiden_controller)),
      store_server_ip_(store_server_ip),
      write_through_pool_(std::make_unique<::tpu_raiden::NumaThreadPool>(4)) {
  if (!global_registry_address.empty()) {
    auto channel = grpc::CreateChannel(std::string(global_registry_address),
                                       grpc::InsecureChannelCredentials());
    registry_client_ =
        std::make_shared<global_registry::GlobalRegistryClient>(channel);
  }

  backends_ = {std::make_shared<HostOffloadBackend>(
      capacity, std::move(metadata), raiden_id_, raiden_controller_.get())};

  if (raiden_controller_) {
    SetRaidenController(raiden_controller_.get());
    RegisterReadRemoteHooks();
    poller_thread_ =
        std::make_unique<std::thread>(&KVCacheStore::PollerLoop, this);
  }
}

void KVCacheStore::SetRaidenController(
    tpu_raiden::controller::RaidenController* controller) {
  // Runs first so that store_server_ip_ decides the bind address: StartServer
  // never rebinds a running server, so whoever starts it first wins.
  EnsureStoreServerAndRegister(controller);

  for (auto& backend : backends_) {
    if (backend != nullptr) {
      backend->SetRaidenController(controller);
    }
  }
}

void KVCacheStore::EnsureStoreServerAndRegister(
    tpu_raiden::controller::RaidenController* controller) {
  if (store_server_ != nullptr || controller == nullptr) {
    return;  // Already done; SetRaidenController may be called repeatedly.
  }

  // Reuse a backend's server if one exists, so a node never serves peers from
  // two ports. Otherwise this store owns one -- every store must be reachable,
  // not just those configured with a backend that happens to host a server.
  KVCacheStoreBackend* serving_backend = nullptr;
  for (auto& backend : backends_) {
    if (backend != nullptr && backend->store_server() != nullptr) {
      store_server_ = backend->store_server();
      serving_backend = backend.get();
      break;
    }
  }
  if (store_server_ == nullptr) {
    // No backend hosts one. Only stand a server up when this store was given
    // an IP to publish -- otherwise it would be an unreachable, unadvertised
    // port on every store in the process, which is a behaviour change for
    // every existing caller.
    if (store_server_ip_.empty()) {
      return;
    }
    // backend() indexes backends_[0] without checking, and an empty or
    // null-headed backends_ is reachable -- capacity() guards for exactly
    // that. A service with no backend can answer nothing, so decline rather
    // than stand up a server that fails every RPC.
    if (backends_.empty() || backends_[0] == nullptr) {
      LOG(WARNING) << "KVCacheStore has no tier-0 backend; not serving peers.";
      return;
    }
    owned_store_server_ = KVCacheStoreServer::Create();
    store_server_ = owned_store_server_.get();
    serving_backend = backends_[0].get();
  }

  // Empty ip keeps the legacy wildcard bind. The port is always gRPC's choice.
  const std::string bind_address =
      store_server_ip_.empty() ? "[::]:0" : absl::StrCat(store_server_ip_, ":0");

  // StartServer on an already-running server is a NO-OP -- it does not rebind
  // and does not re-point the service. So if something started this server
  // first, `bind_address` is ignored. The one in-tree path that does is
  // GlobalMemoryPoolingBackend::Create(config, controller), which starts it
  // from the factory using the host of the controller's address. That happens
  // to agree with us, because the controller address is itself composed from
  // store_server_ip_ -- but only by construction, so say so out loud rather
  // than publish an address nobody verified.
  const bool already_running = store_server_->GetGrpcPort() != 0;
  absl::Status status =
      store_server_->StartServer(serving_backend, controller, bind_address);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to start KVCacheStoreServer on " << bind_address
               << ": " << status.message()
               << ". This store cannot serve peers.";
    store_server_ = nullptr;
    owned_store_server_.reset();
    return;
  }
  if (already_running) {
    LOG(WARNING) << "KVCacheStoreServer was already running on port "
                 << store_server_->GetGrpcPort() << "; requested bind address "
                 << bind_address
                 << " was NOT applied and the service keeps the backend and "
                    "controller it was started with.";
  }

  if (store_server_ip_.empty()) {
    LOG(INFO) << "KVCacheStore has no store_server_ip; serving on "
              << bind_address
              << " but publishing nothing -- peers cannot discover this store.";
    return;
  }

  store_server_address_ =
      absl::StrCat(store_server_ip_, ":", store_server_->GetGrpcPort());

  if (registry_client_ == nullptr) {
    LOG(WARNING) << "KVCacheStore listening on " << store_server_address_
                 << " but no global registry is configured; peers cannot "
                    "discover this store.";
    return;
  }

  absl::Status register_status = registry_client_->RegisterStore(
      raiden_id_, store_server_address_,
      raiden_controller_ != nullptr ? raiden_controller_->controller_address()
                                    : "");
  if (!register_status.ok()) {
    LOG(ERROR) << "Failed to publish store address " << store_server_address_
               << " to the global registry: " << register_status.message();
    return;
  }
  registered_in_global_registry_ = true;
  LOG(INFO) << "KVCacheStore published at " << store_server_address_;
}

KVCacheStore::~KVCacheStore() {
  // Stop being discoverable, then stop serving, and do both BEFORE anything the
  // service dereferences is destroyed. `backends_` is declared before
  // `raiden_controller_`, so member destruction would otherwise free the
  // controller while the server is still accepting RPCs that use it.
  if (registered_in_global_registry_ && registry_client_ != nullptr) {
    absl::Status status = registry_client_->UnregisterStore(raiden_id_);
    if (!status.ok()) {
      LOG(WARNING) << "Failed to unpublish store address "
                   << store_server_address_ << ": " << status.message()
                   << ". Peers may dial a dead address until it is replaced.";
    }
    registered_in_global_registry_ = false;
  }
  if (store_server_ != nullptr) {
    // Shutdown() drains in-flight handlers, so no RPC is mid-flight past this
    // point.
    //
    // We shut the server down even when a backend owns it, because the service
    // holds OUR RaidenController in pointers it cannot re-seat -- once we go,
    // it can only answer RPCs with a dangling controller. The consequence is
    // that sharing one backend between two KVCacheStores is not supported: the
    // first store to be destroyed takes the shared server with it.
    store_server_->Shutdown();
    store_server_ = nullptr;
  }

  if (poller_thread_) {
    stop_poller_.store(true);
    if (poller_thread_->joinable()) {
      poller_thread_->join();
    }
  }
  std::vector<tsl::Future<>> futures_to_await;
  {
    absl::MutexLock lock(&mutex_);
    for (auto& state : active_saves_) {
      futures_to_await.push_back(state.future);
    }
    for (auto& state : active_loads_) {
      futures_to_await.push_back(state.future);
    }
  }
  for (auto& fut : futures_to_await) {
    (void)fut.Await();
  }
}

absl::StatusOr<BlockSliceList> KVCacheStore::Lookup(
    const std::vector<std::string>& block_hashes, bool enable_global) {
  LookupOptions options;
  options.max_tier = enable_global ? -1 : 0;

  BlockSliceList accumulated_results;
  accumulated_results.reserve(block_hashes.size());

  size_t start_idx = 0;
  for (size_t tier_idx = 0; tier_idx < backends_.size(); ++tier_idx) {
    if (options.max_tier >= 0 &&
        static_cast<int>(tier_idx) > options.max_tier) {
      break;
    }

    const auto& backend = backends_[tier_idx];
    if (start_idx >= block_hashes.size()) break;
    if (!backend) continue;

    auto res_or = backend->Lookup(
        absl::MakeSpan(block_hashes).subspan(start_idx), options);
    if (!res_or.ok()) {
      if (!accumulated_results.empty() && absl::IsNotFound(res_or.status())) {
        break;
      }
      return res_or.status();
    }

    const auto& res = res_or.value();
    for (const auto& pair : res) {
      accumulated_results.push_back(pair);
      ++start_idx;
    }
  }

  if (enable_global && registry_client_ &&
      accumulated_results.size() < block_hashes.size()) {
    std::vector<std::string> missing_hashes(
        block_hashes.begin() + accumulated_results.size(), block_hashes.end());
    auto global_res_or = registry_client_->Lookup(missing_hashes);
    if (global_res_or.ok()) {
      const auto& global_res = global_res_or.value();
      for (size_t i = 0; i < global_res.size(); ++i) {
        const auto& item = global_res[i];
        const auto& proto_id = item.raiden_id();
        RaidenId remote_id{
            .job_name = proto_id.job_name(),
            .job_replica_id = proto_id.job_replica_id(),
            .data_name = proto_id.data_name(),
            .data_replica_idx = proto_id.data_replica_idx(),
        };
        accumulated_results.push_back(std::make_pair(
            missing_hashes[i],
            RaidenBlockID(remote_id, item.block_id(), BlockStatus::REMOTE)));
      }
    }
  }

  size_t cap = capacity();
  if (cap > 0 && accumulated_results.size() > cap) {
    accumulated_results.resize(cap);
  }

  return accumulated_results;
}

std::pair<bool, BlockSliceList> KVCacheStore::Insert(
    const std::vector<std::string>& block_hashes,
    const std::vector<RaidenBlockID>& slices, bool on_host) {
  if (backends_.empty()) {
    return std::make_pair(false, BlockSliceList{});
  }
  std::pair<bool, BlockSliceList> primary_result =
      backends_[0]->Insert(block_hashes, slices, on_host);
  for (size_t i = 1; i < backends_.size(); ++i) {
    if (backends_[i]) {
      backends_[i]->Insert(block_hashes, slices, on_host);
    }
  }
  return primary_result;
}

bool KVCacheStore::InsertAndLock(const std::vector<std::string>& block_hashes,
                                 const std::vector<RaidenBlockID>& slices,
                                 bool on_host) {
  if (backends_.empty()) return false;

  std::vector<size_t> locked_backends;
  for (size_t i = 0; i < backends_.size(); ++i) {
    if (!backends_[i]) continue;
    if (!backends_[i]->InsertAndLock(block_hashes, slices, on_host)) {
      for (size_t lb : locked_backends) {
        backends_[lb]->ReleaseAndDelete(block_hashes);
      }
      return false;
    }
    locked_backends.push_back(i);
  }
  return true;
}

size_t KVCacheStore::ReleaseAndDelete(
    const std::vector<std::string>& block_hashes) {
  size_t total_deleted = 0;
  for (auto& backend : backends_) {
    if (backend) {
      total_deleted += backend->ReleaseAndDelete(block_hashes);
    }
  }
  return total_deleted;
}

void KVCacheStore::Delete(const std::vector<std::string>& block_hashes,
                          const std::vector<RaidenBlockID>& slices) {
  for (auto& backend : backends_) {
    if (backend) {
      backend->Delete(block_hashes, slices);
    }
  }
}

bool KVCacheStore::Pin(const std::vector<std::string>& block_hashes) {
  if (backends_.empty()) return false;
  if (backends_.size() == 1) return backends_[0]->Pin(block_hashes);

  if (backends_[0]->Pin(block_hashes)) {
    return true;
  }

  std::vector<std::vector<std::string>> backend_hashes(backends_.size());
  std::vector<std::string> remaining = block_hashes;

  for (size_t i = 0; i < backends_.size() && !remaining.empty(); ++i) {
    if (!backends_[i]) continue;
    auto lookup_or =
        backends_[i]->Lookup(remaining, LookupOptions{.max_tier = -1});
    if (lookup_or.ok()) {
      const auto& matched = lookup_or.value();
      for (const auto& pair : matched) {
        backend_hashes[i].push_back(pair.first);
      }
      remaining.erase(remaining.begin(), remaining.begin() + matched.size());
    }
  }

  if (!remaining.empty()) {
    return false;
  }

  std::vector<size_t> pinned_backends;
  for (size_t i = 0; i < backends_.size(); ++i) {
    if (!backend_hashes[i].empty()) {
      if (!backends_[i]->Pin(backend_hashes[i])) {
        for (size_t pb : pinned_backends) {
          backends_[pb]->Release(backend_hashes[pb]);
        }
        return false;
      }
      pinned_backends.push_back(i);
    }
  }

  return true;
}

void KVCacheStore::Release(const std::vector<std::string>& block_hashes) {
  for (auto& backend : backends_) {
    if (backend) backend->Release(block_hashes);
  }
}

int KVCacheStore::GetPinCount(const std::string& hash) const {
  for (const auto& backend : backends_) {
    if (backend) {
      int count = backend->GetPinCount(hash);
      if (count > 0) return count;
    }
  }
  return 0;
}

size_t KVCacheStore::capacity() const {
  return (!backends_.empty() && backends_[0]) ? backends_[0]->GetCapacity() : 0;
}

std::string KVCacheStore::raiden_controller_address() const {
  if (raiden_controller_) {
    return raiden_controller_->controller_address();
  }
  return "";
}

size_t KVCacheStore::num_registered_workers() const {
  if (!raiden_controller_) {
    return 0;
  }
  auto registry = raiden_controller_->worker_registry();
  if (registry == nullptr) {
    return 0;
  }
  return registry->GetRegisteredWorkers().size();
}

absl::Status KVCacheStore::Save(const std::vector<std::string>& block_hashes) {
  if (!raiden_controller_) {
    return absl::FailedPreconditionError("RaidenController is not initialized");
  }

  std::vector<int64_t> src_device_block_ids;
  src_device_block_ids.reserve(block_hashes.size());

  {
    absl::MutexLock lock(&mutex_);
    auto lookup_or = backend()->Lookup(block_hashes);
    if (!lookup_or.ok()) return lookup_or.status();
    const auto& slices = lookup_or.value();
    if (slices.size() < block_hashes.size()) {
      return absl::NotFoundError(
          absl::StrCat("Block hash not found: ", block_hashes[slices.size()]));
    }
    for (size_t i = 0; i < slices.size(); ++i) {
      const auto& hash = block_hashes[i];
      const auto& existing = slices[i].second;
      if (existing.status != BlockStatus::HBM) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is not in HBM status: ", hash));
      }
      if (existing.device_block_id == -1) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block device_block_id is -1: ", hash));
      }
      if (backend()->GetPinCount(hash) <= 0) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is not pinned: ", hash));
      }
      if (saving_hashes_.contains(hash)) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is already saving: ", hash));
      }
      src_device_block_ids.push_back(existing.device_block_id);
    }
    for (const auto& hash : block_hashes) {
      saving_hashes_.insert(hash);
    }
  }

  auto host_blocks_or = AllocateBlockIds(block_hashes.size());
  if (!host_blocks_or.ok()) {
    absl::MutexLock lock(&mutex_);
    for (const auto& hash : block_hashes) {
      saving_hashes_.erase(hash);
    }
    return host_blocks_or.status();
  }
  const auto& host_block_ids = host_blocks_or.value();

  std::vector<Buffer> src_buffers;
  src_buffers.reserve(src_device_block_ids.size());
  for (int64_t id : src_device_block_ids) {
    src_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                             rpc::MEMORY_TYPE_HBM);
  }
  std::vector<Buffer> dst_buffers;
  dst_buffers.reserve(host_block_ids.size());
  for (int id : host_block_ids) {
    dst_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                             rpc::MEMORY_TYPE_DRAM);
  }

  tsl::Future<> future = raiden_controller_->TransferBuffers(
      src_buffers, dst_buffers, /*staging_host_buffers=*/{},
      /*copy_sizes=*/{});

  {
    absl::MutexLock lock(&mutex_);
    active_saves_.push_back(SaveState{
        .future = std::move(future),
        .block_hashes = block_hashes,
        .host_block_ids = host_block_ids,
    });
  }

  return absl::OkStatus();
}

absl::Status KVCacheStore::Load(const std::vector<std::string>& block_hashes,
                                const std::vector<int>& device_block_ids) {
  if (block_hashes.size() != device_block_ids.size()) {
    return absl::InvalidArgumentError(
        "block_hashes and device_block_ids size mismatch");
  }
  if (!raiden_controller_) {
    return absl::FailedPreconditionError("RaidenController is not initialized");
  }

  std::vector<int64_t> src_host_block_ids;
  src_host_block_ids.reserve(block_hashes.size());

  {
    absl::MutexLock lock(&mutex_);
    auto lookup_or = backend()->Lookup(block_hashes);
    if (!lookup_or.ok()) return lookup_or.status();
    const auto& slices = lookup_or.value();
    if (slices.size() < block_hashes.size()) {
      return absl::NotFoundError(
          absl::StrCat("Block hash not found: ", block_hashes[slices.size()]));
    }
    for (size_t i = 0; i < slices.size(); ++i) {
      const auto& hash = block_hashes[i];
      const auto& existing = slices[i].second;
      if (existing.status != BlockStatus::HOST &&
          existing.status != BlockStatus::HOST_AND_HBM) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is not on host: ", hash));
      }
      if (existing.host_block_id == -1) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block host_block_id is -1: ", hash));
      }
      if (backend()->GetPinCount(hash) <= 0) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is not pinned: ", hash));
      }
      if (loading_hashes_.contains(hash)) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is already loading: ", hash));
      }
      src_host_block_ids.push_back(existing.host_block_id);
    }
    for (const auto& hash : block_hashes) {
      loading_hashes_.insert(hash);
    }
  }

  std::vector<Buffer> src_buffers;
  src_buffers.reserve(src_host_block_ids.size());
  for (int64_t id : src_host_block_ids) {
    src_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                             rpc::MEMORY_TYPE_DRAM);
  }
  std::vector<Buffer> dst_buffers;
  dst_buffers.reserve(device_block_ids.size());
  for (int id : device_block_ids) {
    dst_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                             rpc::MEMORY_TYPE_HBM);
  }

  tsl::Future<> future = raiden_controller_->TransferBuffers(
      src_buffers, dst_buffers, /*staging_host_buffers=*/{},
      /*copy_sizes=*/{});

  {
    absl::MutexLock lock(&mutex_);
    active_loads_.push_back(LoadState{
        .future = std::move(future),
        .block_hashes = block_hashes,
        .device_block_ids = device_block_ids,
    });
  }

  return absl::OkStatus();
}

void KVCacheStore::RegisterReadRemoteHooks() {
  if (!raiden_controller_) {
    return;
  }
  raiden_controller_->SetReadRemoteHooks(
      [this](absl::Span<const std::string> hashes) {
        return this->ValidateAndPinHostBlocks(hashes);
      },
      [this](absl::Span<const std::string> hashes) {
        this->UnpinHostBlocks(hashes);
      });
}

absl::StatusOr<std::vector<int32_t>> KVCacheStore::ValidateAndPinHostBlocks(
    absl::Span<const std::string> block_hashes) {
  absl::MutexLock lock(&mutex_);
  auto lookup_or = backend()->Lookup(block_hashes);
  if (!lookup_or.ok()) return lookup_or.status();
  const auto& slices = lookup_or.value();
  if (slices.size() < block_hashes.size()) {
    return absl::NotFoundError(
        absl::StrCat("BLOCK_HASH_NOT_FOUND: ", block_hashes[slices.size()]));
  }

  std::vector<int32_t> src_host_block_ids;
  src_host_block_ids.reserve(slices.size());
  for (size_t i = 0; i < slices.size(); ++i) {
    const auto& existing = slices[i].second;
    if (existing.status != BlockStatus::HOST &&
        existing.status != BlockStatus::HOST_AND_HBM) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Block not resident in host DRAM (status=",
          static_cast<int>(existing.status), "): ", block_hashes[i]));
    }
    src_host_block_ids.push_back(existing.host_block_id);
  }

  if (!backend()->Pin(block_hashes)) {
    return absl::InternalError("Failed to pin host blocks");
  }

  return src_host_block_ids;
}

void KVCacheStore::UnpinHostBlocks(absl::Span<const std::string> block_hashes) {
  backend()->Release(block_hashes);
}

absl::Status KVCacheStore::ReadRemote(
    const std::vector<std::string>& block_hashes,
    const std::vector<int32_t>& device_block_ids) {
  if (block_hashes.empty()) {
    return absl::OkStatus();
  }

  std::vector<std::string> successfully_marked_as_reading;
  successfully_marked_as_reading.reserve(block_hashes.size());
  auto cleanup = absl::MakeCleanup([this, &successfully_marked_as_reading]() {
    absl::MutexLock lock(&mutex_);
    for (const auto& hash : successfully_marked_as_reading) {
      reading_hashes_.erase(hash);
    }
  });

  auto host_blocks_or = AllocateBlockIds(block_hashes.size());
  if (!host_blocks_or.ok()) {
    return host_blocks_or.status();
  }

  const bool to_hbm = !device_block_ids.empty();
  if (to_hbm && device_block_ids.size() != block_hashes.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "device_block_ids size ", device_block_ids.size(),
        " must be empty (read to host) or match block_hashes size ",
        block_hashes.size()));
  }
  std::vector<int> dst_host_block_ids = host_blocks_or.value();

  struct RemoteReadGroup {
    RaidenId src_raiden_id;
    std::vector<int32_t> src_host_block_ids;
    std::vector<int32_t> dst_host_block_ids;
    std::vector<std::string> block_hashes;
    std::vector<int32_t> device_block_ids;
  };
  std::vector<RemoteReadGroup> groups;

  {
    absl::MutexLock lock(&mutex_);
    auto lookup_or = backend()->Lookup(block_hashes);
    if (!lookup_or.ok()) return lookup_or.status();
    const auto& slices = lookup_or.value();
    if (slices.size() < block_hashes.size()) {
      return absl::NotFoundError(
          absl::StrCat("Block hash not found: ", block_hashes[slices.size()]));
    }
    for (size_t i = 0; i < slices.size(); ++i) {
      const auto& hash = block_hashes[i];
      if (!reading_hashes_.insert(hash).second) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is already reading remote: ", hash));
      }
      successfully_marked_as_reading.push_back(hash);

      const auto& src_id = slices[i].second.raiden_id;
      auto it = std::find_if(groups.begin(), groups.end(),
                             [&src_id](const RemoteReadGroup& g) {
                               return g.src_raiden_id == src_id;
                             });
      if (it == groups.end()) {
        groups.push_back(RemoteReadGroup{.src_raiden_id = src_id});
        it = groups.end() - 1;
      }
      it->src_host_block_ids.push_back(slices[i].second.host_block_id);
      it->dst_host_block_ids.push_back(dst_host_block_ids[i]);
      it->block_hashes.push_back(hash);
      if (to_hbm) {
        it->device_block_ids.push_back(device_block_ids[i]);
      }
    }
  }

  if (!raiden_controller_) {
    DeallocateBlockIds(dst_host_block_ids);
    return absl::FailedPreconditionError(
        "RaidenController is not initialized for ReadRemote");
  }

  // NOTE: the landing block ids are deliberately NOT stamped into the LRU
  // entries here. The entry's host_block_id is the PEER's coordinate until the
  // read commits; overwriting it up front (as this code used to) corrupts the
  // entry on every failure path, because nothing restores it.
  // One lease per owning peer. The per-group futures are joined, so if ANY
  // group fails -- transfer error or a verdict other than HELD -- the whole
  // batch discards, including groups whose bytes landed perfectly. That is
  // fail-closed and consistent with the commit-as-a-unit invariant. Committing
  // only the healthy groups would need per-group RemoteReadState and is
  // exactly where a partial-promote bug would enter; do not "optimise" it
  // without splitting the state first.
  std::vector<tsl::Future<>> futures;
  futures.reserve(groups.size());
  for (const auto& group : groups) {
    futures.push_back(raiden_controller_->ReadRemote(
        group.src_raiden_id, group.src_host_block_ids, group.dst_host_block_ids,
        group.block_hashes, group.device_block_ids));
  }

  tsl::Future<> combined_future;
  if (futures.size() == 1) {
    combined_future = std::move(futures[0]);
  } else {
    combined_future = tsl::JoinFutures(futures);
  }

  {
    absl::MutexLock lock(&mutex_);
    active_remote_reads_.emplace(std::move(combined_future),
                                 RemoteReadState{
                                     .block_hashes = block_hashes,
                                     .host_block_ids = dst_host_block_ids,
                                     .device_block_ids = device_block_ids,
                                 });
  }

  successfully_marked_as_reading.clear();
  return absl::OkStatus();
}

std::tuple<std::vector<std::string>, std::vector<std::string>,
           std::vector<std::string>>
KVCacheStore::PollSaveStatus() {
  PollFuturesInternal();
  absl::MutexLock lock(&mutex_);
  std::vector<std::string> pending;
  for (const auto& state : active_saves_) {
    for (const auto& hash : state.block_hashes) {
      pending.push_back(hash);
    }
  }
  std::vector<std::string> done = std::move(done_saves_);
  std::vector<std::string> failed = std::move(failed_saves_);
  done_saves_.clear();
  failed_saves_.clear();
  return std::make_tuple(done, failed, pending);
}

std::tuple<std::vector<std::string>, std::vector<std::string>,
           std::vector<std::string>>
KVCacheStore::PollLoadStatus() {
  PollFuturesInternal();
  absl::MutexLock lock(&mutex_);
  std::vector<std::string> pending;
  for (const auto& state : active_loads_) {
    for (const auto& hash : state.block_hashes) {
      pending.push_back(hash);
    }
  }
  std::vector<std::string> done = std::move(done_loads_);
  std::vector<std::string> failed = std::move(failed_loads_);
  done_loads_.clear();
  failed_loads_.clear();
  return std::make_tuple(done, failed, pending);
}

std::tuple<std::vector<std::string>, std::vector<std::string>,
           std::vector<std::string>>
KVCacheStore::PollRemoteReadStatus() {
  PollFuturesInternal();
  absl::MutexLock lock(&mutex_);
  std::vector<std::string> pending;
  for (const auto& [fut, state] : active_remote_reads_) {
    for (const auto& hash : state.block_hashes) {
      pending.push_back(hash);
    }
  }
  std::vector<std::string> done = std::move(done_remote_reads_);
  std::vector<std::string> failed = std::move(failed_remote_reads_);
  done_remote_reads_.clear();
  failed_remote_reads_.clear();
  return std::make_tuple(done, failed, pending);
}

absl::StatusOr<size_t> KVCacheStore::RecoverFromLocalManifest() {
  if (!raiden_controller_) {
    return absl::FailedPreconditionError(
        "RaidenController is required for crash recovery");
  }
  if (backend()->GetSize() > 0) {
    return absl::FailedPreconditionError(
        "RecoverFromLocalManifest can only be called on an empty cache store");
  }
  return backend()->RecoverFromLocalManifest();
}

size_t KVCacheStore::Evict(const std::vector<std::string>& block_hashes) {
  if (block_hashes.empty()) {
    return 0;
  }
  std::vector<int> host_ids_to_deallocate;
  {
    absl::MutexLock l(&mutex_);
    host_ids_to_deallocate = backend()->Evict(block_hashes);
  }

  if (host_ids_to_deallocate.empty()) {
    return 0;
  }

  if (registry_client_) {
    auto status = registry_client_->Unregister(block_hashes, raiden_id_);
    if (!status.ok()) {
      LOG(WARNING) << "Failed to unregister proactively evicted blocks: "
                   << status.message();
    }
  }

  DeallocateBlockIds(host_ids_to_deallocate);

  return host_ids_to_deallocate.size();
}

absl::StatusOr<std::vector<int>> KVCacheStore::AllocateBlockIds(int needed) {
  std::vector<std::string> hashes_to_deallocate;
  {
    absl::MutexLock l(&mutex_);
    int free_count = raiden_controller_->block_manager()->num_free_blocks();
    int to_free = needed - free_count;
    if (to_free > 0) {
      hashes_to_deallocate = backend()->GetEvictableKeys(to_free);
      if (hashes_to_deallocate.size() < static_cast<size_t>(to_free)) {
        return absl::ResourceExhaustedError(
            absl::StrCat("Insufficient free blocks and not enough evictable "
                         "blocks. Needed: ",
                         needed, ", Free: ", free_count,
                         ", Evictable: ", hashes_to_deallocate.size()));
      }
    }
  }

  if (!hashes_to_deallocate.empty()) {
    Evict(hashes_to_deallocate);
  }

  return raiden_controller_->AllocateBlockIds(needed);
}

void KVCacheStore::DeallocateBlockIds(absl::Span<const int> block_ids) {
  if (raiden_controller_) {
    auto status = raiden_controller_->DeallocateBlockIds(block_ids);
    if (!status.ok()) {
      LOG(WARNING) << "Failed to deallocate host block IDs: "
                   << status.message();
    }
  }
}

void KVCacheStore::PollerLoop() {
  while (!stop_poller_.load()) {
    PollFuturesInternal();
    absl::SleepFor(absl::Milliseconds(10));
  }
}

void KVCacheStore::PollSavesInternal(std::vector<SaveState> ready_saves) {
  for (auto& state : ready_saves) {
    absl::Status status = state.future.Await();
    absl::MutexLock lock(&mutex_);
    if (status.ok()) {
      std::vector<global_registry::Registration> write_through_regs;
      write_through_regs.reserve(state.block_hashes.size());
      auto lookup_or = backend()->Lookup(state.block_hashes);
      if (lookup_or.ok()) {
        const auto& slices = lookup_or.value();
        std::vector<std::string> update_hashes;
        std::vector<RaidenBlockID> update_slices;
        for (size_t i = 0; i < state.block_hashes.size(); ++i) {
          const auto& hash = state.block_hashes[i];
          if (i < slices.size()) {
            RaidenBlockID block = slices[i].second;
            block.host_block_id = state.host_block_ids[i];
            block.status = BlockStatus::HOST_AND_HBM;
            update_hashes.push_back(hash);
            update_slices.push_back(block);
            if (registry_client_) {
              write_through_regs.push_back({
                  .prefix_hash = hash,
                  .raiden_id = raiden_id_,
                  .block_id = state.host_block_ids[i],
              });
            }
          }
          done_saves_.push_back(hash);
        }
        if (!update_hashes.empty()) {
          backend()->Insert(update_hashes, update_slices, /*on_host=*/true);
        }
      } else {
        DeallocateBlockIds(state.host_block_ids);
      }
      if (!write_through_regs.empty() && registry_client_ &&
          write_through_pool_) {
        write_through_pool_->Schedule([client = registry_client_,
                                       regs = std::move(write_through_regs)]() {
          auto status = client->Register(regs);
          if (!status.ok()) {
            LOG(WARNING) << "Async write-through failed after Save: "
                         << status.message();
          } else {
            LOG(INFO) << "Async write-through succeeded after Save for "
                      << regs.size() << " blocks";
          }
        });
      }
    } else {
      LOG(ERROR) << "Async Save failed: " << status.ToString();
      DeallocateBlockIds(state.host_block_ids);
      for (const auto& hash : state.block_hashes) {
        failed_saves_.push_back(hash);
      }
    }
    for (const auto& hash : state.block_hashes) {
      saving_hashes_.erase(hash);
    }
  }
}

void KVCacheStore::PollLoadsInternal(std::vector<LoadState> ready_loads) {
  for (auto& state : ready_loads) {
    absl::Status status = state.future.Await();
    absl::MutexLock lock(&mutex_);
    if (status.ok()) {
      auto lookup_or = backend()->Lookup(state.block_hashes);
      if (lookup_or.ok()) {
        const auto& slices = lookup_or.value();
        std::vector<std::string> update_hashes;
        std::vector<RaidenBlockID> update_slices;
        for (size_t i = 0; i < state.block_hashes.size(); ++i) {
          const auto& hash = state.block_hashes[i];
          if (i < slices.size()) {
            RaidenBlockID block = slices[i].second;
            block.device_block_id = state.device_block_ids[i];
            block.status = BlockStatus::HOST_AND_HBM;
            update_hashes.push_back(hash);
            update_slices.push_back(block);
          }
          done_loads_.push_back(hash);
        }
        if (!update_hashes.empty()) {
          backend()->Insert(update_hashes, update_slices, /*on_host=*/true);
        }
      }
    } else {
      LOG(ERROR) << "Async Load failed: " << status.ToString();
      for (const auto& hash : state.block_hashes) {
        failed_loads_.push_back(hash);
      }
    }
    for (const auto& hash : state.block_hashes) {
      loading_hashes_.erase(hash);
    }
  }
}

void KVCacheStore::PollRemoteReadsInternal(
    std::vector<std::pair<tsl::Future<>, RemoteReadState>> ready_remote_reads) {
  for (auto& [future, state] : ready_remote_reads) {
    absl::Status status = future.Await();
    absl::MutexLock lock(&mutex_);

    // The batch commits as a UNIT: all hashes promoted, or none. Verify every
    // entry is still present and still pinned BEFORE promoting anything.
    //
    // With pinned entries protected from erase, an entry can only vanish
    // mid-read if the caller broke the contract by releasing its pin early --
    // so this is a bug detector, not a race handler. It replaces a loop that
    // pushed a vanished hash onto done_remote_reads_ anyway, which told the
    // caller "resident in HOST" about a landing block that had just been
    // deallocated and reused.
    std::vector<std::string> contract_violations;
    if (status.ok()) {
      for (const auto& hash : state.block_hashes) {
        if (backend()->GetPinCount(hash) <= 0) {
          contract_violations.push_back(hash);
        }
      }
      if (!contract_violations.empty()) {
        status = absl::FailedPreconditionError(absl::StrCat(
            "read_remote caller released or deleted ", contract_violations.size(),
            " of ", state.block_hashes.size(),
            " entries before poll_remote_read_status reported them terminal; "
            "the whole batch is discarded"));
        LOG(ERROR) << status.message() << " First offending hash: "
                   << absl::BytesToHexString(contract_violations.front());
      }
    }

    if (status.ok()) {
      std::vector<global_registry::Registration> write_through_regs;
      write_through_regs.reserve(state.block_hashes.size());
      auto lookup_or = backend()->Lookup(state.block_hashes);
      if (lookup_or.ok()) {
        const auto& slices = lookup_or.value();
        std::vector<std::string> update_hashes;
        std::vector<RaidenBlockID> update_slices;
        for (size_t i = 0; i < state.block_hashes.size(); ++i) {
          const auto& hash = state.block_hashes[i];
          if (i < slices.size()) {
            RaidenBlockID block = slices[i].second;
            block.host_block_id = state.host_block_ids[i];
            if (i < state.device_block_ids.size()) {
              // Read-to-HBM: the bytes are in the caller's device blocks AND in
              // the host landing blocks (which were the staging hop), so a
              // later local load() can still reuse the host copy.
              block.device_block_id = state.device_block_ids[i];
              block.status = BlockStatus::HOST_AND_HBM;
            } else {
              block.status = BlockStatus::HOST;
            }
            update_hashes.push_back(hash);
            update_slices.push_back(block);
            if (registry_client_) {
              write_through_regs.push_back({
                  .prefix_hash = hash,
                  .raiden_id = raiden_id_,
                  .block_id = state.host_block_ids[i],
              });
            }
          } else {
            DeallocateBlockIds({state.host_block_ids[i]});
          }
          done_remote_reads_.push_back(hash);
        }
        if (!update_hashes.empty()) {
          backend()->Insert(update_hashes, update_slices, /*on_host=*/true);
        }
      } else {
        DeallocateBlockIds(state.host_block_ids);
      }
      if (!write_through_regs.empty() && registry_client_ &&
          write_through_pool_) {
        write_through_pool_->Schedule([client = registry_client_,
                                       regs = std::move(write_through_regs)]() {
          auto status = client->Register(regs);
          if (!status.ok()) {
            LOG(WARNING) << "Async write-through failed after ReadRemote: "
                         << status.message();
          } else {
            LOG(INFO) << "Async write-through succeeded after ReadRemote for "
                      << regs.size() << " blocks";
          }
        });
      }
    } else {
      // Discard path. The entry keeps its ORIGINAL peer host_block_id (never
      // stamped at issue time), so a retry is clean. In read-to-HBM mode the
      // caller's device blocks may hold garbage -- by design: nothing in the
      // LRU points at them, and the caller overwrites device blocks on reuse.
      LOG(WARNING) << "Async ReadRemote failed: " << status.ToString();
      DeallocateBlockIds(state.host_block_ids);
      for (const auto& hash : state.block_hashes) {
        failed_remote_reads_.push_back(hash);
      }
    }
    for (const auto& hash : state.block_hashes) {
      reading_hashes_.erase(hash);
    }
  }
}


void KVCacheStore::PollFuturesInternal() {
  std::vector<SaveState> ready_saves;
  std::vector<LoadState> ready_loads;
  std::vector<std::pair<tsl::Future<>, RemoteReadState>> ready_remote_reads;

  {
    absl::MutexLock lock(&mutex_);
    auto it = active_saves_.begin();
    while (it != active_saves_.end()) {
      if (it->future.IsReady()) {
        ready_saves.push_back(std::move(*it));
        it = active_saves_.erase(it);
      } else {
        ++it;
      }
    }

    auto jt = active_loads_.begin();
    while (jt != active_loads_.end()) {
      if (jt->future.IsReady()) {
        ready_loads.push_back(std::move(*jt));
        jt = active_loads_.erase(jt);
      } else {
        ++jt;
      }
    }

    auto kt = active_remote_reads_.begin();
    while (kt != active_remote_reads_.end()) {
      if (kt->first.IsReady()) {
        ready_remote_reads.push_back({kt->first, std::move(kt->second)});
        active_remote_reads_.erase(kt++);
      } else {
        ++kt;
      }
    }
  }

  PollSavesInternal(std::move(ready_saves));
  PollLoadsInternal(std::move(ready_loads));
  PollRemoteReadsInternal(std::move(ready_remote_reads));
}

}  // namespace kv_cache
}  // namespace tpu_raiden
