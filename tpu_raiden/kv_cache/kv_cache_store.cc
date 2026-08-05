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
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <tuple>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
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

// Every store has a
// controller, and store_server_ip is mandatory and never a wildcard -- it is
// the host of the controller address registered with the orchestrator, so it
// must be routable by peers. Hostnames are allowed (same-host tests use
// "localhost"); empty and wildcard are not.
absl::Status ValidateConstructionRules(absl::string_view store_server_ip,
                                       int num_shards) {
  if (store_server_ip.empty()) {
    return absl::InvalidArgumentError(
        "store_server_ip is required: it is the host peers and the "
        "orchestrator use to reach this store. Same-host use: \"127.0.0.1\".");
  }
  if (store_server_ip == "[::]" || store_server_ip == "::" ||
      store_server_ip == "0.0.0.0" ||
      store_server_ip == "0:0:0:0:0:0:0:0") {
    return absl::InvalidArgumentError(absl::StrCat(
        "store_server_ip may not be a wildcard (got \"", store_server_ip,
        "\"): a wildcard binds but cannot be published or dialled."));
  }
  if (num_shards < 1) {
    return absl::InvalidArgumentError(
        "num_shards must be >= 1: every KVCacheStore has a RaidenController; "
        "the controller-less configuration is not supported.");
  }
  return absl::OkStatus();
}

// ValidateBackends checks that backends_ has at least a tier-0 backend and it
// is not null. Catching this at construction guarantees that a store configured
// with a registry is always registered, by preventing a backend-less store from
// being built: previously such a store constructed, warned, and quietly
// declined to serve or publish.
//
// Validated the way every other construction rule fails -- before any
// resource exists, Status from Create() and abort from the raw constructors.
absl::Status ValidateBackends(
    const std::vector<std::shared_ptr<KVCacheStoreBackend>>& backends) {
  if (backends.empty()) {
    return absl::InvalidArgumentError(
        "KVCacheStore requires at least one backend (tier 0, local host "
        "DRAM).");
  }
  if (backends[0] == nullptr) {
    return absl::InvalidArgumentError(
        "KVCacheStore's tier-0 backend must not be null.");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<KVCacheStore>> KVCacheStore::Create(
    absl::Span<const BackendConfig> backend_configs, size_t capacity,
    absl::string_view global_registry_address, RaidenId raiden_id,
    int num_shards, int64_t shard_size_bytes,
    absl::string_view raiden_orchestrator_address,
    absl::string_view store_server_ip, int raiden_controller_port,
    std::optional<KVCacheMetadata> metadata, int expected_worker_count) {
  if (backend_configs.empty()) {
    return absl::InvalidArgumentError("backend_configs must not be empty");
  }
  // Before any resource is created (violation must not leak).
  RETURN_IF_ERROR(ValidateConstructionRules(store_server_ip, num_shards));

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
    // A custom registration may return OK with a null pointer; that would sail
    // past ValidateBackends for any tier but 0, and crash later.
    if (backend == nullptr) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Backend factory returned a null backend for tier ", i, "."));
    }
    backends.push_back(std::move(backend));
  }

  RETURN_IF_ERROR(ValidateBackends(backends));

  RaidenId effective_raiden_id = !backend_configs[0].raiden_id.empty()
                                     ? backend_configs[0].raiden_id
                                     : raiden_id;

  // CreateTag: the constructor itself must not wire the controller (and
  // FATAL on failure) -- that would defeat Create()'s whole purpose of
  // returning a recoverable Status. Create() does the wiring below instead.
  auto store = absl::WrapUnique(new KVCacheStore(
      std::move(backends), effective_raiden_id, num_shards, shard_size_bytes,
      raiden_orchestrator_address, store_server_ip, raiden_controller_port,
      global_registry_address, expected_worker_count,
      KVCacheStore::CreateTag{}));

  if (store->raiden_controller_ != nullptr) {
    RETURN_IF_ERROR(
        store->SetRaidenController(store->raiden_controller_.get()));
    store->RegisterReadRemoteHooks();
    store->poller_thread_ =
        std::make_unique<std::thread>(&KVCacheStore::PollerLoop, store.get());
  }

  return store;
}

absl::StatusOr<std::unique_ptr<KVCacheStore>> KVCacheStore::Create(
    const BackendConfig& config, size_t capacity,
    absl::string_view global_registry_address, RaidenId raiden_id,
    int num_shards, int64_t shard_size_bytes,
    absl::string_view raiden_orchestrator_address,
    absl::string_view store_server_ip, int raiden_controller_port,
    std::optional<KVCacheMetadata> metadata, int expected_worker_count) {
  return KVCacheStore::Create(absl::MakeConstSpan(&config, 1), capacity,
                              global_registry_address, raiden_id, num_shards,
                              shard_size_bytes, raiden_orchestrator_address,
                              store_server_ip, raiden_controller_port,
                              std::move(metadata), expected_worker_count);
}

KVCacheStore::KVCacheStore(
    std::vector<std::shared_ptr<KVCacheStoreBackend>> backends,
    RaidenId raiden_id, int num_shards, int64_t shard_size_bytes,
    absl::string_view raiden_orchestrator_address,
    absl::string_view store_server_ip, int raiden_controller_port,
    absl::string_view global_registry_address, int expected_worker_count,
    CreateTag)
    : backends_(std::move(backends)),
      raiden_id_(std::move(raiden_id)),
      store_server_ip_(store_server_ip),
      write_through_pool_(std::make_unique<::tpu_raiden::NumaThreadPool>(4)) {
  if (absl::Status v = ValidateConstructionRules(store_server_ip,
                                                 num_shards);
      !v.ok()) {
    LOG(FATAL) << "KVCacheStore construction validation failed: " << v.message()
               << " Use KVCacheStore::Create() for a recoverable error.";
  }
  // Both public backend-taking constructors delegate here, so this is the one
  // place that has to check. Create() checks separately, before constructing,
  // so it can return the error instead of aborting.
  if (absl::Status v = ValidateBackends(backends_); !v.ok()) {
    LOG(FATAL) << "KVCacheStore construction rules violated: " << v.message()
               << " Use KVCacheStore::Create() for a recoverable error.";
  }
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
    // The store's block ids resolve against the manager's own pool, so the
    // legacy physical-buffer pre-provisioning would allocate that pool a
    // second time on the worker.
    raiden_controller_ =
        std::make_unique<::tpu_raiden::controller::RaidenController>(
            unit_proto, cap, num_shards, shard_size_bytes,
            raiden_orchestrator_address,
            ComposeControllerAddress(store_server_ip, raiden_controller_port),
            /*preprovision_worker_buffers=*/false, expected_worker_count);
  }
  // Deliberately does NOT wire the controller here (see CreateTag) -- the
  // caller (Create()) does that itself so a publish failure can return a
  // Status instead of aborting the process.
}

KVCacheStore::KVCacheStore(
    std::vector<std::shared_ptr<KVCacheStoreBackend>> backends,
    RaidenId raiden_id, int num_shards, int64_t shard_size_bytes,
    absl::string_view raiden_orchestrator_address,
    absl::string_view store_server_ip, int raiden_controller_port,
    absl::string_view global_registry_address, int expected_worker_count)
    : KVCacheStore(std::move(backends), std::move(raiden_id), num_shards,
                   shard_size_bytes, raiden_orchestrator_address,
                   store_server_ip, raiden_controller_port,
                   global_registry_address, expected_worker_count,
                   CreateTag{}) {
  if (raiden_controller_) {
    if (absl::Status s = SetRaidenController(raiden_controller_.get());
        !s.ok()) {
      LOG(FATAL) << "KVCacheStore failed to start/publish its store server: "
                 << s.message()
                 << " Use KVCacheStore::Create() for a recoverable error.";
    }
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
                           absl::string_view global_registry_address,
                           int expected_worker_count)
    : KVCacheStore(
          std::vector<std::shared_ptr<KVCacheStoreBackend>>{std::move(backend)},
          std::move(raiden_id), num_shards, shard_size_bytes,
          raiden_orchestrator_address, store_server_ip, raiden_controller_port,
          global_registry_address, expected_worker_count) {}

KVCacheStore::KVCacheStore(
    size_t capacity, absl::string_view global_registry_address,
    RaidenId raiden_id, int num_shards, int64_t shard_size_bytes,
    absl::string_view raiden_orchestrator_address,
    absl::string_view store_server_ip, int raiden_controller_port,
    std::optional<KVCacheMetadata> metadata, int expected_worker_count)
    : raiden_id_(raiden_id),
      store_server_ip_(store_server_ip),
      write_through_pool_(std::make_unique<::tpu_raiden::NumaThreadPool>(4)) {
  if (absl::Status v = ValidateConstructionRules(store_server_ip,
                                                 num_shards);
      !v.ok()) {
    LOG(FATAL) << "KVCacheStore construction validation failed: " << v.message()
               << " Use KVCacheStore::Create() for a recoverable error.";
  }
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
            ComposeControllerAddress(store_server_ip, raiden_controller_port),
            /*preprovision_worker_buffers=*/false, expected_worker_count);
  }

  // registry_client_ is assigned a few lines above, and must be: a backend
  // built without it publishes nothing and, more importantly, cannot RESOLVE a
  // peer -- peer resolution runs through the BACKEND's registry client, not
  // this store's. Leaving it null here gave a store that registered itself
  // perfectly and could reach nobody. Do not move this above that assignment.
  backends_ = {std::make_shared<HostOffloadBackend>(
      capacity, std::move(metadata), raiden_id_, raiden_controller_.get(),
      registry_client_)};

  if (raiden_controller_) {
    if (absl::Status s = SetRaidenController(raiden_controller_.get());
        !s.ok()) {
      LOG(FATAL) << "KVCacheStore failed to start/publish its store server: "
                 << s.message()
                 << " Use KVCacheStore::Create() for a recoverable error.";
    }
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
  if (absl::Status v =
          ValidateConstructionRules(store_server_ip, /*num_shards=*/1);
      !v.ok()) {
    LOG(FATAL) << "KVCacheStore construction validation failed: " << v.message();
  }
  if (raiden_controller_ == nullptr) {
    LOG(FATAL) << "KVCacheStore requires a RaidenController; the "
                  "controller-less configuration is not supported. "
                  "(store_server_ip/controller-host consistency is the "
                  "caller's promise on this constructor.)";
  }
  if (!global_registry_address.empty()) {
    auto channel = grpc::CreateChannel(std::string(global_registry_address),
                                       grpc::InsecureChannelCredentials());
    registry_client_ =
        std::make_shared<global_registry::GlobalRegistryClient>(channel);
  }

  // See the same call in the capacity constructor: the backend needs this
  // store's registry client to resolve peers, and registry_client_ is assigned
  // just above. Do not reorder.
  backends_ = {std::make_shared<HostOffloadBackend>(
      capacity, std::move(metadata), raiden_id_, raiden_controller_.get(),
      registry_client_)};

  if (raiden_controller_) {
    if (absl::Status s = SetRaidenController(raiden_controller_.get());
        !s.ok()) {
      LOG(FATAL) << "KVCacheStore failed to start/publish its store server: "
                 << s.message()
                 << " Use KVCacheStore::Create() for a recoverable error.";
    }
    RegisterReadRemoteHooks();
    poller_thread_ =
        std::make_unique<std::thread>(&KVCacheStore::PollerLoop, this);
  }
}

absl::Status KVCacheStore::SetRaidenController(
    tpu_raiden::controller::RaidenController* controller) {
  // Runs first so that store_server_ip_ decides the bind address: StartServer
  // never rebinds a running server, so whoever starts it first wins.
  RETURN_IF_ERROR(EnsureStoreServerAndRegister(controller));

  for (auto& backend : backends_) {
    if (backend != nullptr) {
      backend->SetRaidenController(controller);
    }
  }
  return absl::OkStatus();
}

absl::Status KVCacheStore::EnsureStoreServerAndRegister(
    tpu_raiden::controller::RaidenController* controller) {
  if (store_server_ != nullptr || controller == nullptr) {
    return absl::OkStatus();  // Already done; may be called repeatedly.
  }

  // The global registry decides whether the P2P plane exists at all. No
  // registry client, no server -- not adopted from a backend, not owned by
  // this store. (store_server_ip_ is never empty here: validation
  // makes it mandatory.)
  if (registry_client_ == nullptr) {
    return absl::OkStatus();
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
    // Unreachable: ValidateBackends rejects a store with no tier-0 backend at
    // construction. Kept as a last line of defence, and as an ERROR rather
    // than the warning-and-carry-on it used to be -- returning OK here left a
    // store that has a registry configured but never publishes itself, which
    // reads to every peer as "that node does not exist".
    if (backends_.empty() || backends_[0] == nullptr) {
      return absl::FailedPreconditionError(
          "KVCacheStore has no tier-0 backend; cannot serve or publish.");
    }
    owned_store_server_ = KVCacheStoreServer::Create();
    store_server_ = owned_store_server_.get();
    serving_backend = backends_[0].get();
  }

  // store_server_ip_ is mandatory; the port is always gRPC's choice.
  const std::string bind_address = absl::StrCat(store_server_ip_, ":0");

  absl::Status status =
      store_server_->StartServer(serving_backend, controller, bind_address);
  if (!status.ok()) {
    store_server_ = nullptr;
    owned_store_server_.reset();
    return absl::Status(
        status.code(),
        absl::StrCat("Failed to start KVCacheStoreServer on ", bind_address,
                     ": ", status.message()));
  }

  // The server is the single source of truth for its own address -- it
  // remembers the host it was actually started with. This is empty only if
  // something adopted (bound before this call) it on a wildcard/empty host --
  // unreachable for any in-tree caller, but checked rather than
  // publishing a blank address.
  store_server_address_ = store_server_->GetServerAddress();
  if (store_server_address_.empty()) {
    return absl::FailedPreconditionError(
        "KVCacheStoreServer has no publishable address (bound on a wildcard "
        "or empty host); cannot register with the global registry.");
  }

  absl::Status register_status = registry_client_->RegisterStore(
      raiden_id_, store_server_address_,
      raiden_controller_ != nullptr ? raiden_controller_->controller_address()
                                    : "");
  if (!register_status.ok()) {
    return absl::Status(
        register_status.code(),
        absl::StrCat("Failed to publish store address ",
                     store_server_address_, " to the global registry: ",
                     register_status.message()));
  }
  registered_in_global_registry_ = true;
  LOG(INFO) << "KVCacheStore published at " << store_server_address_;
  return absl::OkStatus();
}

void KVCacheStore::ShutdownBackendStoreServers(
    KVCacheStoreServer* already_shut) {
  for (auto& backend : backends_) {
    if (backend == nullptr) {
      continue;
    }
    KVCacheStoreServer* server = backend->store_server();
    // Shutdown() is idempotent, but skipping the adopted server says why
    // rather than leaving the reader to check.
    if (server != nullptr && server != already_shut) {
      server->Shutdown();
    }
  }
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
    KVCacheStoreServer* already_shut = store_server_;
    store_server_->Shutdown();
    store_server_ = nullptr;

    // Fall through to the sweep below with this one excluded.
    ShutdownBackendStoreServers(already_shut);
  } else {
    // store_server_ is null whenever this store has no registry (see
    // EnsureStoreServerAndRegister), but a backend's server can still be
    // running: StartServer is public, and a caller may start one AFTER
    // construction, which is too late for the adoption above to see it. That
    // server holds OUR controller in a pointer it cannot re-seat, and
    // raiden_controller_ is about to be destroyed, so it has to be stopped
    // here -- in the destructor BODY, while the controller is still alive.
    // Relying on ~HostOffloadBackend is not enough: backends_ holds
    // shared_ptrs, so a caller keeping its own reference outlives this store.
    ShutdownBackendStoreServers(/*already_shut=*/nullptr);
  }

  if (poller_thread_) {
    stop_poller_.store(true);
    if (poller_thread_->joinable()) {
      poller_thread_->join();
    }
  }
  std::vector<tsl::Future<>> futures_to_await;
  {
    absl::MutexLock lock(mutex_);
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
  options.enable_global = enable_global;

  BlockSliceList accumulated_results;
  accumulated_results.reserve(block_hashes.size());

  size_t start_idx = 0;
  for (size_t tier_idx = 0; tier_idx < backends_.size(); ++tier_idx) {
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
    auto lookup_or = backends_[i]->Lookup(remaining, LookupOptions{});
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

absl::Status KVCacheStore::Save(const std::vector<std::string>& block_hashes) {
  if (!raiden_controller_) {
    return absl::FailedPreconditionError("RaidenController is not initialized");
  }

  std::vector<int64_t> src_device_block_ids;
  src_device_block_ids.reserve(block_hashes.size());

  {
    absl::MutexLock lock(mutex_);
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
    absl::MutexLock lock(mutex_);
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
    absl::MutexLock lock(mutex_);
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
  if (block_hashes.empty()) {
    return absl::OkStatus();
  }

  RaidenId remote_id;
  {
    absl::MutexLock lock(mutex_);
    auto lookup_or = backend()->Lookup(block_hashes);
    if (!lookup_or.ok()) return lookup_or.status();
    const auto& slices = lookup_or.value();
    if (slices.size() < block_hashes.size()) {
      return absl::NotFoundError(
          absl::StrCat("Block hash not found: ", block_hashes[slices.size()]));
    }

    BlockStatus first_status = slices[0].second.status;
    if (first_status == BlockStatus::REMOTE) {
      remote_id = slices[0].second.raiden_id;
    }

    for (size_t i = 0; i < slices.size(); ++i) {
      const auto& hash = block_hashes[i];
      const auto& existing = slices[i].second;
      if (backend()->GetPinCount(hash) <= 0) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is not pinned: ", hash));
      }
      if (loading_hashes_.contains(hash)) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is already loading: ", hash));
      }

      if (first_status == BlockStatus::REMOTE) {
        if (existing.status != BlockStatus::REMOTE) {
          return absl::InvalidArgumentError(
              "Mixed block statuses in a single Load call");
        }
        if (existing.raiden_id != remote_id) {
          return absl::InvalidArgumentError(
              "Mixed remote node IDs in a single Load call");
        }
      } else {
        if (existing.status != BlockStatus::HOST &&
            existing.status != BlockStatus::HOST_AND_HBM) {
          return absl::FailedPreconditionError(
              absl::StrCat("Block is not on host: ", hash));
        }
        if (existing.host_block_id == -1) {
          return absl::FailedPreconditionError(
              absl::StrCat("Block host_block_id is -1: ", hash));
        }
      }
    }
    for (const auto& hash : block_hashes) {
      loading_hashes_.insert(hash);
    }
  }

  tsl::Future<> future = backend()->Load(remote_id, block_hashes,
                                         absl::MakeConstSpan(device_block_ids));

  {
    absl::MutexLock lock(mutex_);
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
  absl::MutexLock lock(mutex_);
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
    absl::MutexLock lock(mutex_);
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
    absl::MutexLock lock(mutex_);
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
    absl::MutexLock lock(mutex_);
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
  absl::MutexLock lock(mutex_);
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
  absl::MutexLock lock(mutex_);
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
  absl::MutexLock lock(mutex_);
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
    absl::MutexLock lock(mutex_);
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
    absl::MutexLock lock(mutex_);
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

// ===========================================================================
// Remote write -- source side.
// ===========================================================================

namespace {

// How long this source keeps its blocks intact and keeps asking. The
// destination is asked for HOLD minus a margin, and clamps that to its own
// cap, so the source always outlives the destination's verdict.
constexpr absl::Duration kDefaultRemoteWriteHold = absl::Seconds(30);

// Covers one-way RPC latency, clock skew between the two hosts, and scheduler
// jitter -- none of which this design controls. Not a tuning knob: the two
// timers live in different processes and are armed at different moments, so
// without a gap the destination's deadline outlives the source's HOLD by the
// time it took the request to arrive. In that window the source has unpinned
// and reported failure while the destination can still commit, and globally
// register, bytes read out of blocks the source may already have reused.
constexpr absl::Duration kRemoteWriteMargin = absl::Seconds(5);

absl::Duration RemoteWriteHold() {
  const char* env = std::getenv("RAIDEN_REMOTE_WRITE_HOLD_S");
  if (env == nullptr) {
    return kDefaultRemoteWriteHold;
  }
  int seconds = 0;
  if (!absl::SimpleAtoi(env, &seconds) || seconds <= 0) {
    LOG(WARNING) << "Ignoring RAIDEN_REMOTE_WRITE_HOLD_S=\"" << env
                 << "\": expected a positive number of seconds.";
    return kDefaultRemoteWriteHold;
  }
  return absl::Seconds(seconds);
}

}  // namespace

absl::Status KVCacheStore::WriteRemote(
    const std::vector<std::string>& block_hashes,
    const RaidenId& dst_raiden_id) {
  if (block_hashes.empty()) {
    return absl::OkStatus();
  }
  if (dst_raiden_id.empty()) {
    return absl::InvalidArgumentError("WriteRemote requires a destination id");
  }
  if (dst_raiden_id == raiden_id_) {
    return absl::InvalidArgumentError(
        "WriteRemote destination is this store; there is nothing to transfer");
  }

  auto* backend = dynamic_cast<HostOffloadBackend*>(this->backend().get());
  if (backend == nullptr) {
    return absl::FailedPreconditionError(
        "WriteRemote requires a HostOffloadBackend at tier 0");
  }

  // Everything before the RPC is undone by this if we do not get as far as
  // recording the operation.
  std::vector<std::string> marked;
  std::vector<std::string> pinned;
  auto rollback = absl::MakeCleanup([&]() {
    if (!pinned.empty()) {
      backend->Release(pinned);
    }
    if (!marked.empty()) {
      absl::MutexLock lock(mutex_);
      for (const auto& hash : marked) {
        writing_hashes_.erase(hash);
      }
    }
  });

  std::vector<int32_t> src_host_block_ids;
  {
    absl::MutexLock lock(mutex_);
    auto lookup_or = backend->Lookup(block_hashes);
    if (!lookup_or.ok()) return lookup_or.status();
    const auto& slices = lookup_or.value();
    if (slices.size() < block_hashes.size()) {
      return absl::NotFoundError(
          absl::StrCat("Block hash not found: ", block_hashes[slices.size()]));
    }
    src_host_block_ids.reserve(slices.size());
    for (size_t i = 0; i < slices.size(); ++i) {
      const auto& [hash, slice] = slices[i];
      if (slice.status != BlockStatus::HOST &&
          slice.status != BlockStatus::HOST_AND_HBM) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Block is not resident in host DRAM, so there is nothing to "
            "offer: ",
            absl::BytesToHexString(hash)));
      }
      if (!writing_hashes_.insert(hash).second) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is already being written remotely: ",
                         absl::BytesToHexString(hash)));
      }
      marked.push_back(hash);
      src_host_block_ids.push_back(slice.host_block_id);
    }
  }

  // The INTERNAL pin, separate from whatever the caller holds. The block ids
  // we are about to send are only authoritative for as long as this holds.
  if (!backend->Pin(block_hashes)) {
    return absl::ResourceExhaustedError(
        "Failed to pin host blocks for a remote write");
  }
  pinned = block_hashes;

  const absl::Duration hold = RemoteWriteHold();
  if (hold <= kRemoteWriteMargin) {
    return absl::FailedPreconditionError(absl::StrCat(
        "RAIDEN_REMOTE_WRITE_HOLD_S (", absl::FormatDuration(hold),
        ") must exceed the ", absl::FormatDuration(kRemoteWriteMargin),
        " margin, or there is no deadline left to ask the destination for."));
  }
  // Armed HERE, with the pin, rather than after the ack: the pin has to be
  // protected even if the RPC itself hangs. The cost of that ordering is that
  // the margin must also cover the round trip.
  const absl::Time hold_expiry = absl::Now() + hold;

  ASSIGN_OR_RETURN(
      HostOffloadBackend::RemoteWriteAck ack,
      backend->BeginWriteRemote(dst_raiden_id, block_hashes, src_host_block_ids,
                                hold - kRemoteWriteMargin));

  if (ack.all_exist) {
    // SUCCESS with nothing to wait for.
    std::move(rollback).Cancel();
    FinishRemoteWrite({.dst_raiden_id = dst_raiden_id,
                       .block_hashes = block_hashes,
                       .hold_expiry = hold_expiry},
                      /*succeeded=*/true, {});
    return absl::OkStatus();
  }
  if (!ack.existing_hashes.empty()) {
    // FAILURE, and this store does not retry the remainder: the caller gets
    // the list and decides.
    std::move(rollback).Cancel();
    FinishRemoteWrite({.dst_raiden_id = dst_raiden_id,
                       .block_hashes = block_hashes,
                       .hold_expiry = hold_expiry},
                      /*succeeded=*/false, std::move(ack.existing_hashes));
    return absl::OkStatus();
  }

  // Belt and braces. The destination clamps to its own cap, which is below the
  // default HOLD, so this should be unreachable -- but the two values live in
  // different processes and neither side can check the invariant alone. If it
  // ever inverts, keep the pin until the granted deadline has elapsed rather
  // than releasing it while the destination may still be pulling.
  if (ack.granted_deadline >= hold) {
    LOG(ERROR) << "Destination granted a deadline of " << ack.granted_deadline
               << ", which is not shorter than this "
               << "source's HOLD of " << hold
               << ". Extending the hold rather than unpinning while the "
                  "destination may still be reading.";
  }

  {
    absl::MutexLock lock(mutex_);
    active_remote_writes_.push_back(RemoteWriteState{
        .dst_raiden_id = dst_raiden_id,
        .operation_id = ack.operation_id,
        .block_hashes = block_hashes,
        .hold_expiry =
            std::max(hold_expiry, absl::Now() + ack.granted_deadline),
    });
  }
  std::move(rollback).Cancel();
  return absl::OkStatus();
}

void KVCacheStore::FinishRemoteWrite(const RemoteWriteState& state,
                                     bool succeeded,
                                     std::vector<std::string> existing,
                                     std::vector<std::string> unregistered) {
  if (auto* backend = this->backend().get(); backend != nullptr) {
    backend->Release(state.block_hashes);
  }
  absl::MutexLock lock(mutex_);
  for (const auto& hash : state.block_hashes) {
    writing_hashes_.erase(hash);
    (succeeded ? done_remote_writes_ : failed_remote_writes_).push_back(hash);
  }
  for (auto& hash : existing) {
    existing_remote_writes_.push_back(std::move(hash));
  }
  for (auto& hash : unregistered) {
    unregistered_remote_writes_.push_back(std::move(hash));
  }
}

void KVCacheStore::PollRemoteWritesInternal() {
  std::vector<RemoteWriteState> still_active;
  std::vector<RemoteWriteState> to_poll;
  {
    absl::MutexLock lock(mutex_);
    to_poll.swap(active_remote_writes_);
  }
  if (to_poll.empty()) {
    return;
  }

  auto* backend = dynamic_cast<HostOffloadBackend*>(this->backend().get());
  const absl::Time now = absl::Now();
  for (auto& state : to_poll) {
    if (backend == nullptr) {
      FinishRemoteWrite(state, /*succeeded=*/false, {});
      continue;
    }
    auto status_or =
        backend->PollWriteRemote(state.dst_raiden_id, state.operation_id);
    if (!status_or.ok()) {
      LOG(WARNING) << "Remote write " << state.operation_id
                   << " could not be polled: " << status_or.status().message();
      FinishRemoteWrite(state, /*succeeded=*/false, {});
      continue;
    }
    switch (status_or->state) {
      case HostOffloadBackend::RemoteWriteState::kPending:
        if (now >= state.hold_expiry) {
          // HOLD expired. Unlike the destination's deadline, which only
          // decides a verdict, this releases the pin immediately -- the
          // blocks become evictable while the destination may still be
          // pulling from them. That window is safe only because the
          // destination refuses to insert or register anything it could not
          // claim inside its own deadline.
          LOG(WARNING) << "Remote write " << state.operation_id
                       << " did not finish within its hold; giving up and "
                          "releasing the source pin.";
          FinishRemoteWrite(state, /*succeeded=*/false, {});
        } else {
          still_active.push_back(std::move(state));
        }
        break;
      case HostOffloadBackend::RemoteWriteState::kCommitted:
      case HostOffloadBackend::RemoteWriteState::kAllExist:
        FinishRemoteWrite(state, /*succeeded=*/true, {});
        break;
      case HostOffloadBackend::RemoteWriteState::kPartialExist:
        FinishRemoteWrite(state, /*succeeded=*/false,
                          std::move(status_or->existing_hashes));
        break;
      case HostOffloadBackend::RemoteWriteState::kStoredUnregistered:
        // The transfer worked; only publication failed. Reported as failed
        // because the safe default is to keep our own copy -- freeing it
        // would move the block from findable-here to findable-nowhere. The
        // caller gets the list and decides; this store does not decide for it.
        LOG(WARNING) << "Remote write " << state.operation_id << " landed "
                     << state.block_hashes.size()
                     << " block(s) on the peer, but the peer could not publish "
                        "them; no lookup will find them there.";
        FinishRemoteWrite(state, /*succeeded=*/false, {},
                          std::move(status_or->unregistered_hashes));
        break;
      case HostOffloadBackend::RemoteWriteState::kFailed:
      case HostOffloadBackend::RemoteWriteState::kUnknown:
        FinishRemoteWrite(state, /*succeeded=*/false, {});
        break;
    }
  }

  if (!still_active.empty()) {
    absl::MutexLock lock(mutex_);
    for (auto& state : still_active) {
      active_remote_writes_.push_back(std::move(state));
    }
  }
}

std::tuple<std::vector<std::string>, std::vector<std::string>,
           std::vector<std::string>, std::vector<std::string>,
           std::vector<std::string>>
KVCacheStore::PollRemoteWriteStatus() {
  absl::MutexLock lock(mutex_);
  std::vector<std::string> pending;
  for (const auto& state : active_remote_writes_) {
    pending.insert(pending.end(), state.block_hashes.begin(),
                   state.block_hashes.end());
  }
  // Drained on read, like the save and load pollers: the consumer of these is
  // in this process, so the store can hand the result over and forget it.
  std::vector<std::string> done;
  std::vector<std::string> failed;
  std::vector<std::string> existing;
  std::vector<std::string> unregistered;
  done.swap(done_remote_writes_);
  failed.swap(failed_remote_writes_);
  existing.swap(existing_remote_writes_);
  unregistered.swap(unregistered_remote_writes_);
  return {std::move(done), std::move(failed), std::move(pending),
          std::move(existing), std::move(unregistered)};
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
    absl::MutexLock lock(mutex_);
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
    absl::MutexLock lock(mutex_);
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
            if (block.status == BlockStatus::REMOTE) {
              block.raiden_id = raiden_id_;
              block.host_block_id = -1;
              block.status = BlockStatus::HBM;
            } else {
              block.status = BlockStatus::HOST_AND_HBM;
            }
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
    absl::MutexLock lock(mutex_);

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
    absl::MutexLock lock(mutex_);
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
  // Unlike the three above, a remote write has no local future to become
  // ready: the work is happening on the destination, so this asks.
  PollRemoteWritesInternal();
}

}  // namespace kv_cache
}  // namespace tpu_raiden
