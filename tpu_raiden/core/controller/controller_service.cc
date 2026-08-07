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

#include "tpu_raiden/core/controller/controller_service.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tpu_raiden/core/controller/worker_registry.h"
#include "tpu_raiden/core/raiden_transfer_endpoint.h"
#include "tpu_raiden/proto/controller_service.pb.h"

namespace tpu_raiden {
namespace core {
namespace controller {

RaidenControllerServiceImpl::RaidenControllerServiceImpl(
    std::shared_ptr<WorkerRegistry> worker_registry)
    : worker_registry_(worker_registry ? std::move(worker_registry)
                                       : std::make_shared<WorkerRegistry>()) {}

grpc::Status RaidenControllerServiceImpl::RegisterWorker(
    grpc::ServerContext* context,
    const ::tpu_raiden::proto::RegisterWorkerRequest* request,
    ::tpu_raiden::proto::RegisterWorkerResponse* response) {
  std::vector<::tpu_raiden::RaidenTransferEndpoint> eps;
  eps.reserve(request->raiden_transfer_endpoints_size());
  for (const auto& desc_proto : request->raiden_transfer_endpoints()) {
    ::tpu_raiden::RaidenTransferEndpoint ep;
    ep.endpoint = desc_proto.endpoint();
    ep.shards.assign(desc_proto.shards().begin(), desc_proto.shards().end());
    eps.push_back(std::move(ep));
  }

  WorkerRegistration reg = {
      .worker_id = request->worker_id(),
      .raiden_worker_endpoint = request->raiden_worker_endpoint(),
      .raiden_transfer_endpoints = std::move(eps),
      .node_id = request->node_id(),
      .block_array_bytes = {request->block_array_bytes().begin(),
                            request->block_array_bytes().end()},
      .num_kv_shards = request->num_kv_shards(),
  };

  std::shared_ptr<WorkerRegistry> registry;
  {
    absl::MutexLock lock(mutex_);
    registry = worker_registry_;
  }

  absl::Status status = registry->RegisterWorker(reg);
  if (!status.ok()) {
    response->set_success(false);
    response->set_error_message(std::string(status.message()));
    return grpc::Status::OK;
  }

  response->set_success(true);
  return grpc::Status::OK;
}

// ---------------------------------------------------------------------------
// Read-lease protocol (source side)
// ---------------------------------------------------------------------------

namespace {

// Reads a duration from an env var holding whole seconds. Returns `fallback`
// if unset, unparseable, or non-positive.
absl::Duration DurationFromEnv(const char* name, absl::Duration fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr) return fallback;
  int64_t seconds = 0;
  if (!absl::SimpleAtoi(raw, &seconds) || seconds <= 0) {
    LOG(ERROR) << name << "=\"" << raw
               << "\" is not a positive integer number of seconds; using "
               << fallback;
    return fallback;
  }
  return absl::Seconds(seconds);
}

std::string RequesterKey(const ::tpu_raiden::rpc::RaidenIdProto& id) {
  return absl::StrCat(id.job_name(), "/", id.job_replica_id(), "/",
                      id.data_name(), "/", id.data_replica_idx());
}

}  // namespace

absl::Duration RaidenControllerServiceImpl::DefaultLeaseTtl() {
  // Flat default. A size-derived TTL (bytes_total / floor_bandwidth) is the
  // obvious extension if this proves insufficient for very large blocks or
  // slow links -- but since the commit-time verdict demoted the TTL from
  // safety to garbage collection, expiry under a live pull costs only a
  // discarded read plus a retry, so a flat value suffices and operators can
  // raise the env var.
  static const absl::Duration kTtl =
      DurationFromEnv("RAIDEN_REMOTE_LOCK_TTL_S", absl::Seconds(120));
  return kTtl;
}

absl::Duration RaidenControllerServiceImpl::MinLeaseTtl() {
  return absl::Seconds(30);
}

absl::Duration RaidenControllerServiceImpl::MaxLeaseTtl() {
  return absl::Seconds(600);
}

absl::Duration RaidenControllerServiceImpl::RevokedLeaseRetention() {
  // Derived from the TTL so that overriding only RAIDEN_REMOTE_LOCK_TTL_S
  // keeps a coherent triple; an explicit env var wins.
  static const absl::Duration kRetention = DurationFromEnv(
      "RAIDEN_REVOKED_LEASE_RETENTION_S", DefaultLeaseTtl() + absl::Minutes(3));
  return kRetention;
}

grpc::Status RaidenControllerServiceImpl::AcquireReadLease(
    grpc::ServerContext* context,
    const ::tpu_raiden::proto::AcquireReadLeaseRequest* request,
    ::tpu_raiden::proto::AcquireReadLeaseResponse* response) {
  if (request->block_hashes().empty()) {
    // Closes the "empty hashes skip validation" hole by construction: there is
    // no path that transfers without validating.
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "AcquireReadLease requires a non-empty block_hashes");
  }

  std::vector<std::string> block_hashes(request->block_hashes().begin(),
                                        request->block_hashes().end());

  std::shared_ptr<const ValidateAndPinCallback> validate_cb;
  std::shared_ptr<const UnpinCallback> unpin_cb;
  std::shared_ptr<WorkerRegistry> registry;
  {
    absl::MutexLock lock(mutex_);
    validate_cb = validate_and_pin_cb_;
    unpin_cb = unpin_cb_;
    registry = worker_registry_;
  }

  if (!validate_cb) {
    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                        "ReadRemote verify hook is not registered");
  }

  absl::StatusOr<std::vector<int32_t>> ids_or = (*validate_cb)(block_hashes);
  if (!ids_or.ok()) {
    // absl and grpc codes share the same canonical integers, so the verify
    // hook's distinctions survive to the destination: NOT_FOUND (hash absent)
    // vs FAILED_PRECONDITION (present but not host-resident).
    return grpc::Status(static_cast<grpc::StatusCode>(ids_or.status().code()),
                        std::string(ids_or.status().message()));
  }

  // Pins are held from here. Anything that fails before the lease record
  // exists must release them, or they leak with no reclaimer -- the sweeper
  // only walks read_leases_.
  bool lease_recorded = false;
  auto pin_guard = absl::MakeCleanup([&] {
    if (!lease_recorded && unpin_cb) {
      (*unpin_cb)(block_hashes);
    }
  });

  absl::Duration ttl = request->ttl_ms() > 0
                           ? absl::Milliseconds(request->ttl_ms())
                           : DefaultLeaseTtl();
  ttl = std::max(MinLeaseTtl(), std::min(MaxLeaseTtl(), ttl));

  const absl::Time now = absl::Now();
  uint64_t lease_id = 0;
  {
    absl::MutexLock lock(mutex_);
    // 128 bits of identity would be cleaner, but a fresh 64-bit draw per grant
    // already makes a post-restart collision (which would return HELD for
    // bytes served unpinned, and steal the colliding lease's pins)
    // vanishingly unlikely. 0 is reserved as "never granted".
    do {
      lease_id = absl::Uniform<uint64_t>(lease_id_gen_);
    } while (lease_id == 0 || read_leases_.contains(lease_id));

    read_leases_[lease_id] = ReadLease{
        .block_hashes = block_hashes,
        .requester_key = RequesterKey(request->requester_raiden_id()),
        .expiration = now + ttl,
        .live = true,
        .revoked = false,
        .retire_at = absl::InfiniteFuture(),
    };
    lease_recorded = true;
    StartSweeperIfNeeded();
    sweeper_cv_.Signal();
  }

  std::shared_ptr<absl::AnyInvocable<void(uint64_t)>> granted_hook;
  {
    absl::MutexLock lock(mutex_);
    granted_hook = lease_granted_hook_;
  }
  if (granted_hook) (*granted_hook)(lease_id);

  response->set_lease_id(lease_id);
  for (int32_t id : *ids_or) {
    response->add_src_host_block_ids(id);
  }
  response->set_granted_ttl_ms(absl::ToInt64Milliseconds(ttl));

  // The destination pulls over the block-transport DATA protocol, so these
  // must be data endpoints. They are: RegisterWorker stores whatever the
  // worker advertised, and a store-configured manager runs without a control
  // server, so get_local_endpoints() yields its data port. If a control server
  // is ever enabled alongside a store, that stops being true and pulls hang
  // with no error on either side.
  if (registry) {
    for (const auto& reg : registry->GetRegisteredWorkers()) {
      auto* group = response->add_src_worker_endpoints();
      group->set_node_id(reg.node_id);
      group->set_worker_id(reg.worker_id);
      for (const auto& ep : reg.raiden_transfer_endpoints) {
        auto* ep_proto = group->add_endpoints();
        ep_proto->set_endpoint(ep.endpoint);
        for (int64_t shard : ep.shards) {
          ep_proto->add_shards(shard);
        }
      }
    }
  }

  return grpc::Status::OK;
}

grpc::Status RaidenControllerServiceImpl::RenewReadLease(
    grpc::ServerContext* context,
    const ::tpu_raiden::proto::RenewReadLeaseRequest* request,
    ::tpu_raiden::proto::RenewReadLeaseResponse* response) {
  absl::Duration extend = request->extend_ms() > 0
                              ? absl::Milliseconds(request->extend_ms())
                              : DefaultLeaseTtl();
  extend = std::max(MinLeaseTtl(), std::min(MaxLeaseTtl(), extend));

  absl::MutexLock lock(mutex_);
  auto it = read_leases_.find(request->lease_id());
  if (it == read_leases_.end()) {
    response->set_verdict(::tpu_raiden::proto::LEASE_UNKNOWN);
    return grpc::Status::OK;
  }
  if (!it->second.live) {
    // Never resurrect: the pins are already gone, so extending the record
    // would promise protection that does not exist.
    response->set_verdict(it->second.revoked
                              ? ::tpu_raiden::proto::LEASE_REVOKED
                              : ::tpu_raiden::proto::LEASE_HELD);
    return grpc::Status::OK;
  }
  // Extension is measured from now, not from the current expiration, and is
  // clamped per extension -- so a live destination CAN hold pins indefinitely
  // by renewing. That is intended: a healthy reader should not lose its blocks.
  it->second.expiration = absl::Now() + extend;
  response->set_verdict(::tpu_raiden::proto::LEASE_HELD);
  response->set_new_ttl_ms(absl::ToInt64Milliseconds(extend));
  sweeper_cv_.Signal();
  return grpc::Status::OK;
}

grpc::Status RaidenControllerServiceImpl::ReleaseReadLease(
    grpc::ServerContext* context,
    const ::tpu_raiden::proto::ReleaseReadLeaseRequest* request,
    ::tpu_raiden::proto::ReleaseReadLeaseResponse* response) {
  std::vector<std::string> to_unpin;
  std::shared_ptr<const UnpinCallback> unpin_cb;
  {
    absl::MutexLock lock(mutex_);
    auto it = read_leases_.find(request->lease_id());
    if (it == read_leases_.end()) {
      response->set_verdict(::tpu_raiden::proto::LEASE_UNKNOWN);
      return grpc::Status::OK;
    }
    if (it->second.revoked) {
      response->set_verdict(::tpu_raiden::proto::LEASE_REVOKED);
      return grpc::Status::OK;
    }
    // Live, or already released: both answer HELD (release is idempotent, and
    // the verdict is retained until the record ages out). Only the first
    // release unpins.
    response->set_verdict(::tpu_raiden::proto::LEASE_HELD);
    to_unpin = TransitionOutOfLive(request->lease_id(), /*revoked=*/false);
    unpin_cb = unpin_cb_;
  }
  if (!to_unpin.empty() && unpin_cb) {
    (*unpin_cb)(to_unpin);
  }
  return grpc::Status::OK;
}

std::vector<std::string> RaidenControllerServiceImpl::TransitionOutOfLive(
    uint64_t lease_id, bool revoked) {
  auto it = read_leases_.find(lease_id);
  if (it == read_leases_.end() || !it->second.live) return {};
  it->second.live = false;
  it->second.revoked = revoked;
  it->second.retire_at = absl::Now() + RevokedLeaseRetention();
  return it->second.block_hashes;
}

bool RaidenControllerServiceImpl::ForceExpireForTest(uint64_t lease_id) {
  std::vector<std::string> to_unpin;
  std::shared_ptr<const UnpinCallback> unpin_cb;
  {
    absl::MutexLock lock(mutex_);
    to_unpin = TransitionOutOfLive(lease_id, /*revoked=*/true);
    unpin_cb = unpin_cb_;
  }
  if (to_unpin.empty()) return false;
  if (unpin_cb) (*unpin_cb)(to_unpin);
  return true;
}

size_t RaidenControllerServiceImpl::LeaseCountForTest() const {
  absl::MutexLock lock(mutex_);
  return read_leases_.size();
}

int64_t RaidenControllerServiceImpl::LeasedBlocksForPeerForTest(
    absl::string_view requester_key) const {
  absl::MutexLock lock(mutex_);
  int64_t total = 0;
  for (const auto& [id, lease] : read_leases_) {
    if (lease.live && lease.requester_key == requester_key) {
      total += static_cast<int64_t>(lease.block_hashes.size());
    }
  }
  return total;
}

void RaidenControllerServiceImpl::StartSweeperIfNeeded() {
  if (sweeper_ != nullptr || sweeper_stop_) return;
  sweeper_ = std::make_unique<std::thread>([this] { SweeperLoop(); });
}

void RaidenControllerServiceImpl::SweeperLoop() {
  while (true) {
    {
      absl::MutexLock lock(mutex_);
      if (sweeper_stop_) return;
      absl::Time wake = absl::InfiniteFuture();
      for (const auto& [id, lease] : read_leases_) {
        wake = std::min(wake, lease.live ? lease.expiration : lease.retire_at);
      }
      if (wake == absl::InfiniteFuture()) {
        wake = absl::Now() + absl::Seconds(1);
      }
      sweeper_cv_.WaitWithDeadline(&mutex_, wake);
      if (sweeper_stop_) return;
    }
    RunSweepOnce();
  }
}

void RaidenControllerServiceImpl::RunSweepOnce() {
  std::vector<std::string> to_unpin;
  std::shared_ptr<const UnpinCallback> unpin_cb;
  {
    absl::MutexLock lock(mutex_);
    const absl::Time now = absl::Now();
    std::vector<uint64_t> expired;
    std::vector<uint64_t> retired;
    for (const auto& [id, lease] : read_leases_) {
      // Re-check expiration inside the same critical section as the
      // transition, so a renewal landing between scan and process is not lost.
      if (lease.live) {
        if (lease.expiration <= now) expired.push_back(id);
      } else if (lease.retire_at <= now) {
        retired.push_back(id);
      }
    }
    for (uint64_t id : expired) {
      std::vector<std::string> hashes =
          TransitionOutOfLive(id, /*revoked=*/true);
      to_unpin.insert(to_unpin.end(), hashes.begin(), hashes.end());
      LOG(WARNING) << "Read lease " << id
                   << " expired under a live read; pins reclaimed. The "
                      "destination will learn REVOKED at commit and discard.";
    }
    for (uint64_t id : retired) {
      read_leases_.erase(id);
    }
    unpin_cb = unpin_cb_;
  }
  // Outside mutex_: the hook takes the store's lock (see the header note).
  if (!to_unpin.empty() && unpin_cb) {
    (*unpin_cb)(to_unpin);
  }
}

void RaidenControllerServiceImpl::ClearReadRemoteHooks() {
  absl::MutexLock lock(mutex_);
  validate_and_pin_cb_.reset();
  unpin_cb_.reset();
}

RaidenControllerServiceImpl::~RaidenControllerServiceImpl() {
  std::unique_ptr<std::thread> sweeper;
  std::vector<std::string> to_unpin;
  std::shared_ptr<const UnpinCallback> unpin_cb;
  {
    absl::MutexLock lock(mutex_);
    sweeper_stop_ = true;
    sweeper_cv_.SignalAll();
    sweeper = std::move(sweeper_);
    std::vector<uint64_t> live;
    for (const auto& [id, lease] : read_leases_) {
      if (lease.live) live.push_back(id);
    }
    for (uint64_t id : live) {
      std::vector<std::string> hashes =
          TransitionOutOfLive(id, /*revoked=*/true);
      to_unpin.insert(to_unpin.end(), hashes.begin(), hashes.end());
    }
    unpin_cb = unpin_cb_;
  }
  if (sweeper && sweeper->joinable()) sweeper->join();
  if (!to_unpin.empty() && unpin_cb) {
    (*unpin_cb)(to_unpin);
  }
}

void RaidenControllerServiceImpl::SetWorkerRegistry(
    std::shared_ptr<WorkerRegistry> worker_registry) {
  if (worker_registry) {
    absl::MutexLock lock(mutex_);
    worker_registry_ = std::move(worker_registry);
  }
}

std::shared_ptr<WorkerRegistry> RaidenControllerServiceImpl::worker_registry()
    const {
  absl::MutexLock lock(mutex_);
  return worker_registry_;
}

}  // namespace controller
}  // namespace core
}  // namespace tpu_raiden
