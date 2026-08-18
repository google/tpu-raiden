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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_STORE_MONITOR_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_STORE_MONITOR_H_

#include <functional>
#include <memory>
#include <thread>  // NOLINT(build/c++11)

#include "absl/status/status.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "tpu_sync/kv_cache/global_registry/global_registry.pb.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_client.h"
#include "tpu_sync/kv_cache/raiden_id.h"

namespace tpu_raiden {
namespace kv_cache {

// The store's periodic reporting thread: heartbeats the store's mutable
// status (see StoreStatus) to the global registry, which both refreshes the
// registration TTL and feeds the registry's capacity ranking. A heartbeat
// answered with NotFound means the registry no longer holds a live
// registration (registry restart or TTL expiry); heartbeats carry no
// coordinates, so the monitor re-registers through the callback instead.
//
// Owned by KVCacheStore; deliberately talks to the store only through the
// two callbacks so it never sees store internals.
class StoreMonitor {
 public:
  static constexpr absl::Duration kDefaultHeartbeatPeriod = absl::Seconds(300);

  struct Options {
    absl::Duration heartbeat_period = kDefaultHeartbeatPeriod;
  };

  // Snapshots the store's current status for one heartbeat.
  using StatusFn = std::function<global_registry::StoreStatus()>;
  // Re-publishes the store's full registration (coordinates and TTL).
  using ReregisterFn = std::function<absl::Status()>;

  StoreMonitor(const Options& options,
               std::shared_ptr<global_registry::GlobalRegistryClient>
                   registry_client,
               RaidenId raiden_id, StatusFn status_fn,
               ReregisterFn reregister_fn);

  // Stops the thread; the callbacks must outlive this call, not the object.
  ~StoreMonitor();

  StoreMonitor(const StoreMonitor&) = delete;
  StoreMonitor& operator=(const StoreMonitor&) = delete;

  // Starts the heartbeat thread. The first heartbeat fires one period from
  // now: the caller registers before starting the monitor, so the registry
  // is already fresh. Call at most once.
  void Start();

  // Stops and joins the thread. Idempotent, and terminal: a stopped monitor
  // cannot be restarted.
  void Stop();

 private:
  void Loop();
  void HeartbeatOnce();

  const Options options_;
  const std::shared_ptr<global_registry::GlobalRegistryClient>
      registry_client_;
  const RaidenId raiden_id_;
  const StatusFn status_fn_;
  const ReregisterFn reregister_fn_;

  absl::Notification stop_;
  std::thread thread_;
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_STORE_MONITOR_H_
