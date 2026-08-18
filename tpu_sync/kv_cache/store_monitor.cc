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

#include "tpu_sync/kv_cache/store_monitor.h"


#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "tpu_sync/kv_cache/global_registry/global_registry_client.h"
#include "tpu_sync/kv_cache/raiden_id.h"

namespace tpu_raiden {
namespace kv_cache {

StoreMonitor::StoreMonitor(
    const Options& options,
    std::shared_ptr<global_registry::GlobalRegistryClient> registry_client,
    RaidenId raiden_id, StatusFn status_fn, ReregisterFn reregister_fn)
    : options_(options),
      registry_client_(std::move(registry_client)),
      raiden_id_(std::move(raiden_id)),
      status_fn_(std::move(status_fn)),
      reregister_fn_(std::move(reregister_fn)) {}

StoreMonitor::~StoreMonitor() { Stop(); }

void StoreMonitor::Start() {
  thread_ = std::thread(&StoreMonitor::Loop, this);
}

void StoreMonitor::Stop() {
  if (!stop_.HasBeenNotified()) {
    stop_.Notify();
  }
  if (thread_.joinable()) {
    thread_.join();
  }
}

void StoreMonitor::Loop() {
  while (!stop_.WaitForNotificationWithTimeout(options_.heartbeat_period)) {
    HeartbeatOnce();
  }
}

void StoreMonitor::HeartbeatOnce() {
  absl::Status status = registry_client_->Heartbeat(raiden_id_, status_fn_());
  if (status.ok()) {
    return;
  }
  if (!absl::IsNotFound(status)) {
    LOG(WARNING) << "Store heartbeat to the global registry failed: " << status
                 << ". Retrying next period; the registration expires if this "
                    "keeps failing.";
    return;
  }
  absl::Status reregistered = reregister_fn_();
  if (reregistered.ok()) {
    LOG(INFO) << "Store registration lapsed at the global registry "
                 "(registry restart or expired TTL); re-registered.";
  } else {
    LOG(WARNING) << "Store registration lapsed at the global registry and "
                    "re-registering failed: "
                 << reregistered << ". Retrying next period.";
  }
}

}  // namespace kv_cache
}  // namespace tpu_raiden
