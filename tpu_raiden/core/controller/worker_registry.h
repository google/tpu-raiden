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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_WORKER_REGISTRY_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_WORKER_REGISTRY_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/functional/any_invocable.h"
#include "tpu_raiden/core/controller/worker_service_client.h"
#include "tpu_raiden/proto/worker_service.pb.h"

namespace tpu_raiden {
namespace core {
namespace controller {

struct WorkerRegistration {
  std::string worker_id;
  std::string raiden_worker_endpoint;
  std::string raiden_transfer_endpoint;
  std::shared_ptr<::tpu_raiden::controller::WorkerServiceClient>
      worker_service_client;
  std::vector<::tpu_raiden::proto::BufferProto> buffers;
};

// Thread-safe registry for worker node registrations in the controller plane.
class WorkerRegistry {
 public:
  WorkerRegistry() = default;
  ~WorkerRegistry() = default;

  WorkerRegistry(const WorkerRegistry&) = delete;
  WorkerRegistry& operator=(const WorkerRegistry&) = delete;

  using OnRegisterCallback =
      absl::AnyInvocable<absl::Status(WorkerRegistration&)>;

  void SetOnRegisterCallback(OnRegisterCallback cb) {
    absl::MutexLock lock(&mutex_);
    on_register_cb_ = std::move(cb);
  }

  // Registers or updates a worker registration.
  absl::Status RegisterWorker(const WorkerRegistration& reg);
  absl::Status RegisterWorker(absl::string_view worker_id,
                              absl::string_view raiden_worker_endpoint,
                              absl::string_view raiden_transfer_endpoint);

  // Retrieves all registered workers.
  std::vector<WorkerRegistration> GetRegisteredWorkers() const;

  // Retrieves a specific worker registration by worker ID.
  absl::StatusOr<WorkerRegistration> GetWorker(
      absl::string_view worker_id) const;

 private:
  mutable absl::Mutex mutex_;
  OnRegisterCallback on_register_cb_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<std::string, WorkerRegistration> workers_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace controller
}  // namespace core
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_CORE_CONTROLLER_WORKER_REGISTRY_H_
