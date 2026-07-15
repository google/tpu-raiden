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

#include "tpu_raiden/core/controller/worker_registry.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "tpu_raiden/core/controller/worker_service_client.h"

namespace tpu_raiden {
namespace core {
namespace controller {

absl::Status WorkerRegistry::RegisterWorker(const WorkerRegistration& reg) {
  if (reg.worker_id.empty()) {
    return absl::InvalidArgumentError("worker_id cannot be empty");
  }
  if (reg.worker_service_client == nullptr &&
      reg.raiden_worker_endpoint.empty() &&
      reg.raiden_transfer_endpoint.empty()) {
    return absl::InvalidArgumentError(
        "at least one of raiden_worker_endpoint or raiden_transfer_endpoint"
        " must be provided, or worker_service_client must be non-null");
  }

  WorkerRegistration entry = reg;
  if (entry.worker_service_client == nullptr &&
      !entry.raiden_worker_endpoint.empty()) {
    auto channel = grpc::CreateChannel(entry.raiden_worker_endpoint,
                                       grpc::InsecureChannelCredentials());
    entry.worker_service_client =
        std::make_shared<::tpu_raiden::controller::WorkerServiceClient>(
            std::move(channel));
  }

  absl::MutexLock lock(&mutex_);
  if (on_register_cb_) {
    absl::Status status = on_register_cb_(entry);
    if (!status.ok()) return status;
  }
  workers_[entry.worker_id] = std::move(entry);
  return absl::OkStatus();
}

absl::Status WorkerRegistry::RegisterWorker(
    absl::string_view worker_id, absl::string_view raiden_worker_endpoint,
    absl::string_view raiden_transfer_endpoint) {
  return RegisterWorker(WorkerRegistration{
      .worker_id = std::string(worker_id),
      .raiden_worker_endpoint = std::string(raiden_worker_endpoint),
      .raiden_transfer_endpoint = std::string(raiden_transfer_endpoint),
  });
}

std::vector<WorkerRegistration> WorkerRegistry::GetRegisteredWorkers() const {
  absl::MutexLock lock(mutex_);
  std::vector<WorkerRegistration> result;
  result.reserve(workers_.size());
  for (const auto& [_, reg] : workers_) {
    result.push_back(reg);
  }
  return result;
}

absl::StatusOr<WorkerRegistration> WorkerRegistry::GetWorker(
    absl::string_view worker_id) const {
  absl::MutexLock lock(mutex_);
  auto it = workers_.find(worker_id);
  if (it == workers_.end()) {
    return absl::NotFoundError(absl::StrCat("Worker not found: ", worker_id));
  }
  return it->second;
}

}  // namespace controller
}  // namespace core
}  // namespace tpu_raiden
