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

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "tpu_raiden/core/raiden_transfer_endpoint.h"

namespace tpu_raiden {
namespace core {
namespace controller {
namespace {

using ::absl_testing::StatusIs;

TEST(WorkerRegistryTest, RegisterAndGetWorkerWorks) {
  WorkerRegistry registry;

  std::vector<RaidenTransferEndpoint> transfer_eps = {{"localhost:10002", {0}}};
  ABSL_ASSERT_OK(
      registry.RegisterWorker("worker_0", "localhost:10001", transfer_eps));

  auto worker_or = registry.GetWorker("worker_0");
  ABSL_ASSERT_OK(worker_or);
  EXPECT_EQ(worker_or->worker_id, "worker_0");
  EXPECT_EQ(worker_or->raiden_worker_endpoint, "localhost:10001");
  EXPECT_EQ(worker_or->raiden_transfer_endpoints.size(), 1);
  EXPECT_EQ(worker_or->raiden_transfer_endpoints[0].endpoint,
            "localhost:10002");
  EXPECT_NE(worker_or->worker_service_client, nullptr);

  std::vector<WorkerRegistration> workers = registry.GetRegisteredWorkers();
  ASSERT_EQ(workers.size(), 1);
  EXPECT_EQ(workers[0].worker_id, "worker_0");
}

TEST(WorkerRegistryTest, RegisterEmptyWorkerIdFails) {
  WorkerRegistry registry;
  absl::Status status = registry.RegisterWorker("", "localhost:10001",
                                                {{"localhost:10002", {0}}});
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(WorkerRegistryTest, RegisterEmptyEndpointsFails) {
  WorkerRegistry registry;
  absl::Status status = registry.RegisterWorker("worker_0", "", {});
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(WorkerRegistryTest, GetNonExistentWorkerFails) {
  WorkerRegistry registry;
  auto worker_or = registry.GetWorker("non_existent");
  EXPECT_THAT(worker_or.status(), StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace controller
}  // namespace core
}  // namespace tpu_raiden
