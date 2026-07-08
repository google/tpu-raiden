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

#include "tpu_raiden/rpc/coordination_client.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tpu_raiden/rpc/coordination.grpc.pb.h"
#include "tpu_raiden/rpc/coordination.pb.h"

namespace tpu_raiden {
namespace rpc {
namespace {

using ::testing::ElementsAre;

class MockCoordinationService : public CoordinationService::Service {
 public:
  grpc::Status CollectReplicaInfo(
      grpc::ServerContext* context, const CollectReplicaInfoRequest* request,
      CollectReplicaInfoResponse* response) override {
    auto* entry1 = response->add_entries();
    entry1->set_device_id(request->device_id());
    for (int32_t v : request->info()) {
      entry1->add_info(v);
    }

    auto* entry2 = response->add_entries();
    entry2->set_device_id(request->device_id() + 1);
    entry2->add_info(42);

    return grpc::Status::OK;
  }
};

TEST(CoordinationClientTest, CollectReplicaInfoWorks) {
  MockCoordinationService service;
  grpc::ServerBuilder builder;
  int selected_port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &selected_port);
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());

  std::string server_address = "localhost:" + std::to_string(selected_port);
  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  CoordinationClient client(channel);

  std::vector<int32_t> info = {1, 2, 3};
  auto status_or_entries = client.CollectReplicaInfo(0, 2, info);

  ASSERT_TRUE(status_or_entries.ok());
  const auto& entries = status_or_entries.value();

  ASSERT_EQ(entries.size(), 2);
  EXPECT_EQ(entries[0].device_id(), 0);
  std::vector<int32_t> info0(entries[0].info().begin(),
                             entries[0].info().end());
  EXPECT_THAT(info0, ElementsAre(1, 2, 3));

  EXPECT_EQ(entries[1].device_id(), 1);
  std::vector<int32_t> info1(entries[1].info().begin(),
                             entries[1].info().end());
  EXPECT_THAT(info1, ElementsAre(42));

  server->Shutdown();
}

}  // namespace
}  // namespace rpc
}  // namespace tpu_raiden
