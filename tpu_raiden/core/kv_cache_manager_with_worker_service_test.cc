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

#include "tpu_raiden/core/kv_cache_manager_with_worker_service.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "xla/tsl/platform/test.h"
#include "tpu_raiden/core/controller/worker_service_client.h"
#include "tpu_raiden/core/kv_cache_manager_with_transfer.h"
#include "tpu_raiden/proto/worker_service.pb.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace {

TEST(KVCacheManagerWithWorkerServiceTest, WrapAndStartServerWorks) {
  // Use metadata-based FFI/CPU constructor of KVCacheManagerWithTransfer.
  auto transfer_mgr = std::make_unique<KVCacheManagerWithTransfer>(
      /*num_layers=*/2, /*num_shards=*/2, /*slice_byte_size=*/1024,
      /*local_port=*/std::nullopt, /*host_blocks_to_allocate=*/16);

  // Construction automatically starts the gRPC server on an ephemeral port
  // (default 0).
  KVCacheManagerWithWorkerService wrapper(std::move(transfer_mgr));
  EXPECT_NE(wrapper.transfer_manager(), nullptr);
  int port = wrapper.GetGrpcPort();
  EXPECT_GT(port, 0);

  // Connect via client and verify gRPC communication.
  std::string server_address = "localhost:" + std::to_string(port);
  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  controller::WorkerServiceClient client(channel);

  proto::CreateBuffersRequest create_req;
  create_req.mutable_unit()->set_job_name("test_job");
  create_req.mutable_unit()->set_job_replica_id("0");
  create_req.mutable_unit()->set_data_name("test_data");
  auto* spec = create_req.add_buffers();
  spec->set_num_shards(1);
  spec->set_size_bytes(512);

  auto resp_or = client.CreateBuffers(create_req);
  ASSERT_TRUE(resp_or.ok());
  EXPECT_TRUE(resp_or->success());
  ASSERT_EQ(resp_or->buffers_size(), 1);

  proto::DeleteBuffersRequest delete_req;
  *delete_req.mutable_unit() = create_req.unit();
  *delete_req.add_sharded_buffers() = resp_or->buffers(0);

  auto del_resp_or = client.DeleteBuffers(delete_req);
  ASSERT_TRUE(del_resp_or.ok());
  EXPECT_TRUE(del_resp_or->success());
}

TEST(KVCacheManagerWithWorkerServiceTest,
     SingletonServerIsReusedAcrossInstances) {
  auto transfer_mgr1 = std::make_unique<KVCacheManagerWithTransfer>(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/512,
      /*local_port=*/std::nullopt, /*host_blocks_to_allocate=*/8);
  KVCacheManagerWithWorkerService wrapper1(std::move(transfer_mgr1));
  int port1 = wrapper1.GetGrpcPort();
  EXPECT_GT(port1, 0);

  // Creating a second wrapper instance reuses the exact same singleton gRPC
  // server and listening port without restarting or re-binding.
  auto transfer_mgr2 = std::make_unique<KVCacheManagerWithTransfer>(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/512,
      /*local_port=*/std::nullopt, /*host_blocks_to_allocate=*/8);
  KVCacheManagerWithWorkerService wrapper2(std::move(transfer_mgr2));
  int port2 = wrapper2.GetGrpcPort();
  EXPECT_EQ(port1, port2);
}

TEST(KVCacheManagerWithWorkerServiceTest, ConstructorThrowsOnInvalidPort) {
  auto transfer_mgr = std::make_unique<KVCacheManagerWithTransfer>(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/512,
      /*local_port=*/std::nullopt, /*host_blocks_to_allocate=*/8);

  // Attempting to construct an instance with an invalid port (-1) throws
  // std::runtime_error because gRPC server startup fails.
  EXPECT_THROW(
      {
        KVCacheManagerWithWorkerService wrapper(std::move(transfer_mgr),
                                                /*host_allocator=*/nullptr,
                                                /*grpc_port=*/-1);
      },
      std::runtime_error);
}

}  // namespace
}  // namespace tpu_raiden
