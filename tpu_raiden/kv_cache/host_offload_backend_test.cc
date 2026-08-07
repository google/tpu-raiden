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

// Unit tests for HostOffloadBackend.
#include "tpu_raiden/kv_cache/host_offload_backend.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server_builder.h"
#include "tpu_raiden/core/controller/controller_client.h"
#include "tpu_raiden/core/controller/orchestrator_service_client.h"
#include "tpu_raiden/core/controller/raiden_controller.h"
#include "tpu_raiden/core/controller/raiden_orchestrator.h"
#include "tpu_raiden/core/controller/test_util.h"
#include "tpu_raiden/core/kv_manager_holder.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_client.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_server.h"
#include "tpu_raiden/kv_cache/global_registry/test_util.h"
#include "tpu_raiden/kv_cache/kv_cache_store_backend.h"
#include "tpu_raiden/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_raiden/kv_cache/kv_cache_store_client.h"
#include "tpu_raiden/kv_cache/kv_cache_store_server.h"
#include "tpu_raiden/kv_cache/raiden_id.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

using ::testing::UnorderedElementsAre;

TEST(HostOffloadBackendTest, BasicInsertAndLookup) {
  HostOffloadBackend backend(/*capacity=*/2);
  EXPECT_EQ(backend.name(), "HostOffloadBackend");
  EXPECT_EQ(backend.GetCapacity(), 2);
  EXPECT_EQ(backend.GetSize(), 0);

  std::vector<std::string> hashes = {"h1", "h2"};
  RaidenId id{"job", "0", "data", 0};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(id, 10, BlockStatus::HOST),
      RaidenBlockID(id, 11, BlockStatus::HOST)};

  auto [all_new, evicted] = backend.Insert(hashes, slices, /*on_host=*/true);
  EXPECT_TRUE(all_new);
  EXPECT_TRUE(evicted.empty());
  EXPECT_EQ(backend.GetSize(), 2);

  // Lookup both
  auto lookup_res = backend.Lookup({"h1", "h2"});
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].first, "h1");
  EXPECT_EQ((*lookup_res)[0].second.host_block_id, 10);
  EXPECT_EQ((*lookup_res)[1].first, "h2");
  EXPECT_EQ((*lookup_res)[1].second.host_block_id, 11);

  // Partial miss at end
  auto partial_res = backend.Lookup({"h1", "h2", "h3"});
  ASSERT_TRUE(partial_res.ok());
  EXPECT_EQ(partial_res->size(), 2);

  // Miss at start
  auto miss_res = backend.Lookup({"h3", "h1"});
  ASSERT_TRUE(miss_res.ok());
  EXPECT_TRUE(miss_res->empty());
}

TEST(HostOffloadBackendTest, LookupUnboundedByAvailableSpace) {
  HostOffloadBackend backend(/*capacity=*/2);
  std::vector<std::string> hashes = {"h1", "h2"};
  RaidenId id{"job", "0", "data", 0};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(id, 10, BlockStatus::HOST),
      RaidenBlockID(id, 11, BlockStatus::HOST)};

  backend.Insert(hashes, slices, /*on_host=*/true);
  EXPECT_TRUE(backend.Pin(hashes));
  EXPECT_EQ(backend.GetAvailableSpace(), 0);

  // Lookup still succeeds completely despite available_space() == 0
  auto lookup_res = backend.Lookup({"h1", "h2"});
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 2);
}

TEST(HostOffloadBackendTest, InsertAndLockRollbackOnCapacityExceeded) {
  HostOffloadBackend backend(/*capacity=*/2);
  std::vector<std::string> hashes = {"h1", "h2", "h3"};
  RaidenId id{"job", "0", "data", 0};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(id, 10, BlockStatus::HOST),
      RaidenBlockID(id, 11, BlockStatus::HOST),
      RaidenBlockID(id, 12, BlockStatus::HOST)};

  // InsertAndLock for 3 items on capacity=2 must fail and rollback
  bool success = backend.InsertAndLock(hashes, slices, /*on_host=*/true);
  EXPECT_FALSE(success);
  EXPECT_EQ(backend.GetSize(), 0);

  // Partial InsertAndLock up to capacity works
  std::vector<std::string> sub_hashes = {"h1", "h2"};
  std::vector<RaidenBlockID> sub_slices = {slices[0], slices[1]};
  EXPECT_TRUE(backend.InsertAndLock(sub_hashes, sub_slices, /*on_host=*/true));
  EXPECT_EQ(backend.GetPinCount("h1"), 1);
  EXPECT_EQ(backend.GetPinCount("h2"), 1);

  // Attempt to InsertAndLock h3 should fail because available_space() is 0
  EXPECT_FALSE(backend.InsertAndLock({"h3"}, {slices[2]}, /*on_host=*/true));
  EXPECT_EQ(backend.GetPinCount("h1"), 1);
  EXPECT_EQ(backend.GetPinCount("h2"), 1);
}

TEST(HostOffloadBackendTest, LookupReturnsRemoteDescriptors) {
  // Setup local gRPC registry server
  auto reg_server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = reg_server->server_address;
  auto registry_client = reg_server->client.get();

  RaidenId remote_node_id{"remote_job", "1", "data", 0};
  std::vector<global_registry::Registration> regs = {
      {.prefix_hash = "r_hash1", .raiden_id = remote_node_id, .block_id = 42},
      {.prefix_hash = "r_hash2", .raiden_id = remote_node_id, .block_id = 43},
  };
  ASSERT_TRUE(registry_client->Register(regs).ok());

  RaidenId local_node_id{"local_job", "0", "data", 0};
  rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(local_node_id.job_name);
  unit_proto.set_job_replica_id(local_node_id.job_replica_id);
  unit_proto.set_data_name(local_node_id.data_name);
  unit_proto.set_data_replica_idx(local_node_id.data_replica_idx);

  ASSERT_OK_AND_ASSIGN(
      auto controller,
      controller::RaidenController::Create(unit_proto, /*num_blocks=*/100,
                                           /*num_shards=*/1,
                                           /*shard_size_bytes=*/1024));

  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.global_registry_address = server_address;
  config.raiden_id = local_node_id;

  auto backend_or = HostOffloadBackend::Create(config, controller.get());
  ASSERT_OK(backend_or.status());
  auto backend = *backend_or;
  EXPECT_EQ(backend->name(), "HostOffloadBackend");

  auto lookup_res = backend->Lookup({"r_hash1", "r_hash2"});
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].first, "r_hash1");
  EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::REMOTE);
  EXPECT_EQ((*lookup_res)[0].second.host_block_id, 42);
  EXPECT_EQ((*lookup_res)[0].second.raiden_id, remote_node_id);

  EXPECT_EQ((*lookup_res)[1].first, "r_hash2");
  EXPECT_EQ((*lookup_res)[1].second.status, BlockStatus::REMOTE);
  EXPECT_EQ((*lookup_res)[1].second.host_block_id, 43);

  // Lookup with miss stops at miss
  auto partial_res = backend->Lookup({"r_hash1", "missing_hash"});
  ASSERT_TRUE(partial_res.ok());
  EXPECT_EQ(partial_res->size(), 1);
}

TEST(HostOffloadBackendTest,
     LookupReturnsHostStatusForMatchingLocalRaidenIdFromGlobalRegistry) {
  // Setup local gRPC registry server
  auto reg_server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = reg_server->server_address;
  auto registry_client = reg_server->client.get();

  RaidenId local_node_id{"local_job", "0", "data", 0};
  std::vector<global_registry::Registration> regs = {
      {.prefix_hash = "local_g_hash",
       .raiden_id = local_node_id,
       .block_id = 99},
  };
  ASSERT_TRUE(registry_client->Register(regs).ok());

  rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(local_node_id.job_name);
  unit_proto.set_job_replica_id(local_node_id.job_replica_id);
  unit_proto.set_data_name(local_node_id.data_name);
  unit_proto.set_data_replica_idx(local_node_id.data_replica_idx);

  ASSERT_OK_AND_ASSIGN(
      auto controller,
      controller::RaidenController::Create(unit_proto, /*num_blocks=*/100,
                                           /*num_shards=*/1,
                                           /*shard_size_bytes=*/1024));

  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.global_registry_address = server_address;
  config.raiden_id = local_node_id;

  auto backend_or = HostOffloadBackend::Create(config, controller.get());
  ASSERT_OK(backend_or.status());
  auto backend = *backend_or;

  auto lookup_res = backend->Lookup({"local_g_hash"});
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 1);
  EXPECT_EQ((*lookup_res)[0].first, "local_g_hash");
  EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::HOST);
  EXPECT_EQ((*lookup_res)[0].second.host_block_id, 99);
  EXPECT_EQ((*lookup_res)[0].second.raiden_id, local_node_id);
}


TEST(HostOffloadBackendTest, CreateRegistersKVTransferSpecFromConfig) {
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(service.get());
  auto server = builder.BuildAndStart();
  std::string server_address = "localhost:" + std::to_string(port);

  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.global_registry_address = server_address;
  config.raiden_id = RaidenId{"node_job", "0", "data", 0};
  config.kv_transfer_spec = KVTransferSpecConfig{
      .block_array_bytes = {4096, 512}, .num_kv_shards = 2, .num_workers = 2};

  auto backend_or = HostOffloadBackend::Create(config);
  ASSERT_OK(backend_or.status());

  // The registry now serves the registered spec.
  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  global_registry::GlobalRegistryClient registry_client(channel);
  auto spec = registry_client.GetKVTransferSpec();
  ASSERT_TRUE(spec.ok()) << spec.status();
  ASSERT_EQ(spec->block_arrays_size(), 2);
  EXPECT_EQ(spec->block_arrays(0).block_bytes(), 4096);
  EXPECT_EQ(spec->block_arrays(1).block_bytes(), 512);
  EXPECT_EQ(spec->num_kv_shards(), 2);
  EXPECT_EQ(spec->num_workers(), 2);

  // Creating another backend with the identical spec is a no-op validation;
  // a differing spec fails creation.
  EXPECT_TRUE(HostOffloadBackend::Create(config).ok());
  config.kv_transfer_spec->num_kv_shards = 4;
  EXPECT_TRUE(
      absl::IsInvalidArgument(HostOffloadBackend::Create(config).status()));

  server->Shutdown();
}

TEST(HostOffloadBackendTest, KVTransferSpecWithoutRegistryFailsCreation) {
  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.kv_transfer_spec = KVTransferSpecConfig{
      .block_array_bytes = {4096}, .num_kv_shards = 1, .num_workers = 1};
  EXPECT_TRUE(absl::IsFailedPrecondition(
      HostOffloadBackend::Create(config).status()));
}

TEST(HostOffloadBackendTest, ServerLifecycleAndControllerInitialization) {
  auto reg_server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = reg_server->server_address;

  RaidenId node_id{"node_job", "0", "data", 0};
  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.global_registry_address = server_address;
  config.raiden_id = node_id;

  // Create RaidenController
  rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(node_id.job_name);
  unit_proto.set_job_replica_id(node_id.job_replica_id);
  unit_proto.set_data_name(node_id.data_name);
  unit_proto.set_data_replica_idx(node_id.data_replica_idx);

  ASSERT_OK_AND_ASSIGN(
      auto controller,
      controller::RaidenController::Create(unit_proto, /*num_blocks=*/100,
                                           /*num_shards=*/1,
                                           /*shard_size_bytes=*/1024));

  auto backend_or = HostOffloadBackend::Create(config, controller.get());
  ASSERT_OK(backend_or.status());
  auto backend = std::dynamic_pointer_cast<HostOffloadBackend>(*backend_or);
  ASSERT_NE(backend, nullptr);

  auto store_server = KVCacheStoreServer::Create();
  // A wildcard bind reports no publishable address,
  // so bind a real, dialable host.
  ASSERT_OK(
      store_server->StartServer(backend.get(), controller.get(), "127.0.0.1"));
  EXPECT_GT(store_server->GetGrpcPort(), 0);
  EXPECT_FALSE(store_server->GetServerAddress().empty());
  store_server->Shutdown();
}

TEST(HostOffloadBackendTest, StartServerStripsControllerPort) {
  RaidenId node_id{"node_job", "0", "data", 0};
  rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(node_id.job_name);
  unit_proto.set_job_replica_id(node_id.job_replica_id);
  unit_proto.set_data_name(node_id.data_name);
  unit_proto.set_data_replica_idx(node_id.data_replica_idx);

  ASSERT_OK_AND_ASSIGN(
      auto controller,
      controller::RaidenController::Create(
          unit_proto, /*num_blocks=*/100, /*num_shards=*/1,
          /*shard_size_bytes=*/1024, /*raiden_orchestrator_address=*/"",
          /*raiden_controller_address=*/"127.0.0.1:12345"));

  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.global_registry_address = "localhost:0";
  config.raiden_id = node_id;

  auto backend_or = HostOffloadBackend::Create(config, controller.get());
  ASSERT_OK(backend_or.status());
  auto backend = std::dynamic_pointer_cast<HostOffloadBackend>(*backend_or);
  ASSERT_NE(backend, nullptr);

  auto store_server = KVCacheStoreServer::Create();
  std::string ctrl_addr = controller->controller_address();
  std::string target_host = ctrl_addr.substr(0, ctrl_addr.rfind(':'));
  ASSERT_OK(store_server->StartServer(backend.get(), controller.get(), target_host));
  EXPECT_GT(store_server->GetGrpcPort(), 0);
  EXPECT_NE(store_server->GetGrpcPort(), 12345);
  store_server->Shutdown();
}

TEST(HostOffloadBackendTest, EndToEndFetchRPC) {
  // 1. Setup global registry server
  auto reg_server = global_registry::CreateTestGlobalRegistryServer();
  std::string reg_address = reg_server->server_address;
  auto registry_client = reg_server->client.get();

  // 2. Setup mock worker server & transfer manager
  auto test_worker_server = controller::CreateTestWorkerServer();
  auto dst_transfer_mock =
      std::make_unique<controller::ShardAwareMockTransferManager>();
  test_worker_server->service->SetTransferManager(
      KVManagerHolder(dst_transfer_mock.get()));

  // 3. Setup orchestrator server
  auto orchestrator_service = std::make_unique<RaidenOrchestrator>();
  grpc::ServerBuilder orch_builder;
  int orch_port = 0;
  orch_builder.AddListeningPort("0.0.0.0:0", grpc::InsecureServerCredentials(),
                                &orch_port);
  orch_builder.RegisterService(orchestrator_service.get());
  auto orchestrator_server = orch_builder.BuildAndStart();
  std::string orchestrator_address = "localhost:" + std::to_string(orch_port);

  // 4. Setup src controller server
  auto src_controller_server = core::controller::CreateTestControllerServer();

  RaidenId src_raiden_id{"src_job", "0", "src_data", 0};
  RaidenId dst_raiden_id{"dst_job", "0", "dst_data", 0};

  rpc::RaidenIdProto src_unit;
  src_unit.set_job_name(src_raiden_id.job_name);
  src_unit.set_job_replica_id(src_raiden_id.job_replica_id);
  src_unit.set_data_name(src_raiden_id.data_name);
  src_unit.set_data_replica_idx(src_raiden_id.data_replica_idx);

  controller::OrchestratorServiceClient orchestrator_client(grpc::CreateChannel(
      orchestrator_address, grpc::InsecureChannelCredentials()));
  ASSERT_OK(orchestrator_client.RegisterController(
      src_unit, src_controller_server->server_address));

  ASSERT_OK(src_controller_server->client->RegisterWorker(
      "worker_0", test_worker_server->server_address,
      {{test_worker_server->server_address, {}}}));

  src_controller_server->service->SetReadRemoteHooks(
      [&](absl::Span<const std::string> h)
          -> absl::StatusOr<std::vector<int32_t>> {
        return std::vector<int32_t>(h.size(), 42);
      },
      [&](absl::Span<const std::string> /*h*/) {});

  // 5. Register remote blocks in GlobalRegistry
  std::vector<global_registry::Registration> registrations = {
      {.prefix_hash = "fetch_hash_1",
       .raiden_id = dst_raiden_id,
       .block_id = 101},
      {.prefix_hash = "fetch_hash_2",
       .raiden_id = dst_raiden_id,
       .block_id = 102},
  };
  ASSERT_OK(registry_client->Register(registrations));

  // 6. Create destination HostOffloadBackend & RaidenController
  rpc::RaidenIdProto dst_unit_proto;
  dst_unit_proto.set_job_name(dst_raiden_id.job_name);
  dst_unit_proto.set_job_replica_id(dst_raiden_id.job_replica_id);
  dst_unit_proto.set_data_name(dst_raiden_id.data_name);
  dst_unit_proto.set_data_replica_idx(dst_raiden_id.data_replica_idx);

  ASSERT_OK_AND_ASSIGN(
      auto dst_controller,
      controller::RaidenController::Create(
          dst_unit_proto, /*num_blocks=*/100, /*num_shards=*/1,
          /*shard_size_bytes=*/1024, orchestrator_address,
          /*raiden_controller_address=*/""));

  BackendConfig dst_config;
  dst_config.type = "HostOffloadBackend";
  dst_config.capacity = 100;
  dst_config.global_registry_address = reg_address;
  dst_config.raiden_id = dst_raiden_id;
  auto backend_or = HostOffloadBackend::Create(dst_config, dst_controller.get());
  ASSERT_OK(backend_or.status());
  auto backend = std::dynamic_pointer_cast<HostOffloadBackend>(*backend_or);
  ASSERT_NE(backend, nullptr);

  std::vector<RaidenBlockID> dst_slices = {
      RaidenBlockID(dst_raiden_id, 101, BlockStatus::HOST),
      RaidenBlockID(dst_raiden_id, 102, BlockStatus::HOST),
  };
  backend->Insert({"fetch_hash_1", "fetch_hash_2"}, dst_slices,
                  /*on_host=*/true);

  core::controller::RaidenControllerClient dst_controller_client(
      dst_controller->controller_address());
  ASSERT_OK(dst_controller_client.RegisterWorker(
      "dst_worker_0", test_worker_server->server_address,
      {{test_worker_server->server_address, {}}}));

  auto store_server = KVCacheStoreServer::Create();
  // A wildcard bind reports no publishable address,
  // so bind a real, dialable host.
  ASSERT_OK(store_server->StartServer(backend.get(), dst_controller.get(),
                                      "127.0.0.1"));
  EXPECT_GT(store_server->GetGrpcPort(), 0);

  // 7. Issue Fetch RPC using KVCacheStoreClient
  auto client_channel = grpc::CreateChannel(store_server->GetServerAddress(),
                                            grpc::InsecureChannelCredentials());
  KVCacheStoreClient client(client_channel);

  std::vector<std::string> hashes = {"fetch_hash_1", "fetch_hash_2"};
  std::vector<int32_t> host_ids = {201, 202};
  auto fetch_res = client
                       .Fetch(hashes, /*device_block_ids=*/{}, host_ids,
                              dst_controller->unit())
                       .Await();
  ASSERT_OK(fetch_res.status());
  EXPECT_THAT(fetch_res->done_block_hashes(),
              UnorderedElementsAre("fetch_hash_1", "fetch_hash_2"));

  store_server->Shutdown();
  orchestrator_server->Shutdown();
}

TEST(HostOffloadBackendTest, LoadMismatchedDeviceBlockCount) {
  auto reg_server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = reg_server->server_address;
  RaidenId node_id{"node_job", "0", "data", 0};

  rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(node_id.job_name);
  unit_proto.set_job_replica_id(node_id.job_replica_id);
  unit_proto.set_data_name(node_id.data_name);
  unit_proto.set_data_replica_idx(node_id.data_replica_idx);

  ASSERT_OK_AND_ASSIGN(
      auto controller,
      controller::RaidenController::Create(unit_proto, /*num_blocks=*/100,
                                           /*num_shards=*/1,
                                           /*shard_size_bytes=*/1024));
  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.global_registry_address = server_address;
  config.raiden_id = node_id;

  auto backend_or = HostOffloadBackend::Create(config, controller.get());
  ASSERT_OK(backend_or.status());
  auto backend = std::dynamic_pointer_cast<HostOffloadBackend>(*backend_or);
  ASSERT_NE(backend, nullptr);

  std::vector<std::string> hashes = {"hash1", "hash2"};
  std::vector<int32_t> dev_ids = {10};  // Mismatched count
  auto load_future = backend->Load(node_id, hashes, dev_ids);
  EXPECT_THAT(load_future.Await(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(HostOffloadBackendTest, LoadSuccess) {
  // Setup GlobalRegistry and register remote block
  auto reg_server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = reg_server->server_address;

  // Setup orchestrator server
  auto orchestrator_service = std::make_unique<RaidenOrchestrator>();
  grpc::ServerBuilder orch_builder;
  int orch_port = 0;
  orch_builder.AddListeningPort("0.0.0.0:0", grpc::InsecureServerCredentials(),
                                &orch_port);
  orch_builder.RegisterService(orchestrator_service.get());
  auto orchestrator_server = orch_builder.BuildAndStart();
  std::string orchestrator_address = "localhost:" + std::to_string(orch_port);

  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  auto registry_client =
      std::make_shared<global_registry::GlobalRegistryClient>(channel);

  RaidenId remote_node_id{"remote_job", "0", "remote_data", 0};
  RaidenId local_node_id{"local_job", "0", "local_data", 0};

  std::vector<global_registry::Registration> regs = {
      {.prefix_hash = "load_hash_1",
       .raiden_id = remote_node_id,
       .block_id = 42},
  };
  ASSERT_OK(registry_client->Register(regs));

  // Setup local RaidenController
  rpc::RaidenIdProto local_unit;
  local_unit.set_job_name(local_node_id.job_name);
  local_unit.set_job_replica_id(local_node_id.job_replica_id);
  local_unit.set_data_name(local_node_id.data_name);
  local_unit.set_data_replica_idx(local_node_id.data_replica_idx);

  ASSERT_OK_AND_ASSIGN(
      auto controller,
      controller::RaidenController::Create(
          local_unit, /*num_blocks=*/100, /*num_shards=*/1,
          /*shard_size_bytes=*/1024, orchestrator_address,
          /*raiden_controller_address=*/""));

  // Setup fake server for remote node to process Fetch
  BackendConfig remote_config;
  remote_config.type = "HostOffloadBackend";
  remote_config.capacity = 100;
  remote_config.global_registry_address = server_address;
  remote_config.raiden_id = remote_node_id;

  auto remote_backend_or =
      HostOffloadBackend::Create(remote_config, controller.get());
  ASSERT_OK(remote_backend_or.status());
  auto remote_backend =
      std::dynamic_pointer_cast<HostOffloadBackend>(*remote_backend_or);
  ASSERT_NE(remote_backend, nullptr);

  std::vector<RaidenBlockID> remote_slices = {
      RaidenBlockID(remote_node_id, 42, BlockStatus::HOST),
  };
  remote_backend->Insert({"load_hash_1"}, remote_slices, /*on_host=*/true);

  auto remote_server = KVCacheStoreServer::Create();
  // A wildcard bind reports no publishable address,
  // so bind a real, dialable host -- this test publishes it below.
  ASSERT_OK(remote_server->StartServer(remote_backend.get(), controller.get(),
                                       "127.0.0.1"));

  ASSERT_OK(registry_client->RegisterStore(
      remote_node_id, remote_server->GetServerAddress(), orchestrator_address));

  BackendConfig local_config;
  local_config.type = "HostOffloadBackend";
  local_config.capacity = 100;
  local_config.global_registry_address = server_address;
  local_config.raiden_id = local_node_id;

  auto local_backend_or =
      HostOffloadBackend::Create(local_config, controller.get());
  ASSERT_OK(local_backend_or.status());
  auto backend =
      std::dynamic_pointer_cast<HostOffloadBackend>(*local_backend_or);
  ASSERT_NE(backend, nullptr);

  // Register local worker in controller
  auto test_worker_server = controller::CreateTestWorkerServer();
  auto dst_transfer_mock =
      std::make_unique<controller::ShardAwareMockTransferManager>();
  test_worker_server->service->SetTransferManager(
      KVManagerHolder(dst_transfer_mock.get()));

  core::controller::RaidenControllerClient controller_client(
      controller->controller_address());
  ASSERT_OK(controller_client.RegisterWorker(
      "worker_0", test_worker_server->server_address,
      {{test_worker_server->server_address, {}}}));

  // Perform Load
  std::vector<std::string> hashes = {"load_hash_1"};
  std::vector<int32_t> dev_ids = {5};
  auto load_future = backend->Load(remote_node_id, hashes, dev_ids);
  EXPECT_OK(load_future.Await());

  remote_server->Shutdown();
  orchestrator_server->Shutdown();
}

TEST(HostOffloadBackendTest, LoadLocalSuccess) {
  RaidenId node_id{"node_job", "0", "data", 0};
  rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(node_id.job_name);
  unit_proto.set_job_replica_id(node_id.job_replica_id);
  unit_proto.set_data_name(node_id.data_name);
  unit_proto.set_data_replica_idx(node_id.data_replica_idx);

  ASSERT_OK_AND_ASSIGN(
      auto controller,
      controller::RaidenController::Create(unit_proto, /*num_blocks=*/100,
                                           /*num_shards=*/1,
                                           /*shard_size_bytes=*/1024));
  auto test_worker_server = controller::CreateTestWorkerServer();
  auto transfer_mock =
      std::make_unique<controller::ShardAwareMockTransferManager>();
  test_worker_server->service->SetTransferManager(
      KVManagerHolder(transfer_mock.get()));

  core::controller::RaidenControllerClient controller_client(
      controller->controller_address());
  ASSERT_OK(controller_client.RegisterWorker(
      "worker_0", test_worker_server->server_address,
      {{test_worker_server->server_address, {}}}));

  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.raiden_id = node_id;

  auto backend_or = HostOffloadBackend::Create(config, controller.get());
  ASSERT_OK(backend_or.status());
  auto backend = std::dynamic_pointer_cast<HostOffloadBackend>(*backend_or);
  ASSERT_NE(backend, nullptr);

  backend->Insert({"local_hash_1"},
                  {RaidenBlockID(node_id, 10, BlockStatus::HOST)},
                  /*on_host=*/true);

  auto load_future = backend->Load(RaidenId{}, {"local_hash_1"}, {5});
  EXPECT_OK(load_future.Await());
}

TEST(HostOffloadBackendTest, LoadLocalMissingBlockError) {
  RaidenId node_id{"node_job", "0", "data", 0};
  rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(node_id.job_name);
  unit_proto.set_job_replica_id(node_id.job_replica_id);
  unit_proto.set_data_name(node_id.data_name);
  unit_proto.set_data_replica_idx(node_id.data_replica_idx);

  ASSERT_OK_AND_ASSIGN(
      auto controller,
      controller::RaidenController::Create(unit_proto, /*num_blocks=*/100,
                                           /*num_shards=*/1,
                                           /*shard_size_bytes=*/1024));
  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.raiden_id = node_id;

  auto backend_or = HostOffloadBackend::Create(config, controller.get());
  ASSERT_OK(backend_or.status());
  auto backend = std::dynamic_pointer_cast<HostOffloadBackend>(*backend_or);
  ASSERT_NE(backend, nullptr);

  auto load_future = backend->Load(RaidenId{}, {"missing_hash"}, {5});
  EXPECT_THAT(load_future.Await(),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST(HostOffloadBackendTest, LoadLocalNonHostBlockError) {
  RaidenId node_id{"node_job", "0", "data", 0};
  rpc::RaidenIdProto unit_proto;
  unit_proto.set_job_name(node_id.job_name);
  unit_proto.set_job_replica_id(node_id.job_replica_id);
  unit_proto.set_data_name(node_id.data_name);
  unit_proto.set_data_replica_idx(node_id.data_replica_idx);

  ASSERT_OK_AND_ASSIGN(
      auto controller,
      controller::RaidenController::Create(unit_proto, /*num_blocks=*/100,
                                           /*num_shards=*/1,
                                           /*shard_size_bytes=*/1024));
  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  config.raiden_id = node_id;

  auto backend_or = HostOffloadBackend::Create(config, controller.get());
  ASSERT_OK(backend_or.status());
  auto backend = std::dynamic_pointer_cast<HostOffloadBackend>(*backend_or);
  ASSERT_NE(backend, nullptr);

  RaidenId remote_id{"remote_job", "0", "data", 0};
  backend->Insert({"remote_hash"},
                  {RaidenBlockID(remote_id, 10, BlockStatus::REMOTE)},
                  /*on_host=*/true);

  auto load_future = backend->Load(RaidenId{}, {"remote_hash"}, {5});
  EXPECT_THAT(load_future.Await(),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(HostOffloadBackendTest, StoreServerOverride) {
  RaidenId local_node_id{"override_job", "0", "cache", 0};
  rpc::RaidenIdProto local_unit;
  local_unit.set_job_name(local_node_id.job_name);
  local_unit.set_job_replica_id(local_node_id.job_replica_id);
  local_unit.set_data_name(local_node_id.data_name);
  local_unit.set_data_replica_idx(local_node_id.data_replica_idx);

  ASSERT_OK_AND_ASSIGN(
      auto controller,
      controller::RaidenController::Create(
          local_unit, /*num_blocks=*/100, /*num_shards=*/1,
          /*shard_size_bytes=*/1024, /*raiden_orchestrator_address=*/"",
          /*raiden_controller_address=*/""));

  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 100;
  auto backend_or = HostOffloadBackend::Create(config, controller.get());
  ASSERT_OK(backend_or.status());
  auto backend = std::dynamic_pointer_cast<HostOffloadBackend>(*backend_or);
  ASSERT_NE(backend, nullptr);
  EXPECT_EQ(backend->store_server(), nullptr);
  ASSERT_TRUE(backend->StartServer("127.0.0.1").ok());
  EXPECT_NE(backend->store_server(), nullptr);
  backend->store_server()->Shutdown();
}

// --- Remote write backend primitives ---------------------------------------

namespace {

rpc::RaidenIdProto ToProto(const RaidenId& id) {
  rpc::RaidenIdProto proto;
  proto.set_job_name(id.job_name);
  proto.set_job_replica_id(id.job_replica_id);
  proto.set_data_name(id.data_name);
  proto.set_data_replica_idx(id.data_replica_idx);
  return proto;
}

}  // namespace

// Lookup cannot answer this question: it stops at the first miss, so a
// scattered subset is unrepresentable in its result.
TEST(HostOffloadBackendWriteRemoteTest, AlreadyPresentReportsAScatteredSubset) {
  HostOffloadBackend backend(/*capacity=*/8);
  RaidenId id{"job", "0", "data", 0};
  backend.Insert({"a", "c"},
                 {RaidenBlockID(id, 1, BlockStatus::HOST),
                  RaidenBlockID(id, 3, BlockStatus::HOST)},
                 /*on_host=*/true);

  EXPECT_THAT(backend.AlreadyPresentHostResident({"a", "b", "c"}),
              UnorderedElementsAre("a", "c"));
  EXPECT_TRUE(backend.AlreadyPresentHostResident({"b"}).empty());
}

// A block that is registered elsewhere but not resident here is not something
// this node can serve, so it must not count as present.
TEST(HostOffloadBackendWriteRemoteTest, AlreadyPresentIgnoresNonHostBlocks) {
  HostOffloadBackend backend(/*capacity=*/8);
  RaidenId id{"job", "0", "data", 0};
  backend.Insert({"remote"}, {RaidenBlockID(id, 1, BlockStatus::REMOTE)},
                 /*on_host=*/true);

  EXPECT_TRUE(backend.AlreadyPresentHostResident({"remote"}).empty());
}

TEST(HostOffloadBackendWriteRemoteTest,
     InsertAllOrNothingInsertsTheWholeBatch) {
  HostOffloadBackend backend(/*capacity=*/8);
  RaidenId id{"job", "0", "data", 0};

  EXPECT_TRUE(backend.InsertAllOrNothing(
      {"a", "b"}, {RaidenBlockID(id, 1, BlockStatus::HOST),
                   RaidenBlockID(id, 2, BlockStatus::HOST)}));
  EXPECT_EQ(backend.GetSize(), 2);
  EXPECT_THAT(backend.AlreadyPresentHostResident({"a", "b"}),
              UnorderedElementsAre("a", "b"));
}

// The whole point of the primitive: one duplicate refuses the batch. Insert()
// would instead rebind the duplicate in place and report partial success,
// which for a remote write orphans the old host block and lets the source
// free blocks the destination does not have.
TEST(HostOffloadBackendWriteRemoteTest, InsertAllOrNothingRefusesADuplicate) {
  HostOffloadBackend backend(/*capacity=*/8);
  RaidenId id{"job", "0", "data", 0};
  backend.Insert({"a"}, {RaidenBlockID(id, 1, BlockStatus::HOST)},
                 /*on_host=*/true);

  EXPECT_FALSE(backend.InsertAllOrNothing(
      {"b", "a"}, {RaidenBlockID(id, 2, BlockStatus::HOST),
                   RaidenBlockID(id, 3, BlockStatus::HOST)}));
  // "b" must not have landed: nothing, not almost-everything.
  EXPECT_TRUE(backend.AlreadyPresentHostResident({"b"}).empty());
  EXPECT_EQ(backend.GetSize(), 1);
}

// available_space(), not capacity(): pinned entries cannot be evicted to make
// room, so a cache full of them has no space no matter how large it is.
TEST(HostOffloadBackendWriteRemoteTest, InsertAllOrNothingRespectsPinnedSpace) {
  HostOffloadBackend backend(/*capacity=*/2);
  RaidenId id{"job", "0", "data", 0};
  ASSERT_TRUE(backend.InsertAndLock({"pinned_a", "pinned_b"},
                                    {RaidenBlockID(id, 1, BlockStatus::HOST),
                                     RaidenBlockID(id, 2, BlockStatus::HOST)},
                                    /*on_host=*/true));

  EXPECT_FALSE(backend.InsertAllOrNothing(
      {"c"}, {RaidenBlockID(id, 3, BlockStatus::HOST)}));
}

// The assertion that matters for E1: a failed registration must leave the LRU
// empty AND the blocks back in the pool. Erasing an entry does not return its
// block, so a rollback that only erased would leak one block per attempt.
TEST(HostOffloadBackendWriteRemoteTest, RollbackInsertErasesAndFreesBlocks) {
  RaidenId id{"job", "0", "data", 0};
  ASSERT_OK_AND_ASSIGN(
      auto controller,
      controller::RaidenController::Create(ToProto(id), /*num_blocks=*/4,
                                           /*num_shards=*/1,
                                           /*shard_size_bytes=*/1024));
  HostOffloadBackend backend(/*capacity=*/8, std::nullopt, id,
                             controller.get());

  auto ids_or = controller->AllocateBlockIds(4);
  ASSERT_TRUE(ids_or.ok()) << ids_or.status().ToString();
  std::vector<int32_t> ids(ids_or->begin(), ids_or->end());
  // The pool is now empty, which is what makes the free observable.
  EXPECT_FALSE(controller->AllocateBlockIds(1).ok());

  std::vector<std::string> hashes = {"a", "b", "c", "d"};
  std::vector<RaidenBlockID> slices;
  for (int32_t block_id : ids) {
    slices.push_back(RaidenBlockID(id, block_id, BlockStatus::HOST));
  }
  ASSERT_TRUE(backend.InsertAllOrNothing(hashes, slices));

  backend.RollbackInsert(hashes, ids);

  EXPECT_TRUE(backend.AlreadyPresentHostResident(hashes).empty());
  EXPECT_EQ(backend.GetSize(), 0);
  auto reallocated = controller->AllocateBlockIds(4);
  EXPECT_TRUE(reallocated.ok())
      << "rollback erased the entries but never returned their blocks: "
      << reallocated.status().ToString();
}

// With no registry there is nothing to advertise, and no way for the blocks to
// be believed reachable when they are not -- so this is success, not an error
// that would fail every write on a registry-less node.
TEST(HostOffloadBackendWriteRemoteTest,
     RegisterBlocksSyncIsOkWithoutARegistry) {
  HostOffloadBackend backend(/*capacity=*/8);
  EXPECT_TRUE(backend.RegisterBlocksSync({"a"}, {1}).ok());
}

TEST(HostOffloadBackendWriteRemoteTest,
     RegisterBlocksSyncPublishesToTheRegistry) {
  auto reg_server = global_registry::CreateTestGlobalRegistryServer();

  RaidenId id{"job_regsync", "0", "data", 0};
  HostOffloadBackend backend(
      /*capacity=*/8, std::nullopt, id, /*raiden_controller=*/nullptr,
      std::make_shared<global_registry::GlobalRegistryClient>(
          reg_server->channel));

  ASSERT_TRUE(backend.RegisterBlocksSync({"a", "b"}, {7, 8}).ok());

  auto looked_up = reg_server->client->Lookup({"a", "b"});
  ASSERT_TRUE(looked_up.ok()) << looked_up.status().ToString();
  ASSERT_EQ(looked_up->size(), 2);
  EXPECT_EQ((*looked_up)[0].block_id(), 7);
  EXPECT_EQ((*looked_up)[1].block_id(), 8);
}

// Unlike Insert's inline Register, which logs and swallows. COMMITTED is only
// allowed to mean "globally reachable", so this failure has to be visible.
TEST(HostOffloadBackendWriteRemoteTest, RegisterBlocksSyncReportsFailure) {
  RaidenId id{"job_regfail", "0", "data", 0};
  // Port 1 is reserved and never listening.
  auto channel =
      grpc::CreateChannel("127.0.0.1:1", grpc::InsecureChannelCredentials());
  HostOffloadBackend backend(
      /*capacity=*/8, std::nullopt, id, /*raiden_controller=*/nullptr,
      std::make_shared<global_registry::GlobalRegistryClient>(channel));

  EXPECT_FALSE(backend.RegisterBlocksSync({"a"}, {1}).ok());
}

TEST(HostOffloadBackendWriteRemoteTest,
     RegisterBlocksSyncRejectsMismatchedSizes) {
  HostOffloadBackend backend(/*capacity=*/8);
  EXPECT_TRUE(
      absl::IsInvalidArgument(backend.RegisterBlocksSync({"a", "b"}, {1})));
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
