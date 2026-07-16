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

#include "tpu_raiden/core/controller/raiden_controller.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/status.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_raiden/core/controller/controller_server.h"
#include "tpu_raiden/core/controller/orchestrator_service_client.h"
#include "tpu_raiden/core/controller/worker_registry.h"
#include "tpu_raiden/core/controller/worker_service_client.h"
#include "tpu_raiden/core/status_macros.h"
#include "tpu_raiden/kv_cache/logical_block_manager.h"
#include "tpu_raiden/kv_cache/raiden_id.h"
#include "tpu_raiden/proto/controller_service.grpc.pb.h"
#include "tpu_raiden/proto/controller_service.pb.h"
#include "tpu_raiden/proto/worker_service.pb.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace controller {
namespace {

bool CompareWorkerIds(absl::string_view a, absl::string_view b) {
  absl::string_view sv_a = a;
  absl::string_view sv_b = b;
  if (absl::ConsumePrefix(&sv_a, "worker_") &&
      absl::ConsumePrefix(&sv_b, "worker_")) {
    int id_a, id_b;
    if (absl::SimpleAtoi(sv_a, &id_a) && absl::SimpleAtoi(sv_b, &id_b)) {
      return id_a < id_b;
    }
  }
  return a < b;
}

rpc::RaidenIdProto ToProto(const kv_cache::RaidenId& id) {
  rpc::RaidenIdProto proto;
  proto.set_job_name(id.job_name);
  proto.set_job_replica_id(id.job_replica_id);
  proto.set_data_name(id.data_name);
  proto.set_data_replica_idx(id.data_replica_idx);
  return proto;
}

absl::StatusOr<proto::CreateBuffersResponse> CreateBuffersForWorker(
    WorkerServiceClient& client, const proto::CreateBuffersRequest& request,
    absl::string_view worker_id, int num_blocks) {
  auto resp_or = client.CreateBuffers(request).Await();
  if (!resp_or.ok()) {
    return absl::Status(resp_or.status().code(),
                        absl::StrCat("Failed to create buffers on worker ",
                                     worker_id, ": ",
                                     resp_or.status().message()));
  }
  if (!resp_or->success()) {
    return absl::FailedPreconditionError(
        absl::StrCat("WorkerService CreateBuffers failed on worker ",
                     worker_id, ": ", resp_or->message()));
  }
  if (resp_or->buffers_size() != num_blocks) {
    return absl::FailedPreconditionError(
        absl::StrCat("WorkerService returned unexpected number of buffers on ",
                     worker_id, ": ", resp_or->buffers_size(),
                     " vs expected ", num_blocks));
  }
  return *resp_or;
}

}  // namespace

void RaidenController::Init(absl::Span<const std::string> worker_addresses,
                            absl::string_view raiden_orchestrator_address) {
  // 1. Set callback on registry
  worker_registry_->SetOnRegisterCallback(
      [this](core::controller::WorkerRegistration& reg) {
        return this->InitializeWorkerBuffers(reg);
      });

  // 2. Start ControllerServer
  controller_server_ = std::make_unique<core::controller::ControllerServer>();
  absl::Status server_status = controller_server_->StartServer(
      worker_registry_, raiden_controller_port_);
  if (!server_status.ok()) {
    LOG(WARNING) << "Failed to start ControllerServer in RaidenController: "
                 << server_status.message();
  }
  // Store the actual port the server bound to
  raiden_controller_port_ = controller_server_->GetGrpcPort();

  controller_server_->SetTransferBuffersCallback(
      [this](rpc::MemoryType src_mem_type, rpc::MemoryType dst_mem_type,
             absl::Span<const int64_t> src_offsets,
             absl::Span<const int64_t> dst_offsets,
             absl::Span<const int64_t> copy_sizes,
             absl::Span<const std::string> peers) {
        return this->TransferBuffers(src_mem_type, dst_mem_type, src_offsets,
                                     dst_offsets, copy_sizes, peers);
      });

  // 3. Register static workers
  for (size_t i = 0; i < worker_addresses.size(); ++i) {
    std::string worker_id = absl::StrCat("worker_", i);
    absl::Status status = worker_registry_->RegisterWorker(
        worker_id, worker_addresses[i], /*raiden_transfer_endpoint=*/"");
    if (!status.ok()) {
      throw std::runtime_error(absl::StrCat(
          "Failed to register static worker ", worker_addresses[i],
          ": ", status.message()));
    }
  }

  // 4. Register with Orchestrator (if address is provided)
  if (!raiden_orchestrator_address.empty()) {
    orchestrator_client_ = std::make_unique<OrchestratorServiceClient>(
        grpc::CreateChannel(std::string(raiden_orchestrator_address),
                            grpc::InsecureChannelCredentials()));
    std::string my_endpoint = absl::StrCat("localhost:", raiden_controller_port_);
    absl::Status status = orchestrator_client_->RegisterController(unit_, my_endpoint);
    if (!status.ok()) {
      throw std::runtime_error(absl::StrCat(
          "Failed to register with orchestrator: ", status.message()));
    }
  }
}

RaidenController::RaidenController(
    const rpc::RaidenIdProto& unit, int num_blocks, int num_shards,
    int64_t shard_size_bytes, int raiden_controller_port,
    absl::string_view raiden_orchestrator_address)
    : unit_(unit),
      num_shards_(num_shards),
      shard_size_bytes_(shard_size_bytes),
      num_total_blocks_(num_blocks),
      worker_registry_(std::make_shared<core::controller::WorkerRegistry>()),
      block_manager_(std::make_unique<kv_cache::LogicalBlockManager>(num_blocks)),
      raiden_controller_port_(raiden_controller_port) {
  Init(/*worker_addresses=*/{}, raiden_orchestrator_address);
}

RaidenController::RaidenController(
    const rpc::RaidenIdProto& unit,
    absl::Span<const std::string> worker_addresses, int num_blocks,
    int num_shards, int64_t shard_size_bytes, int raiden_controller_port,
    absl::string_view raiden_orchestrator_address)
    : unit_(unit),
      num_shards_(num_shards),
      shard_size_bytes_(shard_size_bytes),
      num_total_blocks_(num_blocks),
      worker_registry_(std::make_shared<core::controller::WorkerRegistry>()),
      block_manager_(std::make_unique<kv_cache::LogicalBlockManager>(num_blocks)),
      raiden_controller_port_(raiden_controller_port) {
  Init(worker_addresses, raiden_orchestrator_address);
}

RaidenController::~RaidenController() {
  if (worker_registry_) {
    worker_registry_->SetOnRegisterCallback(nullptr);
  }
  if (controller_server_) {
    controller_server_->SetWorkerRegistry(nullptr);
    controller_server_.reset();
  }

  if (all_sharded_buffers_.empty() || !worker_registry_) return;

  proto::DeleteBuffersRequest request;
  *request.mutable_unit() = unit_;
  for (const auto& sharded_buf : all_sharded_buffers_) {
    *request.add_sharded_buffers() = sharded_buf;
  }

  for (const auto& reg : worker_registry_->GetRegisteredWorkers()) {
    if (reg.worker_service_client) {
      auto resp_or = reg.worker_service_client->DeleteBuffers(request).Await();
      if (!resp_or.ok()) {
        LOG(ERROR) << "Failed to delete buffers on worker " << reg.worker_id
                   << ": " << resp_or.status().message();
      } else if (!resp_or->success()) {
        LOG(ERROR) << "WorkerService DeleteBuffers failed on worker "
                   << reg.worker_id << ": " << resp_or->message();
      }
    }
  }
}

absl::Status RaidenController::InitializeWorkerBuffers(
    core::controller::WorkerRegistration& reg) {
  absl::MutexLock lock(mutex_);

  if (!reg.worker_service_client) {
    return absl::FailedPreconditionError(absl::StrCat(
        "WorkerServiceClient for ", reg.worker_id, " is not initialized"));
  }

  proto::CreateBuffersRequest request;
  *request.mutable_unit() = unit_;
  for (int block_id = 0; block_id < num_total_blocks_; ++block_id) {
    auto* spec = request.add_buffers();
    spec->set_num_shards(num_shards_);
    spec->set_size_bytes(shard_size_bytes_);
  }

  auto resp_or = CreateBuffersForWorker(*reg.worker_service_client, request,
                                        reg.worker_id, num_total_blocks_);
  if (!resp_or.ok()) {
    return absl::FailedPreconditionError(
        absl::StrCat("CreateBuffers RPC failed: ", resp_or.status().message()));
  }
  auto resp = *resp_or;

  std::vector<proto::BufferProto> created_buffers;
  created_buffers.reserve(num_total_blocks_);
  for (int block_id = 0; block_id < num_total_blocks_; ++block_id) {
    created_buffers.push_back(resp.buffers(block_id));
  }

  reg.buffers = std::move(created_buffers);

  if (all_sharded_buffers_.empty()) {
    all_sharded_buffers_ = reg.buffers;
    for (int block_id = 0; block_id < num_total_blocks_; ++block_id) {
      for (const auto& handle_proto : all_sharded_buffers_[block_id].buffer_handles()) {
        handle_to_block_id_[handle_proto.handle()] = block_id;
      }
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<std::vector<proto::BufferProto>> RaidenController::Allocate(
    int num_blocks) {
  absl::MutexLock lock(mutex_);
  ASSIGN_OR_RETURN(std::vector<int> block_ids,
                   block_manager_->Allocate(num_blocks, /*lock=*/true));
  std::vector<proto::BufferProto> result;
  result.reserve(num_blocks);
  for (int block_id : block_ids) {
    result.push_back(all_sharded_buffers_[block_id]);
  }
  return result;
}

absl::Status RaidenController::Deallocate(
    absl::Span<const proto::BufferProto> sharded_buffers) {
  absl::MutexLock lock(mutex_);
  std::vector<int> block_ids;
  block_ids.reserve(sharded_buffers.size());
  for (const auto& sharded_buf : sharded_buffers) {
    if (sharded_buf.buffer_handles().empty()) {
      return absl::InvalidArgumentError("BufferProto has no buffer_handles");
    }
    uint64_t handle = sharded_buf.buffer_handles(0).handle();
    auto it = handle_to_block_id_.find(handle);
    if (it == handle_to_block_id_.end()) {
      return absl::NotFoundError(absl::StrCat(
          "Buffer not found in logical block manager: handle ", handle));
    }
    block_ids.push_back(it->second);
  }
  return block_manager_->Unlock(block_ids);
}

absl::StatusOr<std::vector<int>> RaidenController::AllocateBlockIds(
    int num_blocks) {
  absl::MutexLock lock(mutex_);
  return block_manager_->Allocate(num_blocks, /*lock=*/true);
}

absl::Status RaidenController::DeallocateBlockIds(
    absl::Span<const int> block_ids) {
  absl::MutexLock lock(mutex_);
  return block_manager_->Unlock(block_ids);
}

tsl::Future<> RaidenController::TransferBuffers(
    absl::string_view worker_id, rpc::MemoryType src_mem_type,
    rpc::MemoryType dst_mem_type, absl::Span<const int64_t> src_offsets,
    absl::Span<const int64_t> dst_offsets, absl::Span<const int64_t> copy_sizes,
    absl::string_view peer) {
  if (src_offsets.empty() || src_offsets.size() != dst_offsets.size()) {
    return tsl::Future<>(absl::InvalidArgumentError(
        "Source and destination offsets must have the same non-zero length"));
  }
  if (!copy_sizes.empty() && copy_sizes.size() != src_offsets.size()) {
    return tsl::Future<>(absl::InvalidArgumentError(
        "copy_sizes, if provided, must match the length of src_offsets"));
  }

  auto worker_or = worker_registry_->GetWorker(worker_id);
  if (!worker_or.ok()) {
    return tsl::Future<>(absl::FailedPreconditionError(absl::StrCat(
        "Worker ", worker_id, " is not registered. ",
        "Did you wait for the worker to register?")));
  }
  auto worker_client = worker_or->worker_service_client;
  if (!worker_client) {
    return tsl::Future<>(absl::FailedPreconditionError(absl::StrCat(
        "WorkerServiceClient for ", worker_id, " is not initialized.")));
  }

  proto::TransferBuffersRequest request;
  auto* transfer = request.mutable_transfer();
  transfer->set_src_mem_type(src_mem_type);
  transfer->set_dst_mem_type(dst_mem_type);
  for (int64_t offset : src_offsets) {
    transfer->add_src_offsets(offset);
  }
  for (int64_t offset : dst_offsets) {
    transfer->add_dst_offsets(offset);
  }
  for (int64_t size : copy_sizes) {
    transfer->add_copy_sizes(size);
  }
  // TODO: For remote transfers (e.g. H2H), the peer argument needs to be
  // revised as we do not need the peer address for local D2H and H2D transfers.
  if (!peer.empty()) {
    transfer->set_peer(std::string(peer));
  }
  return worker_client->TransferBuffers(request);
}

tsl::Future<> RaidenController::TransferBuffers(
    rpc::MemoryType src_mem_type, rpc::MemoryType dst_mem_type,
    absl::Span<const int64_t> src_offsets,
    absl::Span<const int64_t> dst_offsets, absl::Span<const int64_t> copy_sizes,
    absl::Span<const std::string> peers) {
  if (src_offsets.empty() || src_offsets.size() != dst_offsets.size()) {
    return tsl::Future<>(absl::InvalidArgumentError(
        "Source and destination offsets must have the same non-zero length"));
  }
  if (!copy_sizes.empty() && copy_sizes.size() != src_offsets.size()) {
    return tsl::Future<>(absl::InvalidArgumentError(
        "copy_sizes, if provided, must match the length of src_offsets"));
  }

  auto workers = worker_registry_->GetRegisteredWorkers();
  if (workers.empty()) {
    return tsl::Future<>(absl::FailedPreconditionError(
        "No registered workers available for TransferBuffers"));
  }

  std::sort(workers.begin(), workers.end(),
            [](const core::controller::WorkerRegistration& a,
               const core::controller::WorkerRegistration& b) {
              return CompareWorkerIds(a.worker_id, b.worker_id);
            });

  if (!peers.empty() && workers.size() != peers.size()) {
    return tsl::Future<>(absl::InvalidArgumentError(
        absl::StrCat("Peers count mismatch: workers has ", workers.size(),
                     ", peers has ", peers.size())));
  }

  std::vector<tsl::Future<>> worker_futures;
  worker_futures.reserve(workers.size());
  for (size_t i = 0; i < workers.size(); ++i) {
    std::string peer = peers.empty() ? "" : peers[i];
    worker_futures.push_back(TransferBuffers(workers[i].worker_id, src_mem_type,
                                             dst_mem_type, src_offsets,
                                             dst_offsets, copy_sizes, peer));
  }

  return tsl::JoinFutures(absl::MakeSpan(worker_futures));
}

absl::StatusOr<std::string> RaidenController::ResolvePeerController(
    const rpc::RaidenIdProto& peer_id) {
  if (!orchestrator_client_) {
    return absl::FailedPreconditionError(
        "Orchestrator client is not initialized. Peer resolution is disabled "
        "when raiden_orchestrator_address is empty.");
  }
  return orchestrator_client_->ResolveController(peer_id);
}

tsl::Future<> RaidenController::ReadRemote(
    const kv_cache::RaidenId& src_raiden_id,
    const std::vector<int32_t>& src_host_block_ids,
    const std::vector<int32_t>& dest_host_block_ids) {
  namespace cproto = ::tpu_raiden::tpu_raiden::proto;

  auto workers = worker_registry_->GetRegisteredWorkers();
  if (workers.empty()) {
    return tsl::Future<>(absl::FailedPreconditionError(
        "No registered workers available for ReadRemote"));
  }

  std::sort(workers.begin(), workers.end(),
            [](const core::controller::WorkerRegistration& a,
               const core::controller::WorkerRegistration& b) {
              return CompareWorkerIds(a.worker_id, b.worker_id);
            });

  std::vector<std::string> dest_worker_endpoints;
  dest_worker_endpoints.reserve(workers.size());
  for (const auto& reg : workers) {
    dest_worker_endpoints.push_back(reg.raiden_transfer_endpoint);
  }

  std::string controller_address;
  {
    absl::MutexLock lock(mutex_);
    auto it = resolved_controllers_.find(src_raiden_id);
    if (it != resolved_controllers_.end()) {
      controller_address = it->second;
    }
  }

  if (controller_address.empty()) {
    rpc::RaidenIdProto src_proto = ToProto(src_raiden_id);
    auto address_or = ResolvePeerController(src_proto);
    if (!address_or.ok()) {
      return tsl::Future<>(address_or.status());
    }
    controller_address = *address_or;
    {
      absl::MutexLock lock(mutex_);
      resolved_controllers_[src_raiden_id] = controller_address;
    }
  }

  std::shared_ptr<cproto::RaidenControllerService::Stub> stub;
  {
    absl::MutexLock lock(mutex_);
    auto it = stubs_.find(controller_address);
    if (it != stubs_.end()) {
      stub = it->second;
    }
  }

  if (stub == nullptr) {
    auto channel = grpc::CreateChannel(controller_address,
                                       grpc::InsecureChannelCredentials());
    stub = cproto::RaidenControllerService::NewStub(channel);
    {
      absl::MutexLock lock(mutex_);
      stubs_[controller_address] = stub;
    }
  }

  cproto::ReadRemoteRequest request;
  request.mutable_src_host_block_ids()->Reserve(src_host_block_ids.size());
  request.mutable_dest_host_block_ids()->Reserve(dest_host_block_ids.size());
  for (int32_t id : src_host_block_ids) {
    request.add_src_host_block_ids(id);
  }
  for (int32_t id : dest_host_block_ids) {
    request.add_dest_host_block_ids(id);
  }
  for (const auto& endpoint : dest_worker_endpoints) {
    request.add_dest_worker_endpoints(endpoint);
  }

  auto [promise, future] = tsl::MakePromise<>();
  auto context = std::make_shared<grpc::ClientContext>();
  auto response = std::make_shared<cproto::ReadRemoteResponse>();

  stub->async()->ReadRemote(
      context.get(), &request, response.get(),
      [context, response, stub,
       promise = std::move(promise).ToShared()](grpc::Status status) {
        if (!status.ok()) {
          promise->Set(absl::InternalError(
              absl::StrCat("ReadRemote RPC failed: ", status.error_message())));
        } else {
          promise->Set(absl::OkStatus());
        }
      });

  return future;
}

}  // namespace controller
}  // namespace tpu_raiden
