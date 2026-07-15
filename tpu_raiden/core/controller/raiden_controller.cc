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

#include <stdexcept>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_raiden/core/controller/controller_server.h"
#include "tpu_raiden/core/controller/orchestrator_service_client.h"
#include "tpu_raiden/core/controller/worker_service_client.h"
#include "tpu_raiden/kv_cache/logical_block_manager.h"
#include "tpu_raiden/proto/worker_service.pb.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace controller {
namespace {

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
  auto& server = core::controller::ControllerServer::GetInstance();
  absl::Status server_status =
      server.StartServer(worker_registry_, raiden_controller_port_);
  if (!server_status.ok()) {
    LOG(WARNING) << "Failed to start ControllerServer in RaidenController: "
                 << server_status.message();
  }
  // Store the actual port the server bound to
  raiden_controller_port_ = server.GetGrpcPort();

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

  worker_registry_->SetOnRegisterCallback(nullptr);
  core::controller::ControllerServer::GetInstance().SetWorkerRegistry(nullptr);
}

absl::Status RaidenController::InitializeWorkerBuffers(
    core::controller::WorkerRegistration& reg) {
  absl::MutexLock lock(&mutex_);

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
  absl::MutexLock lock(&mutex_);
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
  absl::MutexLock lock(&mutex_);
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
  if (!peer.empty()) {
    transfer->set_peer(std::string(peer));
  }
  return worker_client->TransferBuffers(request);
}

tsl::Future<> RaidenController::TransferBuffers(
    rpc::MemoryType src_mem_type, rpc::MemoryType dst_mem_type,
    absl::Span<const int64_t> src_offsets,
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
  if (!peer.empty()) {
    transfer->set_peer(std::string(peer));
  }

  auto workers = worker_registry_->GetRegisteredWorkers();
  if (workers.empty()) {
    return tsl::Future<>(absl::FailedPreconditionError(
        "No registered workers available for TransferBuffers"));
  }

  std::vector<tsl::Future<>> worker_futures;
  for (const auto& reg : workers) {
    if (reg.worker_service_client) {
      worker_futures.push_back(
          reg.worker_service_client->TransferBuffers(request));
    }
  }

  if (worker_futures.empty()) {
    return tsl::Future<>(absl::FailedPreconditionError(
        "No active WorkerServiceClient available for TransferBuffers"));
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

}  // namespace controller
}  // namespace tpu_raiden
