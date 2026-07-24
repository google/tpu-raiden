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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_CONTROLLER_RAIDEN_CONTROLLER_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_CONTROLLER_RAIDEN_CONTROLLER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "grpcpp/channel.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_raiden/core/buffer.h"
#include "tpu_raiden/core/controller/controller_service.h"
#include "tpu_raiden/core/controller/worker_registry.h"
#include "tpu_raiden/core/controller/worker_service_client.h"
#include "tpu_raiden/kv_cache/logical_block_manager.h"
#include "tpu_raiden/kv_cache/raiden_id.h"
#include "tpu_raiden/proto/controller_service.grpc.pb.h"
#include "tpu_raiden/proto/worker_service.pb.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace controller {
class OrchestratorServiceClient;
}  // namespace controller

namespace core {
namespace controller {
class RaidenControllerServiceImpl;
class ControllerServer;
}  // namespace controller
}  // namespace core

namespace controller {

// Raiden Controller responsible for managing logical KV cache block allocations
// and synchronizing them with remote transfer workers via WorkerService gRPC.
class RaidenController {
 public:
  // Constructs a RaidenController for the given unit.
  // The RaidenController sets up its own ControllerService and local
  // LogicalBlockManager. Workers will dynamically register with it.
  //
  // If `raiden_orchestrator_address` is empty, this controller will run without
  // registering itself with the orchestrator. In this mode, peer resolution is
  // disabled (ResolvePeerController will fail), but local transfers (H2D, D2H)
  // and H2H transfers where the peer address is provided directly will function
  // normally.
  RaidenController(const rpc::RaidenIdProto& unit, int num_blocks,
                   int num_shards, int64_t shard_size_bytes,
                   absl::string_view raiden_orchestrator_address = "",
                   absl::string_view raiden_controller_address = "");

  // Constructs a RaidenController for multiple worker addresses.
  // It will also start the ControllerServer to allow dynamic registrations
  // and resolve peers.
  RaidenController(const rpc::RaidenIdProto& unit,
                   absl::Span<const std::string> worker_addresses,
                   int num_blocks, int num_shards, int64_t shard_size_bytes,
                   absl::string_view raiden_orchestrator_address = "",
                   absl::string_view raiden_controller_address = "");
  // Destructor automatically calls DeleteBuffers to clean up all pre-created
  // buffers on the registered workers.
  ~RaidenController();

  // Legacy API (Physical/BufferProto Mode):
  // Allocates `num_blocks` logical blocks and returns their associated
  // `BufferProto`s containing sharded physical buffer handles. Use these if you
  // need worker-level raw buffer handles. No RPC is performed.
  absl::StatusOr<std::vector<proto::BufferProto>> Allocate(int num_blocks);

  // Allocates num_blocks logical blocks from the pre-created buffer pool and
  // returns the corresponding Buffers. No gRPC call is made.
  absl::StatusOr<std::vector<Buffer>> AllocateBuffers(int num_blocks);

  // Unlocks the specified Buffers in the local logical block manager so
  // they can be reused. No gRPC call is made.
  absl::Status DeallocateBuffers(absl::Span<const Buffer> buffers);

  // Legacy API (Physical/BufferProto Mode):
  // Deallocates logical blocks associated with the provided `BufferProto`s.
  // No RPC is performed.
  absl::Status Deallocate(absl::Span<const proto::BufferProto> sharded_buffers);

  // New API (Logical/Block ID Mode):
  // Allocates `num_blocks` logical blocks from the block manager and returns
  // their raw logical block IDs (0..N-1). No RPC is performed.
  absl::StatusOr<std::vector<int>> AllocateBlockIds(int num_blocks);

  // New API (Logical/Block ID Mode):
  // Unlocks/deallocates the given logical block IDs in the block manager.
  // No RPC is performed.
  absl::Status DeallocateBlockIds(absl::Span<const int> block_ids);

  // New API (Logical/Block ID Mode):
  // Allocates exactly the given logical block IDs in the block manager,
  // locked. The whole batch is validated first: fails without changing any
  // state if an ID is out of range, already allocated, or duplicated. No RPC
  // is performed.
  absl::Status AllocateTargetBlockIds(absl::Span<const int> block_ids);

  // Targeted worker transfer
  tsl::Future<> TransferBuffers(absl::string_view worker_id,
                                absl::Span<const Buffer> src_buffers,
                                absl::Span<const Buffer> dst_buffers,
                                absl::Span<const int64_t> copy_sizes = {});

  // Broadcast transfer to all registered workers
  tsl::Future<> TransferBuffers(absl::Span<const Buffer> src_buffers,
                                absl::Span<const Buffer> dst_buffers,
                                absl::Span<const int64_t> copy_sizes = {});

  // Initiates remote read from source controller. block_hashes (parallel to the
  // block ids) let the source verify/pin the blocks in its LRU before transfer.
  tsl::Future<> ReadRemote(const kv_cache::RaidenId& src_raiden_id,
                           const std::vector<int32_t>& src_host_block_ids,
                           const std::vector<int32_t>& dest_host_block_ids,
                           const std::vector<std::string>& block_hashes = {});

  // Registers the ReadRemote step-6a verify/pin and unpin hooks (invoked when
  // this controller acts as the SOURCE of a remote read). Forwards to the hosted
  // ControllerService.
  void SetReadRemoteHooks(
      core::controller::RaidenControllerServiceImpl::ValidateAndPinCallback
          validate_and_pin,
      core::controller::RaidenControllerServiceImpl::UnpinCallback unpin);

  // Resolves a peer controller's ControllerService address via the
  // orchestrator.
  absl::StatusOr<std::string> ResolvePeerController(
      const rpc::RaidenIdProto& peer_id);

  // Accessors for state inspection and testing.
  const rpc::RaidenIdProto& unit() const { return unit_; }
  kv_cache::LogicalBlockManager* block_manager() const {
    return block_manager_.get();
  }
  std::shared_ptr<core::controller::WorkerRegistry> worker_registry() const {
    return worker_registry_;
  }

  int num_shards() const { return num_shards_; }
  int64_t shard_size_bytes() const { return shard_size_bytes_; }
  std::string controller_address() const { return raiden_controller_address_; }
  const std::vector<proto::BufferProto>& all_sharded_buffers() const {
    return all_sharded_buffers_;
  }

 private:
  absl::StatusOr<proto::TransferBuffersRequest> BuildTransferBuffersRequest(
      absl::Span<const Buffer> src_buffers,
      absl::Span<const Buffer> dst_buffers,
      absl::Span<const int64_t> copy_sizes);

  void Init(absl::Span<const std::string> worker_addresses,
            absl::string_view raiden_orchestrator_address,
            absl::string_view raiden_controller_address);

  absl::Status InitializeWorkerBuffers(
      core::controller::WorkerRegistration& reg);
  rpc::RaidenIdProto unit_;

  int num_shards_;
  int64_t shard_size_bytes_;
  int num_total_blocks_;
  std::vector<proto::BufferProto> all_sharded_buffers_;
  std::shared_ptr<core::controller::WorkerRegistry> worker_registry_;
  mutable absl::Mutex mutex_;
  std::unique_ptr<kv_cache::LogicalBlockManager> block_manager_
      ABSL_GUARDED_BY(mutex_);
  std::string raiden_controller_address_;

  std::unique_ptr<core::controller::ControllerServer>
      private_controller_server_;
  // The active ControllerServer hosting this controller's service (either
  // private_controller_server_ or the shared singleton). Used to register the
  // ReadRemote step-6a hooks after construction. Not owned.
  core::controller::ControllerServer* active_server_ = nullptr;
  std::unique_ptr<OrchestratorServiceClient> orchestrator_client_;
  absl::flat_hash_map<kv_cache::RaidenId, std::string, kv_cache::RaidenIdHash>
      resolved_controllers_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<
      std::string,
      std::shared_ptr<
          ::tpu_raiden::proto::RaidenControllerService::Stub>>
      stubs_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace controller
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_CONTROLLER_RAIDEN_CONTROLLER_H_
