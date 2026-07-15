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
#include "tpu_raiden/core/controller/worker_registry.h"
#include "tpu_raiden/core/controller/worker_service_client.h"
#include "tpu_raiden/kv_cache/logical_block_manager.h"
#include "tpu_raiden/proto/worker_service.pb.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace controller {
class OrchestratorServiceClient;
}  // namespace controller

namespace core {
namespace controller {
class RaidenControllerServiceImpl;
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
                   int raiden_controller_port = 0,
                   absl::string_view raiden_orchestrator_address = "");

  // Constructs a RaidenController for multiple worker addresses.
  // It will also start the ControllerServer to allow dynamic registrations
  // and resolve peers.
  RaidenController(const rpc::RaidenIdProto& unit,
                   absl::Span<const std::string> worker_addresses,
                   int num_blocks, int num_shards, int64_t shard_size_bytes,
                   int raiden_controller_port = 0,
                   absl::string_view raiden_orchestrator_address = "");
  // Destructor automatically calls DeleteBuffers to clean up all pre-created
  // buffers on the registered workers.
  ~RaidenController();

  // Allocates num_blocks logical blocks from the pre-created buffer pool and
  // returns the corresponding BufferProtos. No gRPC call is made.
  absl::StatusOr<std::vector<proto::BufferProto>> Allocate(int num_blocks);

  // Unlocks the specified BufferProtos in the local logical block manager so
  // they can be reused. No gRPC call is made.
  absl::Status Deallocate(absl::Span<const proto::BufferProto> sharded_buffers);

  tsl::Future<> TransferBuffers(
      absl::string_view worker_id, rpc::MemoryType src_mem_type,
      rpc::MemoryType dst_mem_type, absl::Span<const int64_t> src_offsets,
      absl::Span<const int64_t> dst_offsets,
      absl::Span<const int64_t> copy_sizes = {}, absl::string_view peer = "");

  // Broadcasts TransferBuffers to all registered workers.
  tsl::Future<> TransferBuffers(
      rpc::MemoryType src_mem_type, rpc::MemoryType dst_mem_type,
      absl::Span<const int64_t> src_offsets,
      absl::Span<const int64_t> dst_offsets,
      absl::Span<const int64_t> copy_sizes = {}, absl::string_view peer = "");

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
  const std::vector<proto::BufferProto>& all_sharded_buffers() const {
    return all_sharded_buffers_;
  }
  int raiden_controller_port() const { return raiden_controller_port_; }

 private:
  void Init(absl::Span<const std::string> worker_addresses,
            absl::string_view raiden_orchestrator_address);

  absl::Status InitializeWorkerBuffers(core::controller::WorkerRegistration& reg);
  rpc::RaidenIdProto unit_;

  int num_shards_;
  int64_t shard_size_bytes_;
  int num_total_blocks_;
  std::vector<proto::BufferProto> all_sharded_buffers_;
  absl::flat_hash_map<uint64_t, int> handle_to_block_id_;
  std::shared_ptr<core::controller::WorkerRegistry> worker_registry_;
  mutable absl::Mutex mutex_;
  std::unique_ptr<kv_cache::LogicalBlockManager> block_manager_
      ABSL_GUARDED_BY(mutex_);

  int raiden_controller_port_;
  std::unique_ptr<OrchestratorServiceClient> orchestrator_client_;
};

}  // namespace controller
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_CONTROLLER_RAIDEN_CONTROLLER_H_
