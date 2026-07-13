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
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "grpcpp/channel.h"
#include "tpu_raiden/core/controller/worker_service_client.h"
#include "tpu_raiden/kv_cache/logical_block_manager.h"
#include "tpu_raiden/proto/worker_service.pb.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace controller {

// Raiden Controller responsible for managing logical KV cache block allocations
// and synchronizing them with remote transfer workers via WorkerService gRPC.
class RaidenController {
 public:
  // Constructs a RaidenController for the given unit by creating an insecure
  // gRPC channel to worker_address, instantiating a WorkerServiceClient, and
  // pre-creating all num_blocks buffers on the remote worker in one shot.
  RaidenController(const rpc::RaidenIdProto& unit,
                   const std::string& worker_address, int num_blocks,
                   int num_shards, int64_t shard_size_bytes);

  // Constructs a RaidenController for the given unit, communicating over
  // worker_channel and pre-creating all num_blocks buffers on the remote
  // worker in one shot.
  RaidenController(const rpc::RaidenIdProto& unit,
                   std::shared_ptr<grpc::Channel> worker_channel,
                   int num_blocks, int num_shards, int64_t shard_size_bytes);

  // Alternative constructor taking an already-instantiated WorkerServiceClient
  // and pre-creating all num_blocks buffers on the remote worker in one shot.
  RaidenController(const rpc::RaidenIdProto& unit,
                   std::unique_ptr<WorkerServiceClient> client, int num_blocks,
                   int num_shards, int64_t shard_size_bytes);

  // Destructor automatically calls DeleteBuffers to clean up all pre-created
  // buffers on the remote worker.
  ~RaidenController();

  // Allocates num_blocks logical blocks from the pre-created buffer pool and
  // returns the corresponding BufferProtos. No gRPC call is made.
  absl::StatusOr<std::vector<proto::BufferProto>> Allocate(int num_blocks);

  // Unlocks the specified BufferProtos in the local logical block manager so
  // they can be reused. No gRPC call is made.
  absl::Status Deallocate(absl::Span<const proto::BufferProto> sharded_buffers);

  // Transfers (copies) disjoint memory regions across memory spaces on the
  // remote transfer worker via WorkerService.
  // If `copy_sizes` is empty, it defaults to copying 1 block for each
  // source/destination offset pair.
  absl::StatusOr<proto::TransferBuffersResponse> TransferBuffers(
      rpc::MemoryType src_mem_type, rpc::MemoryType dst_mem_type,
      absl::Span<const int64_t> src_offsets,
      absl::Span<const int64_t> dst_offsets,
      absl::Span<const int64_t> copy_sizes = {});

  // Accessors for state inspection and testing.
  const rpc::RaidenIdProto& unit() const { return unit_; }
  kv_cache::LogicalBlockManager* block_manager() const {
    return block_manager_.get();
  }
  WorkerServiceClient* client() const { return client_.get(); }
  int num_shards() const { return num_shards_; }
  int64_t shard_size_bytes() const { return shard_size_bytes_; }
  const std::vector<proto::BufferProto>& all_sharded_buffers() const {
    return all_sharded_buffers_;
  }

 private:
  rpc::RaidenIdProto unit_;
  std::unique_ptr<WorkerServiceClient> client_;
  int num_shards_;
  int64_t shard_size_bytes_;
  std::vector<proto::BufferProto> all_sharded_buffers_;
  absl::flat_hash_map<uint64_t, int> handle_to_block_id_;
  mutable absl::Mutex mutex_;
  std::unique_ptr<kv_cache::LogicalBlockManager> block_manager_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace controller
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_CONTROLLER_RAIDEN_CONTROLLER_H_
