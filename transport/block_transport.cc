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

#include "transport/block_transport.h"

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/statusor.h"
#include "core/status_macros.h"
#include "transport/raw_buffer_transport.h"

namespace tpu_raiden {
namespace transport {

namespace {

absl::Status ValidateBlockRange(BlockTransportDelegate* delegate,
                                size_t layer_idx, size_t shard_idx,
                                int block_id, size_t num_blocks,
                                size_t bytes_per_block) {
  if (block_id < 0) {
    return absl::InvalidArgumentError("Negative block id");
  }
  if (bytes_per_block == 0) {
    return absl::InvalidArgumentError("bytes_per_block must be positive");
  }
  uint8_t* base = delegate->GetHostPointer(layer_idx, shard_idx);
  if (base != nullptr) {
    const size_t host_size = delegate->GetHostSize(layer_idx, shard_idx);
    const size_t first_block = static_cast<size_t>(block_id);
    if (first_block > std::numeric_limits<size_t>::max() / bytes_per_block) {
      return absl::OutOfRangeError("Block offset overflows size_t");
    }
    const size_t byte_offset = first_block * bytes_per_block;
    if (num_blocks > std::numeric_limits<size_t>::max() / bytes_per_block) {
      return absl::OutOfRangeError("Block byte size overflows size_t");
    }
    const size_t byte_count = num_blocks * bytes_per_block;
    if (byte_offset > host_size || byte_count > host_size - byte_offset) {
      return absl::OutOfRangeError(absl::StrCat(
          "Block range out of bounds. Block: ", block_id,
          ", Count: ", num_blocks, ", Bytes per block: ", bytes_per_block,
          ", Host size: ", host_size));
    }
  } else {
    for (size_t i = 0; i < num_blocks; ++i) {
      int target_id = block_id + static_cast<int>(i);
      if (delegate->GetBlockHostPointer(layer_idx, shard_idx, target_id) ==
          nullptr) {
        return absl::FailedPreconditionError("Block host pointer is null");
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<MajorOrder> ParseMajorOrder(uint8_t value) {
  switch (value) {
    case static_cast<uint8_t>(MajorOrder::kLayerMajor):
      return MajorOrder::kLayerMajor;
    case static_cast<uint8_t>(MajorOrder::kBlockMajor):
      return MajorOrder::kBlockMajor;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unknown block transport major order: ", value));
  }
}

template <typename Fn>
absl::Status ForEachPayload(MajorOrder major_order, size_t num_layers,
                            size_t num_shards, size_t num_blocks, Fn fn) {
  switch (static_cast<int>(major_order)) {
    case static_cast<int>(MajorOrder::kLayerMajor):
      for (size_t l = 0; l < num_layers; ++l) {
        for (size_t sh = 0; sh < num_shards; ++sh) {
          for (size_t k = 0; k < num_blocks; ++k) {
            RETURN_IF_ERROR(fn(l, sh, k));
          }
        }
      }
      return absl::OkStatus();
    case static_cast<int>(MajorOrder::kBlockMajor):
      for (size_t k = 0; k < num_blocks; ++k) {
        for (size_t l = 0; l < num_layers; ++l) {
          for (size_t sh = 0; sh < num_shards; ++sh) {
            RETURN_IF_ERROR(fn(l, sh, k));
          }
        }
      }
      return absl::OkStatus();
  }
  return absl::InvalidArgumentError("Unknown block transport major order");
}

}  // namespace

BlockTransport::BlockTransport(BlockTransportDelegate* delegate,
                               const std::string& local_ip, int& local_port,
                               bool enable_conn_pool)
    : RawBufferTransport(delegate, local_ip, local_port, enable_conn_pool),
      block_delegate_(delegate) {}

BlockTransport::~BlockTransport() = default;

absl::Status BlockTransport::HandleCustomRequest(int client_fd,
                                                 const PacketHeader& header) {
  size_t bytes_per_block = block_delegate_->bytes_per_block();

  if (header.op == 1) {  // Push
    TF_ASSIGN_OR_RETURN(MajorOrder major_order, ParseMajorOrder(header.flags));
    TF_ASSIGN_OR_RETURN(
        std::vector<int> allocated_ids,
        block_delegate_->AllocateBlocks(header.count_or_size, header.uuid));

    RETURN_IF_ERROR(WriteExact(client_fd, allocated_ids.data(),
                               header.count_or_size * sizeof(int)));

    RETURN_IF_ERROR(ForEachPayload(
        major_order, block_delegate_->num_layers(),
        block_delegate_->num_shards(), header.count_or_size,
        [&](size_t l, size_t sh, size_t k) -> absl::Status {
          ABSL_DCHECK_LT(k, allocated_ids.size());
          const int dst_id = allocated_ids[k];
          RETURN_IF_ERROR(ValidateBlockRange(block_delegate_, l, sh, dst_id, 1,
                                             bytes_per_block));
          uint8_t* dest_ptr =
              block_delegate_->GetBlockHostPointer(l, sh, dst_id);
          return ReadExact(client_fd, dest_ptr, bytes_per_block);
        }));

    RETURN_IF_ERROR(
        block_delegate_->OnBlocksReceived(allocated_ids, header.uuid));
    uint8_t ack = 1;
    RETURN_IF_ERROR(WriteExact(client_fd, &ack, 1));
  } else if (header.op == 6) {  // Explicit Multi-Block Targeted Push
    TF_ASSIGN_OR_RETURN(MajorOrder major_order, ParseMajorOrder(header.flags));
    std::vector<int> allocated_ids(header.count_or_size, 0);
    RETURN_IF_ERROR(ReadExact(client_fd, allocated_ids.data(),
                              header.count_or_size * sizeof(int)));
    uint8_t ack = 1;
    RETURN_IF_ERROR(WriteExact(client_fd, &ack, 1));

    // Notify delegate that a multi-block push has started.
    RETURN_IF_ERROR(
        block_delegate_->OnPushStarted(header.uuid, header.count_or_size));

    RETURN_IF_ERROR(ForEachPayload(
        major_order, block_delegate_->num_layers(),
        block_delegate_->num_shards(), header.count_or_size,
        [&](size_t l, size_t sh, size_t k) -> absl::Status {
          ABSL_DCHECK_LT(k, allocated_ids.size());
          const int dst_id = allocated_ids[k];
          RETURN_IF_ERROR(ValidateBlockRange(block_delegate_, l, sh, dst_id, 1,
                                             bytes_per_block));
          uint8_t* dest_ptr =
              block_delegate_->GetBlockHostPointer(l, sh, dst_id);
          RETURN_IF_ERROR(ReadExact(client_fd, dest_ptr, bytes_per_block));
          // Notify delegate immediately as each block payload is received.
          return block_delegate_->OnBlockPayloadReceived(
              l, sh, dst_id, bytes_per_block, header.uuid);
        }));

    RETURN_IF_ERROR(
        block_delegate_->OnBlocksReceived(allocated_ids, header.uuid));
    ack = 1;
    RETURN_IF_ERROR(WriteExact(client_fd, &ack, 1));
  } else if (header.op == 2) {  // Pull request
    TF_ASSIGN_OR_RETURN(MajorOrder major_order, ParseMajorOrder(header.flags));
    if (block_delegate_->shard_factor() == 0) {
      return absl::InvalidArgumentError("shard_factor must be positive");
    }
    if (header.count_or_size % block_delegate_->shard_factor() != 0) {
      return absl::InvalidArgumentError(
          "Requested remote block count is not divisible by shard_factor");
    }
    PacketHeader resp_header = {};
    resp_header.op = 2;
    resp_header.flags = header.flags;
    resp_header.remote_id = header.local_id;
    resp_header.local_id = 0;
    resp_header.count_or_size = header.count_or_size;

    RETURN_IF_ERROR(WriteExact(client_fd, &resp_header, sizeof(resp_header)));

    size_t local_blocks =
        header.count_or_size / block_delegate_->shard_factor();
    if (header.remote_id >
            static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        local_blocks > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        local_blocks > static_cast<size_t>(std::numeric_limits<int>::max()) -
                           static_cast<size_t>(header.remote_id)) {
      return absl::OutOfRangeError("Requested block range exceeds int range");
    }
    RETURN_IF_ERROR(ForEachPayload(
        major_order, block_delegate_->num_layers(),
        block_delegate_->num_shards(), local_blocks,
        [&](size_t l, size_t sh, size_t k) -> absl::Status {
          const int read_id = static_cast<int>(header.remote_id + k);
          RETURN_IF_ERROR(ValidateBlockRange(block_delegate_, l, sh, read_id, 1,
                                             bytes_per_block));
          RETURN_IF_ERROR(block_delegate_->WaitForBlockRead(l, sh, read_id));
          const uint8_t* src_ptr =
              block_delegate_->GetBlockHostPointer(l, sh, read_id);
          return WriteExact(client_fd, src_ptr, bytes_per_block);
        }));
  } else if (header.op == 4) {  // Single Block Push (Direct)
    int dst_id = static_cast<int>(header.remote_id);
    size_t size_bytes = header.count_or_size;
    uint16_t buf_id = header.buffer_id;

    uint8_t ack = 1;
    RETURN_IF_ERROR(WriteExact(client_fd, &ack, 1));

    RETURN_IF_ERROR(
        ValidateBlockRange(block_delegate_, buf_id, 0, dst_id, 1, size_bytes));
    uint8_t* dest_ptr = block_delegate_->GetBlockHostPointer(buf_id, 0, dst_id);
    if (dest_ptr == nullptr)
      return absl::InvalidArgumentError("Null block pointer");
    RETURN_IF_ERROR(ReadExact(client_fd, dest_ptr, size_bytes));

    RETURN_IF_ERROR(block_delegate_->OnSingleBlockReceived(dst_id, size_bytes));
    ack = 1;
    RETURN_IF_ERROR(WriteExact(client_fd, &ack, 1));
  } else {
    return absl::UnimplementedError(
        absl::StrCat("Unsupported block transport op: ", header.op));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int>> BlockTransport::Push(
    const std::string& peer, const std::vector<int>& src_block_ids,
    const std::vector<int>& dst_block_ids, int parallelism,
    MajorOrder major_order, uint64_t uuid) {
  size_t num_blocks = src_block_ids.size();
  if (num_blocks == 0) {
    return absl::InvalidArgumentError("Block list cannot be empty");
  }

  int P = parallelism;
  if (P <= 0) {
    return absl::InvalidArgumentError("parallelism must be positive");
  }
  if (static_cast<int>(num_blocks) < P) P = num_blocks;

  std::vector<int> allocated_ids(num_blocks, 0);
  std::vector<std::thread> threads;
  std::vector<absl::Status> statuses(P, absl::OkStatus());

  threads.reserve(P);
  const size_t base_blocks_per_stream = num_blocks / P;
  const size_t remainder = num_blocks % P;
  size_t block_offset = 0;
  for (int i = 0; i < P; ++i) {
    const size_t block_count =
        base_blocks_per_stream + (static_cast<size_t>(i) < remainder ? 1 : 0);
    threads.push_back(std::thread(
        &BlockTransport::H2hWriteWorker, this, i, peer, block_offset,
        block_count, std::ref(src_block_ids), std::ref(dst_block_ids),
        std::ref(allocated_ids), std::ref(statuses), major_order, uuid));
    block_offset += block_count;
  }

  for (auto& t : threads) {
    if (t.joinable()) t.join();
  }

  for (int i = 0; i < P; ++i) {
    if (!statuses[i].ok()) return statuses[i];
  }

  return allocated_ids;
}

absl::Status BlockTransport::WriteBlockDirect(const std::string& peer,
                                              int remote_block_id,
                                              const uint8_t* data_ptr,
                                              size_t size_bytes) {
  auto status_or_fd = AcquireConnection(peer);
  if (!status_or_fd.ok()) return status_or_fd.status();
  int fd = status_or_fd.value();
  bool ok_to_pool = false;
  auto fd_cleaner = absl::MakeCleanup([&] {
    if (ok_to_pool) {
      ReleaseConnection(peer, fd);
    } else {
      shutdown(fd, SHUT_RDWR);
      close(fd);
    }
  });

  PacketHeader header = {};
  header.op = 4;
  header.buffer_id = 0;
  header.remote_id = static_cast<uint32_t>(remote_block_id);
  header.count_or_size = static_cast<uint32_t>(size_bytes);

  RETURN_IF_ERROR(WriteExact(fd, &header, sizeof(header)));

  uint8_t ack = 0;
  RETURN_IF_ERROR(ReadExact(fd, &ack, 1));
  if (ack != 1) return absl::InternalError("WriteBlockDirect handshake failed");

  RETURN_IF_ERROR(WriteExact(fd, data_ptr, size_bytes));

  ack = 0;
  RETURN_IF_ERROR(ReadExact(fd, &ack, 1));
  if (ack != 1)
    return absl::InternalError("WriteBlockDirect completion ack failed");

  ok_to_pool = true;
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int>> BlockTransport::Pull(
    const std::string& peer, const std::vector<int>& src_block_ids,
    const std::vector<int>& local_block_ids,
    const std::vector<uint8_t*>& explicit_dst_ptrs, int parallelism,
    MajorOrder major_order, BlockReceivedCallback on_block_received) {
  size_t num_blocks = src_block_ids.size();
  if (num_blocks == 0) {
    return absl::InvalidArgumentError("Block list cannot be empty");
  }

  if (block_delegate_->shard_factor() == 0) {
    return absl::InvalidArgumentError("shard_factor must be positive");
  }
  if (num_blocks % block_delegate_->shard_factor() != 0) {
    return absl::InvalidArgumentError(
        "Block count must be divisible by shard_factor");
  }
  size_t local_blocks = num_blocks / block_delegate_->shard_factor();
  if (local_blocks == 0) {
    return absl::InvalidArgumentError("Local block list cannot be empty");
  }
  if (!explicit_dst_ptrs.empty() &&
      explicit_dst_ptrs.size() !=
          block_delegate_->num_layers() * block_delegate_->num_shards()) {
    return absl::InvalidArgumentError("explicit_dst_ptrs size mismatch");
  }

  std::vector<int> allocated_ids;
  if (!local_block_ids.empty()) {
    if (local_block_ids.size() != local_blocks) {
      return absl::InvalidArgumentError("local_block_ids size mismatch");
    }
    allocated_ids = local_block_ids;
  } else {
    TF_ASSIGN_OR_RETURN(allocated_ids,
                        block_delegate_->AllocateBlocks(local_blocks));
  }

  int P = parallelism;
  if (P <= 0) {
    return absl::InvalidArgumentError("parallelism must be positive");
  }
  if (static_cast<int>(local_blocks) < P) P = local_blocks;

  std::vector<std::thread> threads;
  std::vector<absl::Status> statuses(P, absl::OkStatus());

  threads.reserve(P);
  const size_t base_blocks_per_stream = local_blocks / P;
  const size_t remainder = local_blocks % P;
  size_t local_block_offset = 0;
  size_t remote_block_offset = 0;

  for (int i = 0; i < P; ++i) {
    const size_t local_block_count =
        base_blocks_per_stream + (static_cast<size_t>(i) < remainder ? 1 : 0);
    const size_t remote_block_count =
        local_block_count * block_delegate_->shard_factor();

    threads.push_back(
        std::thread(&BlockTransport::H2hReadWorker, this, i, peer,
                    local_block_offset, local_block_count, remote_block_offset,
                    remote_block_count, std::ref(src_block_ids),
                    std::ref(allocated_ids), std::ref(explicit_dst_ptrs),
                    std::ref(statuses), major_order, on_block_received));
    local_block_offset += local_block_count;
    remote_block_offset += remote_block_count;
  }

  for (auto& t : threads) {
    if (t.joinable()) t.join();
  }

  for (int i = 0; i < P; ++i) {
    if (!statuses[i].ok()) return statuses[i];
  }

  return allocated_ids;
}

void BlockTransport::H2hWriteWorker(int stream_idx, const std::string& peer,
                                    size_t block_offset, size_t block_count,
                                    const std::vector<int>& src_block_ids,
                                    const std::vector<int>& dst_block_ids,
                                    std::vector<int>& allocated_ids,
                                    std::vector<absl::Status>& statuses,
                                    MajorOrder major_order, uint64_t uuid) {
  auto status_or_fd = AcquireConnection(peer);
  if (!status_or_fd.ok()) {
    statuses[stream_idx] = status_or_fd.status();
    return;
  }
  int fd = status_or_fd.value();
  bool ok_to_pool = false;
  auto fd_cleaner = absl::MakeCleanup([&] {
    if (ok_to_pool) {
      ReleaseConnection(peer, fd);
    } else {
      shutdown(fd, SHUT_RDWR);
      close(fd);
    }
  });

  PacketHeader header = {};
  header.op = dst_block_ids.empty() ? 1 : 6;
  header.flags = static_cast<uint8_t>(major_order);
  header.buffer_id = 0;
  header.remote_id = 0;
  header.local_id = 0;
  header.uuid = uuid;
  if (block_count > std::numeric_limits<uint32_t>::max()) {
    statuses[stream_idx] = absl::OutOfRangeError("Block count exceeds uint32");
    return;
  }
  header.count_or_size = static_cast<uint32_t>(block_count);

  absl::Status s = WriteExact(fd, &header, sizeof(header));
  if (!s.ok()) {
    statuses[stream_idx] = s;
    return;
  }

  if (header.op == 6) {
    if (block_offset + block_count > dst_block_ids.size() ||
        block_offset + block_count > allocated_ids.size()) {
      statuses[stream_idx] =
          absl::InvalidArgumentError("Vector sizes mismatch for explicit push");
      return;
    }
    s = WriteExact(fd, &dst_block_ids[block_offset], block_count * sizeof(int));
    if (!s.ok()) {
      statuses[stream_idx] = s;
      return;
    }
    uint8_t ack = 0;
    s = ReadExact(fd, &ack, 1);
    if (!s.ok() || ack != 1) {
      statuses[stream_idx] =
          absl::InternalError("Explicit push destination handshake failed");
      return;
    }
    for (size_t k = 0; k < block_count; ++k) {
      allocated_ids[block_offset + k] = dst_block_ids[block_offset + k];
    }
  } else {
    if (block_offset + block_count > allocated_ids.size()) {
      statuses[stream_idx] =
          absl::InvalidArgumentError("allocated_ids size mismatch for pull");
      return;
    }
    std::vector<int> stream_allocated_ids(block_count, 0);
    s = ReadExact(fd, stream_allocated_ids.data(), block_count * sizeof(int));
    if (!s.ok()) {
      statuses[stream_idx] = s;
      return;
    }

    for (size_t k = 0; k < block_count; ++k) {
      ABSL_DCHECK_LT(block_offset + k, allocated_ids.size());
      ABSL_DCHECK_LT(k, stream_allocated_ids.size());
      allocated_ids[block_offset + k] = stream_allocated_ids[k];
    }
  }

  size_t bytes_per_block = block_delegate_->bytes_per_block();

  s = ForEachPayload(
      major_order, block_delegate_->num_layers(), block_delegate_->num_shards(),
      block_count, [&](size_t l, size_t sh, size_t k) -> absl::Status {
        const uint8_t* base_host_ptr = block_delegate_->GetHostPointer(l, sh);
        ABSL_DCHECK_LT(block_offset + k, src_block_ids.size());
        const int src_id = src_block_ids[block_offset + k];
        RETURN_IF_ERROR(ValidateBlockRange(block_delegate_, l, sh, src_id, 1,
                                           bytes_per_block));
        // Block until the local D2H copy has finished writing to this host
        // buffer block!
        RETURN_IF_ERROR(block_delegate_->WaitForBlockRead(l, sh, src_id));

        const uint8_t* src_ptr = base_host_ptr + src_id * bytes_per_block;
        return WriteExact(fd, src_ptr, bytes_per_block);
      });
  if (!s.ok()) {
    statuses[stream_idx] = s;
    return;
  }

  uint8_t ack = 0;
  s = ReadExact(fd, &ack, 1);
  if (!s.ok() || ack != 1) {
    statuses[stream_idx] = absl::InternalError("Push verification failed");
    return;
  }

  ok_to_pool = true;
}

void BlockTransport::H2hReadWorker(
    int stream_idx, const std::string& peer, size_t local_block_offset,
    size_t local_block_count, size_t remote_block_offset,
    size_t remote_block_count, const std::vector<int>& src_block_ids,
    const std::vector<int>& allocated_ids,
    const std::vector<uint8_t*>& explicit_dst_ptrs,
    std::vector<absl::Status>& statuses, MajorOrder major_order,
    BlockReceivedCallback on_block_received) {
  auto status_or_fd = AcquireConnection(peer);
  if (!status_or_fd.ok()) {
    statuses[stream_idx] = status_or_fd.status();
    return;
  }
  int fd = status_or_fd.value();
  bool ok_to_pool = false;
  auto fd_cleaner = absl::MakeCleanup([&] {
    if (ok_to_pool) {
      ReleaseConnection(peer, fd);
    } else {
      shutdown(fd, SHUT_RDWR);
      close(fd);
    }
  });

  size_t SF = block_delegate_->shard_factor();

  if (remote_block_offset > src_block_ids.size() ||
      remote_block_count > src_block_ids.size() - remote_block_offset) {
    statuses[stream_idx] =
        absl::OutOfRangeError("Remote block range exceeds source block list");
    return;
  }

  struct PullChunk {
    size_t local_start_idx;
    size_t local_count;
    int base_remote_id;
    size_t remote_count;
  };
  std::vector<PullChunk> chunks;

  if (local_block_count > 0) {
    size_t curr_local_start = 0;
    size_t curr_local_count = 1;
    ABSL_DCHECK_LT(remote_block_offset, src_block_ids.size());
    int curr_base_remote_id = src_block_ids[remote_block_offset];

    for (size_t k = 1; k < local_block_count; ++k) {
      ABSL_DCHECK_LT(remote_block_offset + k * SF - 1, src_block_ids.size());
      int prev_last_remote = src_block_ids[remote_block_offset + k * SF - 1];
      ABSL_DCHECK_LT(remote_block_offset + k * SF, src_block_ids.size());
      int curr_first_remote = src_block_ids[remote_block_offset + k * SF];

      if (curr_first_remote == prev_last_remote + 1) {
        curr_local_count++;
      } else {
        chunks.push_back({curr_local_start, curr_local_count,
                          curr_base_remote_id, curr_local_count * SF});
        curr_local_start = k;
        curr_local_count = 1;
        curr_base_remote_id = curr_first_remote;
      }
    }
    chunks.push_back({curr_local_start, curr_local_count, curr_base_remote_id,
                      curr_local_count * SF});
  }

  size_t bytes_per_block = block_delegate_->bytes_per_block();

  for (const auto& chunk : chunks) {
    PacketHeader header = {};
    header.op = 2;  // Pull request
    header.flags = static_cast<uint8_t>(major_order);
    int remote_read_block_id =
        block_delegate_->GetRemoteReadBlockId(chunk.base_remote_id, 0);
    if (remote_read_block_id < 0 ||
        static_cast<uint64_t>(remote_read_block_id) >
            std::numeric_limits<uint32_t>::max() ||
        chunk.remote_count > std::numeric_limits<uint32_t>::max()) {
      statuses[stream_idx] =
          absl::OutOfRangeError("Remote block range exceeds transport header");
      return;
    }
    header.remote_id = static_cast<uint32_t>(remote_read_block_id);
    header.count_or_size = static_cast<uint32_t>(chunk.remote_count);

    absl::Status s = WriteExact(fd, &header, sizeof(header));
    if (!s.ok()) {
      statuses[stream_idx] = s;
      return;
    }

    PacketHeader resp_header = {};
    s = ReadExact(fd, &resp_header, sizeof(resp_header));
    if (!s.ok()) {
      statuses[stream_idx] = s;
      return;
    }
    if (resp_header.op != 2 ||
        resp_header.count_or_size != chunk.remote_count) {
      statuses[stream_idx] =
          absl::InternalError("Unexpected block pull response header");
      return;
    }
    if (resp_header.flags != static_cast<uint8_t>(major_order)) {
      statuses[stream_idx] =
          absl::InternalError("Unexpected block pull response major order");
      return;
    }

    s = ForEachPayload(
        major_order, block_delegate_->num_layers(),
        block_delegate_->num_shards(), chunk.local_count,
        [&](size_t l, size_t sh, size_t k) -> absl::Status {
          uint8_t* base_host_ptr = nullptr;
          if (!explicit_dst_ptrs.empty()) {
            base_host_ptr =
                explicit_dst_ptrs[l * block_delegate_->num_shards() + sh];
          } else {
            base_host_ptr = block_delegate_->GetHostPointer(l, sh);
          }
          if (base_host_ptr == nullptr) {
            return absl::FailedPreconditionError(
                "Destination host pointer is null");
          }

          ABSL_DCHECK_LT(local_block_offset + chunk.local_start_idx + k,
                         allocated_ids.size());
          const int dst_id =
              allocated_ids[local_block_offset + chunk.local_start_idx + k];
          if (dst_id < 0) {
            return absl::InvalidArgumentError(
                "Destination block id is negative");
          }
          if (explicit_dst_ptrs.empty()) {
            RETURN_IF_ERROR(ValidateBlockRange(block_delegate_, l, sh, dst_id,
                                               1, bytes_per_block));
          }
          uint8_t* dst_ptr = base_host_ptr + dst_id * bytes_per_block;
          RETURN_IF_ERROR(ReadExact(fd, dst_ptr, bytes_per_block));
          if (on_block_received != nullptr) {
            RETURN_IF_ERROR(on_block_received(l, sh, dst_id, bytes_per_block));
          }
          return absl::OkStatus();
        });
    if (!s.ok()) {
      statuses[stream_idx] = s;
      return;
    }
  }
  ok_to_pool = true;
}

}  // namespace transport
}  // namespace tpu_raiden