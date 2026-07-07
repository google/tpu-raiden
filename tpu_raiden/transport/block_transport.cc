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

#include "tpu_raiden/transport/block_transport.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_raiden/core/status_macros.h"
#include "tpu_raiden/core/tpu_utils.h"
#include "tpu_raiden/transport/raw_buffer_transport.h"

namespace tpu_raiden {
namespace transport {

namespace {

std::string GetLocalAddr(int fd) {
  struct sockaddr_storage addr;
  socklen_t addr_len = sizeof(addr);
  if (getsockname(fd, (struct sockaddr*)&addr, &addr_len) == -1) {
    LOG(ERROR) << "getsockname failed";
    return "";
  }
  char ip_str[INET6_ADDRSTRLEN];
  int port = 0;
  if (addr.ss_family == AF_INET) {
    struct sockaddr_in* s = (struct sockaddr_in*)&addr;
    inet_ntop(AF_INET, &s->sin_addr, ip_str, sizeof(ip_str));
    port = ntohs(s->sin_port);
    return absl::StrCat(ip_str, ":", port);
  } else if (addr.ss_family == AF_INET6) {
    struct sockaddr_in6* s = (struct sockaddr_in6*)&addr;
    inet_ntop(AF_INET6, &s->sin6_addr, ip_str, sizeof(ip_str));
    port = ntohs(s->sin6_port);
    return absl::StrCat("[", ip_str, "]:", port);
  }
  return "";
}

#ifndef UIO_MAXIOV
#define UIO_MAXIOV 1024
#endif

#ifndef IOV_MAX
#define IOV_MAX UIO_MAXIOV
#endif

absl::Status WriteVExact(int fd, const std::vector<BlockChunk>& chunks) {
  std::vector<struct iovec> iov;
  iov.reserve(chunks.size());
  for (const auto& chunk : chunks) {
    iov.push_back({.iov_base = chunk.ptr, .iov_len = chunk.size});
  }

  size_t iov_idx = 0;
  while (iov_idx < iov.size()) {
    size_t batch_size =
        std::min(iov.size() - iov_idx, static_cast<size_t>(IOV_MAX));

    size_t batch_remaining = batch_size;
    while (batch_remaining > 0) {
      ssize_t written = writev(fd, &iov[iov_idx], batch_remaining);
      if (written < 0) {
        if (errno == EINTR) continue;
        return absl::InternalError(
            absl::StrCat("Socket writev failed: ", std::strerror(errno)));
      }
      if (written == 0) {
        return absl::InternalError("Socket closed unexpectedly during writev");
      }

      size_t remaining = written;
      while (remaining > 0 && batch_remaining > 0) {
        if (remaining >= iov[iov_idx].iov_len) {
          remaining -= iov[iov_idx].iov_len;
          iov_idx++;
          batch_remaining--;
        } else {
          iov[iov_idx].iov_base =
              static_cast<char*>(iov[iov_idx].iov_base) + remaining;
          iov[iov_idx].iov_len -= remaining;
          remaining = 0;
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ReadVExact(int fd, const std::vector<BlockChunk>& chunks) {
  std::vector<struct iovec> iov;
  iov.reserve(chunks.size());
  for (const auto& chunk : chunks) {
    iov.push_back({.iov_base = chunk.ptr, .iov_len = chunk.size});
  }

  size_t iov_idx = 0;
  while (iov_idx < iov.size()) {
    size_t batch_size =
        std::min(iov.size() - iov_idx, static_cast<size_t>(IOV_MAX));

    size_t batch_remaining = batch_size;
    while (batch_remaining > 0) {
      ssize_t bytes_read = readv(fd, &iov[iov_idx], batch_remaining);
      if (bytes_read < 0) {
        if (errno == EINTR) continue;
        return absl::InternalError(
            absl::StrCat("Socket readv failed: ", std::strerror(errno)));
      }
      if (bytes_read == 0) {
        return absl::InternalError("Socket closed unexpectedly during readv");
      }

      size_t remaining = bytes_read;
      while (remaining > 0 && batch_remaining > 0) {
        if (remaining >= iov[iov_idx].iov_len) {
          remaining -= iov[iov_idx].iov_len;
          iov_idx++;
          batch_remaining--;
        } else {
          iov[iov_idx].iov_base =
              static_cast<char*>(iov[iov_idx].iov_base) + remaining;
          iov[iov_idx].iov_len -= remaining;
          remaining = 0;
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateChunks(BlockTransportDelegate* delegate, size_t l,
                            size_t sh, const std::vector<BlockChunk>& chunks) {
  uint8_t* base = delegate->GetHostPointer(l, sh);
  if (base != nullptr) {
    size_t host_size = delegate->GetHostSize(l, sh);
    for (const auto& chunk : chunks) {
      if (chunk.ptr < base || chunk.ptr + chunk.size > base + host_size) {
        return absl::OutOfRangeError(absl::StrCat(
            "Chunk out of bounds. Chunk ptr: ",
            reinterpret_cast<uintptr_t>(chunk.ptr), ", size: ", chunk.size,
            ", Host base: ", reinterpret_cast<uintptr_t>(base),
            ", Host size: ", host_size));
      }
    }
  }
  return absl::OkStatus();
}

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
absl::Status ForEachPayload(MajorOrder major_order,
                            const std::vector<int>& layer_ids,
                            size_t num_shards, size_t num_blocks, Fn fn) {
  switch (static_cast<int>(major_order)) {
    case static_cast<int>(MajorOrder::kLayerMajor):
      for (int l : layer_ids) {
        for (size_t sh = 0; sh < num_shards; ++sh) {
          for (size_t k = 0; k < num_blocks; ++k) {
            RETURN_IF_ERROR(fn(l, sh, k));
          }
        }
      }
      return absl::OkStatus();
    case static_cast<int>(MajorOrder::kBlockMajor):
      for (size_t k = 0; k < num_blocks; ++k) {
        for (int l : layer_ids) {
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

BlockTransport::BlockTransport(BlockTransportDelegate* delegate, int local_port,
                               bool enable_conn_pool,
                               std::optional<std::string> bind_ip)
    : RawBufferTransport(delegate, local_port, enable_conn_pool, bind_ip),
      block_delegate_(delegate) {}

BlockTransport::~BlockTransport() = default;

absl::Status BlockTransport::HandleCustomRequest(int client_fd,
                                                 const PacketHeader& header) {
  LOG(INFO) << "HandleCustomRequest (H2H read start): client_fd=" << client_fd
            << ", op=" << static_cast<int>(header.op)
            << ", uuid=" << header.uuid
            << ", numa=" << block_delegate_->node_id();
  ApplySocketAffinityAndBinding(client_fd);
  const size_t bytes_per_block = block_delegate_->bytes_per_block();

  if (header.op == 1) {  // Push
    LOG(ERROR) << "===H2HDBG recv: op=1 push, count=" << header.count_or_size
               << " uuid=" << header.uuid;
    TF_ASSIGN_OR_RETURN(MajorOrder major_order, ParseMajorOrder(header.flags));
    TF_ASSIGN_OR_RETURN(
        std::vector<int> allocated_ids,
        block_delegate_->AllocateBlocks(header.count_or_size, header.uuid));
    LOG(ERROR) << "===H2HDBG recv: allocated " << allocated_ids.size()
               << " blocks, sending ids";

    RETURN_IF_ERROR(WriteExact(client_fd, allocated_ids.data(),
                               header.count_or_size * sizeof(int)));
    LOG(ERROR) << "===H2HDBG recv: ids sent, reading payload"
               << " (use_chunks=" << block_delegate_->use_block_chunks(header.uuid)
               << ")";

    std::vector<int> target_layers;
    if (header.local_id == 0xFFFFFFFF) {
      target_layers.resize(block_delegate_->num_layers());
      std::iota(target_layers.begin(), target_layers.end(), 0);
    } else {
      target_layers = {static_cast<int>(header.local_id)};
    }

    RETURN_IF_ERROR(ForEachPayload(
        major_order, target_layers, block_delegate_->num_shards(),
        header.count_or_size,
        [&](size_t l, size_t sh, size_t k) -> absl::Status {
          ABSL_DCHECK_LT(k, allocated_ids.size());
          const int dst_id = allocated_ids[k];
          if (block_delegate_->use_block_chunks(header.uuid)) {
            std::string local_addr = GetLocalAddr(client_fd);
            std::vector<BlockChunk> chunks = block_delegate_->GetBlockChunks(
                l, sh, dst_id, header.uuid, header.remote_id, local_addr);
            RETURN_IF_ERROR(ValidateChunks(block_delegate_, l, sh, chunks));

            uint32_t expected_size = 0;
            for (const auto& chunk : chunks) {
              expected_size += chunk.size;
            }

            uint32_t sender_size = 0;
            RETURN_IF_ERROR(
                ReadExact(client_fd, &sender_size, sizeof(sender_size)));

            if (sender_size != expected_size) {
              return absl::InternalError(absl::StrCat(
                  "Block transfer size mismatch! Sender offered: ", sender_size,
                  " bytes, but Receiver expected: ", expected_size,
                  " bytes for Block ID: ", dst_id));
            }

            RETURN_IF_ERROR(ReadVExact(client_fd, chunks));
          } else {
            RETURN_IF_ERROR(ValidateBlockRange(block_delegate_, l, sh, dst_id,
                                               1, bytes_per_block));
            uint8_t* dest_ptr =
                block_delegate_->GetBlockHostPointer(l, sh, dst_id);
            RETURN_IF_ERROR(ReadExact(client_fd, dest_ptr, bytes_per_block));
          }
          return absl::OkStatus();
        }));

    bool trigger_on_layer_received = false;
    const int l =
        (header.local_id == 0xFFFFFFFF) ? 0 : static_cast<int>(header.local_id);
    const size_t expected_chunks = header.reserved;  // P
    {
      absl::MutexLock lock(progress_mu_);
      auto& progress = layer_progress_[{header.uuid, l}];
      progress.completed_chunks++;
      if (progress.completed_chunks == expected_chunks &&
          !progress.on_layer_received_called) {
        progress.on_layer_received_called = true;
        trigger_on_layer_received = true;
        if (l == static_cast<int>(block_delegate_->num_layers()) - 1) {
          for (size_t layer = 0; layer < block_delegate_->num_layers();
               ++layer) {
            layer_progress_.erase({header.uuid, layer});
          }
        }
      }
    }

    if (trigger_on_layer_received) {
      RETURN_IF_ERROR(block_delegate_->OnLayerReceived(l, header.uuid));
    }

    LOG(INFO) << "HandleCustomRequest (H2H read complete): client_fd="
              << client_fd << ", uuid=" << header.uuid
              << ", numa=" << block_delegate_->node_id();
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

    std::vector<int> target_layers;
    if (header.local_id == 0xFFFFFFFF) {
      target_layers.resize(block_delegate_->num_layers());
      std::iota(target_layers.begin(), target_layers.end(), 0);
    } else {
      target_layers = {static_cast<int>(header.local_id)};
    }

    RETURN_IF_ERROR(ForEachPayload(
        major_order, target_layers, block_delegate_->num_shards(),
        header.count_or_size,
        [&](size_t l, size_t sh, size_t k) -> absl::Status {
          ABSL_DCHECK_LT(k, allocated_ids.size());
          const int dst_id = allocated_ids[k];
          if (block_delegate_->use_block_chunks(header.uuid)) {
            std::string local_addr = GetLocalAddr(client_fd);
            std::vector<BlockChunk> chunks = block_delegate_->GetBlockChunks(
                l, sh, dst_id, header.uuid, header.remote_id, local_addr);
            RETURN_IF_ERROR(ValidateChunks(block_delegate_, l, sh, chunks));

            uint32_t expected_size = 0;
            for (const auto& chunk : chunks) {
              expected_size += chunk.size;
            }

            uint32_t sender_size = 0;
            RETURN_IF_ERROR(
                ReadExact(client_fd, &sender_size, sizeof(sender_size)));

            if (sender_size != expected_size) {
              return absl::InternalError(absl::StrCat(
                  "Block transfer size mismatch! Sender offered: ", sender_size,
                  " bytes, but Receiver expected: ", expected_size,
                  " bytes for Block ID: ", dst_id));
            }

            RETURN_IF_ERROR(ReadVExact(client_fd, chunks));
          } else {
            RETURN_IF_ERROR(ValidateBlockRange(block_delegate_, l, sh, dst_id,
                                               1, bytes_per_block));
            uint8_t* dest_ptr =
                block_delegate_->GetBlockHostPointer(l, sh, dst_id);
            RETURN_IF_ERROR(ReadExact(client_fd, dest_ptr, bytes_per_block));
          }
          return absl::OkStatus();
        }));

    bool trigger_on_layer_received = false;
    const int l =
        (header.local_id == 0xFFFFFFFF) ? 0 : static_cast<int>(header.local_id);
    const size_t expected_chunks = header.reserved;  // P
    {
      absl::MutexLock lock(progress_mu_);
      auto& progress = layer_progress_[{header.uuid, l}];
      progress.completed_chunks++;
      if (progress.completed_chunks == expected_chunks &&
          !progress.on_layer_received_called) {
        progress.on_layer_received_called = true;
        trigger_on_layer_received = true;
        if (l == static_cast<int>(block_delegate_->num_layers()) - 1) {
          for (size_t layer = 0; layer < block_delegate_->num_layers();
               ++layer) {
            layer_progress_.erase({header.uuid, layer});
          }
        }
      }
    }

    if (trigger_on_layer_received) {
      RETURN_IF_ERROR(block_delegate_->OnLayerReceived(l, header.uuid));
    }

    LOG(INFO) << "HandleCustomRequest (H2H read complete): client_fd="
              << client_fd << ", uuid=" << header.uuid
              << ", numa=" << block_delegate_->node_id();
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
    std::vector<int> target_layers(block_delegate_->num_layers());
    std::iota(target_layers.begin(), target_layers.end(), 0);
    RETURN_IF_ERROR(ForEachPayload(
        major_order, target_layers, block_delegate_->num_shards(), local_blocks,
        [&](size_t l, size_t sh, size_t k) -> absl::Status {
          const int read_id = static_cast<int>(header.remote_id + k);
          if (block_delegate_->use_block_chunks(header.uuid)) {
            std::vector<BlockChunk> chunks =
                block_delegate_->GetBlockChunks(l, sh, read_id, header.uuid);
            RETURN_IF_ERROR(ValidateChunks(block_delegate_, l, sh, chunks));
            RETURN_IF_ERROR(block_delegate_->WaitForBlockRead(l, sh, read_id));

            uint32_t total_size = 0;
            for (const auto& chunk : chunks) {
              total_size += chunk.size;
            }

            RETURN_IF_ERROR(
                WriteExact(client_fd, &total_size, sizeof(total_size)));

            return WriteVExact(client_fd, chunks);
          } else {
            RETURN_IF_ERROR(ValidateBlockRange(block_delegate_, l, sh, read_id,
                                               1, bytes_per_block));
            RETURN_IF_ERROR(block_delegate_->WaitForBlockRead(l, sh, read_id));
            const uint8_t* src_ptr =
                block_delegate_->GetBlockHostPointer(l, sh, read_id);
            return WriteExact(client_fd, src_ptr, bytes_per_block);
          }
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
    absl::string_view peer, const std::vector<int>& src_block_ids,
    const std::vector<int>& dst_block_ids, int parallelism,
    MajorOrder major_order, uint64_t uuid, int layer_idx) {
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
    threads.push_back(
        std::thread(&BlockTransport::H2hWriteWorker, this, i, peer,
                    block_offset, block_count, std::ref(src_block_ids),
                    std::ref(dst_block_ids), std::ref(allocated_ids),
                    std::ref(statuses), major_order, uuid, layer_idx, P));
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

absl::Status BlockTransport::WriteBlockDirect(absl::string_view peer,
                                              int remote_block_id,
                                              const uint8_t* data_ptr,
                                              size_t size_bytes) {
  TF_ASSIGN_OR_RETURN(int fd, AcquireConnection(peer));
  ApplySocketAffinityAndBinding(fd);
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
  if (ack != 1) {
    return absl::InternalError("WriteBlockDirect handshake failed");
  }

  RETURN_IF_ERROR(WriteExact(fd, data_ptr, size_bytes));

  ack = 0;
  RETURN_IF_ERROR(ReadExact(fd, &ack, 1));
  if (ack != 1) {
    return absl::InternalError("WriteBlockDirect completion ack failed");
  }

  ok_to_pool = true;
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int>> BlockTransport::Pull(
    absl::string_view peer, const std::vector<int>& src_block_ids,
    const std::vector<int>& local_block_ids,
    const std::vector<uint8_t*>& explicit_dst_ptrs, int parallelism,
    MajorOrder major_order, BlockReceivedCallback on_block_received,
    uint64_t uuid) {
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
                    std::ref(statuses), major_order, on_block_received, uuid));

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

void BlockTransport::H2hWriteWorker(int stream_idx, absl::string_view peer,
                                    size_t block_offset, size_t block_count,
                                    const std::vector<int>& src_block_ids,
                                    const std::vector<int>& dst_block_ids,
                                    std::vector<int>& allocated_ids,
                                    std::vector<absl::Status>& statuses,
                                    MajorOrder major_order, uint64_t uuid,
                                    int layer_idx, int parallelism) {
  auto status_or_fd = AcquireConnection(peer);
  if (!status_or_fd.ok()) {
    statuses[stream_idx] = status_or_fd.status();
    return;
  }
  int fd = status_or_fd.value();
  ApplySocketAffinityAndBinding(fd);
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
  header.remote_id = static_cast<uint32_t>(block_delegate_->node_id());
  header.local_id =
      (layer_idx == -1) ? 0xFFFFFFFF : static_cast<uint32_t>(layer_idx);
  header.uuid = uuid;
  header.reserved = static_cast<uint16_t>(parallelism);
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
  LOG(ERROR) << "===H2HDBG sender: connected to " << peer << ", header sent (op="
             << static_cast<int>(header.op) << "), waiting for allocated ids";

  if (header.op == 6) {
    ABSL_DCHECK_LE(block_offset + block_count, dst_block_ids.size());
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
  LOG(ERROR) << "===H2HDBG sender: got allocated ids, sending payload"
             << " (use_chunks=" << block_delegate_->use_block_chunks(uuid) << ")";

  std::vector<int> target_layers;
  if (layer_idx == -1) {
    target_layers.resize(block_delegate_->num_layers());
    std::iota(target_layers.begin(), target_layers.end(), 0);
  } else {
    target_layers = {layer_idx};
  }

  s = ForEachPayload(
      major_order, target_layers, block_delegate_->num_shards(), block_count,
      [&](size_t l, size_t sh, size_t k) -> absl::Status {
        ABSL_DCHECK_LT(block_offset + k, src_block_ids.size());
        const int src_id = src_block_ids[block_offset + k];

        if (block_delegate_->use_block_chunks(uuid)) {
          std::vector<BlockChunk> chunks =
              block_delegate_->GetBlockChunks(l, sh, src_id, uuid, -1, peer);
          RETURN_IF_ERROR(ValidateChunks(block_delegate_, l, sh, chunks));
          RETURN_IF_ERROR(block_delegate_->WaitForBlockRead(l, sh, src_id));

          uint32_t total_size = 0;
          for (const auto& chunk : chunks) {
            total_size += chunk.size;
          }

          RETURN_IF_ERROR(WriteExact(fd, &total_size, sizeof(total_size)));

          return WriteVExact(fd, chunks);
        } else {
          size_t bytes_per_block = block_delegate_->bytes_per_block();
          const uint8_t* base_host_ptr = block_delegate_->GetHostPointer(l, sh);
          RETURN_IF_ERROR(ValidateBlockRange(block_delegate_, l, sh, src_id, 1,
                                             bytes_per_block));
          RETURN_IF_ERROR(block_delegate_->WaitForBlockRead(l, sh, src_id));
          const uint8_t* src_ptr = base_host_ptr + src_id * bytes_per_block;
          return WriteExact(fd, src_ptr, bytes_per_block);
        }
      });
  if (!s.ok()) {
    statuses[stream_idx] = s;
    return;
  }
  LOG(ERROR) << "===H2HDBG sender: payload fully sent, waiting for ack";

  uint8_t ack = 0;
  s = ReadExact(fd, &ack, 1);
  if (!s.ok() || ack != 1) {
    statuses[stream_idx] = absl::InternalError("Push verification failed");
    return;
  }

  ok_to_pool = true;
}

void BlockTransport::H2hReadWorker(
    int stream_idx, absl::string_view peer, size_t local_block_offset,
    size_t local_block_count, size_t remote_block_offset,
    size_t remote_block_count, const std::vector<int>& src_block_ids,
    const std::vector<int>& allocated_ids,
    const std::vector<uint8_t*>& explicit_dst_ptrs,
    std::vector<absl::Status>& statuses, MajorOrder major_order,
    BlockReceivedCallback on_block_received, uint64_t uuid) {
  auto status_or_fd = AcquireConnection(peer);
  if (!status_or_fd.ok()) {
    statuses[stream_idx] = status_or_fd.status();
    return;
  }
  int fd = status_or_fd.value();
  ApplySocketAffinityAndBinding(fd);
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
    header.uuid = uuid;

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

    std::vector<int> target_layers(block_delegate_->num_layers());
    std::iota(target_layers.begin(), target_layers.end(), 0);
    s = ForEachPayload(
        major_order, target_layers, block_delegate_->num_shards(),
        chunk.local_count, [&](size_t l, size_t sh, size_t k) -> absl::Status {
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

          if (block_delegate_->use_block_chunks(uuid)) {
            std::vector<BlockChunk> chunks =
                block_delegate_->GetBlockChunks(l, sh, dst_id, uuid);
            RETURN_IF_ERROR(ValidateChunks(block_delegate_, l, sh, chunks));

            if (!explicit_dst_ptrs.empty()) {
              uint8_t* default_base = block_delegate_->GetHostPointer(l, sh);
              uint8_t* explicit_base = base_host_ptr;
              for (auto& chunk : chunks) {
                size_t offset = chunk.ptr - default_base;
                chunk.ptr = explicit_base + offset;
              }
            }

            uint32_t expected_size = 0;
            for (const auto& chunk : chunks) {
              expected_size += chunk.size;
            }

            uint32_t sender_size = 0;
            RETURN_IF_ERROR(ReadExact(fd, &sender_size, sizeof(sender_size)));

            if (sender_size != expected_size) {
              return absl::InternalError(absl::StrCat(
                  "Block transfer size mismatch! Sender offered: ", sender_size,
                  " bytes, but Receiver expected: ", expected_size,
                  " bytes for Block ID: ", dst_id));
            }

            RETURN_IF_ERROR(ReadVExact(fd, chunks));

            if (on_block_received != nullptr) {
              RETURN_IF_ERROR(on_block_received(l, sh, dst_id, expected_size));
            }
          } else {
            size_t bytes_per_block = block_delegate_->bytes_per_block();
            if (explicit_dst_ptrs.empty()) {
              RETURN_IF_ERROR(ValidateBlockRange(block_delegate_, l, sh, dst_id,
                                                 1, bytes_per_block));
            }
            uint8_t* dst_ptr = base_host_ptr + dst_id * bytes_per_block;
            RETURN_IF_ERROR(ReadExact(fd, dst_ptr, bytes_per_block));
            if (on_block_received != nullptr) {
              RETURN_IF_ERROR(
                  on_block_received(l, sh, dst_id, bytes_per_block));
            }
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
