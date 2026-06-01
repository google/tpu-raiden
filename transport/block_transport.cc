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

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <functional>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace tpu_raiden {
namespace transport {

namespace {

absl::Status WriteExact(int fd, const void* buffer, size_t length) {
  const uint8_t* ptr = static_cast<const uint8_t*>(buffer);
  size_t remaining = length;
  while (remaining > 0) {
    ssize_t written = write(fd, ptr, remaining);
    if (written < 0) {
      if (errno == EINTR) continue;
      return absl::InternalError(
          absl::StrCat("Socket write failed: ", std::strerror(errno)));
    }
    if (written == 0) {
      return absl::InternalError("Socket closed unexpectedly during write");
    }
    ptr += written;
    remaining -= written;
  }
  return absl::OkStatus();
}

absl::Status ReadExact(int fd, void* buffer, size_t length) {
  uint8_t* ptr = static_cast<uint8_t*>(buffer);
  size_t remaining = length;
  while (remaining > 0) {
    ssize_t bytes_read = read(fd, ptr, remaining);
    if (bytes_read < 0) {
      if (errno == EINTR) continue;
      return absl::InternalError(
          absl::StrCat("Socket read failed: ", std::strerror(errno)));
    }
    if (bytes_read == 0) {
      return absl::InternalError("Socket closed unexpectedly during read");
    }
    ptr += bytes_read;
    remaining -= bytes_read;
  }
  return absl::OkStatus();
}

}  // namespace

BlockTransport::BlockTransport(BlockTransportDelegate* delegate, int local_port)
    : delegate_(delegate), local_port_(local_port) {
  server_fd_ = socket(AF_INET6, SOCK_STREAM, 0);
  if (server_fd_ < 0) {
    LOG(FATAL) << "Failed to create server socket: " << std::strerror(errno);
  }
  int opt = 1;
  setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  struct sockaddr_in6 serv_addr;
  std::memset(&serv_addr, 0, sizeof(serv_addr));
  serv_addr.sin6_family = AF_INET6;
  serv_addr.sin6_addr = in6addr_any;
  serv_addr.sin6_port = htons(local_port_);

  if (bind(server_fd_, reinterpret_cast<struct sockaddr*>(&serv_addr),
           sizeof(serv_addr)) < 0) {
    LOG(FATAL) << "Failed to bind server socket to port " << local_port_ << ": "
               << std::strerror(errno);
  }

  socklen_t addr_len = sizeof(serv_addr);
  if (getsockname(server_fd_, reinterpret_cast<struct sockaddr*>(&serv_addr),
                  &addr_len) == 0) {
    local_port_ = ntohs(serv_addr.sin6_port);
  }

  if (listen(server_fd_, 128) < 0) {
    LOG(FATAL) << "Failed to listen on server socket: " << std::strerror(errno);
  }
  listener_thread_ = std::thread(&BlockTransport::ListenerLoop, this);
}

BlockTransport::~BlockTransport() {
  stopping_ = true;
  if (server_fd_ >= 0) {
    shutdown(server_fd_, SHUT_RDWR);
    close(server_fd_);
  }
  {
    absl::MutexLock _(&mu_);
    for (int fd : active_client_fds_) {
      shutdown(fd, SHUT_RDWR);
      close(fd);
    }
    active_client_fds_.clear();
  }
  if (listener_thread_.joinable()) {
    listener_thread_.join();
  }
  for (auto& t : worker_threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
}

absl::StatusOr<int> BlockTransport::ConnectToPeer(const std::string& peer) {
  std::string host;
  std::string port_str;

  if (!peer.empty() && peer.front() == '[') {
    size_t closing_bracket = peer.find(']');
    if (closing_bracket == std::string::npos ||
        closing_bracket + 1 >= peer.size() ||
        peer[closing_bracket + 1] != ':') {
      return absl::InvalidArgumentError(
          "Invalid IPv6 peer bracket string format");
    }
    host = peer.substr(1, closing_bracket - 1);
    port_str = peer.substr(closing_bracket + 2);
  } else {
    std::vector<std::string> parts = absl::StrSplit(peer, ':');
    if (parts.size() != 2) {
      return absl::InvalidArgumentError("Invalid peer string format");
    }
    host = parts[0];
    port_str = parts[1];
  }

  struct addrinfo hints;
  struct addrinfo* result = nullptr;
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  int ret = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result);
  if (ret != 0 || result == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "getaddrinfo failed for host ", host, ": ", gai_strerror(ret)));
  }

  int sock_fd = -1;
  struct addrinfo* rp;
  int last_errno = 0;
  for (rp = result; rp != nullptr; rp = rp->ai_next) {
    sock_fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (sock_fd < 0) {
      last_errno = errno;
      continue;
    }

    int opt = 1;
    setsockopt(sock_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    int buf_opt = 16 * 1024 * 1024;  // 16MB
    setsockopt(sock_fd, SOL_SOCKET, SO_SNDBUF, &buf_opt, sizeof(buf_opt));
    setsockopt(sock_fd, SOL_SOCKET, SO_RCVBUF, &buf_opt, sizeof(buf_opt));

    if (connect(sock_fd, rp->ai_addr, rp->ai_addrlen) >= 0) {
      break; /* Success */
    }

    last_errno = errno;
    close(sock_fd);
    sock_fd = -1;
  }

  freeaddrinfo(result);

  if (sock_fd < 0) {
    return absl::UnavailableError(absl::StrCat(
        "Failed to connect to peer ", peer, ": ", std::strerror(last_errno)));
  }

  return sock_fd;
}

absl::Status BlockTransport::ProcessSingleRequest(int client_fd) {
  BlockPacketHeader header;
  TF_RETURN_IF_ERROR(ReadExact(client_fd, &header, sizeof(header)));

  size_t bytes_per_block =
      delegate_->block_size() * delegate_->slice_byte_size();

  if (header.op == 1) {  // Push
    TF_ASSIGN_OR_RETURN(
        std::vector<int> allocated_ids,
        delegate_->AllocateBlocks(header.num_blocks, /*entity_id=*/0));

    TF_RETURN_IF_ERROR(WriteExact(client_fd, allocated_ids.data(),
                                  header.num_blocks * sizeof(int)));

    for (size_t l = 0; l < delegate_->num_layers(); ++l) {
      for (size_t sh = 0; sh < delegate_->num_shards(); ++sh) {
        uint8_t* base_host_ptr = delegate_->GetHostPointer(l, sh);

        for (size_t k = 0; k < header.num_blocks; ++k) {
          int dst_id = allocated_ids[k];
          uint8_t* dest_ptr = base_host_ptr + dst_id * bytes_per_block;
          TF_RETURN_IF_ERROR(ReadExact(client_fd, dest_ptr, bytes_per_block));
        }
      }
    }

    uint8_t ack = 1;
    TF_RETURN_IF_ERROR(WriteExact(client_fd, &ack, 1));
    TF_RETURN_IF_ERROR(delegate_->OnDataReceived());
  } else if (header.op == 2) {  // Pull request
    BlockPacketHeader resp_header;
    resp_header.op = 2;
    resp_header.remote_block_id = header.local_block_id;
    resp_header.local_block_id = 0;
    resp_header.num_blocks = header.num_blocks;

    TF_RETURN_IF_ERROR(
        WriteExact(client_fd, &resp_header, sizeof(resp_header)));

    size_t local_blocks = header.num_blocks / delegate_->shard_factor();
    for (size_t l = 0; l < delegate_->num_layers(); ++l) {
      for (size_t sh = 0; sh < delegate_->num_shards(); ++sh) {
        const uint8_t* base_host_ptr = delegate_->GetHostPointer(l, sh);

        for (int k = 0; k < local_blocks; ++k) {
          int read_id = header.remote_block_id + k;
          const uint8_t* src_ptr = base_host_ptr + read_id * bytes_per_block;
          TF_RETURN_IF_ERROR(WriteExact(client_fd, src_ptr, bytes_per_block));
        }
      }
    }
  } else if (header.op == 3) {
    // Byte-range Resharding Pull Request!
    uint32_t src_offset = header.remote_block_id;
    uint32_t src_shard_idx = header.local_block_id;
    uint32_t size_bytes = header.num_blocks;

    if (delegate_->num_layers() == 0) {
      return absl::InternalError("Server host buffers are not initialized");
    }
    if (src_shard_idx >= delegate_->num_shards()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid source shard index: ", src_shard_idx,
                       ", total shards: ", delegate_->num_shards()));
    }
    const uint8_t* base_host_ptr = delegate_->GetHostPointer(0, src_shard_idx);
    size_t host_size = delegate_->GetHostSize(0, src_shard_idx);
    if (src_offset + size_bytes > host_size) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Request out of bounds. Offset: ", src_offset, ", Size: ", size_bytes,
          ", Shard Host Size: ", host_size));
    }

    const uint8_t* src_ptr = base_host_ptr + src_offset;
    TF_RETURN_IF_ERROR(WriteExact(client_fd, src_ptr, size_bytes));
  }
  return absl::OkStatus();
}

void BlockTransport::ConnectionWorker(int client_fd) {
  while (!stopping_) {
    struct pollfd pfd;
    pfd.fd = client_fd;
    pfd.events = POLLIN;
    int ret = poll(&pfd, 1, 50);
    if (ret < 0) {
      break;
    }
    if (ret == 0) continue;

    if (!ProcessSingleRequest(client_fd).ok()) {
      break;
    }
  }
  close(client_fd);
  {
    absl::MutexLock _(&mu_);
    active_client_fds_.erase(std::remove(active_client_fds_.begin(),
                                         active_client_fds_.end(), client_fd),
                             active_client_fds_.end());
  }
}

void BlockTransport::ListenerLoop() {
  while (!stopping_) {
    struct pollfd pfd;
    pfd.fd = server_fd_;
    pfd.events = POLLIN;
    int ret = poll(&pfd, 1, 50);
    if (ret < 0) {
      if (stopping_) break;
      continue;
    }
    if (ret == 0) continue;

    struct sockaddr_in6 client_addr;
    socklen_t clilen = sizeof(client_addr);
    int client_fd = accept(
        server_fd_, reinterpret_cast<struct sockaddr*>(&client_addr), &clilen);
    if (client_fd < 0) {
      if (stopping_) break;
      continue;
    }

    int opt = 1;
    setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

    {
      absl::MutexLock _(&mu_);
      active_client_fds_.push_back(client_fd);
    }

    worker_threads_.push_back(
        std::thread([this, client_fd]() { ConnectionWorker(client_fd); }));
  }
}

absl::StatusOr<std::vector<int>> BlockTransport::Push(
    const std::string& peer, const std::vector<int>& src_block_ids,
    int parallelism) {
  size_t num_blocks = src_block_ids.size();
  if (num_blocks == 0) {
    return absl::InvalidArgumentError("Block list cannot be empty");
  }

  int P = parallelism;
  if (static_cast<int>(num_blocks) < P) P = num_blocks;

  if (num_blocks % P != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Block count (", num_blocks,
                     ") must be fully divisible by parallelism (", P, ")"));
  }

  size_t blocks_per_stream = num_blocks / P;
  std::vector<int> allocated_ids(num_blocks, 0);
  std::vector<std::thread> threads;
  std::vector<absl::Status> statuses(P, absl::OkStatus());

  threads.reserve(P);
  for (int i = 0; i < P; ++i) {
    threads.push_back(std::thread(
        &BlockTransport::H2hWriteWorker, this, i, peer, blocks_per_stream,
        std::ref(src_block_ids), std::ref(allocated_ids), std::ref(statuses)));
  }

  for (auto& t : threads) {
    if (t.joinable()) t.join();
  }

  for (int i = 0; i < P; ++i) {
    if (!statuses[i].ok()) return statuses[i];
  }

  return allocated_ids;
}

absl::StatusOr<std::vector<int>> BlockTransport::Pull(
    const std::string& peer, const std::vector<int>& src_block_ids,
    int parallelism) {
  size_t num_blocks = src_block_ids.size();
  if (num_blocks == 0) {
    return absl::InvalidArgumentError("Block list cannot be empty");
  }

  size_t local_blocks = num_blocks / delegate_->shard_factor();
  TF_ASSIGN_OR_RETURN(std::vector<int> allocated_ids,
                      delegate_->AllocateBlocks(local_blocks, /*entity_id=*/0));

  int P = parallelism;
  if (static_cast<int>(local_blocks) < P) P = local_blocks;

  if (local_blocks % P != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Local block count (", local_blocks,
                     ") must be fully divisible by parallelism (", P, ")"));
  }

  size_t blocks_per_stream = local_blocks / P;
  size_t remote_blocks_per_stream = num_blocks / P;

  std::vector<std::thread> threads;
  std::vector<absl::Status> statuses(P, absl::OkStatus());

  threads.reserve(P);
  for (int i = 0; i < P; ++i) {
    threads.push_back(
        std::thread(&BlockTransport::H2hReadWorker, this, i, peer,
                    blocks_per_stream, remote_blocks_per_stream,
                    std::ref(src_block_ids), std::ref(allocated_ids),
                    std::ref(statuses)));
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
                                    size_t blocks_per_stream,
                                    const std::vector<int>& src_block_ids,
                                    std::vector<int>& allocated_ids,
                                    std::vector<absl::Status>& statuses) {
  size_t offset = stream_idx * blocks_per_stream;
  auto status_or_fd = ConnectToPeer(peer);
  if (!status_or_fd.ok()) {
    statuses[stream_idx] = status_or_fd.status();
    return;
  }
  int fd = status_or_fd.value();
  auto fd_cleaner = absl::MakeCleanup([fd] { close(fd); });

  BlockPacketHeader header;
  header.op = 1;  // Push
  header.remote_block_id = 0;
  header.local_block_id = 0;
  header.num_blocks = blocks_per_stream;

  absl::Status s = WriteExact(fd, &header, sizeof(header));
  if (!s.ok()) {
    statuses[stream_idx] = s;
    return;
  }

  std::vector<int> stream_allocated_ids(blocks_per_stream, 0);
  s = ReadExact(fd, stream_allocated_ids.data(),
                blocks_per_stream * sizeof(int));
  if (!s.ok()) {
    statuses[stream_idx] = s;
    return;
  }

  for (size_t k = 0; k < blocks_per_stream; ++k) {
    ABSL_DCHECK_LT(offset + k, allocated_ids.size());
    ABSL_DCHECK_LT(k, stream_allocated_ids.size());
    allocated_ids[offset + k] = stream_allocated_ids[k];
  }

  size_t bytes_per_block =
      delegate_->block_size() * delegate_->slice_byte_size();

  for (size_t l = 0; l < delegate_->num_layers(); ++l) {
    for (size_t sh = 0; sh < delegate_->num_shards(); ++sh) {
      const uint8_t* base_host_ptr = delegate_->GetHostPointer(l, sh);

      for (size_t k = 0; k < blocks_per_stream; ++k) {
        ABSL_DCHECK_LT(offset + k, src_block_ids.size());
        int src_id = src_block_ids[offset + k];
        const uint8_t* src_ptr = base_host_ptr + src_id * bytes_per_block;
        s = WriteExact(fd, src_ptr, bytes_per_block);
        if (!s.ok()) {
          statuses[stream_idx] = s;
          return;
        }
      }
    }
  }

  uint8_t ack = 0;
  s = ReadExact(fd, &ack, 1);
  if (!s.ok()) {
    statuses[stream_idx] = s;
    return;
  }
}

void BlockTransport::H2hReadWorker(
    int stream_idx, const std::string& peer, size_t blocks_per_stream,
    size_t remote_blocks_per_stream, const std::vector<int>& src_block_ids,
    const std::vector<int>& allocated_ids,
    std::vector<absl::Status>& statuses) {
  size_t offset = stream_idx * blocks_per_stream;
  size_t remote_offset = stream_idx * remote_blocks_per_stream;

  auto status_or_fd = ConnectToPeer(peer);
  if (!status_or_fd.ok()) {
    statuses[stream_idx] = status_or_fd.status();
    return;
  }
  int fd = status_or_fd.value();
  auto fd_cleaner = absl::MakeCleanup([fd] { close(fd); });

  size_t SF = delegate_->shard_factor();

  struct PullChunk {
    size_t local_start_idx;
    size_t local_count;
    int base_remote_id;
    size_t remote_count;
  };
  std::vector<PullChunk> chunks;

  if (blocks_per_stream > 0) {
    size_t curr_local_start = 0;
    size_t curr_local_count = 1;
    ABSL_DCHECK_LT(remote_offset, src_block_ids.size());
    int curr_base_remote_id = src_block_ids[remote_offset];

    for (size_t k = 1; k < blocks_per_stream; ++k) {
      ABSL_DCHECK_LT(remote_offset + k * SF - 1, src_block_ids.size());
      int prev_last_remote = src_block_ids[remote_offset + k * SF - 1];
      ABSL_DCHECK_LT(remote_offset + k * SF, src_block_ids.size());
      int curr_first_remote = src_block_ids[remote_offset + k * SF];

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

  size_t bytes_per_block =
      delegate_->block_size() * delegate_->slice_byte_size();

  for (const auto& chunk : chunks) {
    BlockPacketHeader header;
    header.op = 2;  // Pull request
    header.remote_block_id =
        delegate_->GetRemoteReadBlockId(chunk.base_remote_id, 0);
    ABSL_DCHECK_LT(offset + chunk.local_start_idx, allocated_ids.size());
    header.local_block_id = allocated_ids[offset + chunk.local_start_idx];
    header.num_blocks = chunk.remote_count;

    absl::Status s = WriteExact(fd, &header, sizeof(header));
    if (!s.ok()) {
      statuses[stream_idx] = s;
      return;
    }

    BlockPacketHeader resp_header;
    s = ReadExact(fd, &resp_header, sizeof(resp_header));
    if (!s.ok()) {
      statuses[stream_idx] = s;
      return;
    }

    for (size_t l = 0; l < delegate_->num_layers(); ++l) {
      for (size_t sh = 0; sh < delegate_->num_shards(); ++sh) {
        uint8_t* base_host_ptr = delegate_->GetHostPointer(l, sh);

        for (size_t k = 0; k < chunk.local_count; ++k) {
          ABSL_DCHECK_LT(offset + chunk.local_start_idx + k,
                         allocated_ids.size());
          int dst_id = allocated_ids[offset + chunk.local_start_idx + k];
          uint8_t* dest_ptr = base_host_ptr + dst_id * bytes_per_block;
          s = ReadExact(fd, dest_ptr, bytes_per_block);
          if (!s.ok()) {
            statuses[stream_idx] = s;
            return;
          }
        }
      }
    }
  }
}

absl::Status BlockTransport::PullWeightsChunk(
    const std::string& source, size_t src_shard_idx, size_t src_offset_bytes,
    size_t dst_shard_idx, size_t dst_offset_bytes, size_t size_bytes) {
  if (source.empty()) {
    return absl::InvalidArgumentError("Source peer address cannot be empty");
  }

  size_t host_size = delegate_->GetHostSize(0, dst_shard_idx);
  if (dst_offset_bytes + size_bytes > host_size) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Destination offset out of bounds. Offset: ", dst_offset_bytes,
        ", Size: ", size_bytes, ", Shard Host Size: ", host_size));
  }

  // Establish peer socket connection
  TF_ASSIGN_OR_RETURN(int fd, ConnectToPeer(source));

  // Send our customized resharding pull header (op = 3)
  BlockPacketHeader header;
  header.op = 3;
  header.remote_block_id = static_cast<uint32_t>(src_offset_bytes);
  header.local_block_id = static_cast<uint32_t>(src_shard_idx);
  header.num_blocks = static_cast<uint32_t>(size_bytes);

  TF_RETURN_IF_ERROR(WriteExact(fd, &header, sizeof(header)));

  // Read bytes directly into our local Host buffer!
  uint8_t* dest_ptr =
      delegate_->GetHostPointer(0, dst_shard_idx) + dst_offset_bytes;
  TF_RETURN_IF_ERROR(ReadExact(fd, dest_ptr, size_bytes));

  return absl::OkStatus();
}

}  // namespace transport
}  // namespace tpu_raiden
