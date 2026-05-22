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

#include "core/raiden_manager_base.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
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

absl::StatusOr<int> ConnectToPeer(const std::string& peer) {
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

  int sock_fd =
      socket(result->ai_family, result->ai_socktype, result->ai_protocol);
  if (sock_fd < 0) {
    freeaddrinfo(result);
    return absl::InternalError(
        absl::StrCat("Socket creation failed: ", std::strerror(errno)));
  }

  int opt = 1;
  setsockopt(sock_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
  int buf_opt = 16 * 1024 * 1024;  // 16MB
  setsockopt(sock_fd, SOL_SOCKET, SO_SNDBUF, &buf_opt, sizeof(buf_opt));
  setsockopt(sock_fd, SOL_SOCKET, SO_RCVBUF, &buf_opt, sizeof(buf_opt));

  if (connect(sock_fd, result->ai_addr, result->ai_addrlen) < 0) {
    close(sock_fd);
    freeaddrinfo(result);
    return absl::UnavailableError(
        absl::StrCat("Connect failed: ", std::strerror(errno)));
  }

  freeaddrinfo(result);
  return sock_fd;
}

}  // namespace

struct alignas(8) BlockPacketHeader {
  uint8_t op;
  uint32_t remote_block_id;
  uint32_t local_block_id;
  uint32_t num_blocks;
};

struct RaidenManagerBase::BlockTransportServer {
  explicit BlockTransportServer(RaidenManagerBase* parent, int port)
      : parent_(parent), local_port_(port) {
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
      LOG(FATAL) << "Failed to bind server socket to port " << local_port_
                 << ": " << std::strerror(errno);
    }

    socklen_t addr_len = sizeof(serv_addr);
    if (getsockname(server_fd_, reinterpret_cast<struct sockaddr*>(&serv_addr),
                    &addr_len) == 0) {
      local_port_ = ntohs(serv_addr.sin6_port);
    }

    if (listen(server_fd_, 128) < 0) {
      LOG(FATAL) << "Failed to listen on server socket: "
                 << std::strerror(errno);
    }
    listener_thread_ = std::thread(&BlockTransportServer::ListenerLoop, this);
  }

  ~BlockTransportServer() {
    stopping_ = true;
    if (server_fd_ >= 0) {
      shutdown(server_fd_, SHUT_RDWR);
      close(server_fd_);
    }
    if (listener_thread_.joinable()) {
      listener_thread_.join();
    }
    for (auto& t : worker_threads_) {
      if (t.joinable()) {
        t.join();
      }
    }
    absl::MutexLock _(conn_mu_);
    for (const auto& [peer, fd] : connection_pool_) {
      close(fd);
    }
  }

  absl::StatusOr<int> GetOrCreateConnection(const std::string& peer) {
    {
      absl::MutexLock _(conn_mu_);
      auto it = connection_pool_.find(peer);
      if (it != connection_pool_.end()) {
        return it->second;
      }
    }
    std::vector<std::string> parts = absl::StrSplit(peer, ':');
    if (parts.size() != 2) {
      return absl::InvalidArgumentError("Invalid peer string format");
    }
    std::string host = parts[0];
    std::string port_str = parts[1];

    struct addrinfo hints;
    struct addrinfo* result = nullptr;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    int ret = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result);
    if (ret != 0 || result == nullptr) {
      return absl::InvalidArgumentError(absl::StrCat(
          "getaddrinfo failed for host ", host, ": ", gai_strerror(ret)));
    }

    int sock_fd =
        socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (sock_fd < 0) {
      freeaddrinfo(result);
      return absl::InternalError(
          absl::StrCat("Socket creation failed: ", std::strerror(errno)));
    }

    int opt = 1;
    setsockopt(sock_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    int buf_opt = 16 * 1024 * 1024;  // 16MB
    setsockopt(sock_fd, SOL_SOCKET, SO_SNDBUF, &buf_opt, sizeof(buf_opt));
    setsockopt(sock_fd, SOL_SOCKET, SO_RCVBUF, &buf_opt, sizeof(buf_opt));

    if (connect(sock_fd, result->ai_addr, result->ai_addrlen) < 0) {
      close(sock_fd);
      freeaddrinfo(result);
      return absl::UnavailableError(
          absl::StrCat("Connect failed: ", std::strerror(errno)));
    }

    freeaddrinfo(result);

    absl::MutexLock _(conn_mu_);
    connection_pool_[peer] = sock_fd;
    return sock_fd;
  }

  void ListenerLoop() {
    while (!stopping_) {
      struct pollfd pfd;
      pfd.fd = server_fd_;
      pfd.events = POLLIN;
      int ret = poll(&pfd, 1, 50);
      if (ret <= 0) continue;

      struct sockaddr_in client_addr;
      socklen_t clilen = sizeof(client_addr);
      int client_fd =
          accept(server_fd_, reinterpret_cast<struct sockaddr*>(&client_addr),
                 &clilen);
      if (client_fd < 0) continue;

      int opt = 1;
      setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
      int buf_opt = 16 * 1024 * 1024;  // 16MB
      setsockopt(client_fd, SOL_SOCKET, SO_SNDBUF, &buf_opt, sizeof(buf_opt));
      setsockopt(client_fd, SOL_SOCKET, SO_RCVBUF, &buf_opt, sizeof(buf_opt));
      worker_threads_.push_back(
          std::thread([this, client_fd]() { ConnectionWorker(client_fd); }));
    }
  }

  absl::Status ProcessSingleRequest(int client_fd) {
    BlockPacketHeader header;
    TF_RETURN_IF_ERROR(ReadExact(client_fd, &header, sizeof(header)));

    size_t bytes_per_block = parent_->block_size_ * parent_->slice_byte_size_;

    if (header.op == 1) {
      // Push transfer (incoming write). Dynamically allocate blocks locally.
      size_t local_blocks = header.num_blocks / parent_->shard_factor_;
      TF_ASSIGN_OR_RETURN(
          std::vector<int> allocated_ids,
          parent_->AllocateBlocks(local_blocks, /*entity_id=*/0));

      // Send back acknowledgment header containing assigned block IDs.
      TF_RETURN_IF_ERROR(WriteExact(client_fd, allocated_ids.data(),
                                    allocated_ids.size() * sizeof(int)));

      // Read payload blocks across all layers and shards symmetrically.
      for (size_t l = 0; l < parent_->num_layers_; ++l) {
        const auto& layer_info = parent_->layers_[l];
        for (size_t sh = 0; sh < parent_->num_shards_; ++sh) {
          const auto& shard_info = layer_info.shards[sh];
          uint8_t* base_host_ptr = const_cast<uint8_t*>(shard_info.host_ptr);

          for (int k = 0; k < local_blocks; ++k) {
            int assigned_id = allocated_ids[k];
            uint8_t* dest_ptr = base_host_ptr + assigned_id * bytes_per_block;
            TF_RETURN_IF_ERROR(ReadExact(client_fd, dest_ptr, bytes_per_block));
          }
        }
      }

      // 3. Automatically copy received weights from Host onto Device
      // Accelerator HBM E2E!
      TF_RETURN_IF_ERROR(parent_->OnDataReceived());

      // Send final completion acknowledgment byte to unblock client Post loop.
      uint8_t ack = 1;
      TF_RETURN_IF_ERROR(WriteExact(client_fd, &ack, 1));
    } else if (header.op == 2) {
      // Pull transfer (incoming read request).
      // Read data from requested local blocks and push back response writes.
      BlockPacketHeader resp_header;
      resp_header.op = 1;
      resp_header.remote_block_id = header.local_block_id;
      resp_header.local_block_id = 0;
      resp_header.num_blocks = header.num_blocks;

      TF_RETURN_IF_ERROR(
          WriteExact(client_fd, &resp_header, sizeof(resp_header)));

      size_t local_blocks = header.num_blocks / parent_->shard_factor_;
      for (size_t l = 0; l < parent_->num_layers_; ++l) {
        const auto& layer_info = parent_->layers_[l];
        for (size_t sh = 0; sh < parent_->num_shards_; ++sh) {
          const auto& shard_info = layer_info.shards[sh];
          const uint8_t* base_host_ptr = shard_info.host_ptr;

          for (int k = 0; k < local_blocks; ++k) {
            int read_id = header.remote_block_id + k;
            const uint8_t* src_ptr = base_host_ptr + read_id * bytes_per_block;
            TF_RETURN_IF_ERROR(WriteExact(client_fd, src_ptr, bytes_per_block));
          }
        }
      }
    }
    return absl::OkStatus();
  }

  void ConnectionWorker(int client_fd) {
    while (!stopping_) {
      struct pollfd pfd;
      pfd.fd = client_fd;
      pfd.events = POLLIN;
      int ret = poll(&pfd, 1, 50);
      if (ret < 0) break;
      if (ret == 0) continue;

      if (!ProcessSingleRequest(client_fd).ok()) {
        break;
      }
    }
    close(client_fd);
  }

  RaidenManagerBase* parent_;
  int local_port_;
  int server_fd_ = -1;
  std::atomic<bool> stopping_{false};

  absl::Mutex conn_mu_;
  absl::flat_hash_map<std::string, int> connection_pool_
      ABSL_GUARDED_BY(conn_mu_);

  std::thread listener_thread_;
  std::vector<std::thread> worker_threads_;
};

RaidenManagerBase::RaidenManagerBase(size_t num_layers, size_t num_shards,
                                     size_t slice_byte_size, int block_size,
                                     std::optional<int> local_port,
                                     int parallelism)
    : num_layers_(num_layers),
      num_shards_(num_shards),
      slice_byte_size_(slice_byte_size),
      block_size_(block_size),
      parallelism_(parallelism) {
  shard_factor_ = 1;

  if (local_port.has_value()) {
    server_ = std::make_unique<BlockTransportServer>(this, local_port.value());
  }
}

RaidenManagerBase::~RaidenManagerBase() = default;

std::optional<int> RaidenManagerBase::local_port() const {
  if (server_) return server_->local_port_;
  return std::nullopt;
}

const uint8_t* RaidenManagerBase::GetHostPointer(size_t layer_idx,
                                                 size_t shard_idx) const {
  if (layer_idx >= num_layers_ || shard_idx >= num_shards_) {
    return nullptr;
  }
  return layers_[layer_idx].shards[shard_idx].host_ptr;
}

void RaidenManagerBase::SetExternalHostPointers(
    const std::vector<const uint8_t*>& host_ptrs,
    const std::vector<size_t>& host_sizes) {
  size_t idx = 0;
  for (size_t l = 0; l < num_layers_; ++l) {
    for (size_t sh = 0; sh < num_shards_; ++sh) {
      if (idx < host_ptrs.size()) {
        layers_[l].shards[sh].host_ptr = host_ptrs[idx];
        layers_[l].shards[sh].host_size = host_sizes[idx];
        idx++;
      }
    }
  }
}

absl::StatusOr<std::vector<int>> RaidenManagerBase::H2hWriteDirect(
    const std::string& peer, const std::vector<int>& src_block_ids,
    int64_t entity_id) {
  size_t num_blocks = src_block_ids.size();
  if (num_blocks == 0) {
    return absl::InvalidArgumentError("Block list cannot be empty");
  }

  int P = parallelism_;
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
        &RaidenManagerBase::H2hWriteWorker, this, i, peer, blocks_per_stream,
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

absl::StatusOr<std::vector<int>> RaidenManagerBase::H2hReadDirect(
    const std::string& peer, const std::vector<int>& src_block_ids,
    int64_t entity_id) {
  size_t num_blocks = src_block_ids.size();
  if (num_blocks == 0) {
    return absl::InvalidArgumentError("Block list cannot be empty");
  }

  size_t local_blocks = num_blocks / shard_factor_;
  TF_ASSIGN_OR_RETURN(std::vector<int> allocated_ids,
                      AllocateBlocks(local_blocks, entity_id));

  int P = parallelism_;
  if (static_cast<int>(local_blocks) < P) P = local_blocks;

  if (local_blocks % P != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Local block count (", local_blocks,
                     ") must be fully divisible by parallelism (", P, ")"));
  }

  size_t blocks_per_stream = local_blocks / P;
  size_t remote_blocks_per_stream = num_blocks / P;
  int base_remote_id = src_block_ids[0];

  std::vector<std::thread> threads;
  std::vector<absl::Status> statuses(P, absl::OkStatus());

  threads.reserve(P);
  for (int i = 0; i < P; ++i) {
    threads.push_back(std::thread(&RaidenManagerBase::H2hReadWorker, this, i,
                                  peer, blocks_per_stream,
                                  remote_blocks_per_stream, base_remote_id,
                                  std::ref(allocated_ids), std::ref(statuses)));
  }

  for (auto& t : threads) {
    if (t.joinable()) t.join();
  }

  for (int i = 0; i < P; ++i) {
    if (!statuses[i].ok()) return statuses[i];
  }

  return allocated_ids;
}

void RaidenManagerBase::H2hWriteWorker(int stream_idx, const std::string& peer,
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
    allocated_ids[offset + k] = stream_allocated_ids[k];
  }

  size_t bytes_per_block = block_size_ * slice_byte_size_;

  for (size_t l = 0; l < num_layers_; ++l) {
    const auto& layer_info = layers_[l];
    for (size_t sh = 0; sh < num_shards_; ++sh) {
      const auto& shard_info = layer_info.shards[sh];
      const uint8_t* base_host_ptr = shard_info.host_ptr;

      for (size_t k = 0; k < blocks_per_stream; ++k) {
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

void RaidenManagerBase::H2hReadWorker(int stream_idx, const std::string& peer,
                                      size_t blocks_per_stream,
                                      size_t remote_blocks_per_stream,
                                      int base_remote_id,
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

  BlockPacketHeader header;
  header.op = 2;  // Pull request
  header.remote_block_id = GetRemoteReadBlockId(base_remote_id, remote_offset);
  header.local_block_id = allocated_ids[offset];
  header.num_blocks = remote_blocks_per_stream;

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

  size_t bytes_per_block = block_size_ * slice_byte_size_;

  for (size_t l = 0; l < num_layers_; ++l) {
    const auto& layer_info = layers_[l];
    for (size_t sh = 0; sh < num_shards_; ++sh) {
      const auto& shard_info = layer_info.shards[sh];
      uint8_t* base_host_ptr = const_cast<uint8_t*>(shard_info.host_ptr);

      for (size_t k = 0; k < blocks_per_stream; ++k) {
        int dst_id = allocated_ids[offset + k];
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

}  // namespace tpu_raiden
