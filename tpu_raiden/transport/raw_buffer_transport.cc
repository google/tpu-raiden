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

#include "tpu_raiden/transport/raw_buffer_transport.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_raiden/core/status_macros.h"
#include "tpu_raiden/transport/socket_util.h"

namespace tpu_raiden {
namespace transport {

RawBufferTransport::RawBufferTransport(
    RawBufferTransportDelegate* delegate, int local_port, bool enable_conn_pool,
    const std::vector<std::string>& local_ips)
    : raw_delegate_(delegate),
      local_port_(local_port),
      bound_ip_(local_ips.empty() ? "127.0.0.1" : local_ips[0]),
      local_ips_(local_ips),
      pooling_enabled_(enable_conn_pool) {
  // 1. Setup server_fd_ (Always use IPv6 wildcard to listen on all interfaces)
  server_fd_ = socket(AF_INET6, SOCK_STREAM, 0);
  if (server_fd_ < 0) {
    // Fallback to IPv4 wildcard if IPv6 is not supported
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
      throw std::runtime_error("Failed to create server socket: " +
                               std::string(std::strerror(errno)));
    }
    int opt = 1;
    if (setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) <
        0) {
      throw std::runtime_error("Failed to set SO_REUSEADDR: " +
                               std::string(std::strerror(errno)));
    }
    struct sockaddr_in serv_addr;
    std::memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(local_port_);
    if (bind(server_fd_, reinterpret_cast<struct sockaddr*>(&serv_addr),
             sizeof(serv_addr)) < 0) {
      throw std::runtime_error(
          "Failed to bind IPv4 server socket to wildcard:" +
          std::to_string(local_port_) + ": " + std::strerror(errno));
    }
    socklen_t addr_len = sizeof(serv_addr);
    if (getsockname(server_fd_, reinterpret_cast<struct sockaddr*>(&serv_addr),
                    &addr_len) == 0) {
      local_port_ = ntohs(serv_addr.sin_port);
    }
  } else {
    int opt = 1;
    if (setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) <
        0) {
      throw std::runtime_error("Failed to set SO_REUSEADDR: " +
                               std::string(std::strerror(errno)));
    }
    int v6only = 0;
    if (setsockopt(server_fd_, IPPROTO_IPV6, IPV6_V6ONLY, &v6only,
                   sizeof(v6only)) < 0) {
      LOG(WARNING) << "Failed to set IPV6_V6ONLY=0: " << std::strerror(errno);
    }
    struct sockaddr_in6 serv_addr;
    std::memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin6_family = AF_INET6;
    serv_addr.sin6_addr = in6addr_any;
    serv_addr.sin6_port = htons(local_port_);
    if (bind(server_fd_, reinterpret_cast<struct sockaddr*>(&serv_addr),
             sizeof(serv_addr)) < 0) {
      throw std::runtime_error(
          "Failed to bind IPv6 server socket to wildcard:" +
          std::to_string(local_port_) + ": " + std::strerror(errno));
    }
    socklen_t addr_len = sizeof(serv_addr);
    if (getsockname(server_fd_, reinterpret_cast<struct sockaddr*>(&serv_addr),
                    &addr_len) == 0) {
      local_port_ = ntohs(serv_addr.sin6_port);
    }
  }

  if (listen(server_fd_, 128) < 0) {
    LOG(FATAL) << "Failed to listen on server socket: " << std::strerror(errno);
  }

  // 2. Start listener
  listener_thread_ = std::thread(&RawBufferTransport::ListenerLoop, this);
}

RawBufferTransport::~RawBufferTransport() {
  stopping_ = true;
  ClosePooledConnections();
  if (server_fd_ >= 0) {
    shutdown(server_fd_, SHUT_RDWR);
    close(server_fd_);
  }
  {
    absl::MutexLock _( mu_ );
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

static std::string GetPoolKey(absl::string_view peer,
                              absl::string_view local_ip) {
  return absl::StrCat(local_ip, "->", peer);
}

absl::StatusOr<int> RawBufferTransport::AcquireConnection(
    absl::string_view peer, absl::string_view local_ip) {
  if (pooling_enabled_) {
    absl::MutexLock lock( pool_mu_ );
    std::string key = GetPoolKey(peer, local_ip);
    if (auto it = conn_pool_.find(key); it != conn_pool_.end()) {
      while (!it->second.empty()) {
        int fd = it->second.back();
        it->second.pop_back();

        struct pollfd pfd;
        pfd.fd = fd;
        pfd.events = POLLIN;
        if (poll(&pfd, 1, 0) > 0) {
          shutdown(fd, SHUT_RDWR);
          close(fd);
          continue;
        }
        return fd;
      }
    }
  }
  return ConnectToPeer(peer, local_ip);
}

void RawBufferTransport::ReleaseConnection(absl::string_view peer, int fd,
                                           absl::string_view local_ip) {
  if (fd < 0) return;
  absl::MutexLock lock( pool_mu_ );
  if (!pooling_enabled_ || stopping_) {
    shutdown(fd, SHUT_RDWR);
    close(fd);
    return;
  }
  std::string key = GetPoolKey(peer, local_ip);
  auto it = conn_pool_.find(key);
  if (it == conn_pool_.end()) {
    it = conn_pool_.emplace(key, std::vector<int>{}).first;
  }
  it->second.push_back(fd);
}

void RawBufferTransport::ClosePooledConnections() {
  absl::MutexLock lock( pool_mu_ );
  for (auto& entry : conn_pool_) {
    for (int fd : entry.second) {
      shutdown(fd, SHUT_RDWR);
      close(fd);
    }
  }
  conn_pool_.clear();
}

absl::Status RawBufferTransport::ProcessSingleRequest(int client_fd) {
  PacketHeader header = {};
  RETURN_IF_ERROR(ReadExact(client_fd, &header, sizeof(header)));

  if (header.op == 5) {
    uint32_t dst_offset = header.remote_id;
    uint32_t dst_shard_idx = header.local_id;
    uint32_t size_bytes = header.count_or_size;
    uint16_t buf_id = header.buffer_id;

    uint8_t* base_host_ptr =
        raw_delegate_->GetHostPointer(buf_id, dst_shard_idx);
    size_t host_size = raw_delegate_->GetHostSize(buf_id, dst_shard_idx);
    if (base_host_ptr == nullptr || dst_offset + size_bytes > host_size) {
      return absl::InvalidArgumentError("Destination out of bounds");
    }
    uint8_t* dest_ptr = base_host_ptr + dst_offset;
    RETURN_IF_ERROR(ReadExact(client_fd, dest_ptr, size_bytes));

    uint8_t ack = 1;
    RETURN_IF_ERROR(WriteExact(client_fd, &ack, 1));
  } else if (header.op == 3) {
    uint32_t src_offset = header.remote_id;
    uint32_t src_shard_idx = header.local_id;
    uint32_t size_bytes = header.count_or_size;
    uint16_t buf_id = header.buffer_id;

    uint8_t* base_host_ptr =
        raw_delegate_->GetHostPointer(buf_id, src_shard_idx);
    size_t host_size = raw_delegate_->GetHostSize(buf_id, src_shard_idx);
    if (base_host_ptr == nullptr || src_offset + size_bytes > host_size) {
      return absl::InvalidArgumentError("Source out of bounds");
    }
    uint8_t* src_ptr = base_host_ptr + src_offset;
    RETURN_IF_ERROR(WriteExact(client_fd, src_ptr, size_bytes));

  } else {
    return HandleCustomRequest(client_fd, header);
  }
  return absl::OkStatus();
}

absl::Status RawBufferTransport::HandleCustomRequest(
    int client_fd, const PacketHeader& header) {
  return absl::UnimplementedError(
      absl::StrCat("Unsupported raw transport op code: ", header.op));
}

void RawBufferTransport::ConnectionWorker(int client_fd) {
  while (!stopping_) {
    struct pollfd pfd;
    pfd.fd = client_fd;
    pfd.events = POLLIN;
    int ret = poll(&pfd, 1, 50);
    if (ret < 0) {
      // EINTR (interrupted by a signal) and EAGAIN (transient kernel resource
      // pressure) are benign: retry the poll rather than tearing down a healthy
      // connection. Only a genuine error closes the connection.
      if (errno == EINTR || errno == EAGAIN) continue;
      break;
    }
    if (ret == 0) continue;

    if (!ProcessSingleRequest(client_fd).ok()) {
      break;
    }
  }
  close(client_fd);
  {
    absl::MutexLock _( mu_ );
    active_client_fds_.erase(std::remove(active_client_fds_.begin(),
                                         active_client_fds_.end(), client_fd),
                             active_client_fds_.end());
  }
}

void RawBufferTransport::ListenerLoop() {
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
    int buf_opt = 16 * 1024 * 1024;  // 16MB
    setsockopt(client_fd, SOL_SOCKET, SO_SNDBUF, &buf_opt, sizeof(buf_opt));
    setsockopt(client_fd, SOL_SOCKET, SO_RCVBUF, &buf_opt, sizeof(buf_opt));

    {
      absl::MutexLock _( mu_ );
      active_client_fds_.push_back(client_fd);
    }

    worker_threads_.push_back(
        std::thread([this, client_fd]() { ConnectionWorker(client_fd); }));
  }
}

absl::Status RawBufferTransport::PullBuffer(
    absl::string_view source, size_t buffer_id, size_t src_shard_idx,
    size_t src_offset_bytes, size_t dst_shard_idx, size_t dst_offset_bytes,
    size_t size_bytes) {
  if (source.empty()) {
    return absl::InvalidArgumentError("Source peer address cannot be empty");
  }

  size_t host_size = raw_delegate_->GetHostSize(buffer_id, dst_shard_idx);
  if (dst_offset_bytes + size_bytes > host_size) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Destination offset out of bounds. Offset: ", dst_offset_bytes,
        ", Size: ", size_bytes, ", Shard Host Size: ", host_size));
  }

  auto status_or_fd = AcquireConnection(source);
  if (!status_or_fd.ok()) return status_or_fd.status();
  int fd = status_or_fd.value();
  bool ok_to_pool = false;
  auto fd_cleaner = absl::MakeCleanup([&] {
    if (ok_to_pool) {
      ReleaseConnection(source, fd);
    } else {
      shutdown(fd, SHUT_RDWR);
      close(fd);
    }
  });

  PacketHeader header = {};
  header.op = 3;
  header.buffer_id = static_cast<uint16_t>(buffer_id);
  header.remote_id = static_cast<uint32_t>(src_offset_bytes);
  header.local_id = static_cast<uint32_t>(src_shard_idx);
  header.count_or_size = static_cast<uint32_t>(size_bytes);

  RETURN_IF_ERROR(WriteExact(fd, &header, sizeof(header)));

  uint8_t* dest_ptr = raw_delegate_->GetHostPointer(buffer_id, dst_shard_idx) +
                      dst_offset_bytes;
  RETURN_IF_ERROR(ReadExact(fd, dest_ptr, size_bytes));

  ok_to_pool = true;
  return absl::OkStatus();
}

absl::Status RawBufferTransport::PushBuffer(
    absl::string_view peer, size_t buffer_id, size_t dst_shard_idx,
    size_t dst_offset_bytes, const uint8_t* data_ptr, size_t size_bytes) {
  if (peer.empty()) {
    return absl::InvalidArgumentError(
        "Destination peer address cannot be empty");
  }

  TF_ASSIGN_OR_RETURN(const int fd, AcquireConnection(peer));
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
  header.op = 5;
  header.buffer_id = static_cast<uint16_t>(buffer_id);
  header.remote_id = static_cast<uint32_t>(dst_offset_bytes);
  header.local_id = static_cast<uint32_t>(dst_shard_idx);
  header.count_or_size = static_cast<uint32_t>(size_bytes);

  RETURN_IF_ERROR(WriteExact(fd, &header, sizeof(header)));
  RETURN_IF_ERROR(WriteExact(fd, data_ptr, size_bytes));

  uint8_t ack = 0;
  RETURN_IF_ERROR(ReadExact(fd, &ack, 1));
  if (ack != 1) {
    return absl::InternalError("PushBuffer verification failed");
  }

  ok_to_pool = true;
  return absl::OkStatus();
}

// Force warnings check
}  // namespace transport

}  // namespace tpu_raiden
