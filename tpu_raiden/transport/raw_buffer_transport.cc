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
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/statusor.h"
#include "tpu_raiden/core/status_macros.h"

namespace tpu_raiden {
namespace transport {

absl::Status RawBufferTransport::WriteVExact(
    int fd, absl::Span<const struct iovec> iov) {
  std::vector<struct iovec> local_iov(iov.begin(), iov.end());
  size_t iov_idx = 0;
  while (iov_idx < local_iov.size()) {
    size_t batch_size =
        std::min(local_iov.size() - iov_idx, static_cast<size_t>(IOV_MAX));

    size_t batch_remaining = batch_size;
    while (batch_remaining > 0) {
      ssize_t written = writev(fd, &local_iov[iov_idx], batch_remaining);
      if (written < 0) {
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
          struct pollfd pfd;
          pfd.fd = fd;
          pfd.events = POLLOUT;
          int poll_ret = poll(&pfd, 1, 120000);  // 120s timeout
          if (poll_ret < 0) {
            if (errno == EINTR) continue;
            return absl::InternalError(absl::StrCat(
                "Poll failed during WriteVExact: ", std::strerror(errno)));
          }
          if (poll_ret == 0) {
            return absl::DeadlineExceededError(
                "Timeout waiting for socket writability during WriteVExact");
          }
          continue;
        }
        return absl::InternalError(
            absl::StrCat("Socket writev failed: ", std::strerror(errno)));
      }
      if (written == 0) {
        return absl::InternalError("Socket closed unexpectedly during writev");
      }

      size_t remaining = written;
      while (remaining > 0 && batch_remaining > 0) {
        if (remaining >= local_iov[iov_idx].iov_len) {
          remaining -= local_iov[iov_idx].iov_len;
          iov_idx++;
          batch_remaining--;
        } else {
          local_iov[iov_idx].iov_base =
              static_cast<char*>(local_iov[iov_idx].iov_base) + remaining;
          local_iov[iov_idx].iov_len -= remaining;
          remaining = 0;
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status RawBufferTransport::ReadVExact(
    int fd, absl::Span<const struct iovec> iov) {
  std::vector<struct iovec> local_iov(iov.begin(), iov.end());
  size_t iov_idx = 0;
  while (iov_idx < local_iov.size()) {
    size_t batch_size =
        std::min(local_iov.size() - iov_idx, static_cast<size_t>(IOV_MAX));

    size_t batch_remaining = batch_size;
    while (batch_remaining > 0) {
      ssize_t bytes_read = readv(fd, &local_iov[iov_idx], batch_remaining);
      if (bytes_read < 0) {
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
          struct pollfd pfd;
          pfd.fd = fd;
          pfd.events = POLLIN;
          int poll_ret = poll(&pfd, 1, 120000);  // 120s timeout
          if (poll_ret < 0) {
            if (errno == EINTR) continue;
            return absl::InternalError(absl::StrCat(
                "Poll failed during ReadVExact: ", std::strerror(errno)));
          }
          if (poll_ret == 0) {
            return absl::DeadlineExceededError(
                "Timeout waiting for socket readability during ReadVExact");
          }
          continue;
        }
        return absl::InternalError(
            absl::StrCat("Socket readv failed: ", std::strerror(errno)));
      }
      if (bytes_read == 0) {
        return absl::InternalError("Socket closed unexpectedly during readv");
      }

      size_t remaining = bytes_read;
      while (remaining > 0 && batch_remaining > 0) {
        if (remaining >= local_iov[iov_idx].iov_len) {
          remaining -= local_iov[iov_idx].iov_len;
          iov_idx++;
          batch_remaining--;
        } else {
          local_iov[iov_idx].iov_base =
              static_cast<char*>(local_iov[iov_idx].iov_base) + remaining;
          local_iov[iov_idx].iov_len -= remaining;
          remaining = 0;
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status RawBufferTransport::WriteExact(int fd, const void* buffer,
                                            size_t length) {
  struct iovec iov = {.iov_base = const_cast<void*>(buffer), .iov_len = length};
  return WriteVExact(fd, {&iov, 1});
}

absl::Status RawBufferTransport::ReadExact(int fd, void* buffer,
                                           size_t length) {
  struct iovec iov = {.iov_base = buffer, .iov_len = length};
  return ReadVExact(fd, {&iov, 1});
}

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

absl::StatusOr<int> RawBufferTransport::ConnectToPeer(
    absl::string_view peer, absl::string_view local_ip) {
  std::string host;
  std::string port_str;

  if (!peer.empty() && peer.front() == '[') {
    size_t closing_bracket = peer.find(']');
    if (closing_bracket == absl::string_view::npos ||
        closing_bracket + 1 >= peer.size() ||
        peer[closing_bracket + 1] != ':') {
      return absl::InvalidArgumentError(
          "Invalid IPv6 peer bracket string format");
    }
    host = std::string(peer.substr(1, closing_bracket - 1));
    port_str = std::string(peer.substr(closing_bracket + 2));
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

    bool should_bind =
        !local_ip.empty() && local_ip != "0.0.0.0" && local_ip != "::";

    if (should_bind) {
      std::string local_ip_str(local_ip);
      bool is_ipv6 = absl::StrContains(local_ip, ':');
      if (is_ipv6 && rp->ai_family == AF_INET6) {
        struct sockaddr_in6 local_addr;
        std::memset(&local_addr, 0, sizeof(local_addr));
        local_addr.sin6_family = AF_INET6;
        if (inet_pton(AF_INET6, local_ip_str.c_str(), &local_addr.sin6_addr) >
            0) {
          local_addr.sin6_port = 0;
          if (bind(sock_fd, (struct sockaddr*)&local_addr, sizeof(local_addr)) <
              0) {
            LOG(WARNING) << "Client bind IPv6 failed to " << local_ip << ": "
                         << std::strerror(errno);
          }
        }
      } else if (!is_ipv6 && rp->ai_family == AF_INET) {
        struct sockaddr_in local_addr;
        std::memset(&local_addr, 0, sizeof(local_addr));
        local_addr.sin_family = AF_INET;
        if (inet_pton(AF_INET, local_ip_str.c_str(), &local_addr.sin_addr) >
            0) {
          local_addr.sin_port = 0;
          if (bind(sock_fd, (struct sockaddr*)&local_addr, sizeof(local_addr)) <
              0) {
            LOG(WARNING) << "Client bind IPv4 failed to " << local_ip << ": "
                         << std::strerror(errno);
          }
        }
      }
    }

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

    absl::Status s = ProcessSingleRequest(client_fd);
    if (!s.ok()) {
      LOG(ERROR) << "ConnectionWorker: ProcessSingleRequest failed: " << s;
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
