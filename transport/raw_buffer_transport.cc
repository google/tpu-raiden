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

#include "transport/raw_buffer_transport.h"

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/statusor.h"
#include "core/status_macros.h"

namespace tpu_raiden {
namespace transport {

absl::Status RawBufferTransport::WriteExact(int fd, const void* buffer,
                                            size_t length) {
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

absl::Status RawBufferTransport::ReadExact(int fd, void* buffer,
                                           size_t length) {
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

RawBufferTransport::RawBufferTransport(RawBufferTransportDelegate* delegate,
                                       int local_port, bool enable_conn_pool)
    : RawBufferTransport(delegate, "", local_port, enable_conn_pool) {}

RawBufferTransport::RawBufferTransport(RawBufferTransportDelegate* delegate,
                                       const std::string& local_ip,
                                       int& local_port, bool enable_conn_pool)
    : raw_delegate_(delegate), pooling_enabled_(enable_conn_pool) {
  struct addrinfo hints;
  struct addrinfo* result = nullptr;
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;  // Allow IPv4 or IPv6
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;

  std::string port_str = std::to_string(local_port);
  int ret = getaddrinfo(local_ip.empty() ? nullptr : local_ip.c_str(),
                        port_str.c_str(), &hints, &result);
  if (ret != 0 || result == nullptr) {
    LOG(FATAL) << "getaddrinfo failed for local_ip " << local_ip << ": "
               << gai_strerror(ret);
  }

  int sock_fd = -1;
  bool bound = false;
  int bound_port = local_port;

  if (local_port == 0) {
    // Bind once to port 0 to let the OS assign an ephemeral port
    for (struct addrinfo* rp = result; rp != nullptr; rp = rp->ai_next) {
      sock_fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
      if (sock_fd < 0) {
        continue;
      }

      int opt = 1;
      setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

      if (bind(sock_fd, rp->ai_addr, rp->ai_addrlen) == 0) {
        server_fd_ = sock_fd;
        bound = true;
        break;
      }

      close(sock_fd);
    }
  } else {
    // Sequential port hunting for fixed port requests
    for (int port_attempt = 0; port_attempt < 100; ++port_attempt) {
      int target_port = local_port + port_attempt;
      if (target_port > 65535) {
        target_port = 1024 + (target_port % 64512);
      }
      bound_port = target_port;

      for (struct addrinfo* rp = result; rp != nullptr; rp = rp->ai_next) {
        sock_fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sock_fd < 0) {
          continue;
        }

        int opt = 1;
        setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        // Update the port in the resolved sockaddr
        if (rp->ai_family == AF_INET) {
          struct sockaddr_in* addr =
              reinterpret_cast<struct sockaddr_in*>(rp->ai_addr);
          addr->sin_port = htons(bound_port);
        } else if (rp->ai_family == AF_INET6) {
          struct sockaddr_in6* addr =
              reinterpret_cast<struct sockaddr_in6*>(rp->ai_addr);
          addr->sin6_port = htons(bound_port);
        }

        if (bind(sock_fd, rp->ai_addr, rp->ai_addrlen) == 0) {
          server_fd_ = sock_fd;
          bound = true;
          break;
        }

        close(sock_fd);
      }

      if (bound) {
        break;
      }
    }
  }

  freeaddrinfo(result);

  if (!bound) {
    LOG(FATAL) << "Failed to bind server socket to any port in range ["
               << local_port << ", " << local_port + 99 << "] on IP "
               << local_ip;
  }

  struct sockaddr_storage bound_addr;
  socklen_t addr_len = sizeof(bound_addr);
  if (getsockname(server_fd_, reinterpret_cast<struct sockaddr*>(&bound_addr),
                  &addr_len) == 0) {
    if (bound_addr.ss_family == AF_INET) {
      local_port =
          ntohs(reinterpret_cast<struct sockaddr_in*>(&bound_addr)->sin_port);
    } else if (bound_addr.ss_family == AF_INET6) {
      local_port =
          ntohs(reinterpret_cast<struct sockaddr_in6*>(&bound_addr)->sin6_port);
    }
  }
  local_port_ = local_port;

  if (listen(server_fd_, 128) < 0) {
    LOG(FATAL) << "Failed to listen on server socket: " << std::strerror(errno);
  }
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
    absl::MutexLock _(mu_);
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

absl::StatusOr<int> RawBufferTransport::ConnectToPeer(const std::string& peer) {
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

absl::StatusOr<int> RawBufferTransport::AcquireConnection(
    const std::string& peer) {
  if (pooling_enabled_) {
    absl::MutexLock lock(pool_mu_);
    auto it = conn_pool_.find(peer);
    if (it != conn_pool_.end()) {
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
  return ConnectToPeer(peer);
}

void RawBufferTransport::ReleaseConnection(const std::string& peer, int fd) {
  if (fd < 0) return;
  absl::MutexLock lock(pool_mu_);
  if (!pooling_enabled_ || stopping_) {
    shutdown(fd, SHUT_RDWR);
    close(fd);
    return;
  }
  conn_pool_[peer].push_back(fd);
}

void RawBufferTransport::ClosePooledConnections() {
  absl::MutexLock lock(pool_mu_);
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
      break;
    }
    if (ret == 0) continue;

    absl::Status status = ProcessSingleRequest(client_fd);
    if (!status.ok()) {
      LOG(ERROR) << "ConnectionWorker: ProcessSingleRequest failed: " << status;
      break;
    }
  }
  close(client_fd);
  {
    absl::MutexLock _(mu_);
    active_client_fds_.erase(std::remove(active_client_fds_.begin(),
                                         active_client_fds_.end(), client_fd),
                             active_client_fds_.end());
  }
  {
    absl::MutexLock lock(finished_mu_);
    finished_thread_ids_.push_back(std::this_thread::get_id());
  }
}

void RawBufferTransport::ReapFinishedWorkerThreads() {
  std::vector<std::thread::id> finished_ids;
  {
    absl::MutexLock lock(finished_mu_);
    finished_ids.swap(finished_thread_ids_);
  }
  if (finished_ids.empty()) return;

  absl::MutexLock lock(mu_);
  for (auto id : finished_ids) {
    auto it =
        std::find_if(worker_threads_.begin(), worker_threads_.end(),
                     [id](const std::thread& t) { return t.get_id() == id; });
    if (it != worker_threads_.end()) {
      if (it->joinable()) {
        it->join();
      }
      worker_threads_.erase(it);
    }
  }
}

void RawBufferTransport::ListenerLoop() {
  while (!stopping_) {
    ReapFinishedWorkerThreads();
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
      absl::MutexLock _(mu_);
      active_client_fds_.push_back(client_fd);
    }

    worker_threads_.push_back(
        std::thread([this, client_fd]() { ConnectionWorker(client_fd); }));
  }
}

absl::Status RawBufferTransport::PullBuffer(
    const std::string& source, size_t buffer_id, size_t src_shard_idx,
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

absl::Status RawBufferTransport::PushBuffer(const std::string& peer,
                                            size_t buffer_id,
                                            size_t dst_shard_idx,
                                            size_t dst_offset_bytes,
                                            const uint8_t* data_ptr,
                                            size_t size_bytes) {
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

}  // namespace transport
}  // namespace tpu_raiden
