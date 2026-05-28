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

#include "transport/socket_transport.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "third_party/peregrine/src/api/types.h"

namespace tpu_raiden {
namespace transport {

namespace {

// Helper to write exact number of bytes to a socket.
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

// Helper to read exact number of bytes from a socket.
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

SocketTransport::SocketTransport(int local_port) : local_port_(local_port) {
  server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd_ < 0) {
    LOG(FATAL) << "Failed to create server socket: " << std::strerror(errno);
  }

  int opt = 1;
  if (setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
    LOG(WARNING) << "setsockopt SO_REUSEADDR failed: " << std::strerror(errno);
  }

  struct sockaddr_in serv_addr;
  std::memset(&serv_addr, 0, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = INADDR_ANY;
  serv_addr.sin_port = htons(local_port_);

  if (bind(server_fd_, reinterpret_cast<struct sockaddr*>(&serv_addr),
           sizeof(serv_addr)) < 0) {
    LOG(FATAL) << "Failed to bind server socket to port " << local_port_ << ": "
               << std::strerror(errno);
  }

  if (listen(server_fd_, 128) < 0) {
    LOG(FATAL) << "Failed to listen on server socket: " << std::strerror(errno);
  }

  listener_thread_ = std::thread(&SocketTransport::ListenerLoop, this);
  LOG(INFO) << "SocketTransport server listening on port " << local_port_;
}

SocketTransport::~SocketTransport() {
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

  absl::MutexLock _(&conn_mu_);
  for (const auto& [peer, fd] : connection_pool_) {
    close(fd);
  }
  connection_pool_.clear();
}

absl::StatusOr<int> SocketTransport::GetOrCreateConnection(
    std::string_view peer) {
  std::string peer_str(peer);
  {
    absl::MutexLock _(&conn_mu_);
    auto it = connection_pool_.find(peer_str);
    if (it != connection_pool_.end()) {
      return it->second;
    }
  }

  std::vector<std::string> parts = absl::StrSplit(peer_str, ':');
  if (parts.size() != 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid endpoint string expected host:port, got ", peer_str));
  }
  std::string host = parts[0];
  int port = std::stoi(parts[1]);

  int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (sock_fd < 0) {
    return absl::InternalError(
        absl::StrCat("Failed to create client socket: ", std::strerror(errno)));
  }

  int opt = 1;
  setsockopt(sock_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

  struct addrinfo hints;
  struct addrinfo* res;
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;

  std::string port_str = std::to_string(port);
  int err = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res);
  if (err != 0) {
    close(sock_fd);
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to resolve hostname '", host, "': ", gai_strerror(err)));
  }

  if (connect(sock_fd, res->ai_addr, res->ai_addrlen) < 0) {
    freeaddrinfo(res);
    close(sock_fd);
    return absl::UnavailableError(absl::StrCat(
        "Failed to connect to peer ", peer_str, ": ", std::strerror(errno)));
  }
  freeaddrinfo(res);

  absl::MutexLock _(&conn_mu_);
  connection_pool_[peer_str] = sock_fd;
  return sock_fd;
}

absl::StatusOr<peregrine::Handle> SocketTransport::Post(
    std::string_view peer, const peregrine::Request& request) {
  if (!request.IsValid()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid transport request: ", request.ToString()));
  }

  auto status_or_fd = GetOrCreateConnection(peer);
  if (!status_or_fd.ok()) {
    return status_or_fd.status();
  }
  int fd = status_or_fd.value();

  peregrine::Handle handle;
  {
    absl::MutexLock _(&mu_);
    handle = peregrine::Handle(++handle_counter_);
    status_map_[handle] = peregrine::Status::kInProgress;
  }

  absl::Status op_status;
  if (request.op == peregrine::Op::kWrite) {
    op_status = DispatchWrite(fd, request);
  } else if (request.op == peregrine::Op::kRead) {
    op_status = DispatchReadRequest(fd, request);
  } else {
    op_status = absl::InternalError("Unsupported transport operation");
  }

  absl::MutexLock _(&mu_);
  if (op_status.ok()) {
    status_map_[handle] = peregrine::Status::kSuccess;
  } else {
    status_map_[handle] = peregrine::Status::kFailure;
  }

  return handle;
}

absl::Status SocketTransport::DispatchWrite(int fd,
                                            const peregrine::Request& request) {
  PacketHeader header;
  header.op = peregrine::Op::kWrite;
  header.remote_addr = reinterpret_cast<uint64_t>(request.raddr);
  header.local_addr = 0;
  header.length = request.len;

  absl::Status s = WriteExact(fd, &header, sizeof(header));
  if (!s.ok()) return s;

  s = WriteExact(fd, request.laddr, request.len);
  if (!s.ok()) return s;

  // Synchronize: wait for remote acknowledgment byte ensuring application
  // memory copy completion.
  uint8_t ack = 0;
  return ReadExact(fd, &ack, 1);
}

absl::Status SocketTransport::DispatchReadRequest(
    int fd, const peregrine::Request& request) {
  PacketHeader header;
  header.op = peregrine::Op::kRead;
  header.remote_addr = reinterpret_cast<uint64_t>(request.raddr);
  header.local_addr = reinterpret_cast<uint64_t>(request.laddr);
  header.length = request.len;

  absl::Status s = WriteExact(fd, &header, sizeof(header));
  if (!s.ok()) return s;

  // Read back response write packet header containing returned payload bytes.
  PacketHeader resp_header;
  s = ReadExact(fd, &resp_header, sizeof(resp_header));
  if (!s.ok()) return s;

  if (resp_header.length != request.len) {
    return absl::InternalError("Mismatched response payload length");
  }

  return ReadExact(fd, request.laddr, resp_header.length);
}

absl::StatusOr<peregrine::Status> SocketTransport::Poll(peregrine::Handle handle) {
  absl::MutexLock _(&mu_);
  auto it = status_map_.find(handle);
  if (it == status_map_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Transport handle not found: ", handle.value()));
  }
  peregrine::Status s = it->second;
  if (peregrine::IsCompleted(s)) {
    status_map_.erase(it);
  }
  return s;
}

void SocketTransport::ListenerLoop() {
  while (!stopping_) {
    struct pollfd pfd;
    pfd.fd = server_fd_;
    pfd.events = POLLIN;
    int ret = poll(&pfd, 1, 50);
    if (ret < 0) {
      if (stopping_) break;
      if (errno == EINTR) continue;
      LOG(WARNING) << "poll failed: " << std::strerror(errno);
      continue;
    }
    if (ret == 0) {
      continue;  // timeout, check stopping_ again
    }

    struct sockaddr_in client_addr;
    socklen_t clilen = sizeof(client_addr);
    int client_fd = accept(
        server_fd_, reinterpret_cast<struct sockaddr*>(&client_addr), &clilen);
    if (client_fd < 0) {
      if (stopping_) break;
      if (errno == EINTR) continue;
      LOG(WARNING) << "accept failed: " << std::strerror(errno);
      continue;
    }

    int opt = 1;
    setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

    worker_threads_.push_back(
        std::thread([this, client_fd]() { ConnectionWorker(client_fd); }));
  }
}

void SocketTransport::ConnectionWorker(int client_fd) {
  while (!stopping_) {
    struct pollfd pfd;
    pfd.fd = client_fd;
    pfd.events = POLLIN;
    int ret = poll(&pfd, 1, 50);
    if (ret < 0) {
      if (stopping_) break;
      if (errno == EINTR) continue;
      break;
    }
    if (ret == 0) {
      continue;  // timeout, check stopping_ again
    }

    PacketHeader header;
    absl::Status s = ReadExact(client_fd, &header, sizeof(header));
    if (!s.ok()) {
      break;  // connection closed or failed
    }

    if (header.op == peregrine::Op::kWrite) {
      uint8_t* dest_ptr = reinterpret_cast<uint8_t*>(header.remote_addr);
      s = ReadExact(client_fd, dest_ptr, header.length);
      if (!s.ok()) {
        LOG(ERROR) << "Failed reading payload for kWrite: " << s.message();
        break;
      }
      uint8_t ack = 1;
      s = WriteExact(client_fd, &ack, 1);
      if (!s.ok()) break;
    } else if (header.op == peregrine::Op::kRead) {
      // Read local memory to send back as a response write packet
      uint8_t* src_ptr = reinterpret_cast<uint8_t*>(header.remote_addr);

      PacketHeader resp_header;
      resp_header.op = peregrine::Op::kWrite;
      resp_header.remote_addr = header.local_addr;
      resp_header.local_addr = 0;
      resp_header.length = header.length;

      s = WriteExact(client_fd, &resp_header, sizeof(resp_header));
      if (!s.ok()) break;

      s = WriteExact(client_fd, src_ptr, header.length);
      if (!s.ok()) break;
    }
  }
  close(client_fd);
}

}  // namespace transport
}  // namespace tpu_raiden
