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

#ifndef THIRD_PARTY_PEREGRINE_SRC_INTERNAL_SOCKET_SOCKET_BASE_H_
#define THIRD_PARTY_PEREGRINE_SRC_INTERNAL_SOCKET_SOCKET_BASE_H_

#include <fcntl.h>
#include <sys/socket.h>
#include <sys/types.h>

#include "absl/log/check.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/endpoint.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/types.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_util.h"
#include "tpu_raiden/transport/peregrine/src/util/macro.h"

namespace peregrine::internal {

// This class provides common functionalities for TCP/UDP sockets.
// It is not intended to be instantiated directly. Instead, instantiate the
// derived classes `TcpSocket` or `UdpSocket`.
// This class is thread-compatible but not thread-safe.
class SocketBase {
 public:
  // Returns AF_INET for IPv4 and AF_INET6 for IPv6.
  int family() const { return family_; }

  // Returns the socket file descriptor.
  fd_t fd() const { return fd_; }

  // Returns true iff the socket is up and running.
  bool IsValid() const { return IsValidSocket(fd_); }

  // Returns true iff the socket is connected.
  bool IsConnected() const { return connected_; }

  // Returns true iff the socket is in blocking mode.
  bool IsBlocking() const { return IsBlockingMode(fd_); }

  // Returns true iff the socket is in non-blocking mode.
  bool IsNonBlocking() const { return IsNonBlockingMode(fd_); }

 protected:
  // Constructor.
  SocketBase(fd_t fd, int family, bool connected)
      : fd_(fd), family_(family), connected_(connected) {
    DCHECK(invariant());
  }

  // Disables copy since the socket owns OS resource.
  DISALLOW_COPY(SocketBase);

  // Move constructor.
  SocketBase(SocketBase&& o) noexcept
      : fd_(o.fd_), family_(o.family_), connected_(o.connected_) {
    DCHECK(o.invariant());
    o.fd_ = fd_t(-1);
  }

  // Move assignment operator.
  SocketBase& operator=(SocketBase&& o) noexcept {
    DCHECK(o.invariant());
    if (this != &o) {
      fd_ = o.fd_;
      family_ = o.family_;
      connected_ = o.connected_;
      o.fd_ = fd_t(-1);
    }
    return *this;
  }

  // Destructor.
  ~SocketBase() { DCHECK_LT(fd_.value(), 0); }

  // Returns true iff the invariant holds.
  bool invariant() const {
    return fd_.value() >= 0 && (family_ == AF_INET || family_ == AF_INET6) &&
           IsValidSocket(fd_);
  }

 protected:
  // Binds to the `local` endpoint. Returns 0 on success, -1 on error.
  static int Bind(fd_t fd, const Endpoint& local);

  // Connects to the `peer` endpoint. Returns 0 on success, -1 on error.
  static int Connect(fd_t fd, const Endpoint& peer);

 protected:
  fd_t fd_;
  int family_;
  bool connected_;
};

}  // namespace peregrine::internal

#endif  // THIRD_PARTY_PEREGRINE_SRC_INTERNAL_SOCKET_SOCKET_BASE_H_
