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

#ifndef THIRD_PARTY_PEREGRINE_SRC_INTERNAL_SOCKET_SOCKET_TCP_H_
#define THIRD_PARTY_PEREGRINE_SRC_INTERNAL_SOCKET_SOCKET_TCP_H_

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>

#include <cstddef>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>

#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/endpoint.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/types.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_base.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_util.h"

namespace peregrine::internal {

// This class wraps a TCP/IPv{4,6} socket for reliable network communications.
// It is movable but not copyable.
// This class is thread-compatible but not thread-safe.
class TcpSocket final : public SocketBase {
 public:
  // Creates an unconnected tcp socket.
  static std::unique_ptr<TcpSocket> Create(int family);

  // Creates a connected tcp socket.
  static std::unique_ptr<TcpSocket> Create(fd_t fd, int family);

  // Destructor closes the socket.
  ~TcpSocket();

  // Shuts down the socket for both send and recv.
  void Shutdown();

  // Listens on the `local` endpoint.
  bool Listen(const Endpoint& local) const;

  // Accepts a new connection to this listening socket. Returns the new spawn
  // socket file descriptor if successful. Return -2 if the socket is shut down.
  // Otherwise, returns -1.
  fd_t Accept() const;

  // Connects to the `peer` endpoint.
  bool Connect(const Endpoint& peer);

  // Sends `len` bytes of data from the `buf`.
  // Returns the number of bytes sent if successful. Zero byte means no data
  // has been sent due to non-error reasons. Returns -1 on error.
  ssize_t Send(const Byte* buf, size_t len) const;

  // Receives exactly `len` bytes of data into the `buf`.
  // Returns the number of bytes received if successful. Zero byte means the
  // peer side has closed the connection. Returns -1 on error.
  ssize_t Recv(Byte* buf, size_t len) const;

  // Sends on the socket `fd` exactly `len` bytes of data from the `buf`.
  // Returns OK if all the bytes are sent successfully, error otherwise.
  ABSL_DEPRECATED("temporarily for tpu raiden")
  static absl::Status Send(fd_t fd, const Byte* buf, size_t len);

  // Receives on the socket `fd` exactly `len` bytes of data into the `buf`.
  // Returns OK if all the bytes are received successfully, error otherwise.
  ABSL_DEPRECATED("temporarily for tpu raiden")
  static absl::Status Recv(fd_t fd, Byte* buf, size_t len);

  // Returns a self/peer address pair string of the socket.
  std::string ToString() const;

 private:
  // Constructor with a valid file descriptor `fd`.
  // The `fd` comes from a successful `Create()` or `Accept()` call.
  TcpSocket(fd_t fd, int family, bool connected)
      : SocketBase(fd, family, connected) {
    DCHECK(invariant());
  }

 private:
  // Returns a success message for the last socket operation.
  static std::string okMsg(std::string_view func, fd_t fd) {
    return SuccessMsg(kTcp, func, fd);
  }

  // Returns a success message for the last socket operation.
  std::string okMsg(std::string_view func) const {
    return SuccessMsg(kTcp, func, fd_);
  }

  // Returns a success message for the socket send/recv call.
  std::string ioMsg(std::string_view func, size_t bytes) const {
    return SuccessMsg(kTcp, func, fd_, bytes);
  }

  // Returns an error message for the last socket operation.
  std::string errMsg(std::string_view func, int last_errno) const {
    return ErrorMsg(kTcp, func, fd_, last_errno);
  }

  static constexpr std::string_view kTcp = "tcp";
};

inline std::ostream& operator<<(std::ostream& os, const TcpSocket& s) {
  return os << s.ToString();
}

}  // namespace peregrine::internal

#endif  // THIRD_PARTY_PEREGRINE_SRC_INTERNAL_SOCKET_SOCKET_TCP_H_
