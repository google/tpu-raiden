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

#ifndef THIRD_PARTY_PEREGRINE_SRC_INTERNAL_SOCKET_SOCKET_UTIL_H_
#define THIRD_PARTY_PEREGRINE_SRC_INTERNAL_SOCKET_SOCKET_UTIL_H_

#include <fcntl.h>
#include <sys/socket.h>

#include <cerrno>
#include <cstring>
#include <string>
#include <string_view>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/types.h"

namespace peregrine::internal {

// Creates a new socket.
// Returns its file descriptor if successful, or -1 otherwise.
fd_t CreateSocket(int family, int type, bool blocking);

// Sets socket option. Returns true if successful, false otherwise.
inline bool SetOption(fd_t fd, int opt, const void* val, socklen_t len) {
  return ::setsockopt(fd.value(), SOL_SOCKET, opt, val, len) >= 0;
}

// Returns true iff the socket `fd` is valid (not closed).
inline bool IsValidSocket(fd_t fd) { return ::fcntl(fd.value(), F_GETFD) >= 0; }

// Returns true iff the socket `fd` is in blocking mode.
bool IsBlockingMode(fd_t fd);

// Returns true iff the socket `fd` is in non-blocking mode.
bool IsNonBlockingMode(fd_t fd);

// Sets the socket to the specified blocking mode.
// Returns true if successful, false otherwise.
// For internal use only.
bool __set_blocking_mode(fd_t fd, bool blocking);

// Sets the socket to blocking mode.
// Returns true if successful, false otherwise.
inline bool SetBlockingMode(fd_t fd) {
  return __set_blocking_mode(fd, /*blocking=*/true);
}

// Sets the socket to non-blocking mode.
// Returns true if successful, false otherwise.
inline bool SetNonBlockingMode(fd_t fd) {
  return __set_blocking_mode(fd, /*blocking=*/false);
}

// Returns true iff the tcp listen socket Accept() call was shut down.
inline bool IsShutdown(int ret) { return ret == -2; }

// Returns true iff the last socket operation was interrupted by a signal.
inline bool Interrupted(int last_errno) { return last_errno == EINTR; }

// Returns true iff the last socket operation would block.
inline bool WouldBlock(int last_errno) {
  return last_errno == EAGAIN || last_errno == EWOULDBLOCK;
}

// Returns true iff the socket connect operation is in progress.
inline bool InProgress(int last_errno) { return last_errno == EINPROGRESS; }

// Returns a self ip:port string for the socket `fd`.
std::string SelfAddrPort(fd_t fd);

// Returns a peer ip:port string for the socket `fd`.
std::string PeerAddrPort(fd_t fd);

// Returns a string of self/peer ip:port pair for the socket `fd`.
inline std::string AddrPortPair(fd_t fd) {
  return absl::StrCat(SelfAddrPort(fd), " <> ", PeerAddrPort(fd));
}

// Returns a "ipv4:port" or "[ipv6]:port" string.
std::string ToIpAddrPortString(const struct sockaddr_storage& ss);

// Returns a success message for the last socket operation.
inline std::string SuccessMsg(std::string_view who, std::string_view what,
                              fd_t fd) {
  return absl::StrCat(who, " socket ", what, ", fd=", fd.value(), " ",
                      AddrPortPair(fd));
}

// Returns a success message for the last socket send/recv call.
inline std::string SuccessMsg(std::string_view who, std::string_view what,
                              fd_t fd, size_t bytes) {
  return absl::StrCat(who, " socket ", what, ", fd=", fd.value(), " ",
                      AddrPortPair(fd), " #bytes=", bytes);
}

// Returns an error message for the last socket operation.
inline std::string ErrorMsg(std::string_view who, std::string_view what,
                            fd_t fd, int last_errno) {
  return absl::StrFormat("%s socket %s failed: fd=%d %s errno=%d (%s)", who,
                         what, fd.value(), AddrPortPair(fd), last_errno,
                         std::strerror(last_errno));
}

// Returns an error message for the last socket operation.
inline std::string ErrorMsg(std::string_view what, int last_errno) {
  return absl::StrFormat("socket %s failed: errno=%d (%s)", what, last_errno,
                         std::strerror(last_errno));
}

}  // namespace peregrine::internal

#endif  // THIRD_PARTY_PEREGRINE_SRC_INTERNAL_SOCKET_SOCKET_UTIL_H_
