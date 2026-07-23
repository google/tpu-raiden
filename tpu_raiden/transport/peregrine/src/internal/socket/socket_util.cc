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

#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_util.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <cerrno>
#include <cstring>
#include <string>
#include <string_view>

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/types.h"

namespace peregrine::internal {

fd_t CreateSocket(int family, int type, bool blocking) {
  DCHECK(family == AF_INET || family == AF_INET6);
  DCHECK(type == SOCK_STREAM || type == SOCK_DGRAM);
  const int ret = ::socket(family, type | SOCK_CLOEXEC, /*protocol=*/0);
  if (ret < 0) {
    const auto last_errno = errno;
    LOG(WARNING) << ErrorMsg("socket", last_errno);
    return fd_t(-1);
  }
  const fd_t fd(ret);
  DCHECK(IsBlockingMode(fd));
  if (blocking) {
    return fd;
  }
  if (!SetNonBlockingMode(fd)) {
    ::close(fd.value());  // release the socket resource.
    return fd_t(-1);
  }
  DCHECK(IsNonBlockingMode(fd));
  return fd;
}

bool IsBlockingMode(fd_t fd) {
  const int flags = ::fcntl(fd.value(), F_GETFL);
  return flags >= 0 && !(flags & O_NONBLOCK);
}

bool IsNonBlockingMode(fd_t fd) {
  const int flags = ::fcntl(fd.value(), F_GETFL);
  return flags >= 0 && (flags & O_NONBLOCK);
}

bool __set_blocking_mode(fd_t fd, bool blocking) {
  const int flags = ::fcntl(fd.value(), F_GETFL);
  if ABSL_PREDICT_FALSE (flags < 0) return false;
  const int cmd = blocking ? (flags & ~O_NONBLOCK) : (flags | O_NONBLOCK);
  return ::fcntl(fd.value(), F_SETFL, cmd) >= 0;
}

std::string SelfAddrPort(const fd_t fd) {
  struct sockaddr_storage ss;
  socklen_t len = sizeof(ss);
  if (::getsockname(fd.value(), (struct sockaddr*)&ss, &len) == 0) {
    return ToIpAddrPortString(ss);
  } else {
    const auto last_errno = errno;
    LOG(WARNING) << ErrorMsg("getsockname", last_errno);
    return "?";
  }
}

std::string PeerAddrPort(const fd_t fd) {
  struct sockaddr_storage ss;
  socklen_t len = sizeof(ss);
  if (::getpeername(fd.value(), (struct sockaddr*)&ss, &len) == 0) {
    return ToIpAddrPortString(ss);
  } else if (errno == ENOTCONN) {
    return "*";
  } else {
    const auto last_errno = errno;
    LOG(WARNING) << ErrorMsg("getpeername", last_errno);
    return "?";
  }
}

namespace {
std::string NtopErrorMsg(int v, int last_errno) {
  return absl::StrFormat("inet_ntop failed: ipv%d, errno=%d (%s)", v,
                         last_errno, std::strerror(last_errno));
}
}  // namespace

namespace {
template <int kFamily, int kAddrLen, typename T>
std::string ToString(const struct sockaddr_storage& ss) {
  char addr[kAddrLen];
  const T* sa = reinterpret_cast<const T*>(&ss);
  if constexpr (kFamily == AF_INET) {
    if (inet_ntop(AF_INET, &sa->sin_addr, addr, kAddrLen) != nullptr) {
      return absl::StrCat(addr, ":", ntohs(sa->sin_port));
    }
    const auto last_errno = errno;
    LOG(WARNING) << NtopErrorMsg(4, last_errno);
    return "invalid ipv4:port";
  } else {
    static_assert(kFamily == AF_INET6);
    if (inet_ntop(AF_INET6, &sa->sin6_addr, addr, kAddrLen) != nullptr) {
      return absl::StrCat("[", addr, "]:", ntohs(sa->sin6_port));
    }
    const auto last_errno = errno;
    LOG(WARNING) << NtopErrorMsg(6, last_errno);
    return "invalid ipv6:port";
  }
}
}  // namespace

std::string ToIpAddrPortString(const struct sockaddr_storage& ss) {
  switch (ss.ss_family) {
    case AF_INET:
      return ToString<AF_INET, INET_ADDRSTRLEN, struct sockaddr_in>(ss);
    case AF_INET6:
      return ToString<AF_INET6, INET6_ADDRSTRLEN, struct sockaddr_in6>(ss);
    default:
      return absl::StrCat("invalid addr family: ", ss.ss_family);
  }
}

}  // namespace peregrine::internal
