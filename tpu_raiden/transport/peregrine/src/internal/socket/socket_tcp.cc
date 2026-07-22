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

#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_tcp.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <string_view>

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/endpoint.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/types.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_base.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_util.h"

namespace peregrine::internal {

std::unique_ptr<TcpSocket> TcpSocket::Create(int family) {
  const fd_t fd = CreateSocket(family, SOCK_STREAM, /*blocking=*/true);
  if ABSL_PREDICT_FALSE (fd.value() < 0) {
    return nullptr;
  } else {
    LOG(INFO) << okMsg("created", fd);
    return absl::WrapUnique(new TcpSocket(fd, family, /*connected=*/false));
  }
}

std::unique_ptr<TcpSocket> TcpSocket::Create(fd_t fd, int family) {
  return absl::WrapUnique(new TcpSocket(fd, family, /*connected=*/true));
}

TcpSocket::~TcpSocket() {
  DCHECK(invariant());
  if (connected_) Shutdown();
  DCHECK(!connected_);
  LOG(INFO) << okMsg("closing");
  DCHECK(invariant());
  ::close(fd_.value());
  fd_ = fd_t(-1);
}

void TcpSocket::Shutdown() {
  DCHECK(invariant());
  LOG(INFO) << okMsg("shutdown");
  ::shutdown(fd_.value(), SHUT_RDWR);
  connected_ = false;
  DCHECK(invariant());
}

bool TcpSocket::Listen(const Endpoint& local) const {
  DCHECK(invariant());

  int on = 1;
  if ABSL_PREDICT_FALSE (!SetOption(fd_, SO_REUSEADDR, &on, sizeof(on))) {
    const auto last_errno = errno;
    LOG(WARNING) << errMsg("set SO_REUSEADDR", last_errno);
    return false;
  }
  if ABSL_PREDICT_FALSE (SocketBase::Bind(fd_, local) < 0) {
    const auto last_errno = errno;
    LOG(WARNING) << errMsg("bind", last_errno);
    return false;
  } else if (ABSL_PREDICT_FALSE(::listen(fd_.value(), SOMAXCONN) < 0)) {
    const auto last_errno = errno;
    LOG(WARNING) << errMsg("listen", last_errno);
    return false;
  } else {
    LOG(INFO) << okMsg("listening");
    return true;
  }
}

namespace {
int AcceptConn(const int family, const fd_t fd) {
  if (family == AF_INET) {
    struct sockaddr_in sa;
    socklen_t len = sizeof(sa);
    return ::accept4(fd.value(), (struct sockaddr*)&sa, &len, SOCK_CLOEXEC);
  } else {
    DCHECK_EQ(family, AF_INET6);
    struct sockaddr_in6 sa;
    socklen_t len = sizeof(sa);
    return ::accept4(fd.value(), (struct sockaddr*)&sa, &len, SOCK_CLOEXEC);
  }
}
}  // namespace

fd_t TcpSocket::Accept() const {
  DCHECK(invariant());
  DCHECK(IsBlocking());

  const int ret = AcceptConn(family_, fd_);
  if ABSL_PREDICT_FALSE (ret < 0) {
    // At this point, shutdown() is the only reason that can cause EINVAL.
    if (const auto last_errno = errno; last_errno == EINVAL) {
      LOG(WARNING) << okMsg("accept shutdown");
      DCHECK(IsShutdown(-2));
      return fd_t(-2);
    } else {
      LOG(WARNING) << errMsg("accept", last_errno);
      return fd_t(-1);
    }
  } else {
    const fd_t new_fd(ret);
    DCHECK_GE(new_fd.value(), 0);
    LOG(INFO) << okMsg("accepted", new_fd);
    return new_fd;
  }
}

bool TcpSocket::Connect(const Endpoint& peer) {
  DCHECK(invariant());
  DCHECK(IsBlocking());

  if ABSL_PREDICT_FALSE (SocketBase::Connect(fd_, peer) < 0) {
    const auto last_errno = errno;
    LOG(WARNING) << errMsg("connect", last_errno);
    return false;
  } else {
    LOG(INFO) << okMsg("connected");
    connected_ = true;
    return true;
  }
}

ssize_t TcpSocket::Send(const Byte* const buf, const size_t len) const {
  DCHECK(invariant());
  DCHECK_GE(len, 1);
  DCHECK_LE(len, std::numeric_limits<ssize_t>::max());
  DCHECK(IsBlocking());

  const Byte* ptr = buf;
  size_t sent = 0;
  ssize_t left = len;
  while (left > 0) {
    const ssize_t bytes = ::send(fd_.value(), ptr, left, /*flags=*/0);
    if ABSL_PREDICT_TRUE (bytes > 0) {
      DCHECK_LE(bytes, left);
      ptr += bytes;
      left -= bytes;
      sent += bytes;
      DCHECK_EQ(buf + len, ptr + left);
      VLOG(1) << ioMsg("send", bytes);
    } else {
      const auto last_errno = errno;
      if ABSL_PREDICT_TRUE (bytes < 0) {
        if (Interrupted(last_errno)) continue;
        DCHECK(!WouldBlock(last_errno));
        LOG(WARNING) << errMsg("send", last_errno);
        return -1;
      } else {  // rarely happens
        DCHECK_EQ(bytes, 0);
        LOG(WARNING) << errMsg("send zero", last_errno);
        return 0;
      }
    }
  }
  DCHECK_EQ(left, 0);
  DCHECK_EQ(sent, len);
  return sent;
}

ssize_t TcpSocket::Recv(Byte* const buf, const size_t len) const {
  DCHECK(invariant());
  DCHECK_GE(len, 1);
  DCHECK_LE(len, std::numeric_limits<ssize_t>::max());
  DCHECK(IsBlocking());

  Byte* ptr = buf;
  size_t rcvd = 0;
  ssize_t left = len;
  while (left > 0) {
    const ssize_t bytes = ::recv(fd_.value(), ptr, left, /*flags=*/0);
    if ABSL_PREDICT_TRUE (bytes > 0) {
      DCHECK_LE(bytes, left);
      ptr += bytes;
      left -= bytes;
      rcvd += bytes;
      DCHECK_EQ(buf + len, ptr + left);
      VLOG(1) << ioMsg("recv", bytes);
    } else if (bytes == 0) {  // peer closed connection
      LOG(INFO) << ioMsg("recv EoF", 0);
      return 0;
    } else {
      const auto last_errno = errno;
      if (Interrupted(last_errno)) continue;
      DCHECK(!WouldBlock(last_errno));
      LOG(WARNING) << errMsg("recv", last_errno);
      return -1;
    }
  }
  DCHECK_EQ(left, 0);
  DCHECK_EQ(rcvd, len);
  return rcvd;
}

namespace {
inline std::string ErrMsg(std::string_view what, fd_t fd, int last_errno) {
  return absl::StrFormat("tcp socket %s failed: fd=%d %s errno=%d (%s)", what,
                         fd.value(), AddrPortPair(fd), last_errno,
                         std::strerror(last_errno));
}
}  // namespace

/*static*/ absl::Status TcpSocket::Send(const fd_t fd, const Byte* const buf,
                                        const size_t len) {
  DCHECK_GE(len, 1);
  DCHECK_LE(len, std::numeric_limits<ssize_t>::max());
  DCHECK(IsValidSocket(fd));
  DCHECK(IsBlockingMode(fd));

  const Byte* ptr = buf;
  size_t sent = 0;
  ssize_t left = len;
  while (left > 0) {
    const ssize_t bytes = ::send(fd.value(), ptr, left, /*flags=*/0);
    if ABSL_PREDICT_TRUE (bytes > 0) {
      DCHECK_LE(bytes, left);
      ptr += bytes;
      left -= bytes;
      sent += bytes;
      DCHECK_EQ(buf + len, ptr + left);
    } else {
      if ABSL_PREDICT_TRUE (bytes < 0) {
        const auto last_errno = errno;
        if (Interrupted(last_errno)) continue;
        DCHECK(!WouldBlock(last_errno));
        return absl::InternalError(ErrMsg("send", fd, last_errno));
      } else {  // rarely happens
        DCHECK_EQ(bytes, 0);
        return absl::InternalError("send zero");
      }
    }
  }
  DCHECK_EQ(left, 0);
  DCHECK_EQ(sent, len);
  return absl::OkStatus();
}

/*static*/ absl::Status TcpSocket::Recv(const fd_t fd, Byte* const buf,
                                        const size_t len) {
  DCHECK_GE(len, 1);
  DCHECK_LE(len, std::numeric_limits<ssize_t>::max());
  DCHECK(IsValidSocket(fd));
  DCHECK(IsBlockingMode(fd));

  Byte* ptr = buf;
  size_t rcvd = 0;
  ssize_t left = len;
  while (left > 0) {
    const ssize_t bytes = ::recv(fd.value(), ptr, left, /*flags=*/0);
    if ABSL_PREDICT_TRUE (bytes > 0) {
      DCHECK_LE(bytes, left);
      ptr += bytes;
      left -= bytes;
      rcvd += bytes;
      DCHECK_EQ(buf + len, ptr + left);
    } else if (bytes == 0) {  // peer closed connection
      return absl::InternalError("recv EoF");
    } else {
      const auto last_errno = errno;
      if (Interrupted(last_errno)) continue;
      DCHECK(!WouldBlock(last_errno));
      return absl::InternalError(ErrMsg("recv", fd, last_errno));
    }
  }
  DCHECK_EQ(left, 0);
  DCHECK_EQ(rcvd, len);
  return absl::OkStatus();
}

std::string TcpSocket::ToString() const {
  return absl::StrCat("tcp socket: ", AddrPortPair(fd_));
}

}  // namespace peregrine::internal
