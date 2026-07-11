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

#include "tpu_raiden/transport/socket_util.h"

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
#include <cstring>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace tpu_raiden::transport {

absl::Status WriteExact(int fd, const void* buffer, size_t length) {
  struct iovec iov = {.iov_base = const_cast<void*>(buffer), .iov_len = length};
  return WriteVExact(fd, {&iov, 1});
}

absl::Status ReadExact(int fd, void* buffer, size_t length) {
  struct iovec iov = {.iov_base = buffer, .iov_len = length};
  return ReadVExact(fd, {&iov, 1});
}

absl::Status WriteVExact(int fd, absl::Span<const struct iovec> iov) {
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
          CHECK(false) << "unreachable";
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

absl::Status ReadVExact(int fd, absl::Span<const struct iovec> iov) {
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
          CHECK(false) << "unreachable";
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

}  // namespace tpu_raiden::transport
