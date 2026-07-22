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

#ifndef THIRD_PARTY_PEREGRINE_SRC_API_SOCKET_UTIL_H_
#define THIRD_PARTY_PEREGRINE_SRC_API_SOCKET_UTIL_H_

#include <sys/uio.h>

#include <cstddef>

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tpu_raiden/transport/peregrine/src/api/transport_types.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/types.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_tcp.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_util.h"

namespace peregrine {

// Writes exactly `len` bytes of data from the `buf` to the socket `fd`.
// Returns OK if all the bytes are sent successfully, error otherwise.
// Precondition: the caller must ensure the input parameters are valid.
inline absl::Status WriteExact(int fd, const void* buf, size_t len) {
  DCHECK(internal::IsValidSocket(internal::fd_t(fd)));
  const Byte* const buffer = static_cast<const Byte*>(buf);
  return internal::TcpSocket::Send(internal::fd_t(fd), buffer, len);
}

// Writes all the bytes from the `iovs` to the socket `fd`.
// Returns OK if all the bytes are sent successfully, error otherwise.
// Precondition: the caller must ensure the input parameters are valid.
inline absl::Status WriteVExact(int fd, absl::Span<const struct iovec> iovs) {
  DCHECK(internal::IsValidSocket(internal::fd_t(fd)));
  if ABSL_PREDICT_FALSE (iovs.size() > IOV_MAX) {
    return absl::InvalidArgumentError(
        absl::StrCat("#iovs=", iovs.size(), " > IOV_MAX=", IOV_MAX));
  }
  return internal::TcpSocket::SendV(internal::fd_t(fd), iovs);
}

// Reads exactly `len` bytes of data from the socket `fd` into the `buf`.
// Returns OK if all the bytes are received successfully, error otherwise.
// Precondition: the caller must ensure the input parameters are valid.
inline absl::Status ReadExact(int fd, void* buf, size_t len) {
  DCHECK(internal::IsValidSocket(internal::fd_t(fd)));
  Byte* const buffer = static_cast<Byte*>(buf);
  return internal::TcpSocket::Recv(internal::fd_t(fd), buffer, len);
}

// Reads from the socket `fd` into the `iovs`.
// Returns OK if all the bytes are received successfully, error otherwise.
// Precondition: the caller must ensure the input parameters are valid.
inline absl::Status ReadVExact(int fd, absl::Span<const struct iovec> iovs) {
  DCHECK(internal::IsValidSocket(internal::fd_t(fd)));
  if ABSL_PREDICT_FALSE (iovs.size() > IOV_MAX) {
    return absl::InvalidArgumentError(
        absl::StrCat("#iovs=", iovs.size(), " > IOV_MAX=", IOV_MAX));
  }
  return internal::TcpSocket::RecvV(internal::fd_t(fd), iovs);
}

}  // namespace peregrine

#endif  // THIRD_PARTY_PEREGRINE_SRC_API_SOCKET_UTIL_H_
