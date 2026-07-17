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

#ifndef THIRD_PARTY_TPU_RAIDEN_TRANSPORT_LIB_SOCKET_UTIL_H_
#define THIRD_PARTY_TPU_RAIDEN_TRANSPORT_LIB_SOCKET_UTIL_H_

#include <sys/uio.h>

#include <cstddef>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace tpu_raiden::transport::lib {

absl::Status WriteExact(int fd, const void* buffer, size_t length);
absl::Status WriteVExact(int fd, absl::Span<const struct iovec> iov);

absl::Status ReadExact(int fd, void* buffer, size_t length);
absl::Status ReadVExact(int fd, absl::Span<const struct iovec> iov);

absl::StatusOr<int> ConnectToPeer(absl::string_view peer,
                                  absl::string_view local_ip = "");

}  // namespace tpu_raiden::transport::lib

#endif  // THIRD_PARTY_TPU_RAIDEN_TRANSPORT_LIB_SOCKET_UTIL_H_
