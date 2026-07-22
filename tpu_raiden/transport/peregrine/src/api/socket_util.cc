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

#include "tpu_raiden/transport/peregrine/src/api/socket_util.h"

#include <sys/uio.h>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"

namespace peregrine {

absl::Status WriteVExact(int fd, absl::Span<const struct iovec> iovs) {
  for (const auto& iov : iovs) {
    const absl::Status s = WriteExact(fd, iov.iov_base, iov.iov_len);
    if (!s.ok()) return s;
  }
  return absl::OkStatus();
}

absl::Status ReadVExact(int fd, absl::Span<const struct iovec> iovs) {
  for (const auto& iov : iovs) {
    const absl::Status s = ReadExact(fd, iov.iov_base, iov.iov_len);
    if (!s.ok()) return s;
  }
  return absl::OkStatus();
}

}  // namespace peregrine
