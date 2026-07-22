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

#include "tpu_raiden/transport/peregrine/src/internal/util/util.h"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>  // NOLINT

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/types.h"

namespace peregrine::internal {

bool IsValid(absl::Span<const IoVec> iovecs) {
  return std::all_of(iovecs.begin(), iovecs.end(),
                     [](const IoVec& v) { return IsValid(v); });
}

size_t TotalLength(const absl::Span<const IoVec> iovecs) {
  return std::accumulate(iovecs.begin(), iovecs.end(), size_t{0},
                         [](size_t sum, const IoVec& v) {
                           DCHECK(IsValid(v));
                           return sum + v.iov_len;
                         });
}

std::string ThreadId() {
  std::stringstream ss;
  ss << std::this_thread::get_id();
  return ss.str();
}

}  // namespace peregrine::internal
