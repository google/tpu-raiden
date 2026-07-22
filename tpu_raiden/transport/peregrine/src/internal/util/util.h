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

#ifndef THIRD_PARTY_PEREGRINE_SRC_INTERNAL_UTIL_UTIL_H_
#define THIRD_PARTY_PEREGRINE_SRC_INTERNAL_UTIL_UTIL_H_

#include <sys/uio.h>

#include <cstddef>
#include <string>
#include <type_traits>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/types.h"

namespace peregrine::internal {

// Returns true iff the integer is a power of two.
template <typename T>
constexpr bool IsPowerOfTwo(T n) {
  static_assert(std::is_integral_v<T>);
  return n > 0 && (n & (n - 1)) == 0;
}

// end of file
inline constexpr IoVec kEoF = {};
static_assert(kEoF.iov_base == nullptr && kEoF.iov_len == 0);

// Returns true iff the `IoVec` is valid.
inline bool IsValid(const IoVec& v) {
  DCHECK_GE(v.iov_len, 0);
  return v.iov_len == 0 || v.iov_base != nullptr;
}

// Returns true iff all the `iovecs` are valid.
bool IsValid(absl::Span<const IoVec> iovecs);

// Returns the total length of the `iovecs`.
size_t TotalLength(absl::Span<const IoVec> iovecs);

// Returns the total length of the `n` iovecs.
inline size_t TotalLength(const IoVec* iov, int n) {
  return TotalLength(absl::MakeSpan(iov, n));
}

// Returns the thread id where this function is called.
std::string ThreadId();

}  // namespace peregrine::internal

#endif  // THIRD_PARTY_PEREGRINE_SRC_INTERNAL_UTIL_UTIL_H_
