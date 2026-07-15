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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TRANSPORT_LIB_UTIL_UTIL_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TRANSPORT_LIB_UTIL_UTIL_H_

#include <algorithm>
#include <limits>

#include "absl/log/check.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/distributions.h"
#include "absl/types/span.h"
#include "tpu_raiden/transport/lib/base/types.h"

namespace tpu_raiden::transport::lib {

// Returns true iff all the `data` bytes are zero.
inline bool AllZero(absl::Span<const Byte> data) {
  return std::all_of(data.begin(), data.end(),
                     [](const Byte b) { return b == 0; });
}

// Generates a random integer in the range `[min, max]`, inclusively.
template <typename T>
T Random(absl::BitGenRef gen, T min = std::numeric_limits<T>::min(),
         T max = std::numeric_limits<T>::max()) {
  static_assert(std::is_integral_v<T>);
  DCHECK_LE(min, max);
  return absl::Uniform<T>(absl::IntervalClosedClosed, gen, min, max);
}

// Generates random non-zero bytes.
void RandomNonZero(absl::Span<Byte> data);

// Generates random non-zero bytes.
void RandomNonZero(absl::BitGenRef bitgen, absl::Span<Byte> data);

}  // namespace tpu_raiden::transport::lib

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TRANSPORT_LIB_UTIL_UTIL_H_
