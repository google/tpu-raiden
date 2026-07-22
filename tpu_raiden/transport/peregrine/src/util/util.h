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

#ifndef THIRD_PARTY_PEREGRINE_SRC_UTIL_UTIL_H_
#define THIRD_PARTY_PEREGRINE_SRC_UTIL_UTIL_H_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "absl/log/check.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include "absl/types/span.h"
#include "third_party/xxhash/xxhash.h"

namespace peregrine::util {

// `Byte` is an 8-bit unit of data.
using Byte = uint8_t;

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

// Generates a random boolean value.
inline bool Toss(absl::BitGenRef bitgen) {
  return Random<uint8_t>(bitgen, 0, 1) == 0;
}

// Generates random non-zero bytes.
void RandomNonZero(absl::Span<Byte> data);

// Generates random non-zero bytes.
void RandomNonZero(absl::BitGenRef bitgen, absl::Span<Byte> data);

// Calculates the XXH3 hash for the data.
inline uint64_t Xx3Hash(absl::Span<const Byte> data) {
  return XXH3_64bits(data.data(), data.size());
}

// Finds an unused port in the range [10,000, 65,535], inclusively. Returns
// the port number if successful, or 0 otherwise.
// Note: there is no guarantee that the found port is still available when
// the caller actually uses it.
uint16_t FindFreePort(int family, bool tcp);

}  // namespace peregrine::util

#endif  // THIRD_PARTY_PEREGRINE_SRC_UTIL_UTIL_H_
