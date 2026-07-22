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

#ifndef THIRD_PARTY_PEREGRINE_SRC_INTERNAL_BASE_STRONG_INT_H_
#define THIRD_PARTY_PEREGRINE_SRC_INTERNAL_BASE_STRONG_INT_H_

#include <concepts>
#include <cstddef>
#include <functional>
#include <ostream>

namespace peregrine::internal {

template <class Tag, std::integral T>
class StrongInt {
 public:
  using value_type = T;
  using tag_type = Tag;

  constexpr StrongInt() noexcept = default;
  constexpr explicit StrongInt(T value) noexcept : value_(value) {}

  [[nodiscard]]
  constexpr T value() const noexcept {
    return value_;
  }

  [[nodiscard]]
  constexpr explicit operator T() const noexcept {
    return value_;
  }

  friend constexpr bool operator==(StrongInt, StrongInt) = default;
  friend constexpr auto operator<=>(StrongInt, StrongInt) = default;

  constexpr StrongInt& operator+=(StrongInt rhs) noexcept {
    value_ += rhs.value_;
    return *this;
  }

  constexpr StrongInt& operator-=(StrongInt rhs) noexcept {
    value_ -= rhs.value_;
    return *this;
  }

  friend constexpr StrongInt operator+(StrongInt lhs, StrongInt rhs) noexcept {
    return lhs += rhs;
  }

  friend constexpr StrongInt operator-(StrongInt lhs, StrongInt rhs) noexcept {
    return lhs -= rhs;
  }

 private:
  T value_{};
};

template <class Tag, std::integral T>
std::ostream& operator<<(std::ostream& os, StrongInt<Tag, T> v) {
  return os << v.value();
}

#define DEFINE_STRONG_INT_TYPE(Name, UnderlyingType) \
  struct Name##Tag {};                               \
  using Name = StrongInt<Name##Tag, UnderlyingType>

}  // namespace peregrine::internal

// hash
namespace std {
template <class Tag, std::integral T>
struct hash<peregrine::internal::StrongInt<Tag, T>> {
  constexpr size_t operator()(
      peregrine::internal::StrongInt<Tag, T> v) const noexcept {
    return std::hash<T>{}(v.value());
  }
};
}  // namespace std

#endif  // THIRD_PARTY_PEREGRINE_SRC_INTERNAL_BASE_STRONG_INT_H_
