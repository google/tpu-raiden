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

/* Copyright 2026 The TPU Raiden Authors. All Rights Reserved.
==============================================================================*/

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_FRAMEWORKS_JAX_MOCK_NANOBIND_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_FRAMEWORKS_JAX_MOCK_NANOBIND_H_

#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace nanobind {

template <typename T, typename Variant>
struct is_in_variant;

template <typename T, typename... Types>
struct is_in_variant<T, std::variant<Types...>>
    : std::bool_constant<(std::is_same_v<T, Types> || ...)> {};

class list;
class object;

// Simple representation of a Python object in our C++ mock
class object {
 public:
  object() : value_(std::monostate{}) {}

  template <typename T, typename = std::enable_if_t<
                            !std::is_base_of_v<object, std::decay_t<T>>>>
  object(T&& val) : value_(std::forward<T>(val)) {}

  object(const object&) = default;
  object(object&&) = default;
  object& operator=(const object&) = default;
  object& operator=(object&&) = default;

  // Attributes
  void set_attr(const std::string& name, object val) {
    attrs_[name] = std::move(val);
  }

  object attr(const std::string& name) const {
    auto it = attrs_.find(name);
    if (it == attrs_.end()) {
      throw std::runtime_error("Attribute not found: " + name);
    }
    return it->second;
  }

  // Indexing (operator[])
  object operator[](size_t index) const {
    if (elements_.empty()) {
      throw std::runtime_error("Object is not indexable or empty");
    }
    if (index >= elements_.size()) {
      throw std::out_of_range("Index out of range");
    }
    return elements_[index];
  }

  void add_element(object elem) { elements_.push_back(std::move(elem)); }

  size_t size() const { return elements_.size(); }

  // Callable support
  object operator()() const {
    if (callable_) {
      return callable_();
    }
    return *this;
  }

  void set_callable(std::function<object()> callable) {
    callable_ = std::move(callable);
  }

  // Pointer retrieval (ptr())
  void* ptr() const {
    if (std::holds_alternative<void*>(value_)) {
      return std::get<void*>(value_);
    }
    return const_cast<object*>(this);
  }

  // Cast support
  template <typename T>
  T cast() const;

 private:
  std::variant<std::monostate, int, size_t, int64_t, void*, std::string> value_;
  std::unordered_map<std::string, object> attrs_;
  std::vector<object> elements_;
  std::function<object()> callable_;
};

// Inherit list from object to allow list to be passed as object
class list : public object {
 public:
  list() = default;

  // Allow construction with elements
  list(std::vector<object> elements) {
    for (auto& elem : elements) {
      add_element(std::move(elem));
    }
  }
};

template <typename T>
inline T object::cast() const {
  using DecayedT = std::decay_t<T>;
  using RawT = std::remove_reference_t<T>;

  // 1. Handle list specifically
  if constexpr (std::is_same_v<DecayedT, list>) {
    list l;
    *static_cast<object*>(&l) = *this;
    return l;
  }

  // 2. Handle reference or pointer types (assume stored as void*)
  if constexpr (std::is_pointer_v<RawT> || std::is_reference_v<T>) {
    if (std::holds_alternative<void*>(value_)) {
      void* ptr = std::get<void*>(value_);
      if constexpr (std::is_reference_v<T>) {
        return *reinterpret_cast<RawT*>(ptr);
      } else {
        return reinterpret_cast<T>(ptr);
      }
    }
    throw std::runtime_error(
        "Cannot cast non-pointer mock object to pointer/reference");
  }

  // 3. Handle types in variant
  if constexpr (is_in_variant<DecayedT, decltype(value_)>::value) {
    if (std::holds_alternative<DecayedT>(value_)) {
      return std::get<DecayedT>(value_);
    }
  }

  // 4. Handle implicit conversions for numeric types
  if constexpr (std::is_same_v<DecayedT, size_t>) {
    if (std::holds_alternative<int>(value_)) {
      return static_cast<size_t>(std::get<int>(value_));
    }
    if (std::holds_alternative<int64_t>(value_)) {
      return static_cast<size_t>(std::get<int64_t>(value_));
    }
  }
  if constexpr (std::is_same_v<DecayedT, int64_t>) {
    if (std::holds_alternative<int>(value_)) {
      return static_cast<int64_t>(std::get<int>(value_));
    }
    if (std::holds_alternative<size_t>(value_)) {
      return static_cast<int64_t>(std::get<size_t>(value_));
    }
  }

  throw std::runtime_error("Invalid cast");
}

// Gil scoped release stub
class gil_scoped_release {
 public:
  gil_scoped_release() = default;
  ~gil_scoped_release() = default;
};

// Stub functions
inline size_t len(const object& obj) { return obj.size(); }

template <typename T>
inline T cast(const object& obj) {
  return obj.cast<T>();
}

template <typename T>
inline bool isinstance(const object& obj) {
  if constexpr (std::is_same_v<T, list>) {
    return true;  // Assume it is if we are casting it
  }
  return false;
}

// Stub tags
struct arg {
  explicit arg(const char*) {}
};
struct kw_only {};

}  // namespace nanobind

namespace nb = nanobind;

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_FRAMEWORKS_JAX_MOCK_NANOBIND_H_
