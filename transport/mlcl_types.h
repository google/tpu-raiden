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

#ifndef THIRD_PARTY_TPU_RAIDEN_TRANSPORT_MLCL_TYPES_H_
#define THIRD_PARTY_TPU_RAIDEN_TRANSPORT_MLCL_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace mlcl {

enum class Op : uint32_t {
  kWrite = 1,
  kRead = 2,
};

enum class Status : uint32_t {
  kInProgress = 0,
  kSuccess = 1,
  kFailure = 2,
};

inline bool IsCompleted(Status status) {
  return status == Status::kSuccess || status == Status::kFailure;
}

class Endpoint {
 public:
  Endpoint() = default;
  Endpoint(const char* value) : value_(value) {}
  Endpoint(std::string value) : value_(std::move(value)) {}

  operator std::string() const { return value_; }

 private:
  std::string value_;
};

class Handle {
 public:
  Handle() = default;
  explicit Handle(uint32_t value) : value_(value) {}

  uint32_t value() const { return value_; }
  bool operator==(const Handle& other) const { return value_ == other.value_; }

  template <typename H>
  friend H AbslHashValue(H h, const Handle& handle) {
    return H::combine(std::move(h), handle.value_);
  }

 private:
  uint32_t value_ = 0;
};

struct Request {
  Op op = Op::kWrite;
  uint8_t* laddr = nullptr;
  uint8_t* raddr = nullptr;
  size_t len = 0;

  bool IsValid() const {
    return (op == Op::kWrite || op == Op::kRead) && laddr != nullptr &&
           raddr != nullptr && len > 0;
  }

  std::string ToString() const {
    return "Request{op=" + std::to_string(static_cast<uint32_t>(op)) +
           ", len=" + std::to_string(len) + "}";
  }
};

class Transport {
 public:
  virtual ~Transport() = default;
  virtual absl::StatusOr<Handle> Post(Endpoint peer,
                                      const Request& request) = 0;
  virtual absl::StatusOr<Status> Poll(Handle handle) = 0;
};

}  // namespace mlcl

#endif  // THIRD_PARTY_TPU_RAIDEN_TRANSPORT_MLCL_TYPES_H_
