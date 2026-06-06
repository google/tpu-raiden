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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_STATUS_MACROS_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_STATUS_MACROS_H_

#include <utility>
#include "absl/log/log.h"
#include "absl/status/status.h"

namespace tpu_raiden {
namespace status_macro_internal {

class StatusBuilder {
 public:
  explicit StatusBuilder(const absl::Status& status) : status_(status) {}

  StatusBuilder& LogError() {
    LOG(ERROR) << status_;
    return *this;
  }

  template <typename F>
  auto With(F&& f) -> decltype(f(std::declval<absl::Status>())) {
    return f(status_);
  }

  operator absl::Status() const { return status_; }

 private:
  absl::Status status_;
};

}  // namespace status_macro_internal
}  // namespace tpu_raiden

#define RAIDEN_STATUS_MACROS_CONCAT_NAME_INNER(x, y) x##y
#define RAIDEN_STATUS_MACROS_CONCAT_NAME(x, y) \
  RAIDEN_STATUS_MACROS_CONCAT_NAME_INNER(x, y)

// Helper to get variadic macro arguments
#define RAIDEN_STATUS_INTERNAL_GET_VARIADIC(_1, _2, _3, NAME, ...) NAME

#define ASSIGN_OR_RETURN(...)                                          \
  RAIDEN_STATUS_INTERNAL_GET_VARIADIC(__VA_ARGS__, ASSIGN_OR_RETURN_3, \
                                      ASSIGN_OR_RETURN_2)(__VA_ARGS__)

#define ASSIGN_OR_RETURN_2(lhs, rexpr) ASSIGN_OR_RETURN_3(lhs, rexpr, _)

#define ASSIGN_OR_RETURN_3(lhs, rexpr, error_expr)                           \
  ASSIGN_OR_RETURN_IMPL(                                                     \
      RAIDEN_STATUS_MACROS_CONCAT_NAME(_status_or, __COUNTER__), lhs, rexpr, \
      error_expr)

#define ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr, error_expr)              \
  auto statusor = (rexpr);                                                   \
  if (!statusor.ok()) {                                                      \
    ::tpu_raiden::status_macro_internal::StatusBuilder _(statusor.status()); \
    return (error_expr);                                                     \
  }                                                                          \
  lhs = std::move(statusor).value()

#define RETURN_IF_ERROR(expr)                                                  \
  RETURN_IF_ERROR_IMPL(RAIDEN_STATUS_MACROS_CONCAT_NAME(_status, __COUNTER__), \
                       expr)

#define RETURN_IF_ERROR_IMPL(status, expr) \
  auto status = (expr);                    \
  if (!status.ok()) {                      \
    return status;                         \
  }

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_STATUS_MACROS_H_
