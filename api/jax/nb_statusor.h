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

#ifndef THIRD_PARTY_TPU_RAIDEN_API_JAX_NB_STATUS_H_
#define THIRD_PARTY_TPU_RAIDEN_API_JAX_NB_STATUS_H_

#include <Python.h>

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include <nanobind/nanobind.h>

namespace nanobind {
namespace detail {

// A Nanobind type caster for absl::Status. This type caster allows
// Nanobind to automatically convert a non-OK return status to a Python
// exception, or return None if the status is OK.
template <>
struct type_caster<absl::Status> {
  NB_TYPE_CASTER(absl::Status, const_name("Status"))

  bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
    value = absl::OkStatus();
    return true;
  }

  static handle from_cpp(absl::Status&& value, rv_policy policy,
                         cleanup_list* cleanup) noexcept {
    if (!value.ok()) {
      PyErr_Format(PyExc_RuntimeError, "absl::Status not ok: %s",
                   value.ToString().c_str());
      return nullptr;
    }
    return none().release();
  }
};

// A Nanobind type caster for absl::StatusOr<T>. This type caster allows
// Nanobind to automatically convert a non-OK return status to a Python
// exception, or return the value if the status is OK.
template <typename T>
struct type_caster<absl::StatusOr<T>> {
  using ValueCaster = make_caster<T>;

  NB_TYPE_CASTER(absl::StatusOr<T>,
                 const_name("StatusOr[") + ValueCaster::Name + const_name("]"));

  bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
    ValueCaster caster;
    if (!caster.from_python(src, flags, cleanup)) {
      return false;
    }
    value = absl::StatusOr<T>(std::move(caster).operator cast_t<T>());
    return true;
  }

  template <typename U>
  static handle from_cpp(U&& status_or, rv_policy policy,
                         cleanup_list* cleanup) noexcept {
    if (!status_or.ok()) {
      PyErr_Format(PyExc_RuntimeError, "absl::StatusOr not ok: %s",
                   status_or.status().ToString().c_str());
      return nullptr;
    }
    return ValueCaster::from_cpp(forward_like_<U>(*status_or), policy, cleanup);
  }
};

}  // namespace detail
}  // namespace nanobind

#endif  // THIRD_PARTY_TPU_RAIDEN_API_JAX_NB_STATUS_H_
