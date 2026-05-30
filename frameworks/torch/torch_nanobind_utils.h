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

#ifndef THIRD_PARTY_TPU_RAIDEN_FRAMEWORKS_TORCH_TORCH_NANOBIND_UTILS_H_
#define THIRD_PARTY_TPU_RAIDEN_FRAMEWORKS_TORCH_TORCH_NANOBIND_UTILS_H_

#include "ATen/core/TensorBody.h"
#include "nanobind/nanobind.h"
#include "torch/csrc/autograd/python_variable.h"

namespace nanobind::detail {

template <>
struct type_caster<at::Tensor> {
 public:
  NB_TYPE_CASTER(at::Tensor, const_name("torch.Tensor"))

  bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
    (void)flags;
    (void)cleanup;
    if (!THPVariable_Check(src.ptr())) {
      return false;
    }
    value = THPVariable_Unpack(src.ptr());
    return true;
  }

  static handle from_cpp(const at::Tensor& src, rv_policy policy,
                         cleanup_list* cleanup) noexcept {
    (void)policy;
    (void)cleanup;
    return handle(THPVariable_Wrap(src));
  }
};

}  // namespace nanobind::detail

#endif  // THIRD_PARTY_TPU_RAIDEN_FRAMEWORKS_TORCH_TORCH_NANOBIND_UTILS_H_
