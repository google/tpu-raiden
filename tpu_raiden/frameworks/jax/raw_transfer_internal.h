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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_FRAMEWORKS_JAX_RAW_TRANSFER_INTERNAL_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_FRAMEWORKS_JAX_RAW_TRANSFER_INTERNAL_H_

#include "absl/status/statusor.h"
#include "core/raw_transfer_impl.h"  // for PjRtCopyFuture
#ifndef WITHOUT_PYTHON
#include <nanobind/nanobind.h>
#else
#include "tpu_raiden/frameworks/jax/mock_nanobind.h"
#endif

namespace nb = nanobind;

namespace raiden {

absl::StatusOr<PjRtCopyFuture> transfer_d2h_async(
    const nb::object& src_arr, const nb::object& dst_arr,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list(),
    bool unsafe_skip_buffer_lock = false);

absl::StatusOr<PjRtCopyFuture> transfer_h2d_async(
    const nb::object& src_arr, const nb::object& dst_arr,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list(),
    bool unsafe_skip_buffer_lock = false);

absl::StatusOr<PjRtCopyFuture> transfer_d2h_batch_async(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list(),
    bool unsafe_skip_buffer_lock = false);

absl::StatusOr<PjRtCopyFuture> transfer_h2d_batch_async(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list(),
    bool unsafe_skip_buffer_lock = false);

PjRtCopyFuture transfer_d2h_batch_async_naive(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list(),
    bool unsafe_skip_buffer_lock = false);

PjRtCopyFuture transfer_h2d_batch_async_naive(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list(),
    bool unsafe_skip_buffer_lock = false);

void transfer_d2h(const nb::object& src_arr, const nb::object& dst_arr,
                  const nb::list& src_offsets_major_dim = nb::list(),
                  const nb::list& dst_offsets_major_dim = nb::list(),
                  const nb::list& copy_sizes_major_dim = nb::list(),
                  bool unsafe_skip_buffer_lock = false);
void transfer_h2d(const nb::object& src_arr, const nb::object& dst_arr,
                  const nb::list& src_offsets_major_dim = nb::list(),
                  const nb::list& dst_offsets_major_dim = nb::list(),
                  const nb::list& copy_sizes_major_dim = nb::list(),
                  bool unsafe_skip_buffer_lock = false);
void transfer_d2h_batch(const nb::list& src_arrs, const nb::list& dst_arrs,
                        const nb::list& src_offsets_major_dim = nb::list(),
                        const nb::list& dst_offsets_major_dim = nb::list(),
                        const nb::list& copy_sizes_major_dim = nb::list(),
                        bool unsafe_skip_buffer_lock = false);
void transfer_h2d_batch(const nb::list& src_arrs, const nb::list& dst_arrs,
                        const nb::list& src_offsets_major_dim = nb::list(),
                        const nb::list& dst_offsets_major_dim = nb::list(),
                        const nb::list& copy_sizes_major_dim = nb::list(),
                        bool unsafe_skip_buffer_lock = false);
void await_all(const nb::object& future_obj);
bool is_ready(const nb::object& future_obj);

}  // namespace raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_FRAMEWORKS_JAX_RAW_TRANSFER_INTERNAL_H_
