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

#include "xla/pjrt/status_casters.h"
#include "api/jax/jax_utils.h"
#include "core/raw_transfer_impl.h"

namespace raiden {

using ::xla::PjRtBuffer;
using ::xla::PjRtCopyFuture;
using ::xla::Shape;

// Unpack and forward to pure C++ transfer_d2h_core
inline absl::StatusOr<PjRtCopyFuture> transfer_d2h_async(
    const nb::object& src_arr, const nb::object& dst_arr,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list(),
    bool unsafe_skip_buffer_lock = false) {
  std::vector<PjRtBuffer*> src_buffers =
      jax::ExtractPjRtBuffersFromPyArray(src_arr);

  nb::object dst_addressable_shards = dst_arr.attr("addressable_shards");
  size_t num_dst_shards = nb::len(dst_addressable_shards);
  std::vector<uint8_t*> dst_ptrs;
  std::vector<size_t> dst_sizes;
  dst_ptrs.reserve(num_dst_shards);
  dst_sizes.reserve(num_dst_shards);

  for (size_t i = 0; i < num_dst_shards; ++i) {
    nb::object dst_shard_data = dst_addressable_shards[i].attr("data");
    size_t dst_ptr_val =
        nb::cast<size_t>(dst_shard_data.attr("unsafe_buffer_pointer")());
    dst_ptrs.push_back(reinterpret_cast<uint8_t*>(dst_ptr_val));
    dst_sizes.push_back(
        nb::cast<size_t>(dst_shard_data.attr("on_device_size_in_bytes")()));
  }

  return transfer_d2h_core(src_buffers, dst_ptrs, dst_sizes,
                           jax::UnpackListToVector(src_offsets_major_dim),
                           jax::UnpackListToVector(dst_offsets_major_dim),
                           jax::UnpackListToVector(copy_sizes_major_dim),
                           unsafe_skip_buffer_lock);
}

// Unpack and forward to pure C++ transfer_h2d_core
inline absl::StatusOr<PjRtCopyFuture> transfer_h2d_async(
    const nb::object& src_arr, const nb::object& dst_arr,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list(),
    bool unsafe_skip_buffer_lock = false) {
  std::vector<PjRtBuffer*> dst_buffers =
      jax::ExtractPjRtBuffersFromPyArray(dst_arr);

  nb::object src_addressable_shards = src_arr.attr("addressable_shards");
  size_t num_src_shards = nb::len(src_addressable_shards);
  std::vector<const uint8_t*> src_ptrs;
  std::vector<size_t> src_sizes;
  src_ptrs.reserve(num_src_shards);
  src_sizes.reserve(num_src_shards);

  for (size_t i = 0; i < num_src_shards; ++i) {
    nb::object src_shard_data = src_addressable_shards[i].attr("data");
    size_t src_ptr_val =
        nb::cast<size_t>(src_shard_data.attr("unsafe_buffer_pointer")());
    src_ptrs.push_back(reinterpret_cast<const uint8_t*>(src_ptr_val));
    src_sizes.push_back(
        nb::cast<size_t>(src_shard_data.attr("on_device_size_in_bytes")()));
  }

  return transfer_h2d_core(dst_buffers, src_ptrs, src_sizes,
                           jax::UnpackListToVector(src_offsets_major_dim),
                           jax::UnpackListToVector(dst_offsets_major_dim),
                           jax::UnpackListToVector(copy_sizes_major_dim),
                           unsafe_skip_buffer_lock);
}

// Pure FFI JAX helper to run parallel batch transfers using pure C++ core
inline absl::StatusOr<PjRtCopyFuture> transfer_d2h_batch_async_impl(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list(),
    bool unsafe_skip_buffer_lock = false) {
  if (nb::len(src_arrs) != nb::len(dst_arrs)) {
    throw std::runtime_error("Lengths of src_arrs and dst_arrs must match");
  }
  size_t n = nb::len(src_arrs);
  PjRtCopyFuture acc({});
  if (n == 0) return acc;

  std::vector<int64_t> s_offsets =
      jax::UnpackListToVector(src_offsets_major_dim);
  std::vector<int64_t> d_offsets =
      jax::UnpackListToVector(dst_offsets_major_dim);
  std::vector<int64_t> c_sizes = jax::UnpackListToVector(copy_sizes_major_dim);

  for (size_t i = 0; i < n; ++i) {
    TF_ASSIGN_OR_RETURN(
        PjRtCopyFuture f,
        transfer_d2h_async(src_arrs[i], dst_arrs[i], src_offsets_major_dim,
                           dst_offsets_major_dim, copy_sizes_major_dim,
                           unsafe_skip_buffer_lock));
    acc.Append(std::move(f));
  }
  return acc;
}

// Pure FFI JAX helper to run parallel batch transfers using pure C++ core
inline absl::StatusOr<PjRtCopyFuture> transfer_h2d_batch_async_impl(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list(),
    bool unsafe_skip_buffer_lock = false) {
  if (nb::len(src_arrs) != nb::len(dst_arrs)) {
    throw std::runtime_error("Lengths of src_arrs and dst_arrs must match");
  }
  size_t n = nb::len(src_arrs);
  PjRtCopyFuture acc({});
  if (n == 0) return acc;

  std::vector<int64_t> s_offsets =
      jax::UnpackListToVector(src_offsets_major_dim);
  std::vector<int64_t> d_offsets =
      jax::UnpackListToVector(dst_offsets_major_dim);
  std::vector<int64_t> c_sizes = jax::UnpackListToVector(copy_sizes_major_dim);

  for (size_t i = 0; i < n; ++i) {
    TF_ASSIGN_OR_RETURN(
        PjRtCopyFuture f,
        transfer_h2d_async(src_arrs[i], dst_arrs[i], src_offsets_major_dim,
                           dst_offsets_major_dim, copy_sizes_major_dim,
                           unsafe_skip_buffer_lock));
    acc.Append(std::move(f));
  }
  return acc;
}

inline absl::StatusOr<PjRtCopyFuture> transfer_d2h_batch_async(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list(),
    bool unsafe_skip_buffer_lock = false) {
  return transfer_d2h_batch_async_impl(
      src_arrs, dst_arrs, src_offsets_major_dim, dst_offsets_major_dim,
      copy_sizes_major_dim, unsafe_skip_buffer_lock);
}

inline absl::StatusOr<PjRtCopyFuture> transfer_h2d_batch_async(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list(),
    bool unsafe_skip_buffer_lock = false) {
  return transfer_h2d_batch_async_impl(
      src_arrs, dst_arrs, src_offsets_major_dim, dst_offsets_major_dim,
      copy_sizes_major_dim, unsafe_skip_buffer_lock);
}

inline void transfer_d2h_batch(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list(),
    bool unsafe_skip_buffer_lock = false) {
  auto future = xla::ValueOrThrow(transfer_d2h_batch_async(
      src_arrs, dst_arrs, src_offsets_major_dim, dst_offsets_major_dim,
      copy_sizes_major_dim, unsafe_skip_buffer_lock));
  nb::gil_scoped_release release;
  future.Await();
}

inline void transfer_h2d_batch(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list(),
    bool unsafe_skip_buffer_lock = false) {
  auto future = xla::ValueOrThrow(transfer_h2d_batch_async(
      src_arrs, dst_arrs, src_offsets_major_dim, dst_offsets_major_dim,
      copy_sizes_major_dim, unsafe_skip_buffer_lock));
  nb::gil_scoped_release release;
  future.Await();
}

inline void transfer_d2h(const nb::object& src_arr, const nb::object& dst_arr,
                         const nb::list& src_offsets_major_dim = nb::list(),
                         const nb::list& dst_offsets_major_dim = nb::list(),
                         const nb::list& copy_sizes_major_dim = nb::list(),
                         bool unsafe_skip_buffer_lock = false) {
  auto future = xla::ValueOrThrow(transfer_d2h_async(
      src_arr, dst_arr, src_offsets_major_dim, dst_offsets_major_dim,
      copy_sizes_major_dim, unsafe_skip_buffer_lock));
  nb::gil_scoped_release release;
  future.Await();
}

inline void transfer_h2d(const nb::object& src_arr, const nb::object& dst_arr,
                         const nb::list& src_offsets_major_dim = nb::list(),
                         const nb::list& dst_offsets_major_dim = nb::list(),
                         const nb::list& copy_sizes_major_dim = nb::list(),
                         bool unsafe_skip_buffer_lock = false) {
  auto future = xla::ValueOrThrow(transfer_h2d_async(
      src_arr, dst_arr, src_offsets_major_dim, dst_offsets_major_dim,
      copy_sizes_major_dim, unsafe_skip_buffer_lock));
  nb::gil_scoped_release release;
  future.Await();
}

inline PjRtCopyFuture transfer_d2h_batch_async_naive(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list(),
    bool unsafe_skip_buffer_lock = false) {
  if (nb::len(src_arrs) != nb::len(dst_arrs)) {
    throw std::runtime_error("Lengths of src_arrs and dst_arrs must match");
  }
  size_t n = nb::len(src_arrs);
  PjRtCopyFuture acc({});
  for (size_t i = 0; i < n; ++i) {
    acc.Append(xla::ValueOrThrow(transfer_d2h_async(
        src_arrs[i], dst_arrs[i], src_offsets_major_dim, dst_offsets_major_dim,
        copy_sizes_major_dim, unsafe_skip_buffer_lock)));
  }
  return acc;
}

inline PjRtCopyFuture transfer_h2d_batch_async_naive(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list(),
    bool unsafe_skip_buffer_lock = false) {
  if (nb::len(src_arrs) != nb::len(dst_arrs)) {
    throw std::runtime_error("Lengths of src_arrs and dst_arrs must match");
  }
  size_t n = nb::len(src_arrs);
  PjRtCopyFuture acc({});
  for (size_t i = 0; i < n; ++i) {
    acc.Append(xla::ValueOrThrow(transfer_h2d_async(
        src_arrs[i], dst_arrs[i], src_offsets_major_dim, dst_offsets_major_dim,
        copy_sizes_major_dim, unsafe_skip_buffer_lock)));
  }
  return acc;
}

inline void await_all(const nb::object& future_obj) {
  if (nb::isinstance<PjRtCopyFuture>(future_obj)) {
    PjRtCopyFuture& future = nb::cast<PjRtCopyFuture&>(future_obj);
    nb::gil_scoped_release release;
    future.Await();
  } else if (nb::isinstance<nb::list>(future_obj)) {
    nb::list futures = nb::cast<nb::list>(future_obj);
    for (size_t i = 0; i < futures.size(); ++i) {
      PjRtCopyFuture& future = nb::cast<PjRtCopyFuture&>(futures[i]);
      nb::gil_scoped_release release;
      future.Await();
    }
  }
}

inline bool is_ready(const nb::object& future_obj) {
  if (nb::isinstance<PjRtCopyFuture>(future_obj)) {
    return nb::cast<const PjRtCopyFuture&>(future_obj).IsReady();
  } else if (nb::isinstance<nb::list>(future_obj)) {
    nb::list futures = nb::cast<nb::list>(future_obj);
    for (size_t i = 0; i < futures.size(); ++i) {
      if (!nb::cast<const PjRtCopyFuture&>(futures[i]).IsReady()) {
        return false;
      }
    }
    return true;
  }
  return true;
}

NB_MODULE(raw_transfer, m) {
  nb::class_<PjRtCopyFuture>(m, "PjRtCopyFuture")
      .def("Await",
           [](PjRtCopyFuture& future) {
             nb::gil_scoped_release release;
             future.Await();
           })
      .def("IsReady", &PjRtCopyFuture::IsReady);
  m.def("await_all", &await_all, nb::arg("futures"));
  m.def("is_ready", &is_ready, nb::arg("futures"));

  m.def("transfer_d2h_async", xla::ValueOrThrowWrapper(transfer_d2h_async),
        nb::arg("src_arr"), nb::arg("dst_arr"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list(),
        nb::arg("unsafe_skip_buffer_lock") = false);
  m.def("transfer_h2d_async", xla::ValueOrThrowWrapper(transfer_h2d_async),
        nb::arg("src_arr"), nb::arg("dst_arr"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list(),
        nb::arg("unsafe_skip_buffer_lock") = false);
  m.def("transfer_d2h", &transfer_d2h, nb::arg("src_arr"), nb::arg("dst_arr"),
        nb::kw_only(), nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list(),
        nb::arg("unsafe_skip_buffer_lock") = false);
  m.def("transfer_h2d", &transfer_h2d, nb::arg("src_arr"), nb::arg("dst_arr"),
        nb::kw_only(), nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list(),
        nb::arg("unsafe_skip_buffer_lock") = false);

  m.def("transfer_d2h_batch_async_naive", &transfer_d2h_batch_async_naive,
        nb::arg("src_arrs"), nb::arg("dst_arrs"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list(),
        nb::arg("unsafe_skip_buffer_lock") = false);
  m.def("transfer_h2d_batch_async_naive", &transfer_h2d_batch_async_naive,
        nb::arg("src_arrs"), nb::arg("dst_arrs"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list(),
        nb::arg("unsafe_skip_buffer_lock") = false);

  m.def("transfer_d2h_batch_async",
        xla::ValueOrThrowWrapper(transfer_d2h_batch_async), nb::arg("src_arrs"),
        nb::arg("dst_arrs"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list(),
        nb::arg("unsafe_skip_buffer_lock") = false);
  m.def("transfer_h2d_batch_async",
        xla::ValueOrThrowWrapper(transfer_h2d_batch_async), nb::arg("src_arrs"),
        nb::arg("dst_arrs"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list(),
        nb::arg("unsafe_skip_buffer_lock") = false);
  m.def("transfer_d2h_batch", &transfer_d2h_batch, nb::arg("src_arrs"),
        nb::arg("dst_arrs"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list(),
        nb::arg("unsafe_skip_buffer_lock") = false);
  m.def("transfer_h2d_batch", &transfer_h2d_batch, nb::arg("src_arrs"),
        nb::arg("dst_arrs"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list(),
        nb::arg("unsafe_skip_buffer_lock") = false);
}

}  // namespace raiden
