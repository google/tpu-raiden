// Copyright 2026 Google LLC
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

#include "raw_transfer_impl.h"

namespace jax {

inline PjRtCopyFuture transfer_d2h_batch_async(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list()) {
  return transfer_d2h_batch_async_impl(
      src_arrs, dst_arrs, src_offsets_major_dim, dst_offsets_major_dim,
      copy_sizes_major_dim);
}

inline PjRtCopyFuture transfer_h2d_batch_async(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list()) {
  return transfer_h2d_batch_async_impl(
      src_arrs, dst_arrs, src_offsets_major_dim, dst_offsets_major_dim,
      copy_sizes_major_dim);
}

inline void transfer_d2h_batch(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list()) {
  transfer_d2h_batch_async(src_arrs, dst_arrs, src_offsets_major_dim,
                           dst_offsets_major_dim, copy_sizes_major_dim)
      .Await();
}

inline void transfer_h2d_batch(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list()) {
  transfer_h2d_batch_async(src_arrs, dst_arrs, src_offsets_major_dim,
                           dst_offsets_major_dim, copy_sizes_major_dim)
      .Await();
}

void BuildRawTransferAPI(nb::module_& m) {
  nb::class_<PjRtCopyFuture>(m, "PjRtCopyFuture")
      .def("Await", &PjRtCopyFuture::Await);

  m.def("transfer_d2h", &transfer_d2h, nb::arg("src_arr"), nb::arg("dst_arr"),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list());
  m.def("transfer_h2d", &transfer_h2d, nb::arg("src_arr"), nb::arg("dst_arr"),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list());
  m.def("transfer_d2h_async", &transfer_d2h_async, nb::arg("src_arr"),
        nb::arg("dst_arr"), nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list());
  m.def("transfer_h2d_async", &transfer_h2d_async, nb::arg("src_arr"),
        nb::arg("dst_arr"), nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list());
  m.def("transfer_d2h_batch", &transfer_d2h_batch, nb::arg("src_arrs"),
        nb::arg("dst_arrs"), nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list());
  m.def("transfer_h2d_batch", &transfer_h2d_batch, nb::arg("src_arrs"),
        nb::arg("dst_arrs"), nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list());
  m.def("transfer_d2h_batch_async", &transfer_d2h_batch_async,
        nb::arg("src_arrs"), nb::arg("dst_arrs"),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list());
  m.def("transfer_h2d_batch_async", &transfer_h2d_batch_async,
        nb::arg("src_arrs"), nb::arg("dst_arrs"),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list());
  m.def("transfer_d2h_batch_async_naive", &transfer_d2h_batch_async_naive,
        nb::arg("src_arrs"), nb::arg("dst_arrs"),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list());
  m.def("transfer_h2d_batch_async_naive", &transfer_h2d_batch_async_naive,
        nb::arg("src_arrs"), nb::arg("dst_arrs"),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list());
  m.def("await_all", &await_all, nb::arg("futures"));
}

}  // namespace jax

NB_MODULE(raw_transfer, m) {
  jax::BuildRawTransferAPI(m);
}
