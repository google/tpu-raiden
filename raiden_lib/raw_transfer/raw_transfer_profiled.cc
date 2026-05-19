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

#include "base/profiler.h"
#include "perftools/profiles/proto/builder.h"
#include "perftools/profiles/proto/encoder.h"
#include "raiden_lib/raw_transfer/raw_transfer_impl.h"

namespace raiden {

inline absl::StatusOr<PjRtCopyFuture> transfer_d2h_batch_async(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list()) {
  if (!ProfilerStartCollecting(nullptr)) {
    LOG(ERROR) << "Profiler could not be started";
  }

  PjRtCopyFuture acc =
      xla::ValueOrThrow(transfer_d2h_batch_async_impl(
          src_arrs, dst_arrs, src_offsets_major_dim, dst_offsets_major_dim,
          copy_sizes_major_dim));

  auto profile = ProfilerStopCollecting();
  if (profile) {
    auto cpu_profile = perftools::profiles::MakeCpuProfile(std::move(profile));
    if (cpu_profile.ok()) {
      const char* out_dir = getenv("TEST_UNDECLARED_OUTPUTS_DIR");
      std::string profile_path;
      if (out_dir) {
        profile_path = std::string(out_dir) + "/transfer_d2h_batch_async.prof";
      } else {
        profile_path = "/tmp/transfer_d2h_batch_async.prof";
      }
      if (!perftools::profiles::Builder::MarshalToFile(*cpu_profile.value(),
                                                       profile_path.c_str())) {
        LOG(ERROR) << "Could not write profile to file";
      } else {
        LOG(INFO) << "Profile saved to " << profile_path;
      }
    } else {
      LOG(ERROR) << "Conversion failed: " << cpu_profile.status().message();
    }
  } else {
    LOG(ERROR) << "Profiling data could not be collected";
  }

  return acc;
}

inline absl::StatusOr<PjRtCopyFuture> transfer_h2d_batch_async(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list()) {
  if (!ProfilerStartCollecting(nullptr)) {
    LOG(ERROR) << "Profiler could not be started";
  }

  PjRtCopyFuture acc =
      xla::ValueOrThrow(transfer_h2d_batch_async_impl(
          src_arrs, dst_arrs, src_offsets_major_dim, dst_offsets_major_dim,
          copy_sizes_major_dim));

  auto profile = ProfilerStopCollecting();
  if (profile) {
    auto cpu_profile = perftools::profiles::MakeCpuProfile(std::move(profile));
    if (cpu_profile.ok()) {
      const char* out_dir = getenv("TEST_UNDECLARED_OUTPUTS_DIR");
      std::string profile_path;
      if (out_dir) {
        profile_path = std::string(out_dir) + "/transfer_h2d_batch_async.prof";
      } else {
        profile_path = "/tmp/transfer_h2d_batch_async.prof";
      }
      if (!perftools::profiles::Builder::MarshalToFile(*cpu_profile.value(),
                                                       profile_path.c_str())) {
        LOG(ERROR) << "Could not write profile to file";
      } else {
        LOG(INFO) << "Profile saved to " << profile_path;
      }
    } else {
      LOG(ERROR) << "Conversion failed: " << cpu_profile.status().message();
    }
  } else {
    LOG(ERROR) << "Profiling data could not be collected";
  }

  return acc;
}

inline void transfer_d2h_batch(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list()) {
  auto future = xla::ValueOrThrow(
      transfer_d2h_batch_async(src_arrs, dst_arrs, src_offsets_major_dim,
                               dst_offsets_major_dim, copy_sizes_major_dim));
  nb::gil_scoped_release release;
  future.Await();
}

inline void transfer_h2d_batch(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list()) {
  auto future = xla::ValueOrThrow(
      transfer_h2d_batch_async(src_arrs, dst_arrs, src_offsets_major_dim,
                               dst_offsets_major_dim, copy_sizes_major_dim));
  nb::gil_scoped_release release;
  future.Await();
}

NB_MODULE(raw_transfer_profiled, m) {
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
        nb::arg("copy_sizes_major_dim") = nb::list());
  m.def("transfer_h2d_async", xla::ValueOrThrowWrapper(transfer_h2d_async),
        nb::arg("src_arr"), nb::arg("dst_arr"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list());
  m.def("transfer_d2h", &transfer_d2h, nb::arg("src_arr"), nb::arg("dst_arr"),
        nb::kw_only(), nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list());
  m.def("transfer_h2d", &transfer_h2d, nb::arg("src_arr"), nb::arg("dst_arr"),
        nb::kw_only(), nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list());

  m.def("transfer_d2h_batch_async_naive",
        &transfer_d2h_batch_async_naive, nb::arg("src_arrs"),
        nb::arg("dst_arrs"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list());
  m.def("transfer_h2d_batch_async_naive",
        &transfer_h2d_batch_async_naive, nb::arg("src_arrs"),
        nb::arg("dst_arrs"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list());

  m.def("transfer_d2h_batch_async",
        xla::ValueOrThrowWrapper(transfer_d2h_batch_async), nb::arg("src_arrs"),
        nb::arg("dst_arrs"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list());
  m.def("transfer_h2d_batch_async",
        xla::ValueOrThrowWrapper(transfer_h2d_batch_async), nb::arg("src_arrs"),
        nb::arg("dst_arrs"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list());
  m.def("transfer_d2h_batch", &transfer_d2h_batch, nb::arg("src_arrs"),
        nb::arg("dst_arrs"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list());
  m.def("transfer_h2d_batch", &transfer_h2d_batch, nb::arg("src_arrs"),
        nb::arg("dst_arrs"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list());
}

}  // namespace raiden
