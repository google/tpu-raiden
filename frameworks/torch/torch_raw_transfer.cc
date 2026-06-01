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

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/shared_ptr.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"
#include "xla/pjrt/pjrt_client.h"
#include "core/host_memory_allocator.h"
#include "core/raw_transfer_core.h"
#include "core/utils.h"
#include "frameworks/torch/torch_nanobind_utils.h"
#include "frameworks/torch/torch_tpu_utils.h"

namespace nb = nanobind;

namespace raiden {
namespace {

using TensorList = std::vector<at::Tensor>;

using ::tpu_raiden::torch::UnpackTorchTensor;

void DeleteHostBufferAllocation(void* ctx) {
  delete static_cast<std::shared_ptr<tpu_raiden::HostBufferAllocation>*>(ctx);
}

class RawHostBuffer {
 public:
  explicit RawHostBuffer(int64_t size_bytes) {
    if (size_bytes < 0) {
      throw std::invalid_argument(
          "RawHostBuffer size_bytes must be non-negative");
    }
    size_bytes_ = static_cast<size_t>(size_bytes);
  }

  uintptr_t DataPtr() const { return reinterpret_cast<uintptr_t>(data_ptr_); }

  void* MutableData() const { return data_ptr_; }

  const void* Data() const { return data_ptr_; }

  size_t SizeBytes() const { return size_bytes_; }

  bool IsPjRtBacked() const { return pjrt_buffer_ != nullptr; }

  void EnsureBoundToDevice(xla::PjRtDevice* device) {
    if (data_ptr_ != nullptr || size_bytes_ == 0) {
      return;
    }
    if (device == nullptr) {
      throw std::invalid_argument("Cannot bind RawHostBuffer to null device");
    }
    xla::PjRtMemorySpace* pinned_host = nullptr;
    auto memory_or = device->memory_space_by_kind("pinned_host");
    if (memory_or.ok()) {
      pinned_host = memory_or.value();
    } else {
      for (xla::PjRtMemorySpace* memory : device->memory_spaces()) {
        std::string kind(memory->kind());
        if (kind == "pinned_host" || kind == "PINNED_HOST") {
          pinned_host = memory;
          break;
        }
      }
    }

    if (pinned_host != nullptr) {
      xla::Shape shape = xla::ShapeUtil::MakeShape(
          xla::U8, {static_cast<int64_t>(size_bytes_)});
      auto buffer_or =
          device->client()->CreateUninitializedBuffer(shape, pinned_host);
      if (buffer_or.ok()) {
        pjrt_buffer_ = std::move(buffer_or.value());
        auto ptr_or =
            pjrt_buffer_->client()->UnsafeBufferPointer(pjrt_buffer_.get());
        if (!ptr_or.ok()) {
          throw std::runtime_error(
              std::string("Failed to get pinned host buffer pointer: ") +
              std::string(ptr_or.status().message()));
        }
        data_ptr_ = reinterpret_cast<void*>(ptr_or.value());
        return;
      }
    }

    auto allocator_or = tpu_raiden::HostMemoryAllocator::Create(device->client());
    if (!allocator_or.ok()) {
      throw std::runtime_error("Failed to create TPU pinned host allocator: " +
                               allocator_or.status().ToString());
    }
    auto allocator = std::move(allocator_or).value();
    auto status_or_alloc = allocator->Allocate(size_bytes_);
    if (!status_or_alloc.ok()) {
      throw std::runtime_error("Failed to allocate TPU pinned host buffer: " +
                               status_or_alloc.status().ToString());
    }
    auto alloc = std::move(status_or_alloc).value();
    auto* ctx = new std::shared_ptr<tpu_raiden::HostBufferAllocation>(
        std::make_shared<tpu_raiden::HostBufferAllocation>(std::move(alloc)));
    data_ = c10::DataPtr((*ctx)->ptr, ctx, &DeleteHostBufferAllocation,
                         c10::Device(c10::DeviceType::CPU));
    if (data_.get() == nullptr) {
      throw std::runtime_error("Failed to allocate TPU pinned host buffer");
    }
    data_ptr_ = data_.get();
  }

 private:
  size_t size_bytes_ = 0;
  void* data_ptr_ = nullptr;
  std::unique_ptr<xla::PjRtBuffer> pjrt_buffer_;
  c10::DataPtr data_;
};

[[noreturn]] void ThrowStatus(const std::string& context,
                              const absl::Status& status) {
  throw std::runtime_error(context + ": " + std::string(status.message()));
}

template <typename T>
T ValueOrThrow(const std::string& context, absl::StatusOr<T> value_or) {
  if (!value_or.ok()) {
    ThrowStatus(context, value_or.status());
  }
  return std::move(value_or).value();
}

void ValidateCpuTensor(const at::Tensor& tensor, const char* role) {
  if (!tensor.device().is_cpu()) {
    throw std::invalid_argument(std::string(role) + " must be a CPU tensor");
  }
  if (!tensor.is_contiguous()) {
    throw std::invalid_argument(std::string(role) + " must be contiguous");
  }
}

void AwaitReady(xla::PjRtBuffer* buffer, const char* role) {
  (void)buffer;
  (void)role;
}

void KeepTensorAlive(const std::shared_ptr<PjRtCopyFuture>& future,
                     const at::Tensor& tensor) {
  future->AddUserHold(std::make_shared<at::Tensor>(tensor));
}

void IssueD2HCopy(const std::shared_ptr<PjRtCopyFuture>& acc,
                  xla::PjRtBuffer* src_buffer, uint8_t* dst_data,
                  size_t dst_size,
                  const std::vector<int64_t>& src_offsets_major_dim,
                  const std::vector<int64_t>& dst_offsets_major_dim,
                  const std::vector<int64_t>& copy_sizes_major_dim) {
  const bool is_partial = tpu_raiden::IsPartialCopy(
      src_buffer->on_device_shape(), src_offsets_major_dim,
      dst_offsets_major_dim, copy_sizes_major_dim);
  const int64_t physical_size =
      ValueOrThrow("Failed to get source physical buffer size",
                   src_buffer->GetOnDeviceSizeInBytes());
  const int64_t slice_byte_size = GetMajorSliceByteSize(src_buffer);

  if (is_partial) {
    tpu_raiden::ValidatePartialAlignment(src_buffer->on_device_shape(),
                                         slice_byte_size);
  }

  std::vector<tpu_raiden::RawCopyChunk> chunks =
      tpu_raiden::ComputeAndValidateChunks(
          slice_byte_size, physical_size, dst_size, is_partial,
          src_offsets_major_dim, dst_offsets_major_dim, copy_sizes_major_dim,
          /*is_d2h=*/true);

  BufferHoldAndAlias hold =
      ValueOrThrow("Failed to acquire source raw buffer",
                   BufferHoldAndAlias::Acquire(src_buffer));
  std::vector<xla::Future<>> futures;
  futures.reserve(chunks.size());
  for (const auto& chunk : chunks) {
    nb::gil_scoped_release release;
    futures.push_back(hold.CopyRawDeviceToHost(
        dst_data + chunk.dst_offset, chunk.src_offset, chunk.size_bytes));
  }
  acc->Append(std::move(futures), hold);
}

void IssueH2DCopy(const std::shared_ptr<PjRtCopyFuture>& acc,
                  const uint8_t* src_data, size_t src_size,
                  xla::PjRtBuffer* dst_buffer,
                  const std::vector<int64_t>& src_offsets_major_dim,
                  const std::vector<int64_t>& dst_offsets_major_dim,
                  const std::vector<int64_t>& copy_sizes_major_dim) {
  const bool is_partial = tpu_raiden::IsPartialCopy(
      dst_buffer->on_device_shape(), src_offsets_major_dim,
      dst_offsets_major_dim, copy_sizes_major_dim);
  const int64_t physical_size =
      ValueOrThrow("Failed to get destination physical buffer size",
                   dst_buffer->GetOnDeviceSizeInBytes());
  const int64_t slice_byte_size = GetMajorSliceByteSize(dst_buffer);

  if (is_partial) {
    tpu_raiden::ValidatePartialAlignment(dst_buffer->on_device_shape(),
                                         slice_byte_size);
  }

  std::vector<tpu_raiden::RawCopyChunk> chunks =
      tpu_raiden::ComputeAndValidateChunks(
          slice_byte_size, physical_size, src_size, is_partial,
          src_offsets_major_dim, dst_offsets_major_dim, copy_sizes_major_dim,
          /*is_d2h=*/false);

  BufferHoldAndAlias hold =
      ValueOrThrow("Failed to acquire destination raw buffer",
                   BufferHoldAndAlias::Acquire(dst_buffer));
  std::vector<xla::Future<>> futures;
  futures.reserve(chunks.size());
  for (const auto& chunk : chunks) {
    nb::gil_scoped_release release;
    futures.push_back(hold.CopyRawHostToDevice(
        src_data + chunk.src_offset, chunk.dst_offset, chunk.size_bytes));
  }
  acc->Append(std::move(futures), hold);
}

std::shared_ptr<PjRtCopyFuture> TransferD2HBatchAsync(
    const TensorList& src_arrs, const TensorList& dst_arrs,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  if (src_arrs.size() != dst_arrs.size()) {
    throw std::invalid_argument("Lengths of src_arrs and dst_arrs must match");
  }
  tpu_raiden::ValidatePartialSpec(src_offsets_major_dim, dst_offsets_major_dim,
                                  copy_sizes_major_dim);
  auto acc = std::make_shared<PjRtCopyFuture>(std::vector<xla::Future<>>{});
  for (size_t i = 0; i < src_arrs.size(); ++i) {
    ValidateCpuTensor(dst_arrs[i], "Destination");
    xla::PjRtBuffer* src_buffer = UnpackTorchTensor(src_arrs[i]);
    AwaitReady(src_buffer, "Source");
    IssueD2HCopy(acc, src_buffer,
                 reinterpret_cast<uint8_t*>(dst_arrs[i].data_ptr()),
                 dst_arrs[i].nbytes(), src_offsets_major_dim,
                 dst_offsets_major_dim, copy_sizes_major_dim);
    KeepTensorAlive(acc, src_arrs[i]);
    KeepTensorAlive(acc, dst_arrs[i]);
  }
  return acc;
}

std::shared_ptr<PjRtCopyFuture> TransferH2DBatchAsync(
    const TensorList& src_arrs, const TensorList& dst_arrs,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  if (src_arrs.size() != dst_arrs.size()) {
    throw std::invalid_argument("Lengths of src_arrs and dst_arrs must match");
  }
  tpu_raiden::ValidatePartialSpec(src_offsets_major_dim, dst_offsets_major_dim,
                                  copy_sizes_major_dim);
  auto acc = std::make_shared<PjRtCopyFuture>(std::vector<xla::Future<>>{});
  for (size_t i = 0; i < src_arrs.size(); ++i) {
    ValidateCpuTensor(src_arrs[i], "Source");
    xla::PjRtBuffer* dst_buffer = UnpackTorchTensor(dst_arrs[i]);
    AwaitReady(dst_buffer, "Destination");
    IssueH2DCopy(acc, reinterpret_cast<const uint8_t*>(src_arrs[i].data_ptr()),
                 src_arrs[i].nbytes(), dst_buffer, src_offsets_major_dim,
                 dst_offsets_major_dim, copy_sizes_major_dim);
    KeepTensorAlive(acc, src_arrs[i]);
    KeepTensorAlive(acc, dst_arrs[i]);
  }
  return acc;
}

std::shared_ptr<PjRtCopyFuture> TransferD2HAsync(
    const at::Tensor& src_arr, const at::Tensor& dst_arr,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  return TransferD2HBatchAsync({src_arr}, {dst_arr}, src_offsets_major_dim,
                               dst_offsets_major_dim, copy_sizes_major_dim);
}

std::shared_ptr<PjRtCopyFuture> TransferH2DAsync(
    const at::Tensor& src_arr, const at::Tensor& dst_arr,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  return TransferH2DBatchAsync({src_arr}, {dst_arr}, src_offsets_major_dim,
                               dst_offsets_major_dim, copy_sizes_major_dim);
}

class PreparedTorchRawTransfer
    : public std::enable_shared_from_this<PreparedTorchRawTransfer> {
 public:
  PreparedTorchRawTransfer(const at::Tensor& tpu_tensor,
                           std::shared_ptr<RawHostBuffer> host_buffer,
                           bool unsafe_skip_buffer_lock)
      : host_buffer_(std::move(host_buffer)) {
    if (!host_buffer_) {
      throw std::invalid_argument("host_buffer must not be None");
    }
    pjrt_buffer_ = UnpackTorchTensor(tpu_tensor);
    host_buffer_->EnsureBoundToDevice(pjrt_buffer_->device());
    physical_size_ = static_cast<size_t>(
        ValueOrThrow("Failed to get TPU physical buffer size",
                     pjrt_buffer_->GetOnDeviceSizeInBytes()));
    if (host_buffer_->SizeBytes() < physical_size_) {
      throw std::invalid_argument(
          "RawHostBuffer is smaller than TPU physical size");
    }
    auto hold_or = BufferHoldAndAlias::Acquire(pjrt_buffer_, nullptr, nullptr,
                                               unsafe_skip_buffer_lock);
    if (!hold_or.ok()) {
      ThrowStatus("Failed to acquire cached raw buffer", hold_or.status());
    }
    hold_ = std::move(hold_or.value());
  }

  size_t PhysicalSizeBytes() const { return physical_size_; }

  std::shared_ptr<RawHostBuffer> HostBuffer() const { return host_buffer_; }

  std::shared_ptr<PjRtCopyFuture> D2HAsync() {
    auto future =
        std::make_shared<PjRtCopyFuture>(std::vector<xla::Future<>>{});
    xla::Future<> copy_future;
    {
      nb::gil_scoped_release release;
      copy_future = hold_.CopyRawDeviceToHost(host_buffer_->MutableData(), 0,
                                              physical_size_);
    }
    std::vector<xla::Future<>> futures;
    futures.push_back(std::move(copy_future));
    future->Append(std::move(futures));
    future->AddUserHold(shared_from_this());
    return future;
  }

  std::shared_ptr<PjRtCopyFuture> H2DAsync() {
    auto future =
        std::make_shared<PjRtCopyFuture>(std::vector<xla::Future<>>{});
    xla::Future<> copy_future;
    {
      nb::gil_scoped_release release;
      copy_future =
          hold_.CopyRawHostToDevice(host_buffer_->Data(), 0, physical_size_);
    }
    std::vector<xla::Future<>> futures;
    futures.push_back(std::move(copy_future));
    future->Append(std::move(futures));
    future->AddUserHold(shared_from_this());
    return future;
  }

  void D2H() {
    auto future = D2HAsync();
    nb::gil_scoped_release release;
    future->Await();
  }

  void H2D() {
    auto future = H2DAsync();
    nb::gil_scoped_release release;
    future->Await();
  }

 private:
  std::shared_ptr<RawHostBuffer> host_buffer_;
  xla::PjRtBuffer* pjrt_buffer_ = nullptr;
  size_t physical_size_ = 0;
  BufferHoldAndAlias hold_;
};

void AwaitAll(nb::object futures) {
  if (nb::isinstance<PjRtCopyFuture>(futures)) {
    auto future = nb::cast<std::shared_ptr<PjRtCopyFuture>>(futures);
    nb::gil_scoped_release release;
    future->Await();
    return;
  }
  for (nb::handle item : futures) {
    auto future = nb::cast<std::shared_ptr<PjRtCopyFuture>>(item);
    nb::gil_scoped_release release;
    future->Await();
  }
}

bool IsReady(nb::object futures) {
  if (nb::isinstance<PjRtCopyFuture>(futures)) {
    return nb::cast<std::shared_ptr<PjRtCopyFuture>>(futures)->IsReady();
  }
  for (nb::handle item : futures) {
    if (!nb::cast<std::shared_ptr<PjRtCopyFuture>>(item)->IsReady()) {
      return false;
    }
  }
  return true;
}

}  // namespace

NB_MODULE(_torch_raw_transfer, m) {
  nb::class_<RawHostBuffer>(m, "RawHostBuffer")
      .def(nb::init<int64_t>(), nb::arg("size_bytes"))
      .def_prop_ro("size_bytes", &RawHostBuffer::SizeBytes)
      .def_prop_ro("data_ptr", &RawHostBuffer::DataPtr)
      .def_prop_ro("is_pjrt_backed", &RawHostBuffer::IsPjRtBacked);

  auto future_cls = nb::class_<PjRtCopyFuture>(m, "PjRtCopyFuture")
                        .def("Await", &PjRtCopyFuture::Await,
                             nb::call_guard<nb::gil_scoped_release>())
                        .def("wait", &PjRtCopyFuture::Await,
                             nb::call_guard<nb::gil_scoped_release>())
                        .def("IsReady", &PjRtCopyFuture::IsReady)
                        .def("is_ready", &PjRtCopyFuture::IsReady);
  m.attr("PjRtCopyFuture") = future_cls;

  nb::class_<PreparedTorchRawTransfer>(m, "PreparedTorchRawTransfer")
      .def(nb::new_([](const at::Tensor& tpu_tensor,
                       std::shared_ptr<RawHostBuffer> host_buffer,
                       bool unsafe_skip_buffer_lock) {
             return std::make_shared<PreparedTorchRawTransfer>(
                 tpu_tensor, host_buffer, unsafe_skip_buffer_lock);
           }),
           nb::arg("tpu_tensor"), nb::arg("host_buffer"),
           nb::arg("unsafe_skip_buffer_lock") = true)
      .def_prop_ro("physical_size_bytes",
                   &PreparedTorchRawTransfer::PhysicalSizeBytes)
      .def_prop_ro("host_buffer", &PreparedTorchRawTransfer::HostBuffer)
      .def("d2h_async", &PreparedTorchRawTransfer::D2HAsync)
      .def("h2d_async", &PreparedTorchRawTransfer::H2DAsync)
      .def("d2h", &PreparedTorchRawTransfer::D2H)
      .def("h2d", &PreparedTorchRawTransfer::H2D);

  m.def("await_all", &AwaitAll, nb::arg("futures"));
  m.def("is_ready", &IsReady, nb::arg("futures"));

  m.def("transfer_d2h_async", &TransferD2HAsync, nb::arg("src_arr"),
        nb::arg("dst_arr"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
        nb::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
        nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{});
  m.def("transfer_h2d_async", &TransferH2DAsync, nb::arg("src_arr"),
        nb::arg("dst_arr"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
        nb::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
        nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{});
  m.def(
      "transfer_d2h",
      [](const at::Tensor& src_arr, const at::Tensor& dst_arr,
         const std::vector<int64_t>& src_offsets_major_dim,
         const std::vector<int64_t>& dst_offsets_major_dim,
         const std::vector<int64_t>& copy_sizes_major_dim) {
        auto future =
            TransferD2HAsync(src_arr, dst_arr, src_offsets_major_dim,
                             dst_offsets_major_dim, copy_sizes_major_dim);
        nb::gil_scoped_release release;
        future->Await();
      },
      nb::arg("src_arr"), nb::arg("dst_arr"), nb::kw_only(),
      nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
      nb::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
      nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{});
  m.def(
      "transfer_h2d",
      [](const at::Tensor& src_arr, const at::Tensor& dst_arr,
         const std::vector<int64_t>& src_offsets_major_dim,
         const std::vector<int64_t>& dst_offsets_major_dim,
         const std::vector<int64_t>& copy_sizes_major_dim) {
        auto future =
            TransferH2DAsync(src_arr, dst_arr, src_offsets_major_dim,
                             dst_offsets_major_dim, copy_sizes_major_dim);
        nb::gil_scoped_release release;
        future->Await();
      },
      nb::arg("src_arr"), nb::arg("dst_arr"), nb::kw_only(),
      nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
      nb::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
      nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{});

  m.def("transfer_d2h_batch_async", &TransferD2HBatchAsync, nb::arg("src_arrs"),
        nb::arg("dst_arrs"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
        nb::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
        nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{});
  m.def("transfer_h2d_batch_async", &TransferH2DBatchAsync, nb::arg("src_arrs"),
        nb::arg("dst_arrs"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
        nb::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
        nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{});
  m.def(
      "transfer_d2h_batch",
      [](const TensorList& src_arrs, const TensorList& dst_arrs,
         const std::vector<int64_t>& src_offsets_major_dim,
         const std::vector<int64_t>& dst_offsets_major_dim,
         const std::vector<int64_t>& copy_sizes_major_dim) {
        auto future =
            TransferD2HBatchAsync(src_arrs, dst_arrs, src_offsets_major_dim,
                                  dst_offsets_major_dim, copy_sizes_major_dim);
        nb::gil_scoped_release release;
        future->Await();
      },
      nb::arg("src_arrs"), nb::arg("dst_arrs"), nb::kw_only(),
      nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
      nb::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
      nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{});
  m.def(
      "transfer_h2d_batch",
      [](const TensorList& src_arrs, const TensorList& dst_arrs,
         const std::vector<int64_t>& src_offsets_major_dim,
         const std::vector<int64_t>& dst_offsets_major_dim,
         const std::vector<int64_t>& copy_sizes_major_dim) {
        auto future =
            TransferH2DBatchAsync(src_arrs, dst_arrs, src_offsets_major_dim,
                                  dst_offsets_major_dim, copy_sizes_major_dim);
        nb::gil_scoped_release release;
        future->Await();
      },
      nb::arg("src_arrs"), nb::arg("dst_arrs"), nb::kw_only(),
      nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
      nb::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
      nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{});
}

}  // namespace raiden
