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
#include "c10/core/Allocator.h"
#include "c10/core/Device.h"
#include "pybind11/gil.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xla/pjrt/pjrt_client.h"
#include "core/raw_transfer_core.h"
#include "torch/extension.h"  // IWYU pragma: keep
#include "torch/headeronly/core/DeviceType.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/materialize.h"
#include "torch_tpu/eager/tensor_to_buffer.h"

namespace py = pybind11;

namespace raiden {
namespace {

using TensorList = std::vector<at::Tensor>;

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

    data_ = torch_tpu::GetTpuPinnedAllocator()->allocate(size_bytes_);
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

void ValidatePartialSpec(const std::vector<int64_t>& src_offsets_major_dim,
                         const std::vector<int64_t>& dst_offsets_major_dim,
                         const std::vector<int64_t>& copy_sizes_major_dim) {
  bool present = !src_offsets_major_dim.empty() ||
                 !dst_offsets_major_dim.empty() ||
                 !copy_sizes_major_dim.empty();
  if (present &&
      (src_offsets_major_dim.size() != dst_offsets_major_dim.size() ||
       src_offsets_major_dim.size() != copy_sizes_major_dim.size())) {
    throw std::invalid_argument(
        "src_offsets_major_dim, dst_offsets_major_dim, and "
        "copy_sizes_major_dim must have the same length");
  }
  for (size_t i = 0; i < src_offsets_major_dim.size(); ++i) {
    if (src_offsets_major_dim[i] < 0 || dst_offsets_major_dim[i] < 0 ||
        copy_sizes_major_dim[i] < 0) {
      throw std::invalid_argument(
          "raw copy offsets and sizes must be non-negative");
    }
  }
}

bool IsPartial(const xla::Shape& shape,
               const std::vector<int64_t>& src_offsets_major_dim,
               const std::vector<int64_t>& dst_offsets_major_dim,
               const std::vector<int64_t>& copy_sizes_major_dim) {
  if (src_offsets_major_dim.empty()) return false;
  if (shape.dimensions_size() == 0) return true;
  const int64_t full_major_dim = shape.dimensions(0);
  for (size_t i = 0; i < src_offsets_major_dim.size(); ++i) {
    if (src_offsets_major_dim[i] != 0 || dst_offsets_major_dim[i] != 0 ||
        copy_sizes_major_dim[i] != full_major_dim) {
      return true;
    }
  }
  return false;
}

void ValidateCpuTensor(const at::Tensor& tensor, const char* role) {
  if (!tensor.device().is_cpu()) {
    throw std::invalid_argument(std::string(role) + " must be a CPU tensor");
  }
  if (!tensor.is_contiguous()) {
    throw std::invalid_argument(std::string(role) + " must be contiguous");
  }
}

void ValidateTpuTensor(const at::Tensor& tensor, const char* role) {
  if (tensor.device().type() != at::DeviceType::PrivateUse1) {
    throw std::invalid_argument(std::string(role) + " must be a TPU tensor");
  }
  if (!tensor.is_contiguous()) {
    throw std::invalid_argument(std::string(role) + " must be contiguous");
  }
}

torch_tpu::DeviceBufferRef GetMaterializedBufferRef(const at::Tensor& tensor) {
  return ValueOrThrow(
      "Failed to materialize TPU tensor",
      torch_tpu::GetMaterialized(
          tensor, torch_tpu::MaterializationReason::kCpuTransfer));
}

xla::PjRtBuffer* GetPjRtBuffer(const torch_tpu::DeviceBufferRef& buffer_ref) {
  return ValueOrThrow("Failed to get PjRtBuffer", buffer_ref.AwaitBuffer());
}

void AwaitReady(xla::PjRtBuffer* buffer, const char* role) {
  (void)buffer;
  (void)role;
}

void KeepTensorAlive(const std::shared_ptr<PjRtCopyFuture>& future,
                     const at::Tensor& tensor) {
  future->AddUserHold(std::make_shared<at::Tensor>(tensor));
}

void KeepBufferRefAlive(const std::shared_ptr<PjRtCopyFuture>& future,
                        torch_tpu::DeviceBufferRef buffer_ref) {
  future->AddUserHold(
      std::make_shared<torch_tpu::DeviceBufferRef>(std::move(buffer_ref)));
}

void IssueD2HCopy(const std::shared_ptr<PjRtCopyFuture>& acc,
                  xla::PjRtBuffer* src_buffer, uint8_t* dst_data,
                  size_t dst_size,
                  const std::vector<int64_t>& src_offsets_major_dim,
                  const std::vector<int64_t>& dst_offsets_major_dim,
                  const std::vector<int64_t>& copy_sizes_major_dim) {
  const bool is_partial =
      IsPartial(src_buffer->on_device_shape(), src_offsets_major_dim,
                dst_offsets_major_dim, copy_sizes_major_dim);
  const int64_t physical_size =
      ValueOrThrow("Failed to get source physical buffer size",
                   src_buffer->GetOnDeviceSizeInBytes());
  const int64_t slice_byte_size = GetMajorSliceByteSize(src_buffer);

  if (is_partial) {
    if (src_buffer->on_device_shape().dimensions_size() < 3) {
      throw std::invalid_argument(
          "Only rank >= 3 TPU tensors support partial raw copies");
    }
    if (slice_byte_size % 4096 != 0) {
      throw std::invalid_argument(
          "Partial raw copies require a major-dimension slice size aligned to "
          "4096 bytes");
    }
  }

  BufferHoldAndAlias hold =
      ValueOrThrow("Failed to acquire source raw buffer",
                   BufferHoldAndAlias::Acquire(src_buffer));
  std::vector<xla::Future<>> futures;
  if (!is_partial) {
    if (dst_size < static_cast<size_t>(physical_size)) {
      throw std::invalid_argument("Destination CPU tensor is too small");
    }
    py::gil_scoped_release release;
    futures.push_back(hold.CopyRawDeviceToHost(dst_data, 0, physical_size));
  } else {
    for (size_t i = 0; i < src_offsets_major_dim.size(); ++i) {
      const int64_t src_offset = src_offsets_major_dim[i] * slice_byte_size;
      const int64_t dst_offset = dst_offsets_major_dim[i] * slice_byte_size;
      const int64_t size_to_copy = copy_sizes_major_dim[i] * slice_byte_size;
      if (src_offset + size_to_copy > physical_size) {
        throw std::invalid_argument("Copy range exceeds source TPU buffer");
      }
      if (dst_offset + size_to_copy > static_cast<int64_t>(dst_size)) {
        throw std::invalid_argument(
            "Copy range exceeds destination CPU tensor");
      }
      py::gil_scoped_release release;
      futures.push_back(hold.CopyRawDeviceToHost(dst_data + dst_offset,
                                                 src_offset, size_to_copy));
    }
  }
  acc->Append(std::move(futures), hold);
}

void IssueH2DCopy(const std::shared_ptr<PjRtCopyFuture>& acc,
                  const uint8_t* src_data, size_t src_size,
                  xla::PjRtBuffer* dst_buffer,
                  const std::vector<int64_t>& src_offsets_major_dim,
                  const std::vector<int64_t>& dst_offsets_major_dim,
                  const std::vector<int64_t>& copy_sizes_major_dim) {
  const bool is_partial =
      IsPartial(dst_buffer->on_device_shape(), src_offsets_major_dim,
                dst_offsets_major_dim, copy_sizes_major_dim);
  const int64_t physical_size =
      ValueOrThrow("Failed to get destination physical buffer size",
                   dst_buffer->GetOnDeviceSizeInBytes());
  const int64_t slice_byte_size = GetMajorSliceByteSize(dst_buffer);

  if (is_partial) {
    if (dst_buffer->on_device_shape().dimensions_size() < 3) {
      throw std::invalid_argument(
          "Only rank >= 3 TPU tensors support partial raw copies");
    }
    if (slice_byte_size % 4096 != 0) {
      throw std::invalid_argument(
          "Partial raw copies require a major-dimension slice size aligned to "
          "4096 bytes");
    }
  }

  BufferHoldAndAlias hold =
      ValueOrThrow("Failed to acquire destination raw buffer",
                   BufferHoldAndAlias::Acquire(dst_buffer));
  std::vector<xla::Future<>> futures;
  if (!is_partial) {
    if (src_size < static_cast<size_t>(physical_size)) {
      throw std::invalid_argument("Source CPU tensor is too small");
    }
    py::gil_scoped_release release;
    futures.push_back(hold.CopyRawHostToDevice(src_data, 0, physical_size));
  } else {
    for (size_t i = 0; i < src_offsets_major_dim.size(); ++i) {
      const int64_t src_offset = src_offsets_major_dim[i] * slice_byte_size;
      const int64_t dst_offset = dst_offsets_major_dim[i] * slice_byte_size;
      const int64_t size_to_copy = copy_sizes_major_dim[i] * slice_byte_size;
      if (src_offset + size_to_copy > static_cast<int64_t>(src_size)) {
        throw std::invalid_argument("Copy range exceeds source CPU tensor");
      }
      if (dst_offset + size_to_copy > physical_size) {
        throw std::invalid_argument(
            "Copy range exceeds destination TPU buffer");
      }
      py::gil_scoped_release release;
      futures.push_back(hold.CopyRawHostToDevice(src_data + src_offset,
                                                 dst_offset, size_to_copy));
    }
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
  ValidatePartialSpec(src_offsets_major_dim, dst_offsets_major_dim,
                      copy_sizes_major_dim);
  auto acc = std::make_shared<PjRtCopyFuture>(std::vector<xla::Future<>>{});
  for (size_t i = 0; i < src_arrs.size(); ++i) {
    ValidateTpuTensor(src_arrs[i], "Source");
    ValidateCpuTensor(dst_arrs[i], "Destination");
    torch_tpu::DeviceBufferRef src_buffer_ref =
        GetMaterializedBufferRef(src_arrs[i]);
    xla::PjRtBuffer* src_buffer = GetPjRtBuffer(src_buffer_ref);
    AwaitReady(src_buffer, "Source");
    IssueD2HCopy(acc, src_buffer,
                 reinterpret_cast<uint8_t*>(dst_arrs[i].data_ptr()),
                 dst_arrs[i].nbytes(), src_offsets_major_dim,
                 dst_offsets_major_dim, copy_sizes_major_dim);
    KeepBufferRefAlive(acc, std::move(src_buffer_ref));
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
  ValidatePartialSpec(src_offsets_major_dim, dst_offsets_major_dim,
                      copy_sizes_major_dim);
  auto acc = std::make_shared<PjRtCopyFuture>(std::vector<xla::Future<>>{});
  for (size_t i = 0; i < src_arrs.size(); ++i) {
    ValidateCpuTensor(src_arrs[i], "Source");
    ValidateTpuTensor(dst_arrs[i], "Destination");
    torch_tpu::DeviceBufferRef dst_buffer_ref =
        GetMaterializedBufferRef(dst_arrs[i]);
    xla::PjRtBuffer* dst_buffer = GetPjRtBuffer(dst_buffer_ref);
    AwaitReady(dst_buffer, "Destination");
    IssueH2DCopy(acc, reinterpret_cast<const uint8_t*>(src_arrs[i].data_ptr()),
                 src_arrs[i].nbytes(), dst_buffer, src_offsets_major_dim,
                 dst_offsets_major_dim, copy_sizes_major_dim);
    KeepBufferRefAlive(acc, std::move(dst_buffer_ref));
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
      : tpu_tensor_(tpu_tensor), host_buffer_(std::move(host_buffer)) {
    if (!host_buffer_) {
      throw std::invalid_argument("host_buffer must not be None");
    }
    ValidateTpuTensor(tpu_tensor_, "TPU tensor");
    buffer_ref_ = GetMaterializedBufferRef(tpu_tensor_);
    pjrt_buffer_ = GetPjRtBuffer(*buffer_ref_);
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
      py::gil_scoped_release release;
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
      py::gil_scoped_release release;
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
    py::gil_scoped_release release;
    future->Await();
  }

  void H2D() {
    auto future = H2DAsync();
    py::gil_scoped_release release;
    future->Await();
  }

 private:
  at::Tensor tpu_tensor_;
  std::shared_ptr<RawHostBuffer> host_buffer_;
  std::optional<torch_tpu::DeviceBufferRef> buffer_ref_;
  xla::PjRtBuffer* pjrt_buffer_ = nullptr;
  size_t physical_size_ = 0;
  BufferHoldAndAlias hold_;
};

void AwaitAll(py::object futures) {
  if (py::isinstance<PjRtCopyFuture>(futures)) {
    auto future = futures.cast<std::shared_ptr<PjRtCopyFuture>>();
    py::gil_scoped_release release;
    future->Await();
    return;
  }
  for (py::handle item : futures) {
    auto future = item.cast<std::shared_ptr<PjRtCopyFuture>>();
    py::gil_scoped_release release;
    future->Await();
  }
}

bool IsReady(py::object futures) {
  if (py::isinstance<PjRtCopyFuture>(futures)) {
    return futures.cast<std::shared_ptr<PjRtCopyFuture>>()->IsReady();
  }
  for (py::handle item : futures) {
    if (!item.cast<std::shared_ptr<PjRtCopyFuture>>()->IsReady()) {
      return false;
    }
  }
  return true;
}

}  // namespace

PYBIND11_MODULE(_torch_raw_transfer, m) {
  py::class_<RawHostBuffer, std::shared_ptr<RawHostBuffer>>(m, "RawHostBuffer")
      .def(py::init<int64_t>(), py::arg("size_bytes"))
      .def_property_readonly("size_bytes", &RawHostBuffer::SizeBytes)
      .def_property_readonly("data_ptr", &RawHostBuffer::DataPtr)
      .def_property_readonly("is_pjrt_backed", &RawHostBuffer::IsPjRtBacked);

  py::class_<PjRtCopyFuture, std::shared_ptr<PjRtCopyFuture>>(m,
                                                              "PjRtCopyFuture")
      .def("Await", &PjRtCopyFuture::Await,
           py::call_guard<py::gil_scoped_release>())
      .def("wait", &PjRtCopyFuture::Await,
           py::call_guard<py::gil_scoped_release>())
      .def("IsReady", &PjRtCopyFuture::IsReady)
      .def("is_ready", &PjRtCopyFuture::IsReady);

  py::class_<PreparedTorchRawTransfer,
             std::shared_ptr<PreparedTorchRawTransfer>>(
      m, "PreparedTorchRawTransfer")
      .def(py::init<const at::Tensor&, std::shared_ptr<RawHostBuffer>, bool>(),
           py::arg("tpu_tensor"), py::arg("host_buffer"),
           py::arg("unsafe_skip_buffer_lock") = true)
      .def_property_readonly("physical_size_bytes",
                             &PreparedTorchRawTransfer::PhysicalSizeBytes)
      .def_property_readonly("host_buffer",
                             &PreparedTorchRawTransfer::HostBuffer)
      .def("d2h_async", &PreparedTorchRawTransfer::D2HAsync)
      .def("h2d_async", &PreparedTorchRawTransfer::H2DAsync)
      .def("d2h", &PreparedTorchRawTransfer::D2H)
      .def("h2d", &PreparedTorchRawTransfer::H2D);

  m.def("await_all", &AwaitAll, py::arg("futures"));
  m.def("is_ready", &IsReady, py::arg("futures"));

  m.def("transfer_d2h_async", &TransferD2HAsync, py::arg("src_arr"),
        py::arg("dst_arr"), py::kw_only(),
        py::arg("src_offsets_major_dim") = std::vector<int64_t>{},
        py::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
        py::arg("copy_sizes_major_dim") = std::vector<int64_t>{});
  m.def("transfer_h2d_async", &TransferH2DAsync, py::arg("src_arr"),
        py::arg("dst_arr"), py::kw_only(),
        py::arg("src_offsets_major_dim") = std::vector<int64_t>{},
        py::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
        py::arg("copy_sizes_major_dim") = std::vector<int64_t>{});
  m.def(
      "transfer_d2h",
      [](const at::Tensor& src_arr, const at::Tensor& dst_arr,
         const std::vector<int64_t>& src_offsets_major_dim,
         const std::vector<int64_t>& dst_offsets_major_dim,
         const std::vector<int64_t>& copy_sizes_major_dim) {
        auto future =
            TransferD2HAsync(src_arr, dst_arr, src_offsets_major_dim,
                             dst_offsets_major_dim, copy_sizes_major_dim);
        py::gil_scoped_release release;
        future->Await();
      },
      py::arg("src_arr"), py::arg("dst_arr"), py::kw_only(),
      py::arg("src_offsets_major_dim") = std::vector<int64_t>{},
      py::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
      py::arg("copy_sizes_major_dim") = std::vector<int64_t>{});
  m.def(
      "transfer_h2d",
      [](const at::Tensor& src_arr, const at::Tensor& dst_arr,
         const std::vector<int64_t>& src_offsets_major_dim,
         const std::vector<int64_t>& dst_offsets_major_dim,
         const std::vector<int64_t>& copy_sizes_major_dim) {
        auto future =
            TransferH2DAsync(src_arr, dst_arr, src_offsets_major_dim,
                             dst_offsets_major_dim, copy_sizes_major_dim);
        py::gil_scoped_release release;
        future->Await();
      },
      py::arg("src_arr"), py::arg("dst_arr"), py::kw_only(),
      py::arg("src_offsets_major_dim") = std::vector<int64_t>{},
      py::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
      py::arg("copy_sizes_major_dim") = std::vector<int64_t>{});

  m.def("transfer_d2h_batch_async", &TransferD2HBatchAsync, py::arg("src_arrs"),
        py::arg("dst_arrs"), py::kw_only(),
        py::arg("src_offsets_major_dim") = std::vector<int64_t>{},
        py::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
        py::arg("copy_sizes_major_dim") = std::vector<int64_t>{});
  m.def("transfer_h2d_batch_async", &TransferH2DBatchAsync, py::arg("src_arrs"),
        py::arg("dst_arrs"), py::kw_only(),
        py::arg("src_offsets_major_dim") = std::vector<int64_t>{},
        py::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
        py::arg("copy_sizes_major_dim") = std::vector<int64_t>{});
  m.def(
      "transfer_d2h_batch",
      [](const TensorList& src_arrs, const TensorList& dst_arrs,
         const std::vector<int64_t>& src_offsets_major_dim,
         const std::vector<int64_t>& dst_offsets_major_dim,
         const std::vector<int64_t>& copy_sizes_major_dim) {
        auto future =
            TransferD2HBatchAsync(src_arrs, dst_arrs, src_offsets_major_dim,
                                  dst_offsets_major_dim, copy_sizes_major_dim);
        py::gil_scoped_release release;
        future->Await();
      },
      py::arg("src_arrs"), py::arg("dst_arrs"), py::kw_only(),
      py::arg("src_offsets_major_dim") = std::vector<int64_t>{},
      py::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
      py::arg("copy_sizes_major_dim") = std::vector<int64_t>{});
  m.def(
      "transfer_h2d_batch",
      [](const TensorList& src_arrs, const TensorList& dst_arrs,
         const std::vector<int64_t>& src_offsets_major_dim,
         const std::vector<int64_t>& dst_offsets_major_dim,
         const std::vector<int64_t>& copy_sizes_major_dim) {
        auto future =
            TransferH2DBatchAsync(src_arrs, dst_arrs, src_offsets_major_dim,
                                  dst_offsets_major_dim, copy_sizes_major_dim);
        py::gil_scoped_release release;
        future->Await();
      },
      py::arg("src_arrs"), py::arg("dst_arrs"), py::kw_only(),
      py::arg("src_offsets_major_dim") = std::vector<int64_t>{},
      py::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
      py::arg("copy_sizes_major_dim") = std::vector<int64_t>{});
}

}  // namespace raiden
