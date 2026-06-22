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

#include "tpu_raiden/frameworks/torch/torch_raw_transfer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/py/torch/c10/core/Device.h"
#include "third_party/py/torch/torch/headeronly/core/DeviceType.h"
#include "torch_tpu/eager/device_buffer.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tpu_raiden/core/host_memory_allocator.h"
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/core/utils.h"
#include "tpu_raiden/frameworks/torch/torch_tpu_utils.h"

namespace raiden {
namespace {

using TensorList = std::vector<at::Tensor>;

using ::tpu_raiden::torch::UnpackTorchTensor;

void DeleteHostBufferAllocation(void* ctx) {
  delete static_cast<std::shared_ptr<tpu_raiden::HostBufferAllocation>*>(ctx);
}
}  // namespace

RawHostBuffer::RawHostBuffer(int64_t size_bytes) {
  if (size_bytes < 0) {
    throw std::invalid_argument(
        "RawHostBuffer size_bytes must be non-negative");
  }
  size_bytes_ = static_cast<size_t>(size_bytes);
}

uintptr_t RawHostBuffer::DataPtr() const {
  return reinterpret_cast<uintptr_t>(data_ptr_);
}

void* RawHostBuffer::MutableData() const { return data_ptr_; }

const void* RawHostBuffer::Data() const { return data_ptr_; }

size_t RawHostBuffer::SizeBytes() const { return size_bytes_; }

bool RawHostBuffer::IsPjRtBacked() const { return pjrt_buffer_ != nullptr; }

void RawHostBuffer::EnsureBoundToDevice(xla::PjRtDevice* device) {
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
    xla::Shape shape =
        xla::ShapeUtil::MakeShape(xla::U8, {static_cast<int64_t>(size_bytes_)});
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

namespace {
[[noreturn]] void ThrowStatus(absl::string_view context,
                              const absl::Status& status) {
  throw std::runtime_error(absl::StrCat(context, ": ", status.message()));
}

template <typename T>
T ValueOrThrow(absl::string_view context, absl::StatusOr<T> value_or) {
  if (!value_or.ok()) {
    ThrowStatus(context, value_or.status());
  }
  return std::move(value_or).value();
}

void ValidateCpuTensor(const at::Tensor& tensor, absl::string_view role) {
  if (!tensor.device().is_cpu()) {
    throw std::invalid_argument(absl::StrCat(role, " must be a CPU tensor"));
  }
  if (!tensor.is_contiguous()) {
    throw std::invalid_argument(absl::StrCat(role, " must be contiguous"));
  }
}

void AwaitReady(xla::PjRtBuffer* buffer, absl::string_view role) {
  (void)buffer;
  (void)role;
}

// Raw transfer addresses the device buffer as a flat array of equal-size
// major-dimension slices ("blocks"): block i lives at byte offset
// i * GetMajorSliceByteSize(buffer). That mapping is only correct when logical
// dimension 0 is the most-major physical dimension and the buffer's physical
// size is an exact multiple of the slice size (the blocks tile it with no
// remainder). Assert both so a buffer with an unexpected on-device layout fails
// loudly here instead of silently transferring the wrong bytes.
void ValidateMajorDimLayout(xla::PjRtBuffer* buffer, absl::string_view role) {
  const xla::Shape& shape = buffer->on_device_shape();
  const int rank = shape.dimensions().size();
  if (rank < 1) {
    throw std::invalid_argument(
        absl::StrCat(role, " buffer must have rank >= 1 for block transfer"));
  }
  // In xla::Layout, minor_to_major(rank - 1) is the most-major physical dim.
  if (auto pjrt_layout = buffer->layout();
      pjrt_layout && pjrt_layout->xla_layout().minor_to_major(rank - 1) != 0) {
    throw std::invalid_argument(
        absl::StrCat(role, " buffer layout must place logical dimension 0 as the most-major "
                     "physical dimension; block offsetting assumes blocks are the "
                     "outermost, physically contiguous dimension."));
  }
  const int64_t slice = GetMajorSliceByteSize(buffer);
  const int64_t physical_size =
      ValueOrThrow(absl::StrCat(role, " physical buffer size for layout check"),
                   buffer->GetOnDeviceSizeInBytes());
  if (slice <= 0 || physical_size % slice != 0) {
    throw std::invalid_argument(
        absl::StrCat(role, " buffer physical size is not an exact multiple of its major-dimension "
                     "slice size; the block-layout assumption does not hold."));
  }
}

PjRtCopyFuture IssueD2HCopy(xla::PjRtBuffer* src_buffer, uint8_t* dst_data,
                            size_t dst_size,
                            const std::vector<int64_t>& src_offsets_major_dim,
                            const std::vector<int64_t>& dst_offsets_major_dim,
                            const std::vector<int64_t>& copy_sizes_major_dim,
                            std::shared_ptr<void> user_hold = nullptr) {
  ValidateMajorDimLayout(src_buffer, "Source");
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
    futures.push_back(hold.CopyRawDeviceToHost(
        dst_data + chunk.dst_offset, chunk.src_offset, chunk.size_bytes));
  }
  return PjRtCopyFuture(
      xla::JoinFutures(absl::MakeSpan(futures)),
      {BufferHolder{hold.c_hold, hold.common_hold, /*ext_hold=*/nullptr,
                    std::move(user_hold)}});
}

PjRtCopyFuture IssueH2DCopy(const uint8_t* src_data, size_t src_size,
                            xla::PjRtBuffer* dst_buffer,
                            const std::vector<int64_t>& src_offsets_major_dim,
                            const std::vector<int64_t>& dst_offsets_major_dim,
                            const std::vector<int64_t>& copy_sizes_major_dim,
                            std::shared_ptr<void> user_hold = nullptr) {
  ValidateMajorDimLayout(dst_buffer, "Destination");
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
    futures.push_back(hold.CopyRawHostToDevice(
        src_data + chunk.src_offset, chunk.dst_offset, chunk.size_bytes));
  }
  return PjRtCopyFuture(
      xla::JoinFutures(absl::MakeSpan(futures)),
      {BufferHolder{hold.c_hold, hold.common_hold, /*ext_hold=*/nullptr,
                    std::move(user_hold)}});
}
}  // namespace

PjRtCopyFuture TransferD2HBatchAsync(
    const TensorList& src_arrs, const TensorList& dst_arrs,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  if (src_arrs.size() != dst_arrs.size()) {
    throw std::invalid_argument("Lengths of src_arrs and dst_arrs must match");
  }
  tpu_raiden::ValidatePartialSpec(src_offsets_major_dim, dst_offsets_major_dim,
                                  copy_sizes_major_dim);
  std::vector<PjRtCopyFuture> futures;
  futures.reserve(src_arrs.size());
  for (size_t i = 0; i < src_arrs.size(); ++i) {
    ValidateCpuTensor(dst_arrs[i], "Destination");
    auto unpacked = UnpackTorchTensor(src_arrs[i]);
    xla::PjRtBuffer* src_buffer = unpacked.buffer;
    AwaitReady(src_buffer, "Source");

    auto torch_holds = std::make_shared<std::vector<at::Tensor>>();
    torch_holds->push_back(src_arrs[i]);
    torch_holds->push_back(dst_arrs[i]);

    auto fut = IssueD2HCopy(
        src_buffer, reinterpret_cast<uint8_t*>(dst_arrs[i].data_ptr()),
        dst_arrs[i].nbytes(), src_offsets_major_dim, dst_offsets_major_dim,
        copy_sizes_major_dim, std::move(torch_holds));
    // Keep the materialized (possibly view) buffer alive until the copy is
    // done.
    if (unpacked.ref) {
      fut.AddKeepAlive(std::make_shared<torch_tpu::DeviceBufferRef>(
          std::move(*unpacked.ref)));
    }
    futures.push_back(std::move(fut));
  }
  return JoinPjRtCopyFutures(absl::MakeSpan(futures));
}

PjRtCopyFuture TransferH2DBatchAsync(
    const TensorList& src_arrs, const TensorList& dst_arrs,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  if (src_arrs.size() != dst_arrs.size()) {
    throw std::invalid_argument("Lengths of src_arrs and dst_arrs must match");
  }
  tpu_raiden::ValidatePartialSpec(src_offsets_major_dim, dst_offsets_major_dim,
                                  copy_sizes_major_dim);
  std::vector<PjRtCopyFuture> futures;
  futures.reserve(src_arrs.size());
  for (size_t i = 0; i < src_arrs.size(); ++i) {
    ValidateCpuTensor(src_arrs[i], "Source");
    auto unpacked = UnpackTorchTensor(dst_arrs[i]);
    xla::PjRtBuffer* dst_buffer = unpacked.buffer;
    AwaitReady(dst_buffer, "Destination");

    auto torch_holds = std::make_shared<std::vector<at::Tensor>>();
    torch_holds->push_back(src_arrs[i]);
    torch_holds->push_back(dst_arrs[i]);

    auto fut = IssueH2DCopy(
        reinterpret_cast<const uint8_t*>(src_arrs[i].data_ptr()),
        src_arrs[i].nbytes(), dst_buffer, src_offsets_major_dim,
        dst_offsets_major_dim, copy_sizes_major_dim, std::move(torch_holds));
    // Keep the materialized (possibly view) buffer alive until the copy is
    // done.
    if (unpacked.ref) {
      fut.AddKeepAlive(std::make_shared<torch_tpu::DeviceBufferRef>(
          std::move(*unpacked.ref)));
    }
    futures.push_back(std::move(fut));
  }
  return JoinPjRtCopyFutures(absl::MakeSpan(futures));
}

PjRtCopyFuture TransferD2HAsync(
    const at::Tensor& src_arr, const at::Tensor& dst_arr,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  return TransferD2HBatchAsync({src_arr}, {dst_arr}, src_offsets_major_dim,
                               dst_offsets_major_dim, copy_sizes_major_dim);
}

PjRtCopyFuture TransferH2DAsync(
    const at::Tensor& src_arr, const at::Tensor& dst_arr,
    const std::vector<int64_t>& src_offsets_major_dim,
    const std::vector<int64_t>& dst_offsets_major_dim,
    const std::vector<int64_t>& copy_sizes_major_dim) {
  return TransferH2DBatchAsync({src_arr}, {dst_arr}, src_offsets_major_dim,
                               dst_offsets_major_dim, copy_sizes_major_dim);
}

PreparedTorchRawTransfer::PreparedTorchRawTransfer(
    const at::Tensor& tpu_tensor, std::shared_ptr<RawHostBuffer> host_buffer,
    bool unsafe_skip_buffer_lock)
    : host_buffer_(std::move(host_buffer)) {
  if (!host_buffer_) {
    throw std::invalid_argument("host_buffer must not be None");
  }
  auto unpacked = UnpackTorchTensor(tpu_tensor);
  pjrt_buffer_ = unpacked.buffer;
  buffer_ref_ = std::move(unpacked.ref);  // keep the materialized buffer alive
  host_buffer_->EnsureBoundToDevice(pjrt_buffer_->device());
  physical_size_ =
      static_cast<size_t>(ValueOrThrow("Failed to get TPU physical buffer size",
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

size_t PreparedTorchRawTransfer::PhysicalSizeBytes() const {
  return physical_size_;
}

std::shared_ptr<RawHostBuffer> PreparedTorchRawTransfer::HostBuffer() const {
  return host_buffer_;
}

PjRtCopyFuture PreparedTorchRawTransfer::D2HAsync() {
  xla::Future<> copy_future =
      hold_.CopyRawDeviceToHost(host_buffer_->MutableData(), 0, physical_size_);
  return PjRtCopyFuture(
      std::move(copy_future),
      {BufferHolder{hold_.c_hold, hold_.common_hold, /*ext_hold=*/nullptr,
                    shared_from_this()}});
}

PjRtCopyFuture PreparedTorchRawTransfer::H2DAsync() {
  xla::Future<> copy_future =
      hold_.CopyRawHostToDevice(host_buffer_->Data(), 0, physical_size_);
  return PjRtCopyFuture(
      std::move(copy_future),
      {BufferHolder{hold_.c_hold, hold_.common_hold, /*ext_hold=*/nullptr,
                    shared_from_this()}});
}

void PreparedTorchRawTransfer::D2H() {
  PjRtCopyFuture future = D2HAsync();
  absl::Status status = future.Await();
  if (!status.ok()) {
    ThrowStatus("D2H copy failed", status);
  }
}

void PreparedTorchRawTransfer::H2D() {
  PjRtCopyFuture future = H2DAsync();
  absl::Status status = future.Await();
  if (!status.ok()) {
    ThrowStatus("H2D copy failed", status);
  }
}

}  // namespace raiden
