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

#include "raiden_lib/raw_transfer/raw_transfer_core.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include "ATen/core/TensorBody.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "pybind11/gil.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "transport/socket_transport.h"
#include "torch/extension.h"  // IWYU pragma: keep
#include "torch/headeronly/core/DeviceType.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/materialize.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "xla/pjrt/pjrt_client.h"

namespace py = pybind11;

namespace tpu_raiden::kv_cache {
namespace {

using TensorList = std::vector<at::Tensor>;

[[noreturn]] void ThrowStatus(const std::string& context,
                              const absl::Status& status) {
  throw std::runtime_error(context + ": " + std::string(status.message()));
}

void CheckStatus(const std::string& context, const absl::Status& status) {
  if (!status.ok()) {
    ThrowStatus(context, status);
  }
}

void EmitTimingLog(const std::string& message) {
  LOG(INFO) << message;
  std::cerr << message << std::endl;
}

int EnvInt(const char* name, int default_value, int min_value, int max_value) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return default_value;
  }
  char* end = nullptr;
  const long parsed = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0') {
    LOG(WARNING) << "Ignoring invalid " << name << "=" << raw;
    return default_value;
  }
  return static_cast<int>(
      std::max<long>(min_value, std::min<long>(max_value, parsed)));
}

template <typename T>
T ValueOrThrow(const std::string& context, absl::StatusOr<T> value_or) {
  if (!value_or.ok()) {
    ThrowStatus(context, value_or.status());
  }
  return std::move(value_or).value();
}

absl::Status WriteExact(int fd, const void* buffer, size_t length) {
  const uint8_t* ptr = static_cast<const uint8_t*>(buffer);
  size_t remaining = length;
  while (remaining > 0) {
    ssize_t written = write(fd, ptr, remaining);
    if (written < 0) {
      if (errno == EINTR) continue;
      return absl::InternalError(
          "socket write failed: " + std::string(std::strerror(errno)));
    }
    if (written == 0) {
      return absl::InternalError("socket closed during write");
    }
    ptr += written;
    remaining -= written;
  }
  return absl::OkStatus();
}

absl::Status ReadExact(int fd, void* buffer, size_t length) {
  uint8_t* ptr = static_cast<uint8_t*>(buffer);
  size_t remaining = length;
  while (remaining > 0) {
    ssize_t bytes_read = read(fd, ptr, remaining);
    if (bytes_read < 0) {
      if (errno == EINTR) continue;
      return absl::InternalError(
          "socket read failed: " + std::string(std::strerror(errno)));
    }
    if (bytes_read == 0) {
      return absl::InternalError("socket closed during read");
    }
    ptr += bytes_read;
    remaining -= bytes_read;
  }
  return absl::OkStatus();
}

std::pair<std::string, int> SplitEndpoint(const std::string& endpoint) {
  const size_t colon = endpoint.rfind(':');
  if (colon == std::string::npos) {
    throw std::invalid_argument("endpoint must be host:port");
  }
  return {endpoint.substr(0, colon), std::stoi(endpoint.substr(colon + 1))};
}

int ConnectTcp(const std::string& endpoint) {
  auto [host, port] = SplitEndpoint(endpoint);
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    throw std::runtime_error("socket() failed: " + std::string(std::strerror(errno)));
  }
  int opt = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
    close(fd);
    throw std::runtime_error("invalid IPv4 endpoint host: " + host);
  }
  if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    std::string err = std::strerror(errno);
    close(fd);
    throw std::runtime_error("connect(" + endpoint + ") failed: " + err);
  }
  return fd;
}

class MaybeGilRelease {
 public:
  MaybeGilRelease() {
    if (PyGILState_Check()) {
      release_.emplace();
    }
  }

 private:
  std::optional<py::gil_scoped_release> release_;
};

void ValidatePartialSpec(const std::vector<int64_t>& src_offsets_major_dim,
                         const std::vector<int64_t>& dst_offsets_major_dim,
                         const std::vector<int64_t>& copy_sizes_major_dim) {
  const bool present = !src_offsets_major_dim.empty() ||
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
  return ValueOrThrow("Failed to get PjRtBuffer",
                      buffer_ref.GetOrMaterializeBuffer());
}

void KeepTensorAlive(const std::shared_ptr<raiden::PjRtCopyFuture>& future,
                     const at::Tensor& tensor) {
  future->AddUserHold(std::make_shared<at::Tensor>(tensor));
}

class PreparedTpuBuffer : public std::enable_shared_from_this<PreparedTpuBuffer> {
 public:
  PreparedTpuBuffer(const at::Tensor& tpu_tensor, bool unsafe_skip_buffer_lock)
      : tpu_tensor_(tpu_tensor) {
    ValidateTpuTensor(tpu_tensor_, "TPU tensor");
    buffer_ref_ = GetMaterializedBufferRef(tpu_tensor_);
    pjrt_buffer_ = GetPjRtBuffer(*buffer_ref_);
    physical_size_ = static_cast<size_t>(
        ValueOrThrow("Failed to get TPU physical buffer size",
                     pjrt_buffer_->GetOnDeviceSizeInBytes()));
    slice_byte_size_ = static_cast<size_t>(raiden::GetMajorSliceByteSize(pjrt_buffer_));
    auto hold_or = raiden::BufferHoldAndAlias::Acquire(
        pjrt_buffer_, nullptr, nullptr, unsafe_skip_buffer_lock);
    if (!hold_or.ok()) {
      ThrowStatus("Failed to acquire cached raw buffer", hold_or.status());
    }
    hold_ = std::move(hold_or.value());
  }

  std::shared_ptr<raiden::PjRtCopyFuture> D2HToAsync(
      const at::Tensor& dst_arr,
      const std::vector<int64_t>& src_offsets_major_dim,
      const std::vector<int64_t>& dst_offsets_major_dim,
      const std::vector<int64_t>& copy_sizes_major_dim) {
    auto future =
        std::make_shared<raiden::PjRtCopyFuture>(std::vector<xla::Future<>>{});
    AppendD2HTo(future, dst_arr, src_offsets_major_dim, dst_offsets_major_dim,
                copy_sizes_major_dim);
    return future;
  }

  void AppendD2HTo(
      const std::shared_ptr<raiden::PjRtCopyFuture>& future,
      const at::Tensor& dst_arr,
      const std::vector<int64_t>& src_offsets_major_dim,
      const std::vector<int64_t>& dst_offsets_major_dim,
      const std::vector<int64_t>& copy_sizes_major_dim) {
    ValidateCpuTensor(dst_arr, "Destination");
    ValidatePartialSpec(src_offsets_major_dim, dst_offsets_major_dim,
                        copy_sizes_major_dim);
    IssueD2HTo(future, reinterpret_cast<uint8_t*>(dst_arr.data_ptr()),
               dst_arr.nbytes(), src_offsets_major_dim, dst_offsets_major_dim,
               copy_sizes_major_dim);
    KeepTensorAlive(future, dst_arr);
    future->AddUserHold(shared_from_this());
  }

  std::shared_ptr<raiden::PjRtCopyFuture> H2DFromAsync(
      const at::Tensor& src_arr,
      const std::vector<int64_t>& src_offsets_major_dim,
      const std::vector<int64_t>& dst_offsets_major_dim,
      const std::vector<int64_t>& copy_sizes_major_dim) {
    auto future =
        std::make_shared<raiden::PjRtCopyFuture>(std::vector<xla::Future<>>{});
    AppendH2DFrom(future, src_arr, src_offsets_major_dim, dst_offsets_major_dim,
                  copy_sizes_major_dim);
    return future;
  }

  void AppendH2DFrom(
      const std::shared_ptr<raiden::PjRtCopyFuture>& future,
      const at::Tensor& src_arr,
      const std::vector<int64_t>& src_offsets_major_dim,
      const std::vector<int64_t>& dst_offsets_major_dim,
      const std::vector<int64_t>& copy_sizes_major_dim) {
    ValidateCpuTensor(src_arr, "Source");
    ValidatePartialSpec(src_offsets_major_dim, dst_offsets_major_dim,
                        copy_sizes_major_dim);
    IssueH2DFrom(future, reinterpret_cast<const uint8_t*>(src_arr.data_ptr()),
                 src_arr.nbytes(), src_offsets_major_dim,
                 dst_offsets_major_dim, copy_sizes_major_dim);
    KeepTensorAlive(future, src_arr);
    future->AddUserHold(shared_from_this());
  }

 private:
  bool IsPartialPrepared(
      const std::vector<int64_t>& src_offsets_major_dim,
      const std::vector<int64_t>& dst_offsets_major_dim,
      const std::vector<int64_t>& copy_sizes_major_dim) const {
    return IsPartial(pjrt_buffer_->on_device_shape(), src_offsets_major_dim,
                     dst_offsets_major_dim, copy_sizes_major_dim);
  }

  void ValidatePreparedPartial(bool is_partial) const {
    if (!is_partial) {
      return;
    }
    if (pjrt_buffer_->on_device_shape().dimensions_size() < 3) {
      throw std::invalid_argument(
          "Only rank >= 3 TPU tensors support partial raw copies");
    }
    if (slice_byte_size_ % 4096 != 0) {
      throw std::invalid_argument(
          "Partial raw copies require a major-dimension slice size aligned to "
          "4096 bytes");
    }
  }

  void IssueD2HTo(const std::shared_ptr<raiden::PjRtCopyFuture>& future,
                  uint8_t* dst_data, size_t dst_size,
                  const std::vector<int64_t>& src_offsets_major_dim,
                  const std::vector<int64_t>& dst_offsets_major_dim,
                  const std::vector<int64_t>& copy_sizes_major_dim) const {
    const bool is_partial = IsPartialPrepared(
        src_offsets_major_dim, dst_offsets_major_dim, copy_sizes_major_dim);
    ValidatePreparedPartial(is_partial);

    std::vector<xla::Future<>> futures;
    if (!is_partial) {
      if (dst_size < physical_size_) {
        throw std::invalid_argument("Destination CPU tensor is too small");
      }
      MaybeGilRelease release;
      futures.push_back(hold_.CopyRawDeviceToHost(dst_data, 0, physical_size_));
    } else {
      for (size_t i = 0; i < src_offsets_major_dim.size(); ++i) {
        const int64_t src_offset =
            src_offsets_major_dim[i] * static_cast<int64_t>(slice_byte_size_);
        const int64_t dst_offset =
            dst_offsets_major_dim[i] * static_cast<int64_t>(slice_byte_size_);
        const int64_t size_to_copy =
            copy_sizes_major_dim[i] * static_cast<int64_t>(slice_byte_size_);
        if (src_offset + size_to_copy > static_cast<int64_t>(physical_size_)) {
          throw std::invalid_argument("Copy range exceeds source TPU buffer");
        }
        if (dst_offset + size_to_copy > static_cast<int64_t>(dst_size)) {
          throw std::invalid_argument(
              "Copy range exceeds destination CPU tensor");
        }
        MaybeGilRelease release;
        futures.push_back(hold_.CopyRawDeviceToHost(dst_data + dst_offset,
                                                    src_offset, size_to_copy));
      }
    }
    future->Append(std::move(futures));
  }

  void IssueH2DFrom(const std::shared_ptr<raiden::PjRtCopyFuture>& future,
                    const uint8_t* src_data, size_t src_size,
                    const std::vector<int64_t>& src_offsets_major_dim,
                    const std::vector<int64_t>& dst_offsets_major_dim,
                    const std::vector<int64_t>& copy_sizes_major_dim) const {
    const bool is_partial = IsPartialPrepared(
        src_offsets_major_dim, dst_offsets_major_dim, copy_sizes_major_dim);
    ValidatePreparedPartial(is_partial);

    std::vector<xla::Future<>> futures;
    if (!is_partial) {
      if (src_size < physical_size_) {
        throw std::invalid_argument("Source CPU tensor is too small");
      }
      MaybeGilRelease release;
      futures.push_back(hold_.CopyRawHostToDevice(src_data, 0, physical_size_));
    } else {
      for (size_t i = 0; i < src_offsets_major_dim.size(); ++i) {
        const int64_t src_offset =
            src_offsets_major_dim[i] * static_cast<int64_t>(slice_byte_size_);
        const int64_t dst_offset =
            dst_offsets_major_dim[i] * static_cast<int64_t>(slice_byte_size_);
        const int64_t size_to_copy =
            copy_sizes_major_dim[i] * static_cast<int64_t>(slice_byte_size_);
        if (src_offset + size_to_copy > static_cast<int64_t>(src_size)) {
          throw std::invalid_argument("Copy range exceeds source CPU tensor");
        }
        if (dst_offset + size_to_copy > static_cast<int64_t>(physical_size_)) {
          throw std::invalid_argument(
              "Copy range exceeds destination TPU buffer");
        }
        MaybeGilRelease release;
        futures.push_back(hold_.CopyRawHostToDevice(src_data + src_offset,
                                                    dst_offset, size_to_copy));
      }
    }
    future->Append(std::move(futures));
  }

  at::Tensor tpu_tensor_;
  std::optional<torch_tpu::DeviceBufferRef> buffer_ref_;
  xla::PjRtBuffer* pjrt_buffer_ = nullptr;
  size_t physical_size_ = 0;
  size_t slice_byte_size_ = 0;
  raiden::BufferHoldAndAlias hold_;
};

class RaidenTransferFuture {
 public:
  explicit RaidenTransferFuture(
      std::vector<std::shared_ptr<raiden::PjRtCopyFuture>> futures = {})
      : futures_(std::move(futures)) {}

  void Add(std::shared_ptr<raiden::PjRtCopyFuture> future) {
    futures_.push_back(std::move(future));
  }

  void AddAll(const std::shared_ptr<RaidenTransferFuture>& other) {
    futures_.insert(futures_.end(), other->futures_.begin(),
                    other->futures_.end());
  }

  void Await() {
    MaybeGilRelease release;
    for (const auto& future : futures_) {
      if (future) {
        future->Await();
      }
    }
    futures_.clear();
  }

  bool IsReady() const {
    for (const auto& future : futures_) {
      if (future && !future->IsReady()) {
        return false;
      }
    }
    return true;
  }

 private:
  std::vector<std::shared_ptr<raiden::PjRtCopyFuture>> futures_;
};

struct RaidenStageResult {
  std::shared_ptr<RaidenTransferFuture> future;
  TensorList host_views;
  int64_t total_bytes = 0;
  int64_t copy_segments = 0;
};

class RaidenTransferEngine {
 public:
  RaidenTransferEngine(const TensorList& kv_caches, int64_t tp_rank,
                       int64_t local_control_port, int64_t max_blocks,
                       int64_t num_slots, double timeout_s,
                       bool unsafe_skip_buffer_lock)
      : tp_rank_(tp_rank),
        local_control_port_(static_cast<int>(local_control_port)),
        local_data_port_(static_cast<int>(local_control_port) + 1),
        max_blocks_(max_blocks),
        timeout_s_(timeout_s),
        unsafe_skip_buffer_lock_(unsafe_skip_buffer_lock) {
    h2d_issue_threads_ =
        EnvInt("RAIDEN_H2D_ISSUE_THREADS", 4, /*min_value=*/1,
               /*max_value=*/16);
    h2d_batch_max_layers_ =
        EnvInt("RAIDEN_H2D_BATCH_MAX_LAYERS", 4, /*min_value=*/1,
               /*max_value=*/256);
    if (max_blocks_ <= 0) {
      throw std::invalid_argument("max_blocks must be positive");
    }
    if (num_slots <= 0) {
      throw std::invalid_argument("num_slots must be positive");
    }
    RegisterKvCache(kv_caches);
    AllocateHostSlots(num_slots);
    transport_ = std::make_unique<tpu_raiden::transport::SocketTransport>(
        local_data_port_);
    StartControlServer();
  }

  ~RaidenTransferEngine() { StopControlServer(); }

  std::vector<int64_t> RegisterKvCache(const TensorList& kv_caches) {
    kv_caches_ = kv_caches;
    prepared_.clear();
    std::vector<int64_t> region_ids;
    region_ids.reserve(kv_caches_.size());
    for (size_t i = 0; i < kv_caches_.size(); ++i) {
      prepared_.push_back(std::make_shared<PreparedTpuBuffer>(
          kv_caches_[i], unsafe_skip_buffer_lock_));
      region_ids.push_back(static_cast<int64_t>(i));
    }
    return region_ids;
  }

  void RegisterHostBuffers(py::object /*host_pool*/, int64_t tp_rank) {
    tp_rank_ = tp_rank;
  }

  bool UsesPreparedTpuBuffers() const { return true; }

  py::tuple StageD2H(int64_t slot_idx, int64_t num_blocks,
                     const std::vector<int64_t>& block_ids) {
    RaidenStageResult result = IssueD2H(slot_idx, num_blocks, block_ids);
    return py::make_tuple(result.future, kv_caches_, result.host_views,
                          result.total_bytes);
  }

  void StageD2HSync(int64_t slot_idx, int64_t num_blocks,
                    const std::vector<int64_t>& block_ids) {
    RaidenStageResult result = IssueD2H(slot_idx, num_blocks, block_ids);
    result.future->Await();
  }

  py::tuple CommitH2D(int64_t slot_idx, int64_t num_blocks,
                      const std::vector<int64_t>& local_block_ids) {
    if (num_blocks != static_cast<int64_t>(local_block_ids.size())) {
      throw std::invalid_argument("num_blocks must match len(local_block_ids)");
    }
    auto t0 = std::chrono::steady_clock::now();
    RaidenStageResult result = IssueH2D(slot_idx, num_blocks, local_block_ids);
    auto t_issued = std::chrono::steady_clock::now();
    result.future->Await();
    auto t_done = std::chrono::steady_clock::now();
    return py::make_tuple(DurationMs(t0, t_issued),
                          DurationMs(t_issued, t_done), DurationMs(t0, t_done),
                          result.total_bytes);
  }

  py::object RankLayerViews(int64_t slot_idx, int64_t rank,
                            int64_t num_blocks) {
    if (rank != tp_rank_) {
      throw std::invalid_argument("Raiden internal slots are per-rank only");
    }
    py::list views;
    for (const auto& view : LayerViews(slot_idx, num_blocks)) {
      views.append(view);
    }
    return views;
  }

  void UnpackRankLayers(int64_t slot_idx, int64_t rank, int64_t num_blocks,
                        py::object layer_buffers) {
    if (rank != tp_rank_) {
      throw std::invalid_argument("Raiden internal slots are per-rank only");
    }
    TensorList views = LayerViews(slot_idx, num_blocks);
    size_t idx = 0;
    for (py::handle item : layer_buffers) {
      if (idx >= views.size()) {
        throw std::invalid_argument("too many layer buffers");
      }
      py::buffer source = py::reinterpret_borrow<py::buffer>(item);
      py::buffer_info info = source.request();
      if (static_cast<int64_t>(info.size * info.itemsize) != views[idx].nbytes()) {
        throw std::invalid_argument("layer buffer size mismatch");
      }
      std::memcpy(views[idx].data_ptr(), info.ptr, views[idx].nbytes());
      ++idx;
    }
    if (idx != views.size()) {
      throw std::invalid_argument("too few layer buffers");
    }
  }

  int64_t SubmitD2H(int64_t slot_idx, int64_t num_blocks,
                    const std::vector<int64_t>& block_ids) {
    PendingOperation op;
    op.future = IssueD2H(slot_idx, num_blocks, block_ids).future;
    return StorePending(std::move(op));
  }

  int64_t SubmitH2D(int64_t slot_idx, int64_t num_blocks,
                    const std::vector<int64_t>& local_block_ids) {
    PendingOperation op;
    op.future = IssueH2D(slot_idx, num_blocks, local_block_ids).future;
    return StorePending(std::move(op));
  }

  int64_t RegisterSend(const std::string& req_id, uint64_t uuid,
                       const std::vector<int64_t>& block_ids) {
    const auto register_start = std::chrono::steady_clock::now();
    if (block_ids.empty()) {
      return 0;
    }

    {
      std::lock_guard<std::mutex> lock(mu_);
      if (pending_acks_.erase(uuid) > 0) {
        done_sending_.insert(req_id);
        return 0;
      }
    }

    auto entry = std::make_shared<SendEntry>();
    entry->req_id = req_id;
    entry->uuid = uuid;
    entry->num_blocks = static_cast<int64_t>(block_ids.size());
    entry->registered_num_blocks = static_cast<int64_t>(block_ids.size());
    entry->registered_block_ids = block_ids;
    for (int64_t block_id : block_ids) {
      entry->registered_block_set.insert(block_id);
    }
    entry->deadline = DeadlineFromNow();
    entry->register_start = register_start;

    {
      std::lock_guard<std::mutex> lock(mu_);
      send_entries_[uuid] = entry;
    }

    std::ostringstream timing;
    timing << "RAIDEN_TIMING event=producer_register"
           << " req_id=" << req_id << " uuid=" << uuid
           << " rank=" << tp_rank_ << " blocks=" << block_ids.size()
           << " enqueue_ms="
           << DurationMs(register_start, std::chrono::steady_clock::now())
           << " failed=0";
    EmitTimingLog(timing.str());
    return uuid;
  }

  int64_t SubmitLoad(const std::string& req_id, uint64_t uuid,
                     const std::string& remote_endpoint,
                     const std::vector<int64_t>& remote_block_ids,
                     const std::vector<int64_t>& local_block_ids) {
    const int64_t op_id = next_op_id_++;
    const auto submit_start = std::chrono::steady_clock::now();
    CopyPlan load_plan = BuildLoadCopyPlan(remote_block_ids, local_block_ids);
    {
      std::lock_guard<std::mutex> lock(worker_threads_mu_);
      worker_threads_.emplace_back([this, req_id, uuid, remote_endpoint,
                                    submit_start,
                                    load_plan = std::move(load_plan)]() {
        const auto worker_start = std::chrono::steady_clock::now();
        bool failed = false;
        bool report_recv_done = true;
        bool release_only = false;
        int64_t slot_idx = -1;
        int64_t h2h_bytes = 0;
        int64_t h2d_segments = 0;
        size_t h2h_layers = 0;
        double slot_ms = 0.0;
        double control_setup_ms = 0.0;
        double control_ms = 0.0;
        double descriptor_ms = 0.0;
        double h2h_ms = 0.0;
        double host_reorder_ms = 0.0;
        double h2d_issue_ms = 0.0;
        double h2d_issue_wall_ms = 0.0;
        double h2d_wait_ms = 0.0;
        double h2d_total_ms = 0.0;
        double pipeline_ms = 0.0;
        double ack_ms = 0.0;
        try {
          if (load_plan.num_blocks == 0) {
            report_recv_done = false;
            release_only = true;
            const auto ack_start = std::chrono::steady_clock::now();
            AckRemote(remote_endpoint, uuid);
            ack_ms = DurationMs(ack_start, std::chrono::steady_clock::now());
          } else {
            const auto slot_start = std::chrono::steady_clock::now();
            slot_idx = AcquireSlot();
            slot_ms = DurationMs(slot_start, std::chrono::steady_clock::now());
            const auto control_start = std::chrono::steady_clock::now();
            int control_fd = ConnectTcp(remote_endpoint);
            auto control_cleanup = std::unique_ptr<int, void (*)(int*)>(
                &control_fd, [](int* p) {
                  if (p && *p >= 0) close(*p);
                });
            ControlRequestHeader stream_request;
            stream_request.magic = kControlMagic;
            stream_request.op = kOpPullStream;
            stream_request.uuid = uuid;
            stream_request.num_blocks =
                static_cast<uint64_t>(load_plan.num_blocks);
            CheckStatus("control pull stream write",
                        WriteExact(control_fd, &stream_request,
                                   sizeof(stream_request)));
            WriteBlockIds(control_fd, load_plan.producer_remote_block_ids);
            ControlResponseHeader response =
                ReadControlResponseHeader(control_fd);
            control_setup_ms =
                DurationMs(control_start, std::chrono::steady_clock::now());
            control_ms = control_setup_ms;
            const std::string data_endpoint =
                EndpointWithPort(remote_endpoint, response.data_port);
            TensorList local_views =
                LayerViews(slot_idx,
                           static_cast<int64_t>(load_plan.num_blocks));
            if (response.num_layers != local_views.size()) {
              throw std::runtime_error("remote layer descriptor count mismatch");
            }
            h2d_segments =
                static_cast<int64_t>(load_plan.h2d_copy.sizes.size());
            bool h2d_started = false;
            std::chrono::steady_clock::time_point h2d_first_issue_start;
            std::chrono::steady_clock::time_point h2d_last_issue_done;
            std::deque<size_t> h2d_ready_layers;
            std::mutex h2d_ready_mu;
            std::condition_variable h2d_ready_cv;
            bool h2d_input_done = false;
            std::exception_ptr h2d_exception;
            std::mutex h2d_state_mu;
            std::vector<std::shared_ptr<RaidenTransferFuture>>
                h2d_batch_futures;
            h2d_batch_futures.reserve(local_views.size());
            auto issue_h2d_ready_windows = [&]() {
              try {
                while (true) {
                  std::vector<size_t> ready_layers;
                  {
                    std::unique_lock<std::mutex> lock(h2d_ready_mu);
                    h2d_ready_cv.wait(lock, [&]() {
                      return h2d_input_done || !h2d_ready_layers.empty();
                    });
                    if (h2d_ready_layers.empty()) {
                      if (h2d_input_done) break;
                      continue;
                    }
                    const size_t batch_layers = std::min<size_t>(
                        h2d_ready_layers.size(),
                        static_cast<size_t>(h2d_batch_max_layers_));
                    ready_layers.reserve(batch_layers);
                    while (!h2d_ready_layers.empty() &&
                           ready_layers.size() < batch_layers) {
                      ready_layers.push_back(h2d_ready_layers.front());
                      h2d_ready_layers.pop_front();
                    }
                  }

                  H2DBatchIssueResult batch = IssueH2DBatch(
                      local_views, load_plan.h2d_copy, ready_layers);
                  {
                    std::lock_guard<std::mutex> lock(h2d_state_mu);
                    h2d_batch_futures.push_back(std::move(batch.future));
                    if (batch.issued &&
                        (!h2d_started ||
                         batch.first_issue_start < h2d_first_issue_start)) {
                      h2d_first_issue_start = batch.first_issue_start;
                      h2d_started = true;
                    }
                    if (batch.issued &&
                        (h2d_last_issue_done ==
                            std::chrono::steady_clock::time_point() ||
                         batch.last_issue_done > h2d_last_issue_done)) {
                      h2d_last_issue_done = batch.last_issue_done;
                    }
                    h2d_issue_ms += batch.issue_ms;
                  }
                }
              } catch (...) {
                std::lock_guard<std::mutex> lock(h2d_state_mu);
                if (!h2d_exception) {
                  h2d_exception = std::current_exception();
                }
              }
            };
            std::thread h2d_issue_worker(issue_h2d_ready_windows);
            std::exception_ptr pipeline_exception;
            const auto pipeline_start = std::chrono::steady_clock::now();
            try {
              for (size_t i = 0; i < local_views.size(); ++i) {
                PullLayerDescriptor layer;
                const auto descriptor_start = std::chrono::steady_clock::now();
                CheckStatus("control stream layer descriptor read",
                            ReadExact(control_fd, &layer, sizeof(layer)));
                const double layer_descriptor_ms =
                    DurationMs(descriptor_start,
                               std::chrono::steady_clock::now());
                descriptor_ms += layer_descriptor_ms;
                control_ms += layer_descriptor_ms;
                if (layer.len !=
                    static_cast<uint64_t>(local_views[i].nbytes())) {
                  throw std::runtime_error(
                      "remote layer descriptor size mismatch");
                }
                ++h2h_layers;
                h2h_bytes += static_cast<int64_t>(layer.len);
                mlcl::Request request;
                request.op = mlcl::Op::kRead;
                request.laddr =
                    reinterpret_cast<uint8_t*>(local_views[i].data_ptr());
                request.raddr = reinterpret_cast<uint8_t*>(layer.addr);
                request.len = layer.len;
                const auto h2h_start = std::chrono::steady_clock::now();
                auto handle_or = transport_->Post(data_endpoint, request);
                if (!handle_or.ok()) {
                  ThrowStatus("SocketTransport read failed",
                              handle_or.status());
                }
                auto status_or = transport_->Poll(handle_or.value());
                if (!status_or.ok() ||
                    status_or.value() != mlcl::Status::kSuccess) {
                  throw std::runtime_error(
                      "SocketTransport read did not succeed");
                }
                h2h_ms +=
                    DurationMs(h2h_start, std::chrono::steady_clock::now());
                if (load_plan.RequiresHostReorder()) {
                  const auto reorder_start =
                      std::chrono::steady_clock::now();
                  ReorderCompactBlocks(local_views[i],
                                       load_plan.host_dst_to_src);
                  host_reorder_ms += DurationMs(
                      reorder_start, std::chrono::steady_clock::now());
                }
                {
                  std::lock_guard<std::mutex> lock(h2d_ready_mu);
                  h2d_ready_layers.push_back(i);
                }
                h2d_ready_cv.notify_one();
              }
            } catch (...) {
              pipeline_exception = std::current_exception();
            }
            {
              std::lock_guard<std::mutex> lock(h2d_ready_mu);
              h2d_input_done = true;
            }
            h2d_ready_cv.notify_all();
            if (h2d_issue_worker.joinable()) {
              h2d_issue_worker.join();
            }
            if (h2d_exception) {
              std::rethrow_exception(h2d_exception);
            }
            if (pipeline_exception) {
              std::rethrow_exception(pipeline_exception);
            }
            auto h2d_future = std::make_shared<RaidenTransferFuture>();
            for (auto& batch_future : h2d_batch_futures) {
              h2d_future->AddAll(batch_future);
            }
            const auto h2d_issue_done =
                h2d_started ? h2d_last_issue_done
                            : std::chrono::steady_clock::now();
            h2d_issue_wall_ms =
                h2d_started ? DurationMs(h2d_first_issue_start, h2d_issue_done)
                            : 0.0;
            h2d_future->Await();
            const auto h2d_done = std::chrono::steady_clock::now();
            h2d_wait_ms = DurationMs(h2d_issue_done, h2d_done);
            h2d_total_ms = h2d_started
                               ? DurationMs(h2d_first_issue_start, h2d_done)
                               : 0.0;
            pipeline_ms = DurationMs(pipeline_start, h2d_done);
            const auto ack_start = std::chrono::steady_clock::now();
            AckRemote(remote_endpoint, uuid);
            ack_ms = DurationMs(ack_start, std::chrono::steady_clock::now());
          }
        } catch (const std::exception& e) {
          failed = true;
          LOG(ERROR) << "Raiden load failed for req=" << req_id
                     << " uuid=" << uuid << ": " << e.what();
        } catch (...) {
          failed = true;
          LOG(ERROR) << "Raiden load failed for req=" << req_id
                     << " uuid=" << uuid;
        }
        if (slot_idx >= 0) {
          std::lock_guard<std::mutex> lock(mu_);
          ReleaseSlotLocked(slot_idx);
        }
        {
          std::lock_guard<std::mutex> lock(mu_);
          if (report_recv_done) {
            done_recving_.insert(req_id);
            if (failed) failed_recving_.insert(req_id);
          }
        }
        const auto done = std::chrono::steady_clock::now();
        std::ostringstream timing;
        timing << "RAIDEN_TIMING event=consumer_load"
               << " req_id=" << req_id
               << " uuid=" << uuid
               << " rank=" << tp_rank_
               << " endpoint=" << remote_endpoint
               << " blocks=" << load_plan.num_blocks
               << " layers=" << h2h_layers
               << " bytes=" << h2h_bytes
               << " h2d_segments=" << h2d_segments
               << " queue_delay_ms=" << DurationMs(submit_start, worker_start)
               << " slot_ms=" << slot_ms
               << " control_setup_ms=" << control_setup_ms
               << " control_ms=" << control_ms
               << " descriptor_ms=" << descriptor_ms
               << " h2h_ms=" << h2h_ms
               << " host_reorder_ms=" << host_reorder_ms
               << " h2d_issue_ms=" << h2d_issue_ms
               << " h2d_issue_wall_ms=" << h2d_issue_wall_ms
               << " h2d_wait_ms=" << h2d_wait_ms
               << " h2d_total_ms=" << h2d_total_ms
               << " pipeline_ms=" << pipeline_ms
               << " ack_ms=" << ack_ms
               << " total_ms=" << DurationMs(submit_start, done)
               << " release_only=" << (release_only ? 1 : 0)
               << " failed=" << (failed ? 1 : 0);
        EmitTimingLog(timing.str());
      });
    }
    return op_id;
  }

  py::tuple PollFinished() {
    std::vector<std::string> done_sending;
    std::vector<std::string> done_recving;
    std::vector<std::string> failed_recving;
    {
      std::lock_guard<std::mutex> lock(mu_);
      const auto now = std::chrono::steady_clock::now();
      for (auto it = send_entries_.begin(); it != send_entries_.end();) {
        const auto& entry = it->second;
        if (entry->deadline <= now) {
          if (entry->stage_issue_started && !entry->stage_done) {
            entry->cancelled = true;
            ++it;
            continue;
          }
          done_sending_.insert(entry->req_id);
          ReleaseEntrySlotLocked(entry);
          it = send_entries_.erase(it);
        } else {
          ++it;
        }
      }
      done_sending.assign(done_sending_.begin(), done_sending_.end());
      done_recving.assign(done_recving_.begin(), done_recving_.end());
      failed_recving.assign(failed_recving_.begin(), failed_recving_.end());
      done_sending_.clear();
      done_recving_.clear();
      failed_recving_.clear();
    }
    return py::make_tuple(done_sending, done_recving, failed_recving);
  }

  std::vector<int64_t> PollTransferOps() {
    std::vector<int64_t> done;
    for (auto it = pending_.begin(); it != pending_.end();) {
      if (!it->second.future || it->second.future->IsReady()) {
        if (it->second.future) {
          it->second.future->Await();
        }
        done.push_back(it->first);
        it = pending_.erase(it);
      } else {
        ++it;
      }
    }
    return done;
  }

  void WaitTransfer(int64_t op_id) {
    auto it = pending_.find(op_id);
    if (it == pending_.end()) {
      throw std::invalid_argument("unknown Raiden transfer op id");
    }
    if (it->second.future) {
      it->second.future->Await();
    }
    pending_.erase(it);
  }

  int local_control_port() const { return local_control_port_; }
  int local_data_port() const { return local_data_port_; }

  int64_t CountCopySegmentsForTesting(
      const std::vector<int64_t>& block_ids) const {
    return static_cast<int64_t>(
        Offsets(block_ids, /*source_is_compact=*/false).sizes.size());
  }

  int64_t CountCanonicalSendCopySegmentsForTesting(
      const std::vector<int64_t>& block_ids) const {
    return static_cast<int64_t>(
        Offsets(CanonicalSendBlockIds(block_ids), /*source_is_compact=*/false)
            .sizes.size());
  }

  int64_t CountCanonicalLoadCopySegmentsForTesting(
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids) const {
    CopyPlan plan = BuildLoadCopyPlan(remote_block_ids, local_block_ids);
    return static_cast<int64_t>(plan.h2d_copy.sizes.size());
  }

  py::dict SendCopyPlanForTesting(const std::vector<int64_t>& block_ids) const {
    return CopyPlanToDict(BuildProducerCopyPlan(block_ids));
  }

  py::dict LoadCopyPlanForTesting(
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids) const {
    return CopyPlanToDict(BuildLoadCopyPlan(remote_block_ids, local_block_ids));
  }

 private:
  struct CopySpec {
    std::vector<int64_t> src_offsets;
    std::vector<int64_t> dst_offsets;
    std::vector<int64_t> sizes;
  };

  struct CopyPlan {
    int64_t num_blocks = 0;
    std::vector<int64_t> requested_remote_block_ids;
    std::vector<int64_t> requested_local_block_ids;
    std::vector<int64_t> producer_remote_block_ids;
    std::vector<int64_t> h2d_local_block_ids;
    std::vector<size_t> host_dst_to_src;
    CopySpec d2h_copy;
    CopySpec h2d_copy;

    bool RequiresHostReorder() const { return !host_dst_to_src.empty(); }
  };

  struct H2DBatchIssueResult {
    std::shared_ptr<RaidenTransferFuture> future;
    bool issued = false;
    std::chrono::steady_clock::time_point first_issue_start;
    std::chrono::steady_clock::time_point last_issue_done;
    double issue_ms = 0.0;
    size_t layers = 0;
  };

  struct PendingOperation {
    std::string remote_endpoint;
    std::shared_ptr<RaidenTransferFuture> future;
  };

  struct HostSlot {
    TensorList layers;
  };

  struct SendEntry {
    std::string req_id;
    uint64_t uuid = 0;
    int64_t slot_idx = -1;
    int64_t num_blocks = 0;
    int64_t registered_num_blocks = 0;
    int64_t total_bytes = 0;
    int64_t copy_segments = 0;
    int64_t layers_ready = 0;
    std::vector<int64_t> registered_block_ids;
    std::set<int64_t> registered_block_set;
    std::vector<bool> layer_ready;
    std::chrono::steady_clock::time_point register_start;
    std::chrono::steady_clock::time_point d2h_issue_start;
    std::chrono::steady_clock::time_point d2h_issue_done;
    std::chrono::steady_clock::time_point d2h_done;
    std::shared_ptr<RaidenTransferFuture> future;
    bool stage_issue_started = false;
    bool stage_done = false;
    bool failed = false;
    bool cancelled = false;
    bool pull_started = false;
    bool slot_released = false;
    std::chrono::steady_clock::time_point deadline;
  };

  void ReleaseEntrySlotLocked(const std::shared_ptr<SendEntry>& entry) {
    if (!entry || entry->slot_idx < 0 || entry->slot_released) return;
    ReleaseSlotLocked(entry->slot_idx);
    entry->slot_released = true;
  }

  struct PullLayerDescriptor {
    uint64_t addr = 0;
    uint64_t len = 0;
  };

  struct PullResponse {
    int data_port = 0;
    std::vector<PullLayerDescriptor> layers;
  };

  struct alignas(8) ControlRequestHeader {
    uint32_t magic = 0x52414944;  // "RAID"
    uint32_t op = 0;
    uint64_t uuid = 0;
    uint64_t num_blocks = 0;
  };

  struct alignas(8) ControlResponseHeader {
    uint32_t magic = 0x44494152;  // "DIAR"
    int32_t status = 0;
    uint32_t num_layers = 0;
    uint32_t data_port = 0;
    uint64_t message_len = 0;
  };

  static constexpr uint32_t kControlMagic = 0x52414944;
  static constexpr uint32_t kResponseMagic = 0x44494152;
  static constexpr uint32_t kOpPull = 1;
  static constexpr uint32_t kOpAck = 2;
  static constexpr uint32_t kOpPullStream = 3;

  static double DurationMs(std::chrono::steady_clock::time_point start,
                           std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
  }

  static int64_t BytesFromLayers(const std::vector<PullLayerDescriptor>& layers) {
    int64_t total = 0;
    for (const auto& layer : layers) {
      total += static_cast<int64_t>(layer.len);
    }
    return total;
  }

  static void WriteBlockIds(int fd, const std::vector<int64_t>& block_ids) {
    if (block_ids.empty()) return;
    CheckStatus("control block ids write",
                WriteExact(fd, block_ids.data(),
                           block_ids.size() * sizeof(int64_t)));
  }

  static std::vector<int64_t> ReadBlockIds(int fd, uint64_t num_blocks) {
    if (num_blocks == 0) return {};
    if (num_blocks > static_cast<uint64_t>(
                         std::numeric_limits<int64_t>::max())) {
      throw std::invalid_argument("num_blocks is too large");
    }
    std::vector<int64_t> block_ids(static_cast<size_t>(num_blocks));
    CheckStatus("control block ids read",
                ReadExact(fd, block_ids.data(),
                          block_ids.size() * sizeof(int64_t)));
    return block_ids;
  }

  static CopySpec Offsets(const std::vector<int64_t>& block_ids,
                          bool source_is_compact) {
    const int64_t n = static_cast<int64_t>(block_ids.size());
    CopySpec spec;
    spec.src_offsets.reserve(block_ids.size());
    spec.dst_offsets.reserve(block_ids.size());
    spec.sizes.reserve(block_ids.size());
    for (int64_t start = 0; start < n;) {
      int64_t end = start + 1;
      while (end < n && block_ids[end] == block_ids[end - 1] + 1) {
        ++end;
      }
      const int64_t run_size = end - start;
      if (source_is_compact) {
        spec.src_offsets.push_back(start);
        spec.dst_offsets.push_back(block_ids[start]);
      } else {
        spec.src_offsets.push_back(block_ids[start]);
        spec.dst_offsets.push_back(start);
      }
      spec.sizes.push_back(run_size);
      start = end;
    }
    return spec;
  }

  static py::dict CopySpecToDict(const CopySpec& spec) {
    py::dict out;
    out["src_offsets"] = spec.src_offsets;
    out["dst_offsets"] = spec.dst_offsets;
    out["sizes"] = spec.sizes;
    return out;
  }

  static py::dict CopyPlanToDict(const CopyPlan& plan) {
    py::dict out;
    out["num_blocks"] = plan.num_blocks;
    out["requested_remote_block_ids"] = plan.requested_remote_block_ids;
    out["requested_local_block_ids"] = plan.requested_local_block_ids;
    out["producer_remote_block_ids"] = plan.producer_remote_block_ids;
    out["h2d_local_block_ids"] = plan.h2d_local_block_ids;
    out["host_dst_to_src"] = plan.host_dst_to_src;
    out["requires_host_reorder"] = plan.RequiresHostReorder();
    out["d2h_copy"] = CopySpecToDict(plan.d2h_copy);
    out["h2d_copy"] = CopySpecToDict(plan.h2d_copy);
    return out;
  }

  static std::vector<int64_t> CanonicalSendBlockIds(
      const std::vector<int64_t>& block_ids) {
    std::vector<int64_t> ordered = block_ids;
    std::stable_sort(ordered.begin(), ordered.end());
    return ordered;
  }

  static void ValidateRequestedBlocks(
      const SendEntry& entry, const std::vector<int64_t>& requested_block_ids) {
    if (requested_block_ids.empty()) {
      throw std::invalid_argument(
          "pull stream requested no blocks; use ack-only path");
    }
    std::set<int64_t> seen;
    for (int64_t block_id : requested_block_ids) {
      if (entry.registered_block_set.find(block_id) ==
          entry.registered_block_set.end()) {
        throw std::invalid_argument(
            "pull stream requested block not registered by producer");
      }
      if (!seen.insert(block_id).second) {
        throw std::invalid_argument(
            "pull stream requested duplicate producer block id");
      }
    }
  }

  static CopyPlan BuildProducerCopyPlan(
      const std::vector<int64_t>& block_ids) {
    CopyPlan plan;
    plan.num_blocks = static_cast<int64_t>(block_ids.size());
    plan.requested_remote_block_ids = block_ids;
    plan.producer_remote_block_ids = CanonicalSendBlockIds(block_ids);
    plan.d2h_copy =
        Offsets(plan.producer_remote_block_ids, /*source_is_compact=*/false);
    return plan;
  }

  static CopyPlan BuildLoadCopyPlan(
      const std::vector<int64_t>& remote_block_ids,
      const std::vector<int64_t>& local_block_ids) {
    if (remote_block_ids.size() != local_block_ids.size()) {
      throw std::invalid_argument(
          "remote_block_ids and local_block_ids must have same length");
    }
    CopyPlan plan;
    plan.num_blocks = static_cast<int64_t>(remote_block_ids.size());
    plan.requested_remote_block_ids = remote_block_ids;
    plan.requested_local_block_ids = local_block_ids;
    if (remote_block_ids.empty()) {
      return plan;
    }
    std::vector<size_t> remote_order(remote_block_ids.size());
    std::vector<size_t> local_order(local_block_ids.size());
    for (size_t i = 0; i < remote_order.size(); ++i) {
      remote_order[i] = i;
      local_order[i] = i;
    }
    std::stable_sort(
        remote_order.begin(), remote_order.end(),
        [&](size_t a, size_t b) { return remote_block_ids[a] < remote_block_ids[b]; });
    std::stable_sort(
        local_order.begin(), local_order.end(),
        [&](size_t a, size_t b) { return local_block_ids[a] < local_block_ids[b]; });

    plan.producer_remote_block_ids.reserve(remote_order.size());
    plan.h2d_local_block_ids.reserve(local_order.size());
    plan.host_dst_to_src.reserve(local_order.size());
    std::vector<size_t> source_pos_by_original_idx(remote_order.size());
    for (size_t source_pos = 0; source_pos < remote_order.size(); ++source_pos) {
      const size_t original_idx = remote_order[source_pos];
      plan.producer_remote_block_ids.push_back(remote_block_ids[original_idx]);
      source_pos_by_original_idx[original_idx] = source_pos;
    }
    bool identity_reorder = true;
    for (size_t dst_pos = 0; dst_pos < local_order.size(); ++dst_pos) {
      const size_t original_idx = local_order[dst_pos];
      const size_t src_pos = source_pos_by_original_idx[original_idx];
      plan.h2d_local_block_ids.push_back(local_block_ids[original_idx]);
      plan.host_dst_to_src.push_back(src_pos);
      identity_reorder = identity_reorder && (src_pos == dst_pos);
    }
    if (identity_reorder) {
      plan.host_dst_to_src.clear();
    }
    plan.h2d_copy = Offsets(plan.h2d_local_block_ids,
                            /*source_is_compact=*/true);
    return plan;
  }

  static void ReorderCompactBlocks(
      at::Tensor& compact_blocks, const std::vector<size_t>& dst_to_src) {
    if (dst_to_src.empty()) return;
    ValidateCpuTensor(compact_blocks, "Host reorder view");
    const size_t num_blocks = dst_to_src.size();
    if (num_blocks == 0 || compact_blocks.nbytes() % num_blocks != 0) {
      throw std::invalid_argument("host reorder view has invalid block layout");
    }
    const size_t block_bytes = compact_blocks.nbytes() / num_blocks;
    const auto total_bytes = static_cast<size_t>(compact_blocks.nbytes());
    const uint8_t* src = reinterpret_cast<const uint8_t*>(compact_blocks.data_ptr());
    std::vector<uint8_t> reordered(total_bytes);
    for (size_t dst_idx = 0; dst_idx < num_blocks; ++dst_idx) {
      const size_t src_idx = dst_to_src[dst_idx];
      if (src_idx >= num_blocks) {
        throw std::out_of_range("host reorder source index out of range");
      }
      std::memcpy(reordered.data() + dst_idx * block_bytes,
                  src + src_idx * block_bytes, block_bytes);
    }
    std::memcpy(compact_blocks.data_ptr(), reordered.data(), total_bytes);
  }

  TensorList LayerViews(int64_t slot_idx, int64_t num_blocks) {
    if (slot_idx < 0 || slot_idx >= static_cast<int64_t>(host_slots_.size())) {
      throw std::out_of_range("slot_idx out of range");
    }
    if (num_blocks < 0 || num_blocks > max_blocks_) {
      throw std::out_of_range("num_blocks out of range");
    }
    TensorList host_views;
    host_views.reserve(kv_caches_.size());
    for (size_t layer_idx = 0; layer_idx < host_slots_[slot_idx].layers.size();
         ++layer_idx) {
      at::Tensor tensor =
          host_slots_[slot_idx].layers[layer_idx].narrow(0, 0, num_blocks);
      ValidateCpuTensor(tensor, "Host staging view");
      host_views.push_back(std::move(tensor));
    }
    return host_views;
  }

  RaidenStageResult IssueD2H(int64_t slot_idx, int64_t num_blocks,
                             const std::vector<int64_t>& block_ids) {
    if (num_blocks != static_cast<int64_t>(block_ids.size())) {
      throw std::invalid_argument("num_blocks must match len(block_ids)");
    }
    CopySpec copy_spec = Offsets(block_ids, /*source_is_compact=*/false);
    TensorList host_views = LayerViews(slot_idx, num_blocks);
    auto future = std::make_shared<RaidenTransferFuture>();
    int64_t total_bytes = 0;
    for (size_t i = 0; i < prepared_.size(); ++i) {
      total_bytes += static_cast<int64_t>(host_views[i].nbytes());
      future->Add(prepared_[i]->D2HToAsync(host_views[i], copy_spec.src_offsets,
                                           copy_spec.dst_offsets,
                                           copy_spec.sizes));
    }
    return {.future = std::move(future),
            .host_views = std::move(host_views),
            .total_bytes = total_bytes,
            .copy_segments = static_cast<int64_t>(copy_spec.sizes.size())};
  }

  H2DBatchIssueResult IssueH2DBatch(
      const TensorList& host_views, const CopySpec& copy_spec,
      const std::vector<size_t>& layer_indices) {
    H2DBatchIssueResult result;
    result.future = std::make_shared<RaidenTransferFuture>();
    result.layers = layer_indices.size();
    if (layer_indices.empty()) {
      return result;
    }
    for (size_t layer_idx : layer_indices) {
      if (layer_idx >= prepared_.size() || layer_idx >= host_views.size()) {
        throw std::out_of_range("H2D batch layer index out of range");
      }
    }

    const int issue_threads =
        (copy_spec.sizes.size() > 1 && layer_indices.size() > 1)
            ? std::min<int>(h2d_issue_threads_,
                            static_cast<int>(layer_indices.size()))
            : 1;
    result.issued = true;
    result.first_issue_start = std::chrono::steady_clock::now();
    if (issue_threads <= 1) {
      auto raw_future =
          std::make_shared<raiden::PjRtCopyFuture>(std::vector<xla::Future<>>{});
      for (size_t layer_idx : layer_indices) {
        prepared_[layer_idx]->AppendH2DFrom(
            raw_future, host_views[layer_idx], copy_spec.src_offsets,
            copy_spec.dst_offsets, copy_spec.sizes);
      }
      result.last_issue_done = std::chrono::steady_clock::now();
      result.issue_ms =
          DurationMs(result.first_issue_start, result.last_issue_done);
      result.future->Add(std::move(raw_future));
      return result;
    }

    std::atomic<size_t> next_layer{0};
    std::vector<std::shared_ptr<raiden::PjRtCopyFuture>> worker_futures(
        issue_threads);
    std::vector<std::exception_ptr> exceptions(issue_threads);
    std::vector<double> worker_issue_ms(issue_threads, 0.0);
    std::vector<size_t> worker_layers(issue_threads, 0);
    std::vector<std::thread> workers;
    workers.reserve(issue_threads);
    for (int worker_idx = 0; worker_idx < issue_threads; ++worker_idx) {
      workers.emplace_back([&, worker_idx]() {
        try {
          auto raw_future = std::make_shared<raiden::PjRtCopyFuture>(
              std::vector<xla::Future<>>{});
          const auto worker_start = std::chrono::steady_clock::now();
          while (true) {
            const size_t batch_pos = next_layer.fetch_add(1);
            if (batch_pos >= layer_indices.size()) {
              break;
            }
            const size_t layer_idx = layer_indices[batch_pos];
            prepared_[layer_idx]->AppendH2DFrom(
                raw_future, host_views[layer_idx], copy_spec.src_offsets,
                copy_spec.dst_offsets, copy_spec.sizes);
            ++worker_layers[worker_idx];
          }
          const auto worker_done = std::chrono::steady_clock::now();
          worker_issue_ms[worker_idx] = DurationMs(worker_start, worker_done);
          worker_futures[worker_idx] = std::move(raw_future);
        } catch (...) {
          exceptions[worker_idx] = std::current_exception();
        }
      });
    }
    for (auto& worker : workers) {
      worker.join();
    }
    result.last_issue_done = std::chrono::steady_clock::now();
    for (const auto& exception : exceptions) {
      if (exception) {
        std::rethrow_exception(exception);
      }
    }
    for (int worker_idx = 0; worker_idx < issue_threads; ++worker_idx) {
      result.issue_ms += worker_issue_ms[worker_idx];
      if (worker_layers[worker_idx] > 0) {
        result.future->Add(std::move(worker_futures[worker_idx]));
      }
    }
    return result;
  }

  RaidenStageResult IssueH2D(int64_t slot_idx, int64_t num_blocks,
                             const std::vector<int64_t>& local_block_ids) {
    if (num_blocks != static_cast<int64_t>(local_block_ids.size())) {
      throw std::invalid_argument("num_blocks must match len(local_block_ids)");
    }
    CopySpec copy_spec = Offsets(local_block_ids, /*source_is_compact=*/true);
    TensorList host_views = LayerViews(slot_idx, num_blocks);
    auto future = std::make_shared<RaidenTransferFuture>();
    int64_t total_bytes = 0;
    for (const auto& view : host_views) {
      total_bytes += static_cast<int64_t>(view.nbytes());
    }

    std::vector<size_t> layer_indices;
    layer_indices.reserve(prepared_.size());
    for (size_t i = 0; i < prepared_.size(); ++i) {
      layer_indices.push_back(i);
    }
    H2DBatchIssueResult batch =
        IssueH2DBatch(host_views, copy_spec, layer_indices);
    future->AddAll(batch.future);
    return {.future = std::move(future),
            .host_views = std::move(host_views),
            .total_bytes = total_bytes,
            .copy_segments = static_cast<int64_t>(copy_spec.sizes.size())};
  }

  int64_t StorePending(PendingOperation op) {
    const int64_t op_id = next_op_id_++;
    pending_[op_id] = std::move(op);
    return op_id;
  }

  std::chrono::steady_clock::time_point DeadlineFromNow() const {
    return std::chrono::steady_clock::now() +
           std::chrono::milliseconds(static_cast<int64_t>(timeout_s_ * 1000.0));
  }

  void AllocateHostSlots(int64_t num_slots) {
    if (kv_caches_.empty()) return;
    host_slots_.clear();
    host_slots_.reserve(num_slots);
    for (int64_t slot = 0; slot < num_slots; ++slot) {
      HostSlot host_slot;
      host_slot.layers.reserve(kv_caches_.size());
      for (const auto& kv : kv_caches_) {
        std::vector<int64_t> shape;
        shape.reserve(kv.dim());
        shape.push_back(max_blocks_);
        for (int64_t d = 1; d < kv.dim(); ++d) {
          shape.push_back(kv.size(d));
        }
        at::Tensor host = at::empty(
            shape,
            kv.options().device(c10::Device(c10::kCPU)).pinned_memory(true));
        ValidateCpuTensor(host, "Host staging allocation");
        host_slot.layers.push_back(std::move(host));
      }
      host_slots_.push_back(std::move(host_slot));
      free_slots_.push_back(slot);
    }
  }

  int64_t AcquireSlot() {
    std::lock_guard<std::mutex> lock(mu_);
    return AcquireSlotLocked();
  }

  int64_t AcquireSlotLocked() {
    if (free_slots_.empty()) {
      throw std::runtime_error("Raiden host slot pool exhausted");
    }
    int64_t slot = free_slots_.front();
    free_slots_.pop_front();
    return slot;
  }

  void ReleaseSlotLocked(int64_t slot_idx) {
    if (slot_idx < 0) return;
    free_slots_.push_back(slot_idx);
  }

  std::string EndpointWithPort(const std::string& endpoint, int port) const {
    auto [host, ignored_port] = SplitEndpoint(endpoint);
    (void)ignored_port;
    return host + ":" + std::to_string(port);
  }

  std::vector<PullLayerDescriptor> LayerDescriptors(int64_t slot_idx,
                                                    int64_t num_blocks) {
    std::vector<PullLayerDescriptor> out;
    out.reserve(host_slots_[slot_idx].layers.size());
    for (size_t layer_idx = 0; layer_idx < host_slots_[slot_idx].layers.size();
         ++layer_idx) {
      out.push_back(LayerDescriptor(slot_idx, num_blocks, layer_idx));
    }
    return out;
  }

  PullLayerDescriptor LayerDescriptor(int64_t slot_idx, int64_t num_blocks,
                                      size_t layer_idx) {
    if (slot_idx < 0 || slot_idx >= static_cast<int64_t>(host_slots_.size())) {
      throw std::out_of_range("slot_idx out of range");
    }
    if (layer_idx >= host_slots_[slot_idx].layers.size()) {
      throw std::out_of_range("layer_idx out of range");
    }
    at::Tensor view =
        host_slots_[slot_idx].layers[layer_idx].narrow(0, 0, num_blocks);
    ValidateCpuTensor(view, "Host staging layer descriptor");
    return {reinterpret_cast<uint64_t>(view.data_ptr()),
            static_cast<uint64_t>(view.nbytes())};
  }

  void StartControlServer() {
    control_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (control_fd_ < 0) {
      throw std::runtime_error("failed to create Raiden control socket");
    }
    int opt = 1;
    setsockopt(control_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(control_fd_, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(local_control_port_);
    if (bind(control_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
      std::string err = std::strerror(errno);
      close(control_fd_);
      control_fd_ = -1;
      throw std::runtime_error("bind Raiden control socket failed: " + err);
    }
    if (listen(control_fd_, 128) < 0) {
      std::string err = std::strerror(errno);
      close(control_fd_);
      control_fd_ = -1;
      throw std::runtime_error("listen Raiden control socket failed: " + err);
    }
    control_thread_ = std::thread([this]() { ControlLoop(); });
  }

  void StopControlServer() {
    stopping_ = true;
    if (control_fd_ >= 0) {
      shutdown(control_fd_, SHUT_RDWR);
      close(control_fd_);
      control_fd_ = -1;
    }
    if (control_thread_.joinable()) {
      control_thread_.join();
    }
    {
      std::lock_guard<std::mutex> lock(control_workers_mu_);
      for (auto& thread : control_workers_) {
        if (thread.joinable()) thread.join();
      }
      control_workers_.clear();
    }
    {
      std::lock_guard<std::mutex> lock(worker_threads_mu_);
      for (auto& thread : worker_threads_) {
        if (thread.joinable()) thread.join();
      }
      worker_threads_.clear();
    }
  }

  void ControlLoop() {
    while (!stopping_) {
      pollfd pfd;
      pfd.fd = control_fd_;
      pfd.events = POLLIN;
      int ret = poll(&pfd, 1, 50);
      if (ret <= 0) continue;
      sockaddr_in client_addr;
      socklen_t len = sizeof(client_addr);
      int client_fd =
          accept(control_fd_, reinterpret_cast<sockaddr*>(&client_addr), &len);
      if (client_fd < 0) continue;
      int opt = 1;
      setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
      std::lock_guard<std::mutex> lock(control_workers_mu_);
      control_workers_.emplace_back([this, client_fd]() {
        HandleControlConnection(client_fd);
      });
    }
  }

  void HandleControlConnection(int fd) {
    while (!stopping_) {
      ControlRequestHeader request;
      absl::Status s = ReadExact(fd, &request, sizeof(request));
      if (!s.ok()) break;
      if (request.magic != kControlMagic) {
        WriteControlError(fd, "bad control magic");
        break;
      }
      if (request.op == kOpPull) {
        HandlePullRequest(fd, request);
      } else if (request.op == kOpPullStream) {
        HandlePullStreamRequest(fd, request);
      } else if (request.op == kOpAck) {
        AckSend(request.uuid);
        WriteControlOk(fd, {});
      } else {
        WriteControlError(fd, "unknown control op");
      }
    }
    close(fd);
  }

  void WriteControlOk(int fd, const std::vector<PullLayerDescriptor>& layers) {
    ControlResponseHeader response;
    response.magic = kResponseMagic;
    response.status = 0;
    response.num_layers = static_cast<uint32_t>(layers.size());
    response.data_port = static_cast<uint32_t>(local_data_port_);
    response.message_len = 0;
    absl::Status s = WriteExact(fd, &response, sizeof(response));
    if (!s.ok()) return;
    if (!layers.empty()) {
      WriteExact(fd, layers.data(), layers.size() * sizeof(PullLayerDescriptor));
    }
  }

  void WriteControlStreamHeader(int fd, size_t num_layers) {
    ControlResponseHeader response;
    response.magic = kResponseMagic;
    response.status = 0;
    response.num_layers = static_cast<uint32_t>(num_layers);
    response.data_port = static_cast<uint32_t>(local_data_port_);
    response.message_len = 0;
    CheckStatus("control stream response write",
                WriteExact(fd, &response, sizeof(response)));
  }

  void WriteControlError(int fd, const std::string& message) {
    ControlResponseHeader response;
    response.magic = kResponseMagic;
    response.status = 1;
    response.num_layers = 0;
    response.data_port = static_cast<uint32_t>(local_data_port_);
    response.message_len = message.size();
    absl::Status s = WriteExact(fd, &response, sizeof(response));
    if (!s.ok()) return;
    if (!message.empty()) {
      WriteExact(fd, message.data(), message.size());
    }
  }

  void HandlePullRequest(int fd, const ControlRequestHeader& request) {
    const auto request_start = std::chrono::steady_clock::now();
    std::shared_ptr<SendEntry> entry;
    {
      std::unique_lock<std::mutex> lock(mu_);
      const auto deadline = DeadlineFromNow();
      cv_.wait_until(lock, deadline, [&]() {
        auto it = send_entries_.find(request.uuid);
        return it != send_entries_.end() &&
               (it->second->stage_done || it->second->cancelled);
      });
      const auto wait_done = std::chrono::steady_clock::now();
      auto it = send_entries_.find(request.uuid);
      if (it == send_entries_.end() || !it->second->stage_done) {
        const bool cancelled =
            (it != send_entries_.end() && it->second->cancelled);
        const char* reason = cancelled ? "cancelled" : "stage_timeout";
        std::ostringstream timing;
        timing << "RAIDEN_TIMING event=producer_pull"
               << " uuid=" << request.uuid
               << " rank=" << tp_rank_
               << " blocks=" << request.num_blocks
               << " wait_stage_ms=" << DurationMs(request_start, wait_done)
               << " failed=1 reason=" << reason;
        EmitTimingLog(timing.str());
        WriteControlError(fd, cancelled ? "uuid cancelled before staging"
                                        : "uuid not staged before timeout");
        return;
      }
      entry = it->second;
      entry->deadline = DeadlineFromNow();
    }
    if (entry->failed) {
      WriteControlError(fd, "producer stage failed");
      return;
    }
    if (request.num_blocks != static_cast<uint64_t>(entry->num_blocks)) {
      WriteControlError(fd, "num_blocks mismatch");
      return;
    }
    const auto descriptor_start = std::chrono::steady_clock::now();
    WriteControlOk(fd, LayerDescriptors(entry->slot_idx, entry->num_blocks));
    const auto descriptor_done = std::chrono::steady_clock::now();
    std::ostringstream timing;
    timing << "RAIDEN_TIMING event=producer_pull"
           << " req_id=" << entry->req_id
           << " uuid=" << entry->uuid
           << " rank=" << tp_rank_
           << " blocks=" << entry->num_blocks
           << " bytes=" << entry->total_bytes
           << " wait_stage_ms=" << DurationMs(request_start, descriptor_start)
           << " descriptor_ms=" << DurationMs(descriptor_start, descriptor_done)
           << " register_to_pull_ms="
           << DurationMs(entry->register_start, descriptor_start)
           << " stage_to_pull_ms="
           << DurationMs(entry->d2h_done, descriptor_start)
           << " failed=0";
    EmitTimingLog(timing.str());
  }

  void HandlePullStreamRequest(int fd, const ControlRequestHeader& request) {
    const auto request_start = std::chrono::steady_clock::now();
    std::shared_ptr<SendEntry> entry;
    std::vector<int64_t> requested_block_ids;
    try {
      if (request.num_blocks > static_cast<uint64_t>(max_blocks_)) {
        throw std::invalid_argument("requested block count exceeds max_blocks");
      }
      requested_block_ids = ReadBlockIds(fd, request.num_blocks);
    } catch (const std::exception& e) {
      WriteControlError(fd, e.what());
      return;
    }

    const auto requested_plan = BuildProducerCopyPlan(requested_block_ids);
    int64_t slot_idx = -1;
    bool failed = false;
    bool cancelled = false;
    bool header_sent = false;
    std::string failure_reason;
    size_t num_layers = 0;
    size_t layers_sent = 0;
    std::chrono::steady_clock::time_point header_start;
    std::chrono::steady_clock::time_point header_done;
    try {
      {
        std::unique_lock<std::mutex> lock(mu_);
        const auto deadline = DeadlineFromNow();
        cv_.wait_until(lock, deadline, [&]() {
          return stopping_ || send_entries_.find(request.uuid) != send_entries_.end();
        });
        auto it = send_entries_.find(request.uuid);
        if (it == send_entries_.end()) {
          throw std::runtime_error("uuid not registered before timeout");
        }
        entry = it->second;
        if (entry->failed) {
          throw std::runtime_error("producer stage failed");
        }
        if (entry->cancelled) {
          throw std::runtime_error("uuid cancelled before stream");
        }
        if (entry->pull_started) {
          throw std::runtime_error("uuid already has an active pull");
        }
        ValidateRequestedBlocks(*entry, requested_plan.producer_remote_block_ids);
        slot_idx = AcquireSlotLocked();
        entry->slot_idx = slot_idx;
        entry->slot_released = false;
        entry->pull_started = true;
        entry->num_blocks = requested_plan.num_blocks;
        entry->stage_issue_started = true;
        entry->stage_done = false;
        entry->failed = false;
        entry->d2h_issue_start = std::chrono::steady_clock::now();
        entry->deadline = DeadlineFromNow();
      }

      TensorList host_views = LayerViews(slot_idx, requested_plan.num_blocks);
      auto future = std::make_shared<RaidenTransferFuture>();
      std::vector<std::shared_ptr<raiden::PjRtCopyFuture>> layer_futures;
      layer_futures.reserve(prepared_.size());
      int64_t total_bytes = 0;
      for (size_t i = 0; i < prepared_.size(); ++i) {
        total_bytes += static_cast<int64_t>(host_views[i].nbytes());
        auto layer_future = prepared_[i]->D2HToAsync(
            host_views[i], requested_plan.d2h_copy.src_offsets,
            requested_plan.d2h_copy.dst_offsets, requested_plan.d2h_copy.sizes);
        layer_futures.push_back(layer_future);
        future->Add(std::move(layer_future));
      }

      {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = send_entries_.find(request.uuid);
        if (it == send_entries_.end()) {
          if (slot_idx >= 0) {
            ReleaseSlotLocked(slot_idx);
            slot_idx = -1;
          }
          return;
        }
        entry = it->second;
        entry->d2h_issue_done = std::chrono::steady_clock::now();
        entry->total_bytes = total_bytes;
        entry->copy_segments =
            static_cast<int64_t>(requested_plan.d2h_copy.sizes.size());
        entry->layer_ready.assign(layer_futures.size(), false);
        entry->layers_ready = 0;
        entry->future = future;
        num_layers = layer_futures.size();
      }
      cv_.notify_all();

      header_start = std::chrono::steady_clock::now();
      WriteControlStreamHeader(fd, num_layers);
      header_done = std::chrono::steady_clock::now();
      header_sent = true;

      for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        layer_futures[layer_idx]->Await();
        {
          std::lock_guard<std::mutex> lock(mu_);
          auto it = send_entries_.find(request.uuid);
          if (it == send_entries_.end()) {
            if (slot_idx >= 0 && !entry->slot_released) {
              ReleaseSlotLocked(slot_idx);
              entry->slot_released = true;
            }
            return;
          }
          entry = it->second;
          if (entry->failed || entry->cancelled) {
            throw std::runtime_error(entry->cancelled
                                         ? "uuid cancelled during stream"
                                         : "producer stage failed");
          }
          if (layer_idx < entry->layer_ready.size() &&
              !entry->layer_ready[layer_idx]) {
            entry->layer_ready[layer_idx] = true;
            ++entry->layers_ready;
          }
          entry->deadline = DeadlineFromNow();
        }
        cv_.notify_all();
        PullLayerDescriptor descriptor =
            LayerDescriptor(slot_idx, requested_plan.num_blocks, layer_idx);
        CheckStatus("control stream layer descriptor write",
                    WriteExact(fd, &descriptor, sizeof(descriptor)));
        ++layers_sent;
      }
    } catch (const std::exception& e) {
      failed = true;
      failure_reason = e.what();
      LOG(ERROR) << "Raiden pull stream failed for uuid=" << request.uuid
                 << ": " << failure_reason;
      if (!header_sent) {
        WriteControlError(fd, failure_reason);
      }
    } catch (...) {
      failed = true;
      failure_reason = "unknown error";
      LOG(ERROR) << "Raiden pull stream failed for uuid=" << request.uuid;
      if (!header_sent) {
        WriteControlError(fd, failure_reason);
      }
    }
    const auto done = std::chrono::steady_clock::now();
    if (!header_sent) {
      header_start = done;
      header_done = done;
    }
    if (entry) {
      {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = send_entries_.find(request.uuid);
        if (it != send_entries_.end()) {
          if (it->second->d2h_issue_start ==
              std::chrono::steady_clock::time_point()) {
            it->second->d2h_issue_start = done;
          }
          if (it->second->d2h_issue_done ==
              std::chrono::steady_clock::time_point()) {
            it->second->d2h_issue_done = done;
          }
          it->second->stage_done = true;
          it->second->failed = failed;
          it->second->d2h_done = done;
          cancelled = it->second->cancelled;
          if (cancelled) {
            done_sending_.insert(it->second->req_id);
            ReleaseEntrySlotLocked(it->second);
            send_entries_.erase(it);
          }
        } else if (slot_idx >= 0 && entry && !entry->slot_released) {
          ReleaseSlotLocked(slot_idx);
          entry->slot_released = true;
        }
      }
      cv_.notify_all();
      std::ostringstream stage_timing;
      stage_timing << "RAIDEN_TIMING event=producer_stage"
                   << " req_id=" << entry->req_id
                   << " uuid=" << entry->uuid
                   << " rank=" << tp_rank_
                   << " registered_blocks=" << entry->registered_num_blocks
                   << " blocks=" << entry->num_blocks
                   << " bytes=" << entry->total_bytes
                   << " copy_segments=" << entry->copy_segments
                   << " d2h_issue_ms="
                   << DurationMs(entry->d2h_issue_start, entry->d2h_issue_done)
                   << " d2h_wait_ms="
                   << DurationMs(entry->d2h_issue_done, done)
                   << " d2h_total_ms="
                   << DurationMs(entry->d2h_issue_start, done)
                   << " register_to_stage_ms="
                   << DurationMs(entry->register_start, done)
                   << " cancelled=" << (cancelled ? 1 : 0)
                   << " failed=" << (failed ? 1 : 0);
      EmitTimingLog(stage_timing.str());
    }
    std::ostringstream timing;
    timing << "RAIDEN_TIMING event=producer_pull_stream"
           << " req_id=" << (entry ? entry->req_id : "")
           << " uuid=" << request.uuid
           << " rank=" << tp_rank_
           << " registered_blocks="
           << (entry ? entry->registered_num_blocks : 0)
           << " blocks=" << requested_plan.num_blocks
           << " layers=" << num_layers
           << " layers_sent=" << layers_sent
           << " bytes=" << (entry ? entry->total_bytes : 0)
           << " wait_slot_ms=" << DurationMs(request_start, header_start)
           << " header_ms=" << DurationMs(header_start, header_done)
           << " stream_ms=" << DurationMs(header_done, done)
           << " register_to_stream_done_ms="
           << (entry ? DurationMs(entry->register_start, done) : 0.0)
           << " reason=" << failure_reason
           << " failed=" << (failed ? 1 : 0);
    EmitTimingLog(timing.str());
  }

  PullResponse FetchPullDescriptors(const std::string& remote_endpoint,
                                    uint64_t uuid, int64_t num_blocks) {
    int fd = ConnectTcp(remote_endpoint);
    auto cleanup = std::unique_ptr<int, void (*)(int*)>(
        &fd, [](int* p) {
          if (p && *p >= 0) close(*p);
        });
    ControlRequestHeader request;
    request.magic = kControlMagic;
    request.op = kOpPull;
    request.uuid = uuid;
    request.num_blocks = static_cast<uint64_t>(num_blocks);
    CheckStatus("control pull write", WriteExact(fd, &request, sizeof(request)));
    return ReadPullResponse(fd);
  }

  void AckRemote(const std::string& remote_endpoint, uint64_t uuid) {
    int fd = ConnectTcp(remote_endpoint);
    auto cleanup = std::unique_ptr<int, void (*)(int*)>(
        &fd, [](int* p) {
          if (p && *p >= 0) close(*p);
        });
    ControlRequestHeader request;
    request.magic = kControlMagic;
    request.op = kOpAck;
    request.uuid = uuid;
    request.num_blocks = 0;
    CheckStatus("control ack write", WriteExact(fd, &request, sizeof(request)));
    (void)ReadPullResponse(fd);
  }

  ControlResponseHeader ReadControlResponseHeader(int fd) {
    ControlResponseHeader response;
    CheckStatus("control response read",
                ReadExact(fd, &response, sizeof(response)));
    if (response.magic != kResponseMagic) {
      throw std::runtime_error("bad control response magic");
    }
    if (response.status != 0) {
      std::string message(response.message_len, '\0');
      if (response.message_len > 0) {
        CheckStatus("control error body read",
                    ReadExact(fd, message.data(), message.size()));
      }
      throw std::runtime_error("remote Raiden control error: " + message);
    }
    return response;
  }

  PullResponse ReadPullResponse(int fd) {
    ControlResponseHeader response = ReadControlResponseHeader(fd);
    PullResponse out;
    out.data_port = static_cast<int>(response.data_port);
    out.layers.resize(response.num_layers);
    if (!out.layers.empty()) {
      CheckStatus("control layer descriptors read",
                  ReadExact(fd, out.layers.data(),
                            out.layers.size() * sizeof(PullLayerDescriptor)));
    }
    return out;
  }

  void AckSend(uint64_t uuid) {
    std::shared_ptr<SendEntry> entry;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = send_entries_.find(uuid);
      if (it == send_entries_.end()) {
        pending_acks_.insert(uuid);
        return;
      }
      entry = it->second;
      if (!entry->stage_done) {
        entry->cancelled = true;
        if (!entry->stage_issue_started) {
          ReleaseEntrySlotLocked(entry);
          done_sending_.insert(entry->req_id);
          send_entries_.erase(it);
        }
        cv_.notify_all();
        return;
      }
      done_sending_.insert(entry->req_id);
      ReleaseEntrySlotLocked(entry);
      send_entries_.erase(it);
    }
    const auto ack_done = std::chrono::steady_clock::now();
    std::ostringstream timing;
    timing << "RAIDEN_TIMING event=producer_ack"
           << " req_id=" << entry->req_id
           << " uuid=" << entry->uuid
           << " rank=" << tp_rank_
           << " blocks=" << entry->num_blocks
           << " bytes=" << entry->total_bytes
           << " stage_to_ack_ms=" << DurationMs(entry->d2h_done, ack_done)
           << " register_to_ack_ms="
           << DurationMs(entry->register_start, ack_done)
           << " failed=" << (entry->failed ? 1 : 0);
    EmitTimingLog(timing.str());
  }

  TensorList kv_caches_;
  std::vector<std::shared_ptr<PreparedTpuBuffer>> prepared_;
  int64_t tp_rank_ = 0;
  int local_control_port_ = 0;
  int local_data_port_ = 0;
  int64_t max_blocks_ = 0;
  double timeout_s_ = 120.0;
  bool unsafe_skip_buffer_lock_ = true;
  int h2d_issue_threads_ = 1;
  int h2d_batch_max_layers_ = 4;
  int64_t next_op_id_ = 1;
  std::map<int64_t, PendingOperation> pending_;
  std::vector<HostSlot> host_slots_;
  std::deque<int64_t> free_slots_;
  std::map<uint64_t, std::shared_ptr<SendEntry>> send_entries_;
  std::set<uint64_t> pending_acks_;
  std::set<std::string> done_sending_;
  std::set<std::string> done_recving_;
  std::set<std::string> failed_recving_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::unique_ptr<tpu_raiden::transport::SocketTransport> transport_;
  int control_fd_ = -1;
  std::atomic<bool> stopping_{false};
  std::thread control_thread_;
  std::mutex control_workers_mu_;
  std::vector<std::thread> control_workers_;
  std::mutex worker_threads_mu_;
  std::vector<std::thread> worker_threads_;
};

void AwaitAll(py::object futures) {
  if (py::isinstance<RaidenTransferFuture>(futures)) {
    auto future = futures.cast<std::shared_ptr<RaidenTransferFuture>>();
    future->Await();
    return;
  }
  if (py::isinstance<raiden::PjRtCopyFuture>(futures)) {
    auto future = futures.cast<std::shared_ptr<raiden::PjRtCopyFuture>>();
    MaybeGilRelease release;
    future->Await();
    return;
  }
  for (py::handle item : futures) {
    if (py::isinstance<RaidenTransferFuture>(item)) {
      auto future = item.cast<std::shared_ptr<RaidenTransferFuture>>();
      future->Await();
      continue;
    }
    auto future = item.cast<std::shared_ptr<raiden::PjRtCopyFuture>>();
    MaybeGilRelease release;
    future->Await();
  }
}

bool IsReady(py::object futures) {
  if (py::isinstance<RaidenTransferFuture>(futures)) {
    return futures.cast<std::shared_ptr<RaidenTransferFuture>>()->IsReady();
  }
  if (py::isinstance<raiden::PjRtCopyFuture>(futures)) {
    return futures.cast<std::shared_ptr<raiden::PjRtCopyFuture>>()->IsReady();
  }
  for (py::handle item : futures) {
    if (py::isinstance<RaidenTransferFuture>(item)) {
      if (!item.cast<std::shared_ptr<RaidenTransferFuture>>()->IsReady()) {
        return false;
      }
      continue;
    }
    if (!item.cast<std::shared_ptr<raiden::PjRtCopyFuture>>()->IsReady()) {
      return false;
    }
  }
  return true;
}

}  // namespace

PYBIND11_MODULE(_raiden_transfer_engine, m) {
  py::class_<RaidenTransferFuture, std::shared_ptr<RaidenTransferFuture>>(
      m, "RaidenTransferFuture")
      .def("Await", &RaidenTransferFuture::Await)
      .def("wait", &RaidenTransferFuture::Await)
      .def("IsReady", &RaidenTransferFuture::IsReady)
      .def("is_ready", &RaidenTransferFuture::IsReady);

  py::class_<RaidenTransferEngine, std::shared_ptr<RaidenTransferEngine>>(
      m, "RaidenTransferEngine")
      .def(py::init<const TensorList&, int64_t, int64_t, int64_t, int64_t,
                    double, bool>(),
           py::arg("kv_caches"), py::arg("tp_rank"),
           py::arg("local_control_port"), py::arg("max_blocks"),
           py::arg("num_slots"), py::arg("timeout_s") = 120.0,
           py::arg("unsafe_skip_buffer_lock") = true)
      .def_property_readonly("uses_prepared_tpu_buffers",
                             &RaidenTransferEngine::UsesPreparedTpuBuffers)
      .def_property_readonly("local_control_port",
                             &RaidenTransferEngine::local_control_port)
      .def_property_readonly("local_data_port",
                             &RaidenTransferEngine::local_data_port)
      .def("register_kv_cache", &RaidenTransferEngine::RegisterKvCache,
           py::arg("kv_caches"))
      .def("register_host_buffers", &RaidenTransferEngine::RegisterHostBuffers,
           py::arg("host_pool"), py::arg("tp_rank"))
      .def("register_send", &RaidenTransferEngine::RegisterSend,
           py::arg("req_id"), py::arg("uuid"), py::arg("block_ids"))
      .def("submit_load", &RaidenTransferEngine::SubmitLoad, py::arg("req_id"),
           py::arg("uuid"), py::arg("remote_endpoint"),
           py::arg("remote_block_ids"), py::arg("local_block_ids"))
      .def("stage_d2h", &RaidenTransferEngine::StageD2H, py::kw_only(),
           py::arg("slot_idx"), py::arg("num_blocks"), py::arg("block_ids"))
      .def("stage_d2h_sync", &RaidenTransferEngine::StageD2HSync,
           py::kw_only(), py::arg("slot_idx"), py::arg("num_blocks"),
           py::arg("block_ids"))
      .def("commit_h2d", &RaidenTransferEngine::CommitH2D, py::kw_only(),
           py::arg("slot_idx"), py::arg("num_blocks"),
           py::arg("local_block_ids"))
      .def("rank_layer_views", &RaidenTransferEngine::RankLayerViews,
           py::arg("slot_idx"), py::arg("rank"), py::arg("num_blocks"))
      .def("unpack_rank_layers", &RaidenTransferEngine::UnpackRankLayers,
           py::arg("slot_idx"), py::arg("rank"), py::arg("num_blocks"),
           py::arg("layer_buffers"))
      .def("submit_d2h", &RaidenTransferEngine::SubmitD2H, py::kw_only(),
           py::arg("slot_idx"), py::arg("num_blocks"), py::arg("block_ids"))
      .def("submit_h2d", &RaidenTransferEngine::SubmitH2D, py::kw_only(),
           py::arg("slot_idx"), py::arg("num_blocks"),
           py::arg("local_block_ids"))
      .def("poll_finished", &RaidenTransferEngine::PollFinished)
      .def("poll_transfer_ops", &RaidenTransferEngine::PollTransferOps)
      .def("wait_transfer", &RaidenTransferEngine::WaitTransfer,
           py::arg("op_id"))
      .def("_count_copy_segments_for_testing",
           &RaidenTransferEngine::CountCopySegmentsForTesting,
           py::arg("block_ids"))
      .def("_count_canonical_send_copy_segments_for_testing",
           &RaidenTransferEngine::CountCanonicalSendCopySegmentsForTesting,
           py::arg("block_ids"))
      .def("_count_canonical_load_copy_segments_for_testing",
           &RaidenTransferEngine::CountCanonicalLoadCopySegmentsForTesting,
           py::arg("remote_block_ids"), py::arg("local_block_ids"))
      .def("_send_copy_plan_for_testing",
           &RaidenTransferEngine::SendCopyPlanForTesting, py::arg("block_ids"))
      .def("_load_copy_plan_for_testing",
           &RaidenTransferEngine::LoadCopyPlanForTesting,
           py::arg("remote_block_ids"), py::arg("local_block_ids"));

  m.def("await_all", &AwaitAll, py::arg("futures"));
  m.def("is_ready", &IsReady, py::arg("futures"));
}

}  // namespace tpu_raiden::kv_cache
