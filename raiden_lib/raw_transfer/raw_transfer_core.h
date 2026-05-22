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

#ifndef THIRD_PARTY_TPU_RAIDEN_RAIDEN_LIB_RAW_TRANSFER_RAW_TRANSFER_CORE_H_
#define THIRD_PARTY_TPU_RAIDEN_RAIDEN_LIB_RAW_TRANSFER_RAW_TRANSFER_CORE_H_

#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/future.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_external.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace raiden {

struct RawBufferHolder {
  const PJRT_Api* c_api;
  const PJRT_RawBuffer_Extension* extension;
  PJRT_RawBuffer* buffer;

  RawBufferHolder(const PJRT_Api* api, const PJRT_RawBuffer_Extension* ext,
                  PJRT_RawBuffer* buf)
      : c_api(api), extension(ext), buffer(buf) {}

  ~RawBufferHolder() {
    if (buffer) {
      pjrt::PjRtCApiRawBuffer_Destroy(c_api, extension, buffer);
    }
  }
};

inline const PJRT_RawBuffer_Extension* GetRawBufferExtension(
    const xla::PjRtBuffer* buffer, const PJRT_Api** out_c_api = nullptr) {
  auto* capi_buffer = dynamic_cast<const xla::PjRtCApiBuffer*>(buffer);
  if (!capi_buffer && buffer->client()->plugin_attributes().has_value()) {
    capi_buffer = static_cast<const xla::PjRtCApiBuffer*>(buffer);
  }
  if (!capi_buffer) {
    return nullptr;
  }
  const PJRT_Api* c_api = capi_buffer->pjrt_c_api();
  if (out_c_api) {
    *out_c_api = c_api;
  }
  for (PJRT_Extension_Base* ext = c_api->extension_start; ext != nullptr;
       ext = ext->next) {
    if (ext->type == PJRT_Extension_Type::PJRT_Extension_Type_RawBuffer) {
      return reinterpret_cast<const PJRT_RawBuffer_Extension*>(ext);
    }
  }
  return nullptr;
}

inline xla::PjRtCApiBuffer* AsPjRtCApiBuffer(xla::PjRtBuffer* buffer) {
  auto* capi_buffer = dynamic_cast<xla::PjRtCApiBuffer*>(buffer);
  if (capi_buffer) {
    return capi_buffer;
  }
  if (buffer->client()->plugin_attributes().has_value()) {
    return static_cast<xla::PjRtCApiBuffer*>(buffer);
  }
  return nullptr;
}

inline xla::CommonPjRtBuffer* AsCommonPjRtBuffer(xla::PjRtBuffer* buffer) {
  return dynamic_cast<xla::CommonPjRtBuffer*>(buffer);
}

inline int64_t GetMajorSliceByteSize(const xla::PjRtBuffer* buffer) {
  const xla::Shape& shape = buffer->on_device_shape();
  if (shape.dimensions_size() == 0) return 0;

  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  int64_t stride = 1;
  for (int i = 1; i < shape.dimensions_size(); ++i) {
    stride *= shape.dimensions(i);
  }

  auto pjrt_layout = buffer->layout();
  const xla::Layout* xla_layout = nullptr;
  if (pjrt_layout) {
    xla_layout = &pjrt_layout->xla_layout();
  }

  if (xla_layout && !xla_layout->tiles().empty() &&
      shape.dimensions_size() >= 3) {
    const xla::Tile& tile = xla_layout->tiles()[0];
    auto tile_dims = tile.dimensions();
    if (tile_dims.size() != 2) {
      throw std::runtime_error("Only 2D tiling supported for now");
    }
    int64_t tH = tile_dims[0];
    int64_t tW = tile_dims[1];
    int64_t rank = shape.dimensions_size();

    // Find the two most minor logical dimensions physically.
    int64_t logical_minor_0 = xla_layout->minor_to_major(0);
    int64_t logical_minor_1 = xla_layout->minor_to_major(1);

    int64_t num_tiles_0 = (shape.dimensions(logical_minor_0) + tW - 1) / tW;
    int64_t num_tiles_1 = (shape.dimensions(logical_minor_1) + tH - 1) / tH;
    int64_t tiled_2d_block_size =
        num_tiles_0 * num_tiles_1 * tH * tW * itemsize;

    int64_t multiplier = 1;
    for (int i = 1; i < rank; ++i) {
      if (i != logical_minor_0 && i != logical_minor_1) {
        multiplier *= shape.dimensions(i);
      }
    }

    int64_t size_per_major_dim = tiled_2d_block_size * multiplier;

    return size_per_major_dim;
  }

  return stride * itemsize;
}

struct BufferHoldAndAlias {
  xla::PjRtBuffer* buffer = nullptr;
  bool is_common_buffer = false;

  // For CommonPjRtBuffer:
  tsl::RCReference<xla::CommonPjRtRawBuffer> common_raw_buffer;
  std::shared_ptr<xla::CommonPjRtBuffer::ScopedHold> common_hold;

  // For PjRtCApiBuffer:
  PJRT_RawBuffer* c_raw_buffer = nullptr;
  std::shared_ptr<RawBufferHolder> c_hold;

  static absl::StatusOr<BufferHoldAndAlias> Acquire(
      xla::PjRtBuffer* buf, const PJRT_Api* c_api = nullptr,
      const PJRT_RawBuffer_Extension* extension = nullptr,
      bool unsafe_skip_buffer_lock = false) {
    BufferHoldAndAlias result;
    result.buffer = buf;
    auto* common_buf = AsCommonPjRtBuffer(buf);
    auto* capi_buf = AsPjRtCApiBuffer(buf);

    if (common_buf) {
      result.is_common_buffer = true;
      auto hold = common_buf->GetBufferWithHold(
          xla::CommonPjRtBuffer::ScopedHold::kUsage);
      if (!hold.ok()) {
        return hold.status();
      }
      result.common_raw_buffer = hold.buffer()->raw_buffer();
      if (!unsafe_skip_buffer_lock) {
        result.common_hold =
            std::make_shared<xla::CommonPjRtBuffer::ScopedHold>(
                std::move(hold));
      }
      return result;
    }

    if (capi_buf) {
      result.is_common_buffer = false;
      if (!extension) {
        extension = GetRawBufferExtension(buf, &c_api);
        if (!extension) {
          return absl::InternalError("RawBuffer extension missing");
        }
      }
      auto status_or_raw = pjrt::PjRtCApiBuffer_CreateRawAliasOfBuffer(
          c_api, extension, capi_buf->c_buffer());
      if (!status_or_raw.ok()) {
        return status_or_raw.status();
      }
      result.c_raw_buffer = status_or_raw.value();
      result.c_hold = std::make_shared<RawBufferHolder>(c_api, extension,
                                                        result.c_raw_buffer);
      return result;
    }

    return absl::InvalidArgumentError("Unsupported PjRtBuffer type");
  }

  xla::Future<> CopyRawHostToDevice(const void* src, int64_t device_offset,
                                    int64_t size) const {
    if (is_common_buffer) {
      return common_raw_buffer->CopyRawHostToDevice(src, device_offset, size);
    }
    return pjrt::PjRtCApiRawBuffer_CopyRawHostToDevice(
        c_hold->c_api, c_hold->extension, c_raw_buffer, src, device_offset,
        size);
  }

  xla::Future<> CopyRawDeviceToHost(void* host_ptr, int64_t device_offset,
                                    int64_t size) const {
    if (is_common_buffer) {
      return buffer->CopyRawToHost(host_ptr, device_offset, size);
    }
    return pjrt::PjRtCApiRawBuffer_CopyRawDeviceToHost(
        c_hold->c_api, c_hold->extension, c_raw_buffer, host_ptr, device_offset,
        size);
  }
};

class PjRtCopyFuture {
 public:
  explicit PjRtCopyFuture(
      std::vector<xla::Future<>> futures,
      std::vector<std::shared_ptr<RawBufferHolder>> c_api_holds = {},
      std::vector<std::shared_ptr<xla::CommonPjRtBuffer::ScopedHold>> holds =
          {})
      : futures_(std::move(futures)),
        c_api_holds_(std::move(c_api_holds)),
        holds_(std::move(holds)) {}

  explicit PjRtCopyFuture(
      std::vector<xla::Future<>> futures,
      std::shared_ptr<RawBufferHolder> c_api_hold,
      std::shared_ptr<xla::CommonPjRtBuffer::ScopedHold> hold = nullptr)
      : futures_(std::move(futures)) {
    if (c_api_hold) c_api_holds_.push_back(std::move(c_api_hold));
    if (hold) holds_.push_back(std::move(hold));
  }

  void Await() {
    for (auto& f : futures_) {
      if (!f.IsValid()) {
        continue;
      }
      absl::Status status = f.Await();
      if (!status.ok()) {
        throw std::runtime_error(std::string("Async copy failed: ") +
                                 std::string(status.message()));
      }
    }
    futures_.clear();
    c_api_holds_.clear();
    holds_.clear();
    ext_holds_.clear();
    user_holds_.clear();
  }

  bool IsReady() const {
    for (const auto& f : futures_) {
      if (!f.IsValid()) {
        continue;
      }
      if (!f.IsReady()) {
        return false;
      }
    }
    return true;
  }

  void Append(PjRtCopyFuture other) {
    futures_.insert(futures_.end(),
                    std::make_move_iterator(other.futures_.begin()),
                    std::make_move_iterator(other.futures_.end()));
    c_api_holds_.insert(c_api_holds_.end(),
                        std::make_move_iterator(other.c_api_holds_.begin()),
                        std::make_move_iterator(other.c_api_holds_.end()));
    holds_.insert(holds_.end(), std::make_move_iterator(other.holds_.begin()),
                  std::make_move_iterator(other.holds_.end()));
    ext_holds_.insert(ext_holds_.end(),
                      std::make_move_iterator(other.ext_holds_.begin()),
                      std::make_move_iterator(other.ext_holds_.end()));
    user_holds_.insert(user_holds_.end(),
                       std::make_move_iterator(other.user_holds_.begin()),
                       std::make_move_iterator(other.user_holds_.end()));
  }

  void Append(
      std::vector<xla::Future<>> other_futures,
      std::shared_ptr<RawBufferHolder> other_c_api_hold = nullptr,
      std::shared_ptr<xla::CommonPjRtBuffer::ScopedHold> other_hold = nullptr) {
    futures_.insert(futures_.end(),
                    std::make_move_iterator(other_futures.begin()),
                    std::make_move_iterator(other_futures.end()));
    if (other_c_api_hold) c_api_holds_.push_back(std::move(other_c_api_hold));
    if (other_hold) holds_.push_back(std::move(other_hold));
  }

  void Append(std::vector<xla::Future<>> other_futures,
              std::vector<std::shared_ptr<RawBufferHolder>> other_c_api_holds,
              std::vector<std::shared_ptr<xla::CommonPjRtBuffer::ScopedHold>>
                  other_holds) {
    futures_.insert(futures_.end(),
                    std::make_move_iterator(other_futures.begin()),
                    std::make_move_iterator(other_futures.end()));
    c_api_holds_.insert(c_api_holds_.end(),
                        std::make_move_iterator(other_c_api_holds.begin()),
                        std::make_move_iterator(other_c_api_holds.end()));
    holds_.insert(holds_.end(), std::make_move_iterator(other_holds.begin()),
                  std::make_move_iterator(other_holds.end()));
  }

  void Append(std::vector<xla::Future<>> other_futures,
              const BufferHoldAndAlias& hold) {
    futures_.insert(futures_.end(),
                    std::make_move_iterator(other_futures.begin()),
                    std::make_move_iterator(other_futures.end()));
    if (hold.c_hold) c_api_holds_.push_back(hold.c_hold);
    if (hold.common_hold) holds_.push_back(hold.common_hold);
  }

  void AddExtHold(std::unique_ptr<xla::PjRtBuffer::ExternalReference> hold) {
    ext_holds_.push_back(std::move(hold));
  }

  void AddUserHold(std::shared_ptr<void> hold) {
    user_holds_.push_back(std::move(hold));
  }

 private:
  std::vector<xla::Future<>> futures_;
  std::vector<std::shared_ptr<RawBufferHolder>> c_api_holds_;
  std::vector<std::shared_ptr<xla::CommonPjRtBuffer::ScopedHold>> holds_;
  std::vector<std::shared_ptr<xla::PjRtBuffer::ExternalReference>> ext_holds_;
  std::vector<std::shared_ptr<void>> user_holds_;
};

}  // namespace raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_RAIDEN_LIB_RAW_TRANSFER_RAW_TRANSFER_CORE_H_
