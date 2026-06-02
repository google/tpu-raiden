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
#include "xla/layout.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_external.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/shape.h"
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
  if (!capi_buffer) return nullptr;
  if (out_c_api) *out_c_api = capi_buffer->pjrt_c_api();
  auto* capi_client = dynamic_cast<xla::PjRtCApiClient*>(
      const_cast<xla::PjRtClient*>(capi_buffer->client()));
  if (!capi_client) return nullptr;
  return capi_client->FindExtension<PJRT_RawBuffer_Extension>(
      PJRT_Extension_Type::PJRT_Extension_Type_RawBuffer);
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
    auto* common_buf = dynamic_cast<xla::CommonPjRtBuffer*>(buf);
    auto* capi_buf = dynamic_cast<xla::PjRtCApiBuffer*>(buf);

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

struct BufferHolder {
  std::shared_ptr<RawBufferHolder> c_api_hold;
  std::shared_ptr<xla::CommonPjRtBuffer::ScopedHold> hold;
  std::shared_ptr<xla::PjRtBuffer::ExternalReference> ext_hold;
  std::shared_ptr<void> user_hold;
};

using BufferHolders = std::vector<BufferHolder>;
using PjRtCopyFuture = xla::Future<BufferHolders>;

inline xla::Future<BufferHolder> CreateBufferFuture(
    std::vector<xla::Future<>> futures,
    std::shared_ptr<RawBufferHolder> c_api_hold = nullptr,
    std::shared_ptr<xla::CommonPjRtBuffer::ScopedHold> hold = nullptr,
    std::shared_ptr<xla::PjRtBuffer::ExternalReference> ext_hold = nullptr,
    std::shared_ptr<void> user_hold = nullptr) {
  auto join_future = xla::JoinFutures(futures);
  if (!join_future.IsValid()) {
    return xla::Future<BufferHolder>(
        BufferHolder{std::move(c_api_hold), std::move(hold),
                     std::move(ext_hold), std::move(user_hold)});
  }
  return join_future.Map([c_api_hold = std::move(c_api_hold),
                          hold = std::move(hold),
                          ext_hold = std::move(ext_hold),
                          user_hold = std::move(user_hold)]() mutable {
    return BufferHolder{std::move(c_api_hold), std::move(hold),
                        std::move(ext_hold), std::move(user_hold)};
  });
}

inline xla::Future<BufferHolders> FlattenPjRtFutures(
    xla::Future<std::vector<BufferHolders>> futures) {
  if (!futures.IsValid()) {
    return xla::Future<BufferHolders>(BufferHolders{});
  }
  return futures.Map([](std::vector<BufferHolders> vecs) {
    std::vector<BufferHolder> result;
    for (auto& vec : vecs) {
      for (auto& h : vec) {
        result.push_back(std::move(h));
      }
    }
    return result;
  });
}

}  // namespace raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_RAIDEN_LIB_RAW_TRANSFER_RAW_TRANSFER_CORE_H_
