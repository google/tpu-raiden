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

#include "tpu_raiden/weight_sync/tiling_utils.h"

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "xla/layout.h"

#include "absl/status/status.h"
#include "xla/index_util.h"
#include "xla/layout_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace tpu_raiden::weight_sync {

namespace {

int64_t GetTiledBufferElements(const xla::Shape& shape) {
  const int num_dims = shape.dimensions().size();
  if (num_dims == 0) {
    return 1;
  }

  std::vector<int64_t> current_shape;
  current_shape.reserve(num_dims);
  for (int64_t i = num_dims - 1; i >= 0; --i) {
    int64_t logical_dim = shape.layout().minor_to_major(i);
    current_shape.push_back(shape.dimensions(logical_dim));
  }

  for (const xla::Tile& tile : shape.layout().tiles()) {
    const int64_t tile_rank = tile.dimensions().size();
    if (tile_rank > current_shape.size()) {
      int64_t pad_size = tile_rank - current_shape.size();
      current_shape.insert(current_shape.begin(), pad_size, 1);
    }

    const int64_t suffix_start = current_shape.size() - tile_rank;
    std::vector<int64_t> next_shape;
    next_shape.reserve(current_shape.size() + tile_rank);

    for (int i = 0; i < suffix_start; ++i) {
      next_shape.push_back(current_shape[i]);
    }

    for (int i = 0; i < tile_rank; ++i) {
      int64_t d = current_shape[suffix_start + i];
      int64_t t = tile.dimension(i);
      next_shape.push_back(xla::CeilOfRatio(d, t));
    }

    for (int i = 0; i < tile_rank; ++i) {
      int64_t t = tile.dimension(i);
      next_shape.push_back(t);
    }

    current_shape = std::move(next_shape);
  }

  int64_t total_elements = 1;
  for (int64_t dim_size : current_shape) {
    total_elements *= dim_size;
  }
  return total_elements;
}

}  // namespace

absl::Status DetileBuffer(const uint8_t* src_tiled, uint8_t* dst_linear,
                          const xla::Shape& shape, const xla::Layout& layout) {
  if (layout.tiles().empty()) {
    return absl::InternalError("Buffer is not tiled");
  }

  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  xla::Shape standard_shape =
      xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions());

  xla::ShapeUtil::ForEachIndexNoStatus(
      shape, [&](absl::Span<const int64_t> indices) -> bool {
        int64_t linear_offset =
            xla::IndexUtil::MultidimensionalIndexToLinearIndex(standard_shape,
                                                               indices) *
            itemsize;
        int64_t physical_offset =
            xla::LayoutUtil::LinearIndexForNestedTiling(shape, indices) *
            itemsize;
        std::memcpy(dst_linear + linear_offset, src_tiled + physical_offset,
                    itemsize);
        return true;
      });

  return absl::OkStatus();
}

absl::Status TileBuffer(const uint8_t* src_linear, uint8_t* dst_tiled,
                        const xla::Shape& shape, const xla::Layout& layout) {
  if (layout.tiles().empty()) {
    return absl::InternalError("Buffer is not tiled");
  }

  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  int64_t total_physical_elements = GetTiledBufferElements(shape);
  std::memset(dst_tiled, 0, total_physical_elements * itemsize);

  xla::Shape standard_shape =
      xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions());

  xla::ShapeUtil::ForEachIndexNoStatus(
      shape, [&](absl::Span<const int64_t> indices) -> bool {
        int64_t linear_offset =
            xla::IndexUtil::MultidimensionalIndexToLinearIndex(standard_shape,
                                                               indices) *
            itemsize;
        int64_t physical_offset =
            xla::LayoutUtil::LinearIndexForNestedTiling(shape, indices) *
            itemsize;
        std::memcpy(dst_tiled + physical_offset, src_linear + linear_offset,
                    itemsize);
        return true;
      });

  return absl::OkStatus();
}

}  // namespace tpu_raiden::weight_sync
