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
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/layout.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace tpu_raiden::weight_sync {
namespace {

TEST(TilingUtilsTest, Standard2D) {
  // 2D matrix of shape 8x8, element type float (4 bytes).
  // Layout has minor_to_major={1, 0} (row-major), and tiling with tile
  // dimensions 4x4.
  const int64_t H = 8;
  const int64_t W = 8;
  const int64_t tH = 4;
  const int64_t tW = 4;

  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {H, W}, {1, 0}, {xla::Tile({tH, tW})});

  const int64_t num_elements = H * W;
  std::vector<float> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<float>(i);
  }

  // Tiled buffer size should be H * W * sizeof(float)
  const int64_t tiled_size_bytes = H * W * sizeof(float);
  std::vector<uint8_t> dst_tiled(tiled_size_bytes);

  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  // Verify tiled structure manually for a few values.
  // Original 8x8 matrix is divided into 2x2 grid of 4x4 tiles:
  // Tile(0, 0) covers rows 0-3, cols 0-3.
  // Tile(0, 1) covers rows 0-3, cols 4-7.
  // Tile(1, 0) covers rows 4-7, cols 0-3.
  // Tile(1, 1) covers rows 4-7, cols 4-7.
  //
  // Let's check row 0, col 4: linear index is 0*8 + 4 = 4. Value is 4.0.
  // In tiling: tile_h = 0, tile_w = 1. Tile index = 0 * 2 + 1 = 1.
  // Within tile: row_offset = 0, col_offset = 0. Offset within tile = 0*4 + 0 =
  // 0. Physical offset in tiled buffer: (tile_index * (tH * tW) +
  // offset_within_tile) * sizeof(float) = (1 * 16 + 0) * 4 = 64 bytes. Float
  // value at 64 bytes should be 4.0f.
  float* dst_tiled_float = reinterpret_cast<float*>(dst_tiled.data());
  EXPECT_EQ(dst_tiled_float[16], 4.0f);

  // Let's check row 4, col 1: linear index is 4*8 + 1 = 33. Value is 33.0.
  // In tiling: tile_h = 1, tile_w = 0. Tile index = 1 * 2 + 0 = 2.
  // Within tile: row_offset = 0, col_offset = 1. Offset within tile = 0*4 + 1
  // = 1. Physical offset: (2 * 16 + 1) * 4 = 132 bytes (index 33 in float
  // array). Float value at index 33 in tiled array should be 33.0f.
  EXPECT_EQ(dst_tiled_float[33], 33.0f);

  // Now detile.
  std::vector<float> dst_linear(num_elements, 0.0f);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST(TilingUtilsTest, Padding2D) {
  // 2D matrix of shape 6x6, element type float (4 bytes).
  // Layout has minor_to_major={1, 0} (row-major), and tiling with tile
  // dimensions 4x4. The dimensions do not divide the tile size. Number of tiles
  // in W: ceil(6 / 4) = 2. Number of tiles in H: ceil(6 / 4) = 2. Total tiles
  // = 4. Total size in tiled buffer = 2 * 2 * 4 * 4 * 4 = 256 bytes (64
  // floats).
  const int64_t H = 6;
  const int64_t W = 6;
  const int64_t tH = 4;
  const int64_t tW = 4;

  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {H, W}, {1, 0}, {xla::Tile({tH, tW})});

  const int64_t num_elements = H * W;
  std::vector<float> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<float>(i);
  }

  const int64_t tiled_size_bytes = 2 * 2 * tH * tW * sizeof(float);
  std::vector<uint8_t> dst_tiled(tiled_size_bytes);

  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  // Verify that padding elements are 0.
  // E.g., Tile(1, 1) covers rows 4-7, cols 4-7. But matrix only goes up to
  // index 5. Row 5, col 5 is in matrix. Row 5, col 6 is padding. Tile(1, 1)
  // index is 1 * 2 + 1 = 3. Within Tile(1, 1), Row 5 (offset 1), Col 6 (offset
  // 2) -> offset = 1 * 4 + 2 = 6. Float index: 3 * 16 + 6 = 54. It should be
  // 0.0f.
  float* dst_tiled_float = reinterpret_cast<float*>(dst_tiled.data());
  EXPECT_EQ(dst_tiled_float[54], 0.0f);

  // Now detile back.
  std::vector<float> dst_linear(num_elements, 0.0f);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST(TilingUtilsTest, PermutedOuterLayout) {
  // Shape: F32[2, 3, 8, 8]
  // Outer dimensions: 0 (size 2), 1 (size 3)
  // Tiled dimensions: 2 (size 8), 3 (size 8)
  // Tile dimensions: 4x4
  // Layout minor_to_major: {3, 2, 0, 1}
  const int64_t D0 = 2;
  const int64_t D1 = 3;
  const int64_t D2 = 8;
  const int64_t D3 = 8;
  const int64_t tH = 4;
  const int64_t tW = 4;

  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {D0, D1, D2, D3}, {3, 2, 0, 1},
      {xla::Tile({tH, tW})});

  const int64_t num_elements = D0 * D1 * D2 * D3;
  std::vector<float> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<float>(i);
  }

  const int64_t tiled_size_bytes = num_elements * sizeof(float);
  std::vector<uint8_t> dst_tiled(tiled_size_bytes);

  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  const float* dst_tiled_float =
      reinterpret_cast<const float*>(dst_tiled.data());
  EXPECT_EQ(dst_tiled_float[128], 64.0f);

  // Detile back.
  std::vector<float> dst_linear(num_elements, 0.0f);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST(TilingUtilsTest, Bf16SubTiling) {
  // Shape: BF16[16, 256]
  // Tile 1: 8x128
  // Tile 2: 2x1
  const int64_t H = 16;
  const int64_t W = 256;

  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {H, W}, {1, 0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});

  const int64_t num_elements = H * W;
  std::vector<uint16_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<uint16_t>(i);
  }

  const int64_t tiled_size_bytes = num_elements * sizeof(uint16_t);
  std::vector<uint8_t> dst_tiled(tiled_size_bytes);

  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  const uint16_t* dst_tiled_uint16 =
      reinterpret_cast<const uint16_t*>(dst_tiled.data());
  EXPECT_EQ(dst_tiled_uint16[1], 256);

  // Detile back.
  std::vector<uint16_t> dst_linear(num_elements, 0);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

}  // namespace
}  // namespace tpu_raiden::weight_sync
