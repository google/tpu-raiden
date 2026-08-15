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

#include "tpu_sync/weight_sync/tiling_utils.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/span.h"
#include "xla/index_util.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tpu_sync/core/numa_thread_pool.h"

namespace tpu_raiden::weight_sync {

namespace {

constexpr int64_t kParallelizationThresholdBytes = 2 * 1024 * 1024;

constexpr int64_t kMaxNumThreads = 4;

tpu_raiden::NumaThreadPool* GetThreadPool() {
  static absl::NoDestructor<tpu_raiden::NumaThreadPool> global_pool(
      kMaxNumThreads);
  return global_pool.get();
}

bool IsStandardRowMajorTiled(const xla::Shape& shape,
                             const xla::Layout& layout) {
  if (layout.tiles().size() != 1) return false;
  if (layout.tiles(0).dimensions().size() != 2) return false;

  const int R = shape.dimensions().size();
  if (R < 1) return false;

  for (int i = 0; i < R; ++i) {
    if (layout.minor_to_major(i) != R - 1 - i) {
      return false;
    }
  }
  return true;
}

// --- BF16 {Tile(8,128),Tile(2,1)} fast detile (PR-E) ---------------------
// Real bf16 weights carry a nested layout {Tile(8,128), Tile(2,1)} rather
// than the single-tile layout IsStandardRowMajorTiled() fast-paths, so they
// otherwise fall through to the scalar ForEachIndexNoStatus() detile. The
// inner Tile(2,1) interleaves pairs of adjacent rows: within each 8x128 tile
// the physical element order is [rowpair(4)][col(128)][parity(2)], i.e. the
// two rows of a pair are stored with a fixed stride of 2. This is a pure
// fixed-stride-2 row de-interleave, so it can be done with tight contiguous
// output writes and one strided read per element -- byte-exact with, but far
// faster than, the scalar per-element index-math fallback.
//
// Matches ONLY BF16, rank-2, standard row-major (minor_to_major = {1,0}),
// tiles == {Tile(8,128), Tile(2,1)}. Every other layout returns false and
// falls through to the existing paths unchanged.
bool IsBf16SubTiled_8_128_2_1(const xla::Shape& shape,
                              const xla::Layout& layout) {
  if (shape.element_type() != xla::PrimitiveType::BF16) return false;
  if (shape.dimensions().size() != 2) return false;
  if (layout.minor_to_major().size() != 2) return false;
  if (layout.minor_to_major(0) != 1 || layout.minor_to_major(1) != 0) {
    return false;
  }
  if (layout.tiles().size() != 2) return false;
  const xla::Tile& t0 = layout.tiles(0);
  const xla::Tile& t1 = layout.tiles(1);
  if (t0.dimensions().size() != 2 || t1.dimensions().size() != 2) return false;
  if (t0.dimension(0) != 8 || t0.dimension(1) != 128) return false;
  if (t1.dimension(0) != 2 || t1.dimension(1) != 1) return false;
  return true;
}

// Byte-exact fast detile for the BF16 {Tile(8,128),Tile(2,1)} layout. The
// caller has already verified the layout via IsBf16SubTiled_8_128_2_1().
void DetileBufferBf16SubTiled(const uint8_t* src_tiled, uint8_t* dst_linear,
                              int64_t H, int64_t W) {
  constexpr int64_t kTileH = 8;
  constexpr int64_t kTileW = 128;
  constexpr int64_t kTileElems = kTileH * kTileW;  // 1024
  const int64_t col_tiles = (W + kTileW - 1) / kTileW;
  const uint16_t* src = reinterpret_cast<const uint16_t*>(src_tiled);
  uint16_t* dst = reinterpret_cast<uint16_t*>(dst_linear);

  for (int64_t r = 0; r < H; ++r) {
    const int64_t tr = r / kTileH;
    const int64_t rin = r % kTileH;
    const int64_t rowpair = rin / 2;   // 0..3
    const int64_t parity = rin % 2;    // 0..1
    uint16_t* dst_row = dst + r * W;
    for (int64_t tc = 0; tc < col_tiles; ++tc) {
      const int64_t c0 = tc * kTileW;
      const int64_t cw = std::min<int64_t>(kTileW, W - c0);
      const int64_t tile_index = tr * col_tiles + tc;
      // Physical base of this row's elements within the tile:
      //   tile_index*1024 + rowpair*256 + parity, then stride 2 across cols.
      const uint16_t* src_row =
          src + tile_index * kTileElems + rowpair * (kTileW * 2) + parity;
      uint16_t* out = dst_row + c0;
      for (int64_t cin = 0; cin < cw; ++cin) {
        out[cin] = src_row[cin * 2];
      }
    }
  }
}

// Dispatches row copy operations to compile-time specialized fixed-width
// vector copy loops for common TPU tile row sizes (FP8, BF16, FP32 with tile_W
// 128 or 8), falling back to generic std::memcpy for arbitrary dimensions.
template <typename F>
decltype(auto) DispatchByRowBytes(int64_t row_bytes, F&& f) {
  switch (row_bytes) {
    case 8:
      return f(std::integral_constant<size_t, 8>{});
    case 16:
      return f(std::integral_constant<size_t, 16>{});
    case 32:
      return f(std::integral_constant<size_t, 32>{});
    case 64:
      return f(std::integral_constant<size_t, 64>{});
    case 128:
      return f(std::integral_constant<size_t, 128>{});
    case 256:
      return f(std::integral_constant<size_t, 256>{});
    case 512:
      return f(std::integral_constant<size_t, 512>{});
    default:
      return f(std::integral_constant<size_t, 0>{});
  }
}

// Copies a single tile from the source batch pointer to the destination tile
// pointer when no padding is needed. Template-specialized on compile-time row
// byte width to enable fully inlined SIMD load/store vector instructions.
template <size_t kRowBytes>
void CopyTileNoPadding(const uint8_t* src_batch_ptr, uint8_t* dst_tile_ptr,
                       int64_t tile_row, int64_t tile_col, int64_t tile_H,
                       int64_t tile_W, int64_t W, int64_t itemsize) {
  int64_t logical_col_start = tile_col * tile_W;
  int64_t row_stride = (kRowBytes > 0) ? kRowBytes : (tile_W * itemsize);
  for (int64_t r = 0; r < tile_H; ++r) {
    int64_t logical_row = tile_row * tile_H + r;
    const uint8_t* src_row_ptr =
        src_batch_ptr + (logical_row * W + logical_col_start) * itemsize;
    uint8_t* dst_row_ptr = dst_tile_ptr + r * row_stride;

    if constexpr (kRowBytes > 0) {
      std::memcpy(dst_row_ptr, src_row_ptr, kRowBytes);
    } else {
      std::memcpy(dst_row_ptr, src_row_ptr, row_stride);
    }
  }
}

// Copies a single tile from the source batch pointer to the destination tile
// pointer, handling padding if the tile is partially or fully out of bounds.
// Out-of-bounds elements in the destination tile are zero-initialized.
void CopyTileWithPadding(const uint8_t* src_batch_ptr, uint8_t* dst_tile_ptr,
                         int64_t tile_row, int64_t tile_col, int64_t tile_H,
                         int64_t tile_W, int64_t H, int64_t W, int64_t itemsize,
                         int64_t tile_size_bytes) {
  int64_t logical_col_start = tile_col * tile_W;
  int64_t valid_elements = std::min(tile_W, W - logical_col_start);
  if (valid_elements <= 0) {
    std::memset(dst_tile_ptr, 0, tile_size_bytes);
    return;
  }

  for (int64_t r = 0; r < tile_H; ++r) {
    int64_t logical_row = tile_row * tile_H + r;
    uint8_t* dst_row_ptr = dst_tile_ptr + (r * tile_W) * itemsize;
    if (logical_row >= H) {
      std::memset(dst_row_ptr, 0, tile_W * itemsize);
      continue;
    }
    const uint8_t* src_row_ptr =
        src_batch_ptr + (logical_row * W + logical_col_start) * itemsize;

    std::memcpy(dst_row_ptr, src_row_ptr, valid_elements * itemsize);
    if (valid_elements < tile_W) {
      std::memset(dst_row_ptr + valid_elements * itemsize, 0,
                  (tile_W - valid_elements) * itemsize);
    }
  }
}

template <size_t kRowBytes>
void DetileSingleTileNoPadding(const uint8_t* src_tile_ptr,
                               uint8_t* dst_batch_ptr, int64_t tile_row,
                               int64_t tile_col, int64_t tile_H, int64_t tile_W,
                               int64_t W, int64_t itemsize) {
  int64_t logical_col_start = tile_col * tile_W;
  int64_t row_stride = (kRowBytes > 0) ? kRowBytes : (tile_W * itemsize);
  for (int64_t r = 0; r < tile_H; ++r) {
    int64_t logical_row = tile_row * tile_H + r;
    uint8_t* dst_row_ptr =
        dst_batch_ptr + (logical_row * W + logical_col_start) * itemsize;
    const uint8_t* src_row_ptr = src_tile_ptr + r * row_stride;

    if constexpr (kRowBytes > 0) {
      std::memcpy(dst_row_ptr, src_row_ptr, kRowBytes);
    } else {
      std::memcpy(dst_row_ptr, src_row_ptr, row_stride);
    }
  }
}

void DetileSingleTileWithPadding(const uint8_t* src_tile_ptr,
                                 uint8_t* dst_batch_ptr, int64_t tile_row,
                                 int64_t tile_col, int64_t tile_H,
                                 int64_t tile_W, int64_t H, int64_t W,
                                 int64_t itemsize) {
  int64_t logical_col_start = tile_col * tile_W;
  int64_t valid_elements = std::min(tile_W, W - logical_col_start);
  if (valid_elements <= 0) {
    return;
  }

  for (int64_t r = 0; r < tile_H; ++r) {
    int64_t logical_row = tile_row * tile_H + r;
    if (logical_row >= H) {
      continue;
    }
    uint8_t* dst_row_ptr =
        dst_batch_ptr + (logical_row * W + logical_col_start) * itemsize;
    const uint8_t* src_row_ptr = src_tile_ptr + (r * tile_W) * itemsize;

    std::memcpy(dst_row_ptr, src_row_ptr, valid_elements * itemsize);
  }
}

// Tiles a buffer using an optimized path for standard row-major layouts.
// It avoids global zero-initialization of the destination buffer to prevent
// CPU cache pollution, instead zeroing padding elements locally per tile.
absl::Status TileBufferNDOptimized(const uint8_t* src_linear,
                                   uint8_t* dst_tiled, const xla::Shape& shape,
                                   const xla::Layout& layout) {
  const int R = shape.dimensions().size();
  int64_t H = (R == 1) ? 1 : shape.dimensions(layout.minor_to_major(1));
  int64_t W = shape.dimensions(layout.minor_to_major(0));
  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  const xla::Tile& tile = layout.tiles(0);
  int64_t tile_H = tile.dimension(0);
  int64_t tile_W = tile.dimension(1);

  int64_t num_tiles_0 = xla::CeilOfRatio(H, tile_H);
  int64_t num_tiles_1 = xla::CeilOfRatio(W, tile_W);
  int64_t tile_size_bytes = tile_H * tile_W * itemsize;

  int64_t batch_size = 1;
  for (int i = 2; i < R; ++i) {
    batch_size *= shape.dimensions(layout.minor_to_major(i));
  }

  int64_t matrix_size_bytes = H * W * itemsize;
  int64_t tiled_matrix_size_bytes = num_tiles_0 * num_tiles_1 * tile_size_bytes;

  bool has_padding = (H % tile_H != 0) || (W % tile_W != 0);

  // Fast-path: When the tiled layout is byte-for-byte identical to the linear
  // layout (e.g. single column of tiles W == tile_W with no vertical padding
  // H % tile_H == 0, or 1D tensor with tile_H == 1 and no padding).
  if (!has_padding && (W == tile_W || (H == 1 && tile_H == 1))) {
    std::memcpy(dst_tiled, src_linear, batch_size * matrix_size_bytes);
    return absl::OkStatus();
  }

  // Fast-path for 1D tensors (H == 1) with arbitrary tile dimensions.
  if (H == 1) {
    for (int64_t b = 0; b < batch_size; ++b) {
      const uint8_t* src_batch_ptr = src_linear + b * matrix_size_bytes;
      uint8_t* dst_batch_ptr = dst_tiled + b * tiled_matrix_size_bytes;
      for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
        int64_t logical_col_start = tile_col * tile_W;
        int64_t valid_elements = std::min(tile_W, W - logical_col_start);
        uint8_t* dst_tile_ptr = dst_batch_ptr + tile_col * tile_size_bytes;
        if (valid_elements > 0) {
          std::memcpy(dst_tile_ptr,
                      src_batch_ptr + logical_col_start * itemsize,
                      valid_elements * itemsize);
          if (valid_elements < tile_W) {
            std::memset(dst_tile_ptr + valid_elements * itemsize, 0,
                        (tile_W - valid_elements) * itemsize);
          }
        } else {
          std::memset(dst_tile_ptr, 0, tile_W * itemsize);
        }
        if (tile_H > 1) {
          std::memset(dst_tile_ptr + tile_W * itemsize, 0,
                      (tile_H - 1) * tile_W * itemsize);
        }
      }
    }
    return absl::OkStatus();
  }

  int64_t total_tasks = batch_size * num_tiles_0;
  int64_t total_bytes = batch_size * H * W * itemsize;
  int64_t row_bytes = tile_W * itemsize;

  if (total_bytes < kParallelizationThresholdBytes || total_tasks <= 1) {
    DispatchByRowBytes(row_bytes, [&](auto kRowBytesTag) {
      constexpr size_t kRowBytes = decltype(kRowBytesTag)::value;
      if (!has_padding) {
        for (int64_t b = 0; b < batch_size; ++b) {
          const uint8_t* src_batch_ptr = src_linear + b * matrix_size_bytes;
          uint8_t* dst_batch_ptr = dst_tiled + b * tiled_matrix_size_bytes;
          for (int64_t tile_row = 0; tile_row < num_tiles_0; ++tile_row) {
            for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
              int64_t tile_index = tile_row * num_tiles_1 + tile_col;
              uint8_t* dst_tile_ptr =
                  dst_batch_ptr + tile_index * tile_size_bytes;
              CopyTileNoPadding<kRowBytes>(src_batch_ptr, dst_tile_ptr,
                                           tile_row, tile_col, tile_H, tile_W,
                                           W, itemsize);
            }
          }
        }
      } else {
        for (int64_t b = 0; b < batch_size; ++b) {
          const uint8_t* src_batch_ptr = src_linear + b * matrix_size_bytes;
          uint8_t* dst_batch_ptr = dst_tiled + b * tiled_matrix_size_bytes;
          for (int64_t tile_row = 0; tile_row < num_tiles_0; ++tile_row) {
            for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
              int64_t tile_index = tile_row * num_tiles_1 + tile_col;
              uint8_t* dst_tile_ptr =
                  dst_batch_ptr + tile_index * tile_size_bytes;
              CopyTileWithPadding(src_batch_ptr, dst_tile_ptr, tile_row,
                                  tile_col, tile_H, tile_W, H, W, itemsize,
                                  tile_size_bytes);
            }
          }
        }
      }
    });
  } else {
    DispatchByRowBytes(row_bytes, [&](auto kRowBytesTag) {
      constexpr size_t kRowBytes = decltype(kRowBytesTag)::value;
      auto run_task = [&](int64_t b, int64_t tile_row) {
        const uint8_t* src_batch_ptr = src_linear + b * matrix_size_bytes;
        uint8_t* dst_batch_ptr = dst_tiled + b * tiled_matrix_size_bytes;

        if (!has_padding) {
          for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
            int64_t tile_index = tile_row * num_tiles_1 + tile_col;
            uint8_t* dst_tile_ptr =
                dst_batch_ptr + tile_index * tile_size_bytes;
            CopyTileNoPadding<kRowBytes>(src_batch_ptr, dst_tile_ptr, tile_row,
                                         tile_col, tile_H, tile_W, W, itemsize);
          }
        } else {
          for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
            int64_t tile_index = tile_row * num_tiles_1 + tile_col;
            uint8_t* dst_tile_ptr =
                dst_batch_ptr + tile_index * tile_size_bytes;
            CopyTileWithPadding(src_batch_ptr, dst_tile_ptr, tile_row, tile_col,
                                tile_H, tile_W, H, W, itemsize,
                                tile_size_bytes);
          }
        }
      };

      tpu_raiden::NumaThreadPool* pool = GetThreadPool();
      int64_t chunk_size = (total_tasks + kMaxNumThreads - 1) / kMaxNumThreads;
      int64_t num_chunks = (total_tasks + chunk_size - 1) / chunk_size;
      absl::BlockingCounter counter(num_chunks);

      for (int64_t t = 0; t < kMaxNumThreads; ++t) {
        int64_t begin = t * chunk_size;
        int64_t end = std::min(begin + chunk_size, total_tasks);
        if (begin >= end) break;
        pool->Schedule([&, begin, end]() {
          for (int64_t i = begin; i < end; ++i) {
            int64_t b = i / num_tiles_0;
            int64_t tile_row = i % num_tiles_0;
            run_task(b, tile_row);
          }
          counter.DecrementCount();
        });
      }
      counter.Wait();
    });
  }

  return absl::OkStatus();
}

absl::Status DetileBufferNDOptimized(const uint8_t* src_tiled,
                                     uint8_t* dst_linear,
                                     const xla::Shape& shape,
                                     const xla::Layout& layout) {
  const int R = shape.dimensions().size();
  int64_t H = (R == 1) ? 1 : shape.dimensions(layout.minor_to_major(1));
  int64_t W = shape.dimensions(layout.minor_to_major(0));
  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  const xla::Tile& tile = layout.tiles(0);
  int64_t tile_H = tile.dimension(0);
  int64_t tile_W = tile.dimension(1);

  int64_t num_tiles_0 = xla::CeilOfRatio(H, tile_H);
  int64_t num_tiles_1 = xla::CeilOfRatio(W, tile_W);
  int64_t tile_size_bytes = tile_H * tile_W * itemsize;

  int64_t batch_size = 1;
  for (int i = 2; i < R; ++i) {
    batch_size *= shape.dimensions(layout.minor_to_major(i));
  }

  int64_t matrix_size_bytes = H * W * itemsize;
  int64_t tiled_matrix_size_bytes = num_tiles_0 * num_tiles_1 * tile_size_bytes;

  bool has_padding = (H % tile_H != 0) || (W % tile_W != 0);

  // Fast-path: When the tiled layout is byte-for-byte identical to the linear
  // layout (e.g. single column of tiles W == tile_W with no vertical padding
  // H % tile_H == 0, or 1D tensor with tile_H == 1 and no padding).
  if (!has_padding && (W == tile_W || (H == 1 && tile_H == 1))) {
    std::memcpy(dst_linear, src_tiled, batch_size * matrix_size_bytes);
    return absl::OkStatus();
  }

  // Fast-path for 1D tensors (H == 1) with arbitrary tile dimensions.
  if (H == 1) {
    for (int64_t b = 0; b < batch_size; ++b) {
      const uint8_t* src_batch_ptr = src_tiled + b * tiled_matrix_size_bytes;
      uint8_t* dst_batch_ptr = dst_linear + b * matrix_size_bytes;
      for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
        int64_t logical_col_start = tile_col * tile_W;
        int64_t valid_elements = std::min(tile_W, W - logical_col_start);
        if (valid_elements <= 0) {
          break;
        }
        std::memcpy(dst_batch_ptr + logical_col_start * itemsize,
                    src_batch_ptr + tile_col * tile_size_bytes,
                    valid_elements * itemsize);
      }
    }
    return absl::OkStatus();
  }

  int64_t total_tasks = batch_size * num_tiles_0;
  int64_t total_bytes = batch_size * H * W * itemsize;
  int64_t row_bytes = tile_W * itemsize;

  if (total_bytes < kParallelizationThresholdBytes || total_tasks <= 1) {
    DispatchByRowBytes(row_bytes, [&](auto kRowBytesTag) {
      constexpr size_t kRowBytes = decltype(kRowBytesTag)::value;
      if (!has_padding) {
        for (int64_t b = 0; b < batch_size; ++b) {
          const uint8_t* src_batch_ptr =
              src_tiled + b * tiled_matrix_size_bytes;
          uint8_t* dst_batch_ptr = dst_linear + b * matrix_size_bytes;
          for (int64_t tile_row = 0; tile_row < num_tiles_0; ++tile_row) {
            for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
              int64_t tile_index = tile_row * num_tiles_1 + tile_col;
              const uint8_t* src_tile_ptr =
                  src_batch_ptr + tile_index * tile_size_bytes;
              DetileSingleTileNoPadding<kRowBytes>(
                  src_tile_ptr, dst_batch_ptr, tile_row, tile_col, tile_H,
                  tile_W, W, itemsize);
            }
          }
        }
      } else {
        for (int64_t b = 0; b < batch_size; ++b) {
          const uint8_t* src_batch_ptr =
              src_tiled + b * tiled_matrix_size_bytes;
          uint8_t* dst_batch_ptr = dst_linear + b * matrix_size_bytes;
          for (int64_t tile_row = 0; tile_row < num_tiles_0; ++tile_row) {
            for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
              int64_t tile_index = tile_row * num_tiles_1 + tile_col;
              const uint8_t* src_tile_ptr =
                  src_batch_ptr + tile_index * tile_size_bytes;
              DetileSingleTileWithPadding(
                  src_tile_ptr, dst_batch_ptr, tile_row, tile_col, tile_H,
                  tile_W, H, W, itemsize);
            }
          }
        }
      }
    });
  } else {
    DispatchByRowBytes(row_bytes, [&](auto kRowBytesTag) {
      constexpr size_t kRowBytes = decltype(kRowBytesTag)::value;
      auto run_detile_task = [&](int64_t b, int64_t tile_row) {
        const uint8_t* src_batch_ptr = src_tiled + b * tiled_matrix_size_bytes;
        uint8_t* dst_batch_ptr = dst_linear + b * matrix_size_bytes;

        if (!has_padding) {
          for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
            int64_t tile_index = tile_row * num_tiles_1 + tile_col;
            const uint8_t* src_tile_ptr =
                src_batch_ptr + tile_index * tile_size_bytes;
            DetileSingleTileNoPadding<kRowBytes>(
                src_tile_ptr, dst_batch_ptr, tile_row, tile_col, tile_H,
                tile_W, W, itemsize);
          }
        } else {
          for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
            int64_t tile_index = tile_row * num_tiles_1 + tile_col;
            const uint8_t* src_tile_ptr =
                src_batch_ptr + tile_index * tile_size_bytes;
            DetileSingleTileWithPadding(
                src_tile_ptr, dst_batch_ptr, tile_row, tile_col, tile_H,
                tile_W, H, W, itemsize);
          }
        }
      };

      tpu_raiden::NumaThreadPool* pool = GetThreadPool();
      int64_t chunk_size = (total_tasks + kMaxNumThreads - 1) / kMaxNumThreads;
      int64_t num_chunks = (total_tasks + chunk_size - 1) / chunk_size;
      absl::BlockingCounter counter(num_chunks);

      for (int64_t t = 0; t < kMaxNumThreads; ++t) {
        int64_t begin = t * chunk_size;
        int64_t end = std::min(begin + chunk_size, total_tasks);
        if (begin >= end) break;
        pool->Schedule([&, begin, end]() {
          for (int64_t i = begin; i < end; ++i) {
            int64_t b = i / num_tiles_0;
            int64_t tile_row = i % num_tiles_0;
            run_detile_task(b, tile_row);
          }
          counter.DecrementCount();
        });
      }
      counter.Wait();
    });
  }

  return absl::OkStatus();
}

}  // namespace

int64_t GetTiledBufferElements(const xla::Shape& shape) {
  const int num_dims = shape.dimensions().size();
  if (num_dims == 0) {
    return 1;
  }

  std::vector<int64_t> current_shape;
  current_shape.reserve(std::max(num_dims, 2));
  if (num_dims == 1) {
    current_shape.push_back(1);
    current_shape.push_back(shape.dimensions(0));
  } else {
    for (int64_t i = num_dims - 1; i >= 0; --i) {
      int64_t logical_dim = shape.layout().minor_to_major(i);
      current_shape.push_back(shape.dimensions(logical_dim));
    }
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

absl::Status DetileBuffer(const uint8_t* src_tiled, uint8_t* dst_linear,
                          const xla::Shape& shape, const xla::Layout& layout) {
  if (layout.tiles().empty()) {
    std::memcpy(dst_linear, src_tiled, xla::ShapeUtil::ByteSizeOf(shape));
    return absl::OkStatus();
  }

  if (IsStandardRowMajorTiled(shape, layout)) {
    return DetileBufferNDOptimized(src_tiled, dst_linear, shape, layout);
  }

  // Fast byte-exact path for the real bf16 weight layout
  // {Tile(8,128), Tile(2,1)}, which otherwise hits the scalar fallback below.
  // Auto-dispatched (no env gate); any non-matching layout falls through.
  if (IsBf16SubTiled_8_128_2_1(shape, layout)) {
    DetileBufferBf16SubTiled(src_tiled, dst_linear, shape.dimensions(0),
                             shape.dimensions(1));
    return absl::OkStatus();
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
    std::memcpy(dst_tiled, src_linear, xla::ShapeUtil::ByteSizeOf(shape));
    return absl::OkStatus();
  }

  if (IsStandardRowMajorTiled(shape, layout)) {
    return TileBufferNDOptimized(src_linear, dst_tiled, shape, layout);
  }

  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  int64_t total_physical_elements = GetTiledBufferElements(shape);
  int64_t logical_elements = 1;
  for (int64_t dim : shape.dimensions()) {
    logical_elements *= dim;
  }
  if (total_physical_elements > logical_elements) {
    std::memset(dst_tiled, 0, total_physical_elements * itemsize);
  }

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
