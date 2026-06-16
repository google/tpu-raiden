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

#include "core/raw_transfer_impl.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "core/raw_transfer_core.h"
#include "core/tpu_pjrt_manager.h"

namespace raiden {
namespace {

using ::absl_testing::IsOk;

TEST(RawTransferImplTest, FullTransferD2HAndH2D) {
  TF_ASSERT_OK_AND_ASSIGN(tpu_raiden::TpuPjrtManager * manager,
                          tpu_raiden::TpuPjrtManager::GetDefault());

  // 1. Create initial device buffer
  std::vector<float> host_data = {1.1f, 2.2f, 3.3f, 4.4f};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      manager->BufferFromHost(host_data.data(), xla::F32, {4}));

  ASSERT_THAT(buffer->GetReadyFuture().Await(), IsOk());

  TF_ASSERT_OK_AND_ASSIGN(int64_t physical_size,
                          buffer->GetOnDeviceSizeInBytes());
  std::cout << "1D buffer logical size: " << 4 * sizeof(float)
            << ", physical size: " << physical_size << std::endl;

  // 2. D2H full transfer using physical size
  std::vector<uint8_t> dst_data(physical_size, 0);
  std::vector<uint8_t*> dst_ptrs = {dst_data.data()};
  std::vector<size_t> dst_sizes = {static_cast<size_t>(physical_size)};

  TF_ASSERT_OK_AND_ASSIGN(
      PjRtCopyFuture d2h_future,
      transfer_d2h_core({buffer.get()}, dst_ptrs, dst_sizes,
                        /*src_offsets_major_dim=*/{0},
                        /*dst_offsets_major_dim=*/{0},
                        /*copy_sizes_major_dim=*/{4}));  // major dim size is 4
  ASSERT_THAT(d2h_future.Await(), IsOk());

  // Verify logical data at the beginning (assuming no tiling for 1D)
  float* dst_floats = reinterpret_cast<float*>(dst_data.data());
  EXPECT_EQ(dst_floats[0], 1.1f);
  EXPECT_EQ(dst_floats[1], 2.2f);
  EXPECT_EQ(dst_floats[2], 3.3f);
  EXPECT_EQ(dst_floats[3], 4.4f);

  // 3. H2D full transfer
  // Prepare host buffer of physical size, with new data at the beginning
  std::vector<uint8_t> new_host_bytes(physical_size, 0);
  float* new_host_floats = reinterpret_cast<float*>(new_host_bytes.data());
  new_host_floats[0] = 9.9f;
  new_host_floats[1] = 8.8f;
  new_host_floats[2] = 7.7f;
  new_host_floats[3] = 6.6f;

  std::vector<const uint8_t*> src_ptrs = {new_host_bytes.data()};
  std::vector<size_t> src_sizes = {static_cast<size_t>(physical_size)};

  TF_ASSERT_OK_AND_ASSIGN(PjRtCopyFuture h2d_future,
                          transfer_h2d_core({buffer.get()}, src_ptrs, src_sizes,
                                            /*src_offsets_major_dim=*/{0},
                                            /*dst_offsets_major_dim=*/{0},
                                            /*copy_sizes_major_dim=*/{4}));
  ASSERT_THAT(h2d_future.Await(), IsOk());

  // Verify H2D worked using normal ToLiteralSync
  TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer->ToLiteral().Await());
  auto read_back = literal->data<float>();
  ASSERT_GE(read_back.size(), 4);
  EXPECT_EQ(read_back[0], 9.9f);
  EXPECT_EQ(read_back[1], 8.8f);
  EXPECT_EQ(read_back[2], 7.7f);
  EXPECT_EQ(read_back[3], 6.6f);
}

TEST(RawTransferImplTest, PartialTransferD2HAndH2D) {
  TF_ASSERT_OK_AND_ASSIGN(tpu_raiden::TpuPjrtManager * manager,
                          tpu_raiden::TpuPjrtManager::GetDefault());

  // Rank 3 shape: {2, 32, 32} of float.
  // Major dim size: 2.
  std::vector<int64_t> shape_dims = {2, 32, 32};
  int64_t elements_per_slice = 32 * 32;             // 1024
  int64_t total_elements = 2 * elements_per_slice;  // 2048

  // Buffer 1: initialized with 0, 1, 2, ...
  std::vector<float> host_data1(total_elements);
  for (int i = 0; i < total_elements; ++i) {
    host_data1[i] = static_cast<float>(i);
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer1,
      manager->BufferFromHost(host_data1.data(), xla::F32, shape_dims));

  // Buffer 2: initialized with -1.0f
  std::vector<float> host_data2(total_elements, -1.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer2,
      manager->BufferFromHost(host_data2.data(), xla::F32, shape_dims));

  ASSERT_THAT(buffer1->GetReadyFuture().Await(), IsOk());
  ASSERT_THAT(buffer2->GetReadyFuture().Await(), IsOk());

  int64_t slice_byte_size = GetMajorSliceByteSize(buffer1.get());
  std::cout << "Rank 3 buffer slice byte size: " << slice_byte_size
            << std::endl;

  // Allocate host holder of slice_byte_size
  std::vector<uint8_t> host_holder(slice_byte_size, 0);

  // 1. Partial D2H: Copy the second slice (index 1 of major dim) of buffer1 to
  // host_holder
  std::vector<uint8_t*> dst_ptrs = {host_holder.data()};
  std::vector<size_t> dst_sizes = {static_cast<size_t>(slice_byte_size)};

  TF_ASSERT_OK_AND_ASSIGN(
      PjRtCopyFuture d2h_future,
      transfer_d2h_core({buffer1.get()}, dst_ptrs, dst_sizes,
                        /*src_offsets_major_dim=*/{1},
                        /*dst_offsets_major_dim=*/{0},
                        /*copy_sizes_major_dim=*/{1}));
  ASSERT_THAT(d2h_future.Await(), IsOk());

  // 2. Partial H2D: Copy host_holder to the first slice (index 0 of major dim)
  // of buffer2
  std::vector<const uint8_t*> src_ptrs = {host_holder.data()};
  std::vector<size_t> src_sizes = {static_cast<size_t>(slice_byte_size)};

  TF_ASSERT_OK_AND_ASSIGN(
      PjRtCopyFuture h2d_future,
      transfer_h2d_core({buffer2.get()}, src_ptrs, src_sizes,
                        /*src_offsets_major_dim=*/{0},
                        /*dst_offsets_major_dim=*/{0},
                        /*copy_sizes_major_dim=*/{1}));
  ASSERT_THAT(h2d_future.Await(), IsOk());

  // 3. Verify buffer2 using normal ToLiteralSync
  TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer2->ToLiteral().Await());
  auto read_back = literal->data<float>();
  ASSERT_EQ(read_back.size(), total_elements);

  // First slice of buffer2 should now contain what was in second slice of
  // buffer1 (1024, 1025, ...)
  for (int i = 0; i < elements_per_slice; ++i) {
    EXPECT_EQ(read_back[i], static_cast<float>(elements_per_slice + i));
  }
  // Second slice of buffer2 should remain untouched (-1.0f)
  for (int i = elements_per_slice; i < total_elements; ++i) {
    EXPECT_EQ(read_back[i], -1.0f);
  }
}

}  // namespace
}  // namespace raiden
