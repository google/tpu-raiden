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

/* Copyright 2026 The TPU Raiden Authors. All Rights Reserved.
==============================================================================*/

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "core/raw_transfer_core.h"
#include "frameworks/jax/mock_nanobind.h"
#include "frameworks/jax/raw_transfer_internal.h"

namespace raiden {
namespace {

// Helper to construct a mock JAX Array for Host pointer (Destination in D2H,
// Source in H2D)
nb::object CreateMockHostArray(uint8_t* raw_ptr, size_t size) {
  nb::object arr;
  nb::list shards;
  nb::object shard;
  nb::object shard_data;

  nb::object unsafe_buffer_pointer_fn;
  unsafe_buffer_pointer_fn.set_callable(
      [raw_ptr]() { return nb::object(reinterpret_cast<size_t>(raw_ptr)); });
  shard_data.set_attr("unsafe_buffer_pointer", unsafe_buffer_pointer_fn);

  nb::object on_device_size_in_bytes_fn;
  on_device_size_in_bytes_fn.set_callable(
      [size]() { return nb::object(size); });
  shard_data.set_attr("on_device_size_in_bytes", on_device_size_in_bytes_fn);

  shard.set_attr("data", shard_data);
  shards.add_element(shard);
  arr.set_attr("addressable_shards", shards);
  return arr;
}

// Helper to construct a mock JAX Array for PJRT Buffer (Source in D2H,
// Destination in H2D)
nb::object CreateMockDeviceArray(xla::PjRtBuffer* pjrt_buffer) {
  nb::object arr;
  nb::list shards;
  nb::object shard;
  nb::object shard_data(reinterpret_cast<void*>(pjrt_buffer));

  shard.set_attr("data", shard_data);
  shards.add_element(shard);
  arr.set_attr("addressable_shards", shards);
  return arr;
}

class RawTransferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize Host PJRT Client
    TF_ASSERT_OK_AND_ASSIGN(client_,
                            xla::GetXlaPjrtCpuClient(xla::CpuClientOptions()));
  }

  std::unique_ptr<xla::PjRtClient> client_;
};

TEST_F(RawTransferTest, TestTransferD2HAsync) {
  constexpr size_t kSize = 1024;
  std::vector<uint8_t> src_data(kSize);
  for (size_t i = 0; i < kSize; ++i) {
    src_data[i] = static_cast<uint8_t>(i % 256);
  }

  TF_ASSERT_OK_AND_ASSIGN(
      xla::PjRtMemorySpace * memory_space,
      client_->addressable_devices()[0]->default_memory_space());

  // Create Host PjRtBuffer
  TF_ASSERT_OK_AND_ASSIGN(
      auto src_buffer,
      client_->BufferFromHostBuffer(
          src_data.data(), xla::U8, {static_cast<int64_t>(kSize)},
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::
              kImmutableUntilTransferCompletes,
          /*on_done_with_host_buffer=*/nullptr, memory_space,
          /*device_layout=*/nullptr));

  // Destination Host buffer
  std::vector<uint8_t> dst_data(kSize, 0);

  // Wrap in mock JAX arrays
  nb::object src_arr = CreateMockDeviceArray(src_buffer.get());
  nb::object dst_arr = CreateMockHostArray(dst_data.data(), kSize);

  // Trigger async transfer
  TF_ASSERT_OK_AND_ASSIGN(PjRtCopyFuture future,
                          transfer_d2h_async(src_arr, dst_arr));

  // Wait
  future.Await();

  // Verify data
  EXPECT_EQ(dst_data, src_data);
}

TEST_F(RawTransferTest, TestTransferH2DAsync) {
  constexpr size_t kSize = 1024;
  std::vector<uint8_t> src_data(kSize);
  for (size_t i = 0; i < kSize; ++i) {
    src_data[i] = static_cast<uint8_t>(i % 256);
  }

  TF_ASSERT_OK_AND_ASSIGN(
      xla::PjRtMemorySpace * memory_space,
      client_->addressable_devices()[0]->default_memory_space());

  // Create Destination Host PjRtBuffer (initially zero)
  std::vector<uint8_t> initial_dst_data(kSize, 0);
  TF_ASSERT_OK_AND_ASSIGN(
      auto dst_buffer,
      client_->BufferFromHostBuffer(
          initial_dst_data.data(), xla::U8, {static_cast<int64_t>(kSize)},
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::
              kImmutableUntilTransferCompletes,
          /*on_done_with_host_buffer=*/nullptr, memory_space,
          /*device_layout=*/nullptr));

  // Wrap in mock JAX arrays
  nb::object src_arr = CreateMockHostArray(src_data.data(), kSize);
  nb::object dst_arr = CreateMockDeviceArray(dst_buffer.get());

  // Trigger async transfer
  TF_ASSERT_OK_AND_ASSIGN(PjRtCopyFuture future,
                          transfer_h2d_async(src_arr, dst_arr));

  // Wait
  future.Await();

  // Verify data by reading back from PJRT buffer
  std::vector<uint8_t> dst_data(kSize, 0);
  ASSERT_OK(dst_buffer->CopyRawToHost(dst_data.data(), 0, kSize).Await());
  EXPECT_EQ(dst_data, src_data);
}

TEST_F(RawTransferTest, TestTransferD2HBatchAsync) {
  constexpr size_t kSize = 512;
  constexpr size_t kBatch = 3;

  std::vector<std::vector<uint8_t>> src_datas(kBatch,
                                              std::vector<uint8_t>(kSize));
  std::vector<std::unique_ptr<xla::PjRtBuffer>> src_buffers;
  src_buffers.reserve(kBatch);

  nb::list src_arrs;
  nb::list dst_arrs;

  std::vector<std::vector<uint8_t>> dst_datas(kBatch,
                                              std::vector<uint8_t>(kSize, 0));

  TF_ASSERT_OK_AND_ASSIGN(
      xla::PjRtMemorySpace * memory_space,
      client_->addressable_devices()[0]->default_memory_space());

  for (size_t b = 0; b < kBatch; ++b) {
    for (size_t i = 0; i < kSize; ++i) {
      src_datas[b][i] = static_cast<uint8_t>((b + i) % 256);
    }

    TF_ASSERT_OK_AND_ASSIGN(
        auto src_buffer,
        client_->BufferFromHostBuffer(
            src_datas[b].data(), xla::U8, {static_cast<int64_t>(kSize)},
            /*byte_strides=*/std::nullopt,
            xla::PjRtClient::HostBufferSemantics::
                kImmutableUntilTransferCompletes,
            /*on_done_with_host_buffer=*/nullptr, memory_space,
            /*device_layout=*/nullptr));

    src_arrs.add_element(CreateMockDeviceArray(src_buffer.get()));
    dst_arrs.add_element(CreateMockHostArray(dst_datas[b].data(), kSize));

    src_buffers.push_back(std::move(src_buffer));
  }

  // Trigger batch transfer
  TF_ASSERT_OK_AND_ASSIGN(PjRtCopyFuture future,
                          transfer_d2h_batch_async(src_arrs, dst_arrs));

  // Wait
  future.Await();

  // Verify all
  for (size_t b = 0; b < kBatch; ++b) {
    EXPECT_EQ(dst_datas[b], src_datas[b]);
  }
}

TEST_F(RawTransferTest, TestTransferH2DBatchAsync) {
  constexpr size_t kSize = 512;
  constexpr size_t kBatch = 3;

  std::vector<std::vector<uint8_t>> src_datas(kBatch,
                                              std::vector<uint8_t>(kSize));
  std::vector<std::unique_ptr<xla::PjRtBuffer>> dst_buffers;
  dst_buffers.reserve(kBatch);

  nb::list src_arrs;
  nb::list dst_arrs;

  std::vector<uint8_t> initial_dst_data(kSize, 0);

  TF_ASSERT_OK_AND_ASSIGN(
      xla::PjRtMemorySpace * memory_space,
      client_->addressable_devices()[0]->default_memory_space());

  for (size_t b = 0; b < kBatch; ++b) {
    for (size_t i = 0; i < kSize; ++i) {
      src_datas[b][i] = static_cast<uint8_t>((b + i) % 256);
    }

    TF_ASSERT_OK_AND_ASSIGN(
        auto dst_buffer,
        client_->BufferFromHostBuffer(
            initial_dst_data.data(), xla::U8, {static_cast<int64_t>(kSize)},
            /*byte_strides=*/std::nullopt,
            xla::PjRtClient::HostBufferSemantics::
                kImmutableUntilTransferCompletes,
            /*on_done_with_host_buffer=*/nullptr, memory_space,
            /*device_layout=*/nullptr));

    src_arrs.add_element(CreateMockHostArray(src_datas[b].data(), kSize));
    dst_arrs.add_element(CreateMockDeviceArray(dst_buffer.get()));

    dst_buffers.push_back(std::move(dst_buffer));
  }

  // Trigger batch transfer
  TF_ASSERT_OK_AND_ASSIGN(PjRtCopyFuture future,
                          transfer_h2d_batch_async(src_arrs, dst_arrs));

  // Wait
  future.Await();

  // Verify all by reading back
  for (size_t b = 0; b < kBatch; ++b) {
    std::vector<uint8_t> dst_data(kSize, 0);
    ASSERT_OK(dst_buffers[b]->CopyRawToHost(dst_data.data(), 0, kSize).Await());
    EXPECT_EQ(dst_data, src_datas[b]);
  }
}

}  // namespace
}  // namespace raiden
