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

#include "tpu_raiden/frameworks/jax/weight_synchronizer_ffi.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "absl/log/log.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tsl/platform/test.h"
#include "tpu_raiden/frameworks/jax/weight_synchronizer_ffi_internal.h"
#include "weight_sync/weight_synchronizer_base.h"

namespace tpu_raiden {
namespace weight_sync {
namespace {

// Keep ffi buffers alive during test scope
struct FfiBufferFixture {
  XLA_FFI_Buffer ffi_buf;
  std::vector<int64_t> dims;

  FfiBufferFixture(XLA_FFI_DataType dtype, void* data,
                   std::vector<int64_t> dimensions)
      : dims(dimensions) {
    ffi_buf.struct_size = sizeof(XLA_FFI_Buffer);
    ffi_buf.extension_start = nullptr;
    ffi_buf.dtype = dtype;
    ffi_buf.data = data;
    ffi_buf.rank = dims.size();
    ffi_buf.dims = dims.data();
  }

  xla::ffi::AnyBuffer AsAnyBuffer() { return xla::ffi::AnyBuffer(&ffi_buf); }
};

class WeightSynchronizerFfiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Ensure Host platform is registered and active
    auto platform = stream_executor::PlatformManager::PlatformWithName("Host");
    ASSERT_TRUE(platform.ok())
        << "Host Platform must be available on CPU sandboxes";
  }

  void TearDown() override {
    // Clean up global registry between tests
    for (int i = 0; i < 32; ++i) {
      if (g_weight_synchronizers[i] != nullptr) {
        delete g_weight_synchronizers[i];
        g_weight_synchronizers[i] = nullptr;
      }
      g_streams[i].reset();
    }
  }
};

enum class FfiType { kInit, kInitAndD2h };

class WeightSynchronizerFfiParamTest : public WeightSynchronizerFfiTest,
                                       public ::testing::WithParamInterface<FfiType> {
 protected:
  xla::ffi::Error CallInit(xla::ffi::AnyBuffer x, xla::ffi::AnyBuffer shard_idx_buf,
                            int64_t slice_byte_size, int32_t local_port,
                            int32_t parallelism, int32_t num_layers,
                            xla::ffi::Result<xla::ffi::AnyBuffer> out) {
    if (GetParam() == FfiType::kInit) {
      return TriggerWeightSynchronizerInitImpl(x, shard_idx_buf, slice_byte_size,
                                                local_port, parallelism, num_layers, out);
    } else {
      return TriggerWeightSynchronizerInitAndD2hImpl(x, shard_idx_buf, slice_byte_size,
                                                      local_port, parallelism, num_layers, out);
    }
  }
};

TEST_P(WeightSynchronizerFfiParamTest, TriggerWSInitSucceeds) {
  int32_t shard_idx = 0;
  FfiBufferFixture shard_idx_fixture(XLA_FFI_DataType_S32, &shard_idx, {1});

  std::vector<int32_t> anchor(256, 0);
  FfiBufferFixture anchor_fixture(XLA_FFI_DataType_S32, anchor.data(), {256});

  xla::ffi::AnyBuffer x = anchor_fixture.AsAnyBuffer();
  xla::ffi::AnyBuffer shard_idx_buf = shard_idx_fixture.AsAnyBuffer();

  int64_t slice_byte_size = 1024;
  int32_t local_port = 0;  // Allocate dynamic free port
  int32_t parallelism = 1;
  int32_t num_layers = 1;

  // Output buffer for IP (16 bytes) and port (4 bytes) -> 20 bytes -> 5 int32
  std::vector<int32_t> out_data(5, 0);
  FfiBufferFixture out_fixture(XLA_FFI_DataType_S32, out_data.data(), {5});
  xla::ffi::Result<xla::ffi::AnyBuffer> out = out_fixture.AsAnyBuffer();

  xla::ffi::Error err = CallInit(
      x, shard_idx_buf, slice_byte_size, local_port, parallelism, num_layers,
      out);

  EXPECT_TRUE(err.success()) << "WS Init failed: " << err.message();
  EXPECT_NE(g_weight_synchronizers[0], nullptr);
  EXPECT_NE(g_streams[0], nullptr);

  // Verify output contains some IP (can be all zeros if no external IP, but
  // port should be > 0)
  int32_t port = out_data[4];
  EXPECT_GT(port, 0) << "Assigned port should be positive";
  VLOG(1) << "Assigned port: " << port;
}

TEST_P(WeightSynchronizerFfiParamTest, TriggerExecuteReshardingDMAOrchestration) {
  int64_t slice_byte_size = 1024;
  int32_t parallelism = 1;
  int32_t num_layers = 1;

  // 1. Initialize Source Synchronizer (Shard 0)
  int32_t shard_idx_0 = 0;
  FfiBufferFixture shard_idx_0_fixture(XLA_FFI_DataType_S32, &shard_idx_0, {1});
  std::vector<int32_t> anchor_0(256, 0);
  FfiBufferFixture anchor_0_fixture(XLA_FFI_DataType_S32, anchor_0.data(),
                                    {256});
  std::vector<int32_t> out_data_0(5, 0);
  FfiBufferFixture out_0_fixture(XLA_FFI_DataType_S32, out_data_0.data(), {5});

  xla::ffi::Error init_err_0 = CallInit(
      anchor_0_fixture.AsAnyBuffer(), shard_idx_0_fixture.AsAnyBuffer(),
      slice_byte_size, /*local_port=*/0, parallelism, num_layers,
      out_0_fixture.AsAnyBuffer());
  ASSERT_TRUE(init_err_0.success()) << init_err_0.message();

  int32_t port_0 = out_data_0[4];
  ASSERT_GT(port_0, 0);

  // 2. Initialize Destination Synchronizer (Shard 1)
  int32_t shard_idx_1 = 1;
  FfiBufferFixture shard_idx_1_fixture(XLA_FFI_DataType_S32, &shard_idx_1, {1});
  std::vector<int32_t> anchor_1(256, 0);
  FfiBufferFixture anchor_1_fixture(XLA_FFI_DataType_S32, anchor_1.data(),
                                    {256});
  std::vector<int32_t> out_data_1(5, 0);
  FfiBufferFixture out_1_fixture(XLA_FFI_DataType_S32, out_data_1.data(), {5});

  xla::ffi::Error init_err_1 = CallInit(
      anchor_1_fixture.AsAnyBuffer(), shard_idx_1_fixture.AsAnyBuffer(),
      slice_byte_size, /*local_port=*/0, parallelism, num_layers,
      out_1_fixture.AsAnyBuffer());
  ASSERT_TRUE(init_err_1.success()) << init_err_1.message();

  // 3. Populate Source Host Buffer with test data
  // The host buffer size is slice_byte_size + scratchpad (256KB)
  // We write test data to the beginning of the host buffer of Shard 0
  uint8_t* h_base_0 =
      const_cast<uint8_t*>(g_weight_synchronizers[0]->GetHostBufferPtr(0, 0));
  ASSERT_NE(h_base_0, nullptr);
  for (size_t i = 0; i < 256; ++i) {
    h_base_0[i] = static_cast<uint8_t>(i);
  }

  // 4. Setup Resharding Parameters for Dest (Shard 1) to pull from Source
  // (Shard 0) We will pull 256 bytes from Shard 0's offset 0 to Shard 1's
  // offset 128.

  // src_ips: 4 * int32 for IPv6. We use the IP returned by Shard 0 init.
  FfiBufferFixture src_ips_fixture(XLA_FFI_DataType_S32, out_data_0.data(),
                                   {4});

  // src_ports: 1 * int32
  FfiBufferFixture src_ports_fixture(XLA_FFI_DataType_S32, &port_0, {1});

  // src_offsets: 1 * int32
  int32_t src_offset = 0;
  FfiBufferFixture src_offsets_fixture(XLA_FFI_DataType_S32, &src_offset, {1});

  // dst_offsets: 1 * int32
  int32_t dst_offset = 128;
  FfiBufferFixture dst_offsets_fixture(XLA_FFI_DataType_S32, &dst_offset, {1});

  // sizes: 1 * int32
  int32_t size = 256;
  FfiBufferFixture sizes_fixture(XLA_FFI_DataType_S32, &size, {1});

  // dst_shard_indices: 1 * int32 (we are Shard 1)
  int32_t dst_shard_idx = 1;
  FfiBufferFixture dst_shard_indices_fixture(XLA_FFI_DataType_S32,
                                             &dst_shard_idx, {1});

  // anchor/out buffer for JAX on Shard 1. It represents the on-device weights
  // buffer. Size should be at least dst_offset + size = 128 + 256 = 384 bytes.
  std::vector<uint8_t> device_buffer_data(512, 0);
  FfiBufferFixture device_buffer_fixture(XLA_FFI_DataType_U8,
                                         device_buffer_data.data(), {512});

  // 5. Execute Resharding on Shard 1
  xla::ffi::Error reshard_err = TriggerExecuteReshardingImpl(
      device_buffer_fixture.AsAnyBuffer(), shard_idx_1_fixture.AsAnyBuffer(),
      src_ips_fixture.AsAnyBuffer(), src_ports_fixture.AsAnyBuffer(),
      src_offsets_fixture.AsAnyBuffer(), dst_offsets_fixture.AsAnyBuffer(),
      sizes_fixture.AsAnyBuffer(), dst_shard_indices_fixture.AsAnyBuffer(),
      device_buffer_fixture.AsAnyBuffer());

  EXPECT_TRUE(reshard_err.success())
      << "Resharding failed: " << reshard_err.message();

  // 6. Verify Destination Device Buffer contains the pulled weights
  // The resharding handler should have pulled the weights from Shard 0's host
  // buffer to Shard 1's host buffer scratchpad, and then copied them to the
  // device buffer (device_buffer_data) at dst_offset.
  for (size_t i = 0; i < 256; ++i) {
    EXPECT_EQ(device_buffer_data[dst_offset + i], static_cast<uint8_t>(i))
        << "Mismatch at index " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(FfiTypeTests, WeightSynchronizerFfiParamTest,
                         ::testing::Values(FfiType::kInit, FfiType::kInitAndD2h));

}  // namespace
}  // namespace weight_sync
}  // namespace tpu_raiden
