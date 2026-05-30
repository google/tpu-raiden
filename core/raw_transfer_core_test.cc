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

#include "core/raw_transfer_core.h"

#include "xla/tsl/platform/test.h"

#include <memory>
#include <vector>

#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/statusor.h"
#include "core/tpu_pjrt_manager.h"

namespace raiden {
namespace {

using ::absl_testing::IsOk;

TEST(RawTransferCoreTest, GetMajorSliceByteSizeFor1D) {
  TF_ASSERT_OK_AND_ASSIGN(tpu_raiden::TpuPjrtManager * manager,
                          tpu_raiden::TpuPjrtManager::GetDefault());

  std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      manager->BufferFromHost(host_data.data(), xla::F32, {4}));

  EXPECT_EQ(GetMajorSliceByteSize(buffer.get()), 4);
}

TEST(RawTransferCoreTest, GetMajorSliceByteSizeFor2D) {
  TF_ASSERT_OK_AND_ASSIGN(tpu_raiden::TpuPjrtManager * manager,
                          tpu_raiden::TpuPjrtManager::GetDefault());

  std::vector<float> host_data(128 * 256, 1.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      manager->BufferFromHost(host_data.data(), xla::F32, {128, 256}));

  EXPECT_EQ(GetMajorSliceByteSize(buffer.get()), 1024);
}

TEST(RawTransferCoreTest, AcquireHoldAndRawCopy) {
  TF_ASSERT_OK_AND_ASSIGN(tpu_raiden::TpuPjrtManager * manager,
                          tpu_raiden::TpuPjrtManager::GetDefault());

  // 1. Create an initial device buffer with some data
  std::vector<float> host_data = {1.1f, 2.2f, 3.3f, 4.4f};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      manager->BufferFromHost(host_data.data(), xla::F32, {4}));

  // Wait until ready to ensure it's fully on device
  ASSERT_THAT(buffer->GetReadyFuture().Await(), IsOk());

  // 2. Acquire hold using BufferHoldAndAlias
  TF_ASSERT_OK_AND_ASSIGN(BufferHoldAndAlias hold,
                          BufferHoldAndAlias::Acquire(buffer.get()));

  // 3. Perform raw copy Device-to-Host (Read)
  std::vector<float> read_data(4, 0.0f);
  xla::Future<> read_future = hold.CopyRawDeviceToHost(
      read_data.data(), /*device_offset=*/0, /*size=*/4 * sizeof(float));

  PjRtCopyFuture read_copy_future({read_future}, hold.c_hold, hold.common_hold);
  read_copy_future.Await();

  EXPECT_EQ(read_data[0], 1.1f);
  EXPECT_EQ(read_data[1], 2.2f);
  EXPECT_EQ(read_data[2], 3.3f);
  EXPECT_EQ(read_data[3], 4.4f);

  // 4. Perform raw copy Host-to-Device (Write)
  std::vector<float> write_data = {9.9f, 8.8f, 7.7f, 6.6f};
  xla::Future<> write_future = hold.CopyRawHostToDevice(
      write_data.data(), /*device_offset=*/0, /*size=*/4 * sizeof(float));

  PjRtCopyFuture write_copy_future({write_future}, hold.c_hold,
                                   hold.common_hold);
  write_copy_future.Await();

  // 5. Read back again using normal ToLiteralSync to verify it actually
  // modified the buffer!
  TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer->ToLiteral().Await());
  auto read_back = literal->data<float>();
  ASSERT_EQ(read_back.size(), 4);
  EXPECT_EQ(read_back[0], 9.9f);
  EXPECT_EQ(read_back[1], 8.8f);
  EXPECT_EQ(read_back[2], 7.7f);
  EXPECT_EQ(read_back[3], 6.6f);
}

}  // namespace
}  // namespace raiden
