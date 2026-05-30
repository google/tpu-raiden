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

#include "core/tpu_pjrt_manager.h"

#include <memory>
#include <vector>

#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/statusor.h"

namespace tpu_raiden {
namespace {

using ::absl_testing::IsOk;

TEST(TpuPjrtManagerTest, BasicLifecycleAndBufferCreation) {
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * manager,
                          TpuPjrtManager::GetDefault());

  EXPECT_NE(manager->client(), nullptr);

  // Device
  xla::PjRtDevice* device = manager->GetDefaultDevice();
  EXPECT_NE(device, nullptr);

  // Create buffer from host
  std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      manager->BufferFromHost(host_data.data(), xla::F32, {4}));
  EXPECT_NE(buffer, nullptr);

  // Wait for ready
  ASSERT_THAT(buffer->GetReadyFuture().Await(), IsOk());

  // Read back and verify
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> literal,
                          buffer->ToLiteral().Await());

  auto read_data = literal->data<float>();
  ASSERT_EQ(read_data.size(), 4);
  EXPECT_EQ(read_data[0], 1.0f);
  EXPECT_EQ(read_data[1], 2.0f);
  EXPECT_EQ(read_data[2], 3.0f);
  EXPECT_EQ(read_data[3], 4.0f);
}

}  // namespace
}  // namespace tpu_raiden
