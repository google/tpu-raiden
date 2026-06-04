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

#include "frameworks/torch/raiden_transfer_engine.h"

#include <memory>
#include <vector>

#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "torch/torch.h"
#include "frameworks/torch/torch_tpu_utils_mock.h"

namespace tpu_raiden::torch {
namespace {

class RaidenTransferEngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(client_,
                            xla::GetXlaPjrtCpuClient(xla::CpuClientOptions()));
  }

  std::unique_ptr<xla::PjRtClient> client_;
};

TEST_F(RaidenTransferEngineTest, ConstructorAndRegisterSucceeds) {
  // Create a real CPU PJRT buffer
  TF_ASSERT_OK_AND_ASSIGN(
      xla::PjRtMemorySpace * memory_space,
      client_->addressable_devices()[0]->default_memory_space());
  std::vector<float> data(8 * 1024, 1.0f);
  TF_ASSERT_OK_AND_ASSIGN(
      auto pjrt_buffer, client_->BufferFromHostBuffer(
                            data.data(), xla::F32, {8, 1024},
                            /*byte_strides=*/std::nullopt,
                            xla::PjRtClient::HostBufferSemantics::
                                kImmutableUntilTransferCompletes,
                            /*on_done_with_host_buffer=*/nullptr, memory_space,
                            /*device_layout=*/nullptr));

  // Create a CPU PyTorch tensor.
  at::Tensor tensor = ::torch::zeros({8, 1024}, ::torch::kFloat32);

  // Register the mapping
  RegisterMockTensor(tensor, pjrt_buffer.get());

  std::vector<at::Tensor> kv_caches = {tensor};

  // Construct RaidenTransferEngine
  RaidenTransferEngine engine(kv_caches, /*tp_rank=*/0,
                              /*local_control_port=*/0, /*max_blocks=*/2,
                              /*num_slots=*/2, /*timeout_s=*/10.0,
                              /*unsafe_skip_buffer_lock=*/true);

  EXPECT_TRUE(engine.UsesPreparedTpuBuffers());

  // Test RegisterKvCache
  std::vector<int64_t> region_ids = engine.RegisterKvCache(kv_caches);
  EXPECT_EQ(region_ids.size(), 1);
  EXPECT_EQ(region_ids[0], 0);
}

}  // namespace
}  // namespace tpu_raiden::torch
