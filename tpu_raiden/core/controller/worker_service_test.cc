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

#include <stdint.h>

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "tpu_raiden/core/controller/test_util.h"
#include "tpu_raiden/core/controller/worker_service_client.h"
#include "tpu_raiden/core/controller/worker_service_impl.h"
#include "tpu_raiden/proto/worker_service.pb.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace controller {
namespace {

using ::testing::HasSubstr;

class WorkerServiceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_server_ = CreateTestServer();
    unit_.set_job_name("test_job");
    unit_.set_job_replica_id("0");
    unit_.set_data_name("test_data");
  }

  rpc::RaidenIdProto unit_;
  std::unique_ptr<TestServer> test_server_;
};

TEST_F(WorkerServiceTest, CreateAndDeleteBuffersSuccess) {
  proto::CreateBuffersRequest create_req;
  *create_req.mutable_unit() = unit_;
  auto* spec1 = create_req.add_buffers();
  spec1->set_num_shards(2);
  spec1->set_size_bytes(1024);
  auto* spec2 = create_req.add_buffers();
  spec2->set_num_shards(2);
  spec2->set_size_bytes(1024);

  auto create_resp_or = test_server_->client->CreateBuffers(create_req);
  ASSERT_TRUE(create_resp_or.ok());
  EXPECT_TRUE(create_resp_or->success());
  ASSERT_EQ(create_resp_or->buffers_size(), 2);
  EXPECT_EQ(test_server_->service->GetBufferCount(), 4);

  const auto& buf1 = create_resp_or->buffers(0);
  const auto& buf2 = create_resp_or->buffers(1);
  ASSERT_EQ(buf1.buffer_handles_size(), 2);
  ASSERT_EQ(buf2.buffer_handles_size(), 2);

  uint64_t handle1 = buf1.buffer_handles(0).handle();
  uint64_t handle2 = buf1.buffer_handles(1).handle();
  EXPECT_NE(handle1, handle2);

  auto alloc_or = test_server_->service->GetBuffer(BufferHandle(handle1));
  ASSERT_TRUE(alloc_or.ok());
  EXPECT_EQ(alloc_or->size, 1024);
  EXPECT_NE(alloc_or->ptr, nullptr);

  proto::DeleteBuffersRequest delete_req;
  *delete_req.mutable_unit() = unit_;
  *delete_req.add_sharded_buffers() = buf1;
  *delete_req.add_sharded_buffers() = buf2;

  auto delete_resp_or = test_server_->client->DeleteBuffers(delete_req);
  ASSERT_TRUE(delete_resp_or.ok());
  EXPECT_TRUE(delete_resp_or->success());
  EXPECT_EQ(test_server_->service->GetBufferCount(), 0);
}

TEST_F(WorkerServiceTest, CreateBuffersWithInvalidSpecFails) {
  proto::CreateBuffersRequest create_req;
  *create_req.mutable_unit() = unit_;
  auto* spec = create_req.add_buffers();
  spec->set_num_shards(0);
  spec->set_size_bytes(512);

  auto create_resp_or = test_server_->client->CreateBuffers(create_req);
  ASSERT_TRUE(create_resp_or.ok());
  EXPECT_FALSE(create_resp_or->success());
  EXPECT_THAT(create_resp_or->message(), HasSubstr("must be positive"));
}

TEST_F(WorkerServiceTest, DeleteNonExistentBufferFails) {
  proto::DeleteBuffersRequest delete_req;
  *delete_req.mutable_unit() = unit_;
  auto* sharded_buf = delete_req.add_sharded_buffers();
  sharded_buf->add_buffer_handles()->set_handle(9999);

  auto delete_resp_or = test_server_->client->DeleteBuffers(delete_req);
  ASSERT_TRUE(delete_resp_or.ok());
  EXPECT_FALSE(delete_resp_or->success());
  EXPECT_THAT(delete_resp_or->message(), HasSubstr("not found"));
}

}  // namespace
}  // namespace controller
}  // namespace tpu_raiden
