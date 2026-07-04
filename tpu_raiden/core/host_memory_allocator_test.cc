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

#include "tpu_raiden/core/host_memory_allocator.h"

#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <string>

#include "absl/strings/str_format.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tpu_raiden/core/tpu_pjrt_manager.h"

namespace tpu_raiden {
namespace {

TEST(HostMemoryAllocatorTest, FallbackAllocationWithoutClient) {
  TF_ASSERT_OK_AND_ASSIGN(auto allocator, HostMemoryAllocator::Create(nullptr));

  // Allocate 1024 bytes
  TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc,
                          allocator->Allocate(1024));
  EXPECT_NE(alloc.ptr, nullptr);
  EXPECT_EQ(alloc.size, 1024);
  EXPECT_NE(alloc.owner, nullptr);

  // Verify 64-byte alignment
  EXPECT_EQ(reinterpret_cast<uintptr_t>(alloc.ptr) % 64, 0);

  // Write and read back to verify it's usable memory
  std::memset(alloc.ptr, 0xAB, 1024);
  for (size_t i = 0; i < 1024; ++i) {
    EXPECT_EQ(alloc.ptr[i], 0xAB);
  }

  // Zero allocation should work and return a nullptr or empty alloc safely
  TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation zero_alloc,
                          allocator->Allocate(0));
  EXPECT_EQ(zero_alloc.ptr, nullptr);
  EXPECT_EQ(zero_alloc.size, 0);
}

TEST(HostMemoryAllocatorTest, AllocationWithTpuClient) {
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * manager,
                          TpuPjrtManager::GetDefault());
  ASSERT_NE(manager->client(), nullptr);

  TF_ASSERT_OK_AND_ASSIGN(auto allocator,
                          HostMemoryAllocator::Create(manager->client()));

  // Allocate 4096 bytes
  TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc,
                          allocator->Allocate(4096));
  EXPECT_NE(alloc.ptr, nullptr);
  EXPECT_EQ(alloc.size, 4096);
  EXPECT_NE(alloc.owner, nullptr);

  // Verify 64-byte alignment
  EXPECT_EQ(reinterpret_cast<uintptr_t>(alloc.ptr) % 64, 0);

  // Write and read back to verify it's usable memory
  std::memset(alloc.ptr, 0xCD, 4096);
  for (size_t i = 0; i < 4096; ++i) {
    EXPECT_EQ(alloc.ptr[i], 0xCD);
  }
}

TEST(HostMemoryAllocatorTest, SharedMemoryColdAndWarmBoot) {
  std::string shm_key = "/test_raiden_shm_key_" + std::to_string(getpid());

  SharedMemoryHeader schema1 = {};
  schema1.magic = 0x52414944454E;
  schema1.version = 1;
  absl::SNPrintF(schema1.model_uid, sizeof(schema1.model_uid), "test_model_v1");
  schema1.num_blocks = 128;
  schema1.block_size = 4096;
  schema1.num_heads = 32;
  schema1.head_dim = 128;
  schema1.itemsize = 2;

  {
    shm_unlink(shm_key.c_str());

    TF_ASSERT_OK_AND_ASSIGN(
        auto allocator1,
        SharedMemoryHostMemoryAllocator::Create(nullptr, shm_key, schema1));

    TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc1,
                            allocator1->Allocate(1024));
    EXPECT_NE(alloc1.ptr, nullptr);
    EXPECT_EQ(alloc1.size, 1024);

    std::memset(alloc1.ptr, 0x55, 1024);

    SharedMemoryHeader* header1 = reinterpret_cast<SharedMemoryHeader*>(
        static_cast<uint8_t*>(alloc1.ptr) - sizeof(SharedMemoryHeader));
    EXPECT_EQ(header1->reference_count, 1);
    EXPECT_EQ(header1->version, 1);
    EXPECT_STREQ(header1->model_uid, "test_model_v1");

    {
      TF_ASSERT_OK_AND_ASSIGN(
          auto allocator2,
          SharedMemoryHostMemoryAllocator::Create(nullptr, shm_key, schema1));

      TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc2,
                              allocator2->Allocate(1024));
      EXPECT_NE(alloc2.ptr, nullptr);
      EXPECT_EQ(alloc2.size, 1024);

      for (size_t i = 0; i < 1024; ++i) {
        ASSERT_EQ(alloc2.ptr[i], 0x55);
      }

      SharedMemoryHeader* header2 = reinterpret_cast<SharedMemoryHeader*>(
          static_cast<uint8_t*>(alloc2.ptr) - sizeof(SharedMemoryHeader));
      EXPECT_EQ(header2->reference_count, 2);
    }

    EXPECT_EQ(header1->reference_count, 1);

    {
      SharedMemoryHeader schema2 = schema1;
      schema2.version = 2;
      absl::SNPrintF(schema2.model_uid, sizeof(schema2.model_uid),
                     "test_model_v2");

      TF_ASSERT_OK_AND_ASSIGN(
          auto allocator3,
          SharedMemoryHostMemoryAllocator::Create(nullptr, shm_key, schema2));

      TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc3,
                              allocator3->Allocate(1024));
      EXPECT_NE(alloc3.ptr, nullptr);

      SharedMemoryHeader* header3 = reinterpret_cast<SharedMemoryHeader*>(
          static_cast<uint8_t*>(alloc3.ptr) - sizeof(SharedMemoryHeader));
      EXPECT_EQ(header3->reference_count, 1);
      EXPECT_EQ(header3->version, 2);
      EXPECT_STREQ(header3->model_uid, "test_model_v2");

      for (size_t i = 0; i < 1024; ++i) {
        ASSERT_EQ(alloc3.ptr[i], 0);
      }
    }
  }

  shm_unlink(shm_key.c_str());
}

}  // namespace
}  // namespace tpu_raiden
