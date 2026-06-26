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

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

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

TEST(HostMemoryAllocatorTest, FileWriteReadVerifyWithTpuClient) {
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * manager,
                          TpuPjrtManager::GetDefault());
  ASSERT_NE(manager->client(), nullptr);

  TF_ASSERT_OK_AND_ASSIGN(auto allocator,
                          HostMemoryAllocator::Create(manager->client()));

  // Allocate 10MB
  const size_t kSize = 10 * 1024 * 1024;
  TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc,
                          allocator->Allocate(kSize));
  EXPECT_NE(alloc.ptr, nullptr);
  EXPECT_EQ(alloc.size, kSize);
  EXPECT_NE(alloc.owner, nullptr);

  // Fill buffer with a pattern
  for (size_t i = 0; i < kSize; ++i) {
    alloc.ptr[i] = static_cast<uint8_t>(i % 256);
  }

  // Write to /tmp/
  std::string filename = "/tmp/host_memory_allocator_test_file_10mb.bin";
  {
    std::ofstream ofs(filename, std::ios::binary);
    ASSERT_TRUE(ofs.is_open());
    ofs.write(reinterpret_cast<const char*>(alloc.ptr), kSize);
    ASSERT_TRUE(ofs.good());
  }

  // Read back into a different buffer to verify
  std::vector<uint8_t> read_buffer(kSize);
  {
    std::ifstream ifs(filename, std::ios::binary);
    ASSERT_TRUE(ifs.is_open());
    ifs.read(reinterpret_cast<char*>(read_buffer.data()), kSize);
    ASSERT_TRUE(ifs.good());
  }

  // Validate contents
  EXPECT_EQ(std::memcmp(alloc.ptr, read_buffer.data(), kSize), 0);

  // Clean up
  std::remove(filename.c_str());
}

}  // namespace
}  // namespace tpu_raiden
