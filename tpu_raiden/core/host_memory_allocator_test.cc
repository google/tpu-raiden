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
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>


#include <cstdint>
#include <cstring>


#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tpu_raiden/core/tpu_pjrt_manager.h"

namespace tpu_raiden {
namespace {

using ::absl_testing::IsOk;

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
bool SendFd(int sock, int fd) {
  struct msghdr msg = {0};
  char buf[1] = {0};
  struct iovec iov = {.iov_base = buf, .iov_len = 1};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;

  char cmsg_buf[CMSG_SPACE(sizeof(int))];
  msg.msg_control = cmsg_buf;
  msg.msg_controllen = sizeof(cmsg_buf);

  struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  *reinterpret_cast<int*>(CMSG_DATA(cmsg)) = fd;

  return sendmsg(sock, &msg, 0) > 0;
}

int ReceiveFd(int sock) {
  struct msghdr msg = {0};
  char buf[1];
  struct iovec iov = {.iov_base = buf, .iov_len = 1};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;

  char cmsg_buf[CMSG_SPACE(sizeof(int))];
  msg.msg_control = cmsg_buf;
  msg.msg_controllen = sizeof(cmsg_buf);

  if (recvmsg(sock, &msg, 0) < 0) {
    return -1;
  }

  struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
  if (!cmsg || cmsg->cmsg_type != SCM_RIGHTS) {
    return -1;
  }

  return *reinterpret_cast<int*>(CMSG_DATA(cmsg));
}

TEST(HostMemoryAllocatorTest, CrossProcessSharedMemory) {
  TF_ASSERT_OK_AND_ASSIGN(TpuPjrtManager * manager,
                          TpuPjrtManager::GetDefault());
  ASSERT_NE(manager->client(), nullptr);

  TF_ASSERT_OK_AND_ASSIGN(auto allocator,
                          HostMemoryAllocator::Create(manager->client()));

  // Allocate shared memory
  const size_t kSize = 1024;
  TF_ASSERT_OK_AND_ASSIGN(HostBufferAllocation alloc,
                          allocator->AllocateShared(kSize));
  EXPECT_NE(alloc.ptr, nullptr);
  EXPECT_EQ(alloc.size, kSize);
  EXPECT_NE(alloc.owner, nullptr);
  EXPECT_NE(alloc.fd, -1);

  // Set up Unix Domain Socket pair for IPC
  int sv[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);

  // Fork process
  pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    // Child process (Process B)
    close(sv[0]);  // Close parent's end

    // To simulate fd sharing via IPC, we close the inherited FD.
    // Process B must receive it via UDS.
    int inherited_fd = alloc.fd;
    close(inherited_fd);

    // Receive FD over UDS
    int received_fd = ReceiveFd(sv[1]);
    if (received_fd < 0) {
      _exit(1);
    }


    // Map the received fd
    void* child_ptr = mmap(nullptr, kSize, PROT_READ | PROT_WRITE,
                             MAP_SHARED, received_fd, 0);

    if (child_ptr == MAP_FAILED) {
      close(received_fd);
      _exit(4);
    }

    // Phase 1: Verify Parent's write
    uint8_t* u8_child_ptr = static_cast<uint8_t*>(child_ptr);
    for (size_t i = 0; i < kSize; ++i) {
      if (u8_child_ptr[i] != static_cast<uint8_t>(i % 255)) {
        _exit(5);
      }
    }

    // Phase 2: Child writes
    for (size_t i = 0; i < kSize; ++i) {
      u8_child_ptr[i] = static_cast<uint8_t>((i + 1) % 255);
    }

    // Send Ack to parent
    char ack = 'K';
    if (write(sv[1], &ack, 1) != 1) {
      _exit(6);
    }


    munmap(child_ptr, kSize);
    close(received_fd);
    close(sv[1]);
    _exit(0);
  } else {
    // Parent process (Process A)
    close(sv[1]);  // Close child's end

    // Phase 1: Parent writes
    for (size_t i = 0; i < kSize; ++i) {
      alloc.ptr[i] = static_cast<uint8_t>(i % 255);
    }


    // Send FD over UDS
    EXPECT_TRUE(SendFd(sv[0], alloc.fd));


    // Wait for child to finish writing (Ack)
    char ack;
    EXPECT_EQ(read(sv[0], &ack, 1), 1);

    // Phase 2: Verify Child's write
    for (size_t i = 0; i < kSize; ++i) {
      EXPECT_EQ(alloc.ptr[i], static_cast<uint8_t>((i + 1) % 255));
    }

    // Wait for child
    int status;
    waitpid(pid, &status, 0);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);


    close(sv[0]);
  }
}

}  // namespace
}  // namespace tpu_raiden
