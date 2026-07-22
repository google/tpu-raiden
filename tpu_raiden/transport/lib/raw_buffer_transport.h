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

#ifndef THIRD_PARTY_TPU_RAIDEN_TRANSPORT_LIB_RAW_BUFFER_TRANSPORT_H_
#define THIRD_PARTY_TPU_RAIDEN_TRANSPORT_LIB_RAW_BUFFER_TRANSPORT_H_

#include <sys/uio.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace tpu_raiden::transport::lib {

// Foundational delegate interface for RawBufferTransport to query base host
// memory.
class RawBufferTransportDelegate {
 public:
  virtual ~RawBufferTransportDelegate() = default;

  // Authoritative physical Host / pinned HBM base starting pointer.
  // The buffer_id parameter allows multidimensional indexing across
  // layers/buffers.
  virtual uint8_t* GetHostPointer(size_t buffer_id, size_t shard_idx) = 0;

  // Authoritative total byte capacity of the target shard staging buffer.
  virtual size_t GetHostSize(size_t buffer_id, size_t shard_idx) = 0;

  // Notification triggered upon verified data chunk arrival.
  virtual absl::Status OnDataReceived() { return absl::OkStatus(); }
};

// Standalone raw buffer POSIX TCP socket transport engine.
class RawBufferTransport {
 public:
  // Compact 32-byte binary packet header layout.
  struct alignas(8) PacketHeader {
    uint8_t op;     // 3=BytePull, 5=ByteSlicePush, 1,2,4,6=HigherLevelBlockOps
    uint8_t flags;  // Holds major_order or protocol flags
    uint16_t buffer_id;      // Multidimensional Buffer / Layer ID coordinate
    uint16_t reserved;       // Holds parallelism/expected chunks count
    uint16_t padding;        // Unused padding to align fields
    uint32_t remote_id;      // Remote block ID or linear memory offset
    uint32_t local_id;       // Local block ID or target shard index
    uint32_t count_or_size;  // Number of blocks or continuous payload bytes
    uint64_t uuid;           // Globally unique transaction routing ID
  };

  RawBufferTransport(RawBufferTransportDelegate* delegate, int local_port,
                     const std::vector<std::string>& local_ips = {});
  virtual ~RawBufferTransport();

  // Directly pushes an arbitrary continuous byte array into a specific offset
  // of a remote peer's buffer.
  absl::Status PushBuffer(absl::string_view peer, size_t buffer_id,
                          size_t dst_shard_idx, size_t dst_offset_bytes,
                          const uint8_t* data_ptr, size_t size_bytes);

  // Synchronously requests an arbitrary continuous byte slice from a remote
  // peer's staging memory.
  absl::Status PullBuffer(absl::string_view peer, size_t buffer_id,
                          size_t src_shard_idx, size_t src_offset_bytes,
                          size_t dst_shard_idx, size_t dst_offset_bytes,
                          size_t size_bytes);

  int local_port() const { return local_port_; }
  const std::string& bound_ip() const { return bound_ip_; }

 protected:
  virtual absl::StatusOr<int> BorrowConnection(absl::string_view peer,
                                               absl::string_view local_ip = "");
  virtual void ReturnConnection(bool ok, int fd, absl::string_view peer,
                                absl::string_view local_ip = "");
  void ClosePooledConnections();

  virtual absl::Status ProcessSingleRequest(int client_fd);
  virtual absl::Status HandleCustomRequest(int client_fd,
                                           const PacketHeader& header);

  void ConnectionWorker(int client_fd);
  void ListenerLoop();

  RawBufferTransportDelegate* raw_delegate_;
  int local_port_;
  int server_fd_ = -1;
  std::string bound_ip_ = "127.0.0.1";
  std::vector<std::string> local_ips_;
  std::atomic<bool> stopping_{false};

  absl::Mutex mu_;
  absl::flat_hash_set<int> active_client_fds_ ABSL_GUARDED_BY(mu_);

  absl::Mutex pool_mu_;
  absl::flat_hash_map<std::string, std::vector<int>> conn_pool_
      ABSL_GUARDED_BY(pool_mu_);

  std::thread listener_thread_;
  std::vector<std::thread> worker_threads_;
};

}  // namespace tpu_raiden::transport::lib

#endif  // THIRD_PARTY_TPU_RAIDEN_TRANSPORT_LIB_RAW_BUFFER_TRANSPORT_H_
