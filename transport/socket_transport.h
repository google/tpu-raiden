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

#ifndef THIRD_PARTY_TPU_RAIDEN_TRANSPORT_SOCKET_TRANSPORT_H_
#define THIRD_PARTY_TPU_RAIDEN_TRANSPORT_SOCKET_TRANSPORT_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>  // NOLINT
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "third_party/peregrine/src/api/transport.h"
#include "third_party/peregrine/src/api/types.h"

namespace tpu_raiden {
namespace transport {

// Prototyped POSIX TCP socket implementation of peregrine::Transport interface.
// Spins up a dedicated background server thread listening on a local port to
// receive cross-host data transfer requests. Uses a custom binary packet
// protocol to push and pull payload byte arrays mapping remote virtual memory
// addresses.
class SocketTransport final : public peregrine::Transport {
 public:
  // Binary packet header layout for remote socket stream transmission.
  struct alignas(8) PacketHeader {
    peregrine::Op op;
    uint64_t remote_addr;
    uint64_t local_addr;
    uint64_t length;
  };

  // Constructs a SocketTransport listening on `local_port`.
  explicit SocketTransport(int local_port);

  // Destructor closing streams and joining background threads cleanly.
  ~SocketTransport() override;

  // Non-copyable and non-movable.
  SocketTransport(const SocketTransport&) = delete;
  SocketTransport& operator=(const SocketTransport&) = delete;

  // Posts an asynchronous transport request to communicate with `peer`.
  // Returns a process-unique handle to poll completion.
  absl::StatusOr<peregrine::Handle> Post(std::string_view peer,
                                    const peregrine::Request& request) override;

  // Queries the completion status of `handle`. Removes handle if completed.
  absl::StatusOr<peregrine::Status> Poll(peregrine::Handle handle) override;

  // Expose listening port.
  int local_port() const { return local_port_; }

 private:
  // Resolves peer endpoint string to IP and port. Returns cached socket or
  // connects.
  absl::StatusOr<int> GetOrCreateConnection(std::string_view peer)
      ABSL_LOCKS_EXCLUDED(conn_mu_);

  // Background listener loop accepting incoming connection sockets.
  void ListenerLoop();

  // Background worker loop processing incoming requests from an accepted socket
  // stream.
  void ConnectionWorker(int client_fd);

  // Dispatches socket write operation for Post.
  absl::Status DispatchWrite(int fd, const peregrine::Request& request);

  // Dispatches socket read request for Post.
  absl::Status DispatchReadRequest(int fd, const peregrine::Request& request);

  int local_port_;
  int server_fd_ = -1;
  std::atomic<bool> stopping_{false};

  absl::Mutex mu_;
  uint32_t handle_counter_ ABSL_GUARDED_BY(mu_) = 0;
  absl::flat_hash_map<peregrine::Handle, peregrine::Status> status_map_
      ABSL_GUARDED_BY(mu_);

  absl::Mutex conn_mu_;
  absl::flat_hash_map<std::string, int> connection_pool_
      ABSL_GUARDED_BY(conn_mu_);

  std::thread listener_thread_;
  std::vector<std::thread> worker_threads_;
};

}  // namespace transport
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TRANSPORT_SOCKET_TRANSPORT_H_
