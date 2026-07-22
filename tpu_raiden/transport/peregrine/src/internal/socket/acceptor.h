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

#ifndef THIRD_PARTY_PEREGRINE_SRC_INTERNAL_SOCKET_ACCEPTOR_H_
#define THIRD_PARTY_PEREGRINE_SRC_INTERNAL_SOCKET_ACCEPTOR_H_

#include <atomic>
#include <memory>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/endpoint.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_tcp.h"

namespace peregrine::internal {

// This class listens on a local endpoint, accepts incoming connections, and
// creates a new tcp socket for each connection.
// It is thread-compatible but not thread-safe.
class TcpAcceptor {
  using AcceptCallback = absl::AnyInvocable<void(std::unique_ptr<TcpSocket>)>;

 public:
  // Creates a tcp acceptor with a socket listening on the `local` endpoint.
  static std::unique_ptr<TcpAcceptor> Create(const Endpoint& local);

  // Returns the underlying tcp listen socket.
  const TcpSocket& Socket() const { return *listener_; }

  // Starts running the acceptor.
  void Start(AcceptCallback accept);

  // Stops the acceptor.
  void Stop();

 private:
  // Constructor with a valid tcp listen socket.
  explicit TcpAcceptor(std::unique_ptr<TcpSocket> socket)
      : stop_(false), listener_(std::move(socket)) {
    DCHECK(invariant());
  }

  // Returns true iff the acceptor is in a valid state.
  bool invariant() const {
    return listener_ != nullptr && listener_->IsValid();
  }

 private:
  std::atomic<bool> stop_;
  std::unique_ptr<TcpSocket> listener_;
};

}  // namespace peregrine::internal

#endif  // THIRD_PARTY_PEREGRINE_SRC_INTERNAL_SOCKET_ACCEPTOR_H_
