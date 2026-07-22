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

#ifndef THIRD_PARTY_PEREGRINE_SRC_INTERNAL_SOCKET_CONNECTOR_H_
#define THIRD_PARTY_PEREGRINE_SRC_INTERNAL_SOCKET_CONNECTOR_H_

#include <memory>

#include "tpu_raiden/transport/peregrine/src/internal/base/endpoint.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_tcp.h"

namespace peregrine::internal {

// This is a utility class that creates a tcp socket and connects it to
// a peer endpoint which has a tcp acceptor socket listening.
// It is thread-safe since it has no state.
class TcpConnector {
 public:
  // Connects to the `peer` endpoint. Returns a connected tcp socket
  // if successful. Otherwise, returns a null pointer.
  static std::unique_ptr<TcpSocket> Create(const Endpoint& peer);
};

}  // namespace peregrine::internal

#endif  // THIRD_PARTY_PEREGRINE_SRC_INTERNAL_SOCKET_CONNECTOR_H_
