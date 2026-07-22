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

#include "tpu_raiden/transport/peregrine/src/internal/socket/connector.h"

#include <memory>
#include <string_view>

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/endpoint.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_tcp.h"

namespace peregrine::internal {

namespace {
constexpr std::string_view kConnector = "tcp connector ";
}  // namespace

std::unique_ptr<TcpSocket> TcpConnector::Create(const Endpoint& peer) {
  DCHECK(peer.IsValid());

  const int family = peer.GetIpAddr().AddressFamily();
  std::unique_ptr<TcpSocket> socket = TcpSocket::Create(family);
  if ABSL_PREDICT_FALSE (socket == nullptr) {
    return nullptr;
  }

  DCHECK(socket->IsBlocking());
  if (!socket->Connect(peer)) {
    return nullptr;
  }

  DCHECK(socket->IsConnected());
  LOG(INFO) << kConnector << "made " << *socket;
  return socket;
}

}  // namespace peregrine::internal
