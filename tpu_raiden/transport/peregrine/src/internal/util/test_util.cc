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

#include "tpu_raiden/transport/peregrine/src/internal/util/test_util.h"

#include <memory>

#include "absl/log/check.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/endpoint.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/ipaddr.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/types.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_tcp.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_udp.h"
#include "tpu_raiden/transport/peregrine/src/util/util.h"

namespace peregrine::internal::testing {

port_t TestOnly_FindFreeTcpPort(int family) {
  const port_t port = util::FindFreePort(family, /*kTcp=*/true);
  CHECK_GT(port, 0);  // Crash OK
  return port;
}

port_t TestOnly_FindFreeUdpPort(int family) {
  const port_t port = util::FindFreePort(family, /*kTcp=*/false);
  CHECK_GT(port, 0);  // Crash OK
  return port;
}

std::unique_ptr<TcpSocket> TestOnly_CreateTcpSocket(int family) {
  std::unique_ptr<TcpSocket> socket = TcpSocket::Create(family);
  CHECK_NE(socket, nullptr);  // Crash OK
  DCHECK_EQ(socket->family(), family);
  DCHECK(socket->IsValid());
  DCHECK(socket->IsBlocking());
  DCHECK(!socket->IsConnected());
  return socket;
}

std::unique_ptr<UdpSocket> TestOnly_CreateUdpSocket(int family) {
  std::unique_ptr<UdpSocket> socket = UdpSocket::Create(family);
  CHECK_NE(socket, nullptr);  // Crash OK
  DCHECK_EQ(socket->family(), family);
  DCHECK(socket->IsValid());
  DCHECK(socket->IsBlocking());
  DCHECK(!socket->IsConnected());
  return socket;
}

Endpoint TestOnly_LocalEndpoint(int family, bool tcp) {
  const IpAddr ipaddr = IpLocalhost(family);
  if (tcp) {
    return Endpoint(ipaddr, TestOnly_FindFreeTcpPort(family));
  } else {
    return Endpoint(ipaddr, TestOnly_FindFreeUdpPort(family));
  }
}

}  // namespace peregrine::internal::testing
