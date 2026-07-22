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

#ifndef THIRD_PARTY_PEREGRINE_SRC_INTERNAL_UTIL_TEST_UTIL_H_
#define THIRD_PARTY_PEREGRINE_SRC_INTERNAL_UTIL_TEST_UTIL_H_

#include <memory>
#include <string_view>

#include "tpu_raiden/transport/peregrine/src/internal/base/endpoint.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/ipaddr.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/types.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_tcp.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_udp.h"

namespace peregrine::internal::testing {

// ip addresses
inline constexpr std::string_view kIPv4AnyAddr = "0.0.0.0";
inline constexpr std::string_view kIPv6AnyAddr = "::";
inline constexpr std::string_view kIPv4Localhost = "127.0.0.1";
inline constexpr std::string_view kIPv6Localhost = "::1";

// Returns the ipv4 `ANY_ADDR` (all 0's).
inline IpAddr IPv4AnyAddr() {
  return IpAddr(ParseIPv4Addr(kIPv4AnyAddr).value());
}

// Returns the ipv6 `ANY_ADDR` (all 0's).
inline IpAddr IPv6AnyAddr() {
  return IpAddr(ParseIPv6Addr(kIPv6AnyAddr).value());
}

// Returns the ipv4 localhost address.
inline IpAddr IPv4Localhost() {
  return IpAddr(ParseIPv4Addr(kIPv4Localhost).value());
}

// Returns the ipv6 localhost address.
inline IpAddr IPv6Localhost() {
  return IpAddr(ParseIPv6Addr(kIPv6Localhost).value());
}

// Returns an ipv4 or ipv6 localhost address in the given address `family`.
inline IpAddr IpLocalhost(int family) {
  return family == AF_INET ? IPv4Localhost() : IPv6Localhost();
}

// Creates a localhost endpoint in the given address `family` and protocol.
Endpoint TestOnly_LocalEndpoint(int family, bool tcp);

// Finds an unused TCP port in the given address `family`.
// Return a non-zero port if successful, otherwise crashes.
port_t TestOnly_FindFreeTcpPort(int family);

// Finds an unused UDP port in the given address `family`.
// Return a non-zero port if successful, otherwise crashes.
port_t TestOnly_FindFreeUdpPort(int family);

// Creates a TCP socket in the given address `family`.
// Return a non-null socket if successful, otherwise crashes.
std::unique_ptr<TcpSocket> TestOnly_CreateTcpSocket(int family);

// Creates a UDP socket in the given address `family`.
// Return a non-null socket if successful, otherwise crashes.
std::unique_ptr<UdpSocket> TestOnly_CreateUdpSocket(int family);

}  // namespace peregrine::internal::testing

#endif  // THIRD_PARTY_PEREGRINE_SRC_INTERNAL_UTIL_TEST_UTIL_H_
