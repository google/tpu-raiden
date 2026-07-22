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

#ifndef THIRD_PARTY_PEREGRINE_SRC_INTERNAL_BASE_ENDPOINT_H_
#define THIRD_PARTY_PEREGRINE_SRC_INTERNAL_BASE_ENDPOINT_H_

#include <ostream>
#include <string>
#include <string_view>

#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/ipaddr.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/types.h"
#include "tpu_raiden/transport/peregrine/src/util/macro.h"

namespace peregrine::internal {

// This class represents a network endpoint, which is a combination of
// an ipv{4,6} address and a port number. It is used to uniquely identify
// a process, whose control channel listens on the `ip:port`.
// It is thread-compatible and but not thread-safe.
class Endpoint final {
 public:
  // Parses and creates an endpoint from a string, eg. "127.0.0.1:56789" or
  // "[::1]:56789". Returns an invalid endpoint if the string parsing fails.
  static Endpoint Create(std::string_view ipaddr_port);

  // Default constructor creates an invalid endpoint.
  Endpoint() : ipaddr_(), port_(0) { DCHECK(!IsValid()); }

  // Constructor for ipv4.
  Endpoint(ipv4_t ip4, port_t port) : ipaddr_(ip4), port_(port) {}

  // Constructor for ipv6.
  Endpoint(ipv6_t ip6, port_t port) : ipaddr_(ip6), port_(port) {}

  // Constructor for ipv{4,6}.
  Endpoint(const IpAddr& ip, port_t port) : ipaddr_(ip), port_(port) {}

  // Allows copy/move.
  ALLOW_COPY(Endpoint);
  ALLOW_MOVE(Endpoint);

  // Destructor.
  ~Endpoint() = default;

  // Returns true iff the endpoint is valid.
  bool IsValid() const {
    DCHECK(0 <= port_ && port_ <= 65535);
    return 1 <= port_;  // [1, 65535]
  }

  // Returns the ip address of the endpoint.
  const IpAddr& GetIpAddr() const { return ipaddr_; };

  // Returns true iff the endpoint has an ipv4 address.
  bool IsIPv4() const { return ipaddr_.IsIPv4(); }

  // Returns true iff the endpoint has an ipv6 address.
  bool IsIPv6() const { return ipaddr_.IsIPv6(); }

  // Returns the ipv4 address of the ipv4 endpoint.
  const ipv4_t& IPv4Addr() const {
    DCHECK(IsIPv4());
    return ipaddr_.IPv4Addr();
  }

  // Returns the ipv6 address of the ipv6 endpoint.
  const ipv6_t& IPv6Addr() const {
    DCHECK(IsIPv6());
    return ipaddr_.IPv6Addr();
  }

  // Returns the port of the endpoint.
  port_t Port() const { return port_; };

  // Builds a `sockaddr_in` struct for the ipv4 endpoint.
  struct sockaddr_in BuildIPv4Sockaddr() const;

  // Builds a `sockaddr_in6` struct for the ipv6 endpoint.
  struct sockaddr_in6 BuildIPv6Sockaddr() const;

  // Equality operator.
  friend bool operator==(const Endpoint& a, const Endpoint& b) {
    return a.port_ == b.port_ && a.ipaddr_ == b.ipaddr_;
  }

  // Returns a hash signature of the endpoint.
  HashValue Hash() const { return Hash(*this); }

  // Returns a hash signature of the endpoint.
  static HashValue Hash(const Endpoint& e) { return absl::Hash<Endpoint>{}(e); }

  // Returns a string representation of the endpoint.
  std::string ToString() const;

 private:
  // Calculates a hash value for the endpoint.
  template <typename H>
  friend H AbslHashValue(H h, const Endpoint& e) {
    return H::combine(std::move(h), e.ipaddr_, e.port_);
  }

 private:
  IpAddr ipaddr_;
  port_t port_;
};

inline std::ostream& operator<<(std::ostream& os, const Endpoint& e) {
  return os << e.ToString();
}

}  // namespace peregrine::internal

#endif  // THIRD_PARTY_PEREGRINE_SRC_INTERNAL_BASE_ENDPOINT_H_
