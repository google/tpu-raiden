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

#include "tpu_raiden/transport/peregrine/src/internal/base/endpoint.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/ipaddr.h"

namespace peregrine::internal {

namespace {
inline bool LooksLikeIPv6(std::string_view addr) {
  return addr.starts_with('[') && addr.ends_with(']');
}
}  // namespace

Endpoint Endpoint::Create(const std::string_view ipaddr_port) {
  // "127.0.0.1:56789" or "[::1]:56789", or sth invalid
  const Endpoint invalid;
  DCHECK(!invalid.IsValid());

  const auto pos = ipaddr_port.rfind(':');
  if (pos == std::string_view::npos) {
    LOG(WARNING) << "invalid ip:port " << ipaddr_port;
    return invalid;
  }

  const std::string_view a = ipaddr_port.substr(0, pos);
  const std::string_view p = ipaddr_port.substr(pos + 1);

  uint16_t port;
  if (!(absl::SimpleAtoi(p, &port) && 1 <= port && port <= 65535)) {
    LOG(WARNING) << "invalid port in " << ipaddr_port;
    return invalid;
  }

  std::string_view ipaddr = a;
  if (LooksLikeIPv6(ipaddr)) {
    ipaddr = a.substr(1, a.size() - 2);  // "[...]" -> "..."
    const std::optional<ipv6_t> ipv6 = ParseIPv6Addr(ipaddr);
    if (!ipv6.has_value()) {
      LOG(WARNING) << "invalid ipv6 addr in " << ipaddr_port;
      return invalid;
    }
    const Endpoint e(ipv6.value(), port);
    DCHECK(e.IsValid());
    return e;

  } else {
    const std::optional<ipv4_t> ipv4 = ParseIPv4Addr(ipaddr);
    if (!ipv4.has_value()) {
      LOG(WARNING) << "invalid ipv4 addr in " << ipaddr_port;
      return invalid;
    }
    const Endpoint e(ipv4.value(), port);
    DCHECK(e.IsValid());
    return e;
  }
}

struct sockaddr_in Endpoint::BuildIPv4Sockaddr() const {
  DCHECK(IsIPv4());
  return sockaddr_in{
      .sin_family = AF_INET,
      .sin_port = htons(port_),
      .sin_addr = IPv4Addr(),
  };
}

struct sockaddr_in6 Endpoint::BuildIPv6Sockaddr() const {
  DCHECK(IsIPv6());
  return sockaddr_in6{
      .sin6_family = AF_INET6,
      .sin6_port = htons(port_),
      .sin6_addr = IPv6Addr(),
  };
}

std::string Endpoint::ToString() const {
  const std::string addr = ipaddr_.ToString();
  if (ipaddr_.IsIPv4()) {
    return absl::StrCat(addr, ":", port_);
  } else {
    DCHECK(ipaddr_.IsIPv6());
    return absl::StrCat("[", addr, "]:", port_);
  }
};

}  // namespace peregrine::internal
