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

#include "tpu_raiden/transport/peregrine/src/internal/base/ipaddr.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <cerrno>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"

namespace peregrine::internal {

namespace {
std::string PtonErrorMsg(std::string_view ip, int v, int last_errno) {
  return absl::StrFormat("inet_pton failed: ipv%d_addr=%s, errno=%d (%s)", v,
                         ip, last_errno, std::strerror(last_errno));
}

std::string NtopErrorMsg(int last_errno) {
  return absl::StrFormat("inet_ntop failed: errno=%d (%s)", last_errno,
                         std::strerror(last_errno));
}
}  // namespace

std::optional<ipv4_t> ParseIPv4Addr(std::string_view ip) {
  ipv4_t addr;
  switch (inet_pton(AF_INET, std::string(ip).c_str(), &addr)) {
    case 1:
      return addr;
    case 0:
      LOG(WARNING) << "invalid ipv4 addr: " << ip;
      return std::nullopt;
    default:
      const auto last_errno = errno;
      LOG(WARNING) << PtonErrorMsg(ip, 4, last_errno);
      return std::nullopt;
  }
}

std::optional<ipv6_t> ParseIPv6Addr(const std::string_view ip) {
  ipv6_t addr;
  switch (inet_pton(AF_INET6, std::string(ip).c_str(), &addr)) {
    case 1:
      return addr;
    case 0:
      LOG(WARNING) << "invalid ipv6 addr: " << ip;
      return std::nullopt;
    default:
      const auto last_errno = errno;
      LOG(WARNING) << PtonErrorMsg(ip, 6, last_errno);
      return std::nullopt;
  }
}

std::string ToIPv4String(const ipv4_t& ip4) {
  char addr[INET_ADDRSTRLEN];
  if (inet_ntop(AF_INET, &ip4, addr, INET_ADDRSTRLEN) != nullptr) {
    return addr;
  } else {
    const auto last_errno = errno;
    LOG(WARNING) << NtopErrorMsg(last_errno);
    return "invalid ipv4 addr";
  }
}

std::string ToIPv6String(const ipv6_t& ip6) {
  char addr[INET6_ADDRSTRLEN];
  if (inet_ntop(AF_INET6, &ip6, addr, INET6_ADDRSTRLEN) != nullptr) {
    return addr;
  } else {
    const auto last_errno = errno;
    LOG(WARNING) << NtopErrorMsg(last_errno);
    return "invalid ipv6 addr";
  }
}

std::optional<IpAddr> IpAddr::Create(std::string_view ip) {
  if (auto v4 = ParseIPv4Addr(ip); v4.has_value()) {
    return IpAddr(v4.value());
  } else if (auto v6 = ParseIPv6Addr(ip); v6.has_value()) {
    return IpAddr(v6.value());
  } else {
    return std::nullopt;
  }
}

bool operator==(const IpAddr& a, const IpAddr& b) {
  if (a.IsIPv4() && b.IsIPv4()) {
    return a.IPv4Addr().s_addr == b.IPv4Addr().s_addr;
  } else if (a.IsIPv6() && b.IsIPv6()) {
    return std::memcmp(&a.IPv6Addr(), &b.IPv6Addr(), sizeof(ipv6_t)) == 0;
  } else {
    return false;
  }
}

std::string IpAddr::ToString() const {
  return IsIPv4() ? ToIPv4String(IPv4Addr()) : ToIPv6String(IPv6Addr());
}

}  // namespace peregrine::internal
