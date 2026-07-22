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

#include "tpu_raiden/transport/peregrine/src/util/util.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <cstdint>
#include <cstring>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include "absl/types/span.h"

namespace peregrine::util {

void RandomNonZero(absl::Span<Byte> data) {
  absl::BitGen bitgen;
  RandomNonZero(bitgen, data);
}

void RandomNonZero(absl::BitGenRef bitgen, absl::Span<Byte> data) {
  for (int i = 0; i < data.size(); ++i) {
    data[i] = Random<Byte>(bitgen, 0x01, 0xff);
    DCHECK_NE(data[i], 0);
  }
}

namespace {
using port_t = uint16_t;

int Bind(const int fd, const int family, const port_t port) {
  if (family == AF_INET) {
    struct sockaddr_in sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons(port);
    sa.sin_addr = {.s_addr = INADDR_ANY};
    return bind(fd, (struct sockaddr*)&sa, sizeof(sa));
  } else {
    DCHECK_EQ(family, AF_INET6);
    struct sockaddr_in6 sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sin6_family = AF_INET6;
    sa.sin6_port = htons(port);
    sa.sin6_addr = IN6ADDR_ANY_INIT;
    return bind(fd, (struct sockaddr*)&sa, sizeof(sa));
  }
}

port_t GetPort(const struct sockaddr_storage& ss) {
  if (ss.ss_family == AF_INET) {
    return ntohs(reinterpret_cast<const struct sockaddr_in&>(ss).sin_port);
  } else {
    DCHECK_EQ(ss.ss_family, AF_INET6);
    return ntohs(reinterpret_cast<const struct sockaddr_in6&>(ss).sin6_port);
  }
}
}  // namespace

port_t FindFreePort(const int family, const bool tcp) {
  DCHECK(family == AF_INET || family == AF_INET6);
  const int type = tcp ? SOCK_STREAM : SOCK_DGRAM;
  const int protocol = tcp ? IPPROTO_TCP : IPPROTO_UDP;

  absl::BitGen bitgen;
  for (int i = 0; i < 100; ++i) {
    // Create a socket.
    const int fd = socket(family, type, protocol);
    if (fd < 0) {
      continue;
    }

    // Set SO_REUSEADDR to avoid "Address already in use" error.
    int on = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) < 0) {
      close(fd);
      continue;
    }

    // Bind the socket to a randomly chosen port.
    constexpr port_t kMinPort = 10'000;
    constexpr port_t kMaxPort = 65'535;
    const port_t port = Random(bitgen, kMinPort, kMaxPort);
    if (Bind(fd, family, port) < 0) {
      close(fd);
      continue;
    }

    // Check the bound socket.
    struct sockaddr_storage ss;
    socklen_t len = sizeof(ss);
    std::memset(&ss, 0, len);
    if (getsockname(fd, (struct sockaddr*)&ss, &len) < 0 ||
        ss.ss_family != family || GetPort(ss) != port) {
      close(fd);
      continue;
    }

    // Check that the tcp socket can listen.
    if (tcp && listen(fd, SOMAXCONN) < 0) {
      close(fd);
      continue;
    }

    close(fd);
    return port;
  }

  return 0;
}

}  // namespace peregrine::util
