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

#include "net/util/ports.h"

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace net_util {

int PickUnusedPortOrDie() {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    perror("PickUnusedPortOrDie: socket");
    std::abort();
  }
  int opt = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  struct sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = 0;  // Ask the kernel for a free ephemeral port.

  if (bind(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
    perror("PickUnusedPortOrDie: bind");
    close(fd);
    std::abort();
  }

  socklen_t len = sizeof(addr);
  if (getsockname(fd, reinterpret_cast<struct sockaddr*>(&addr), &len) != 0) {
    perror("PickUnusedPortOrDie: getsockname");
    close(fd);
    std::abort();
  }

  int port = ntohs(addr.sin_port);
  close(fd);
  return port;
}

}  // namespace net_util
