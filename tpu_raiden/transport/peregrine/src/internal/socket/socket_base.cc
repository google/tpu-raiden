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

#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_base.h"

#include <netinet/in.h>

#include "absl/log/check.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/endpoint.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/types.h"

namespace peregrine::internal {

int SocketBase::Bind(const fd_t fd, const Endpoint& local) {
  if (local.IsIPv4()) {
    const struct sockaddr_in sa = local.BuildIPv4Sockaddr();
    return ::bind(fd.value(), (struct sockaddr*)&sa, sizeof(sa));
  } else {
    DCHECK(local.IsIPv6());
    const struct sockaddr_in6 sa = local.BuildIPv6Sockaddr();
    return ::bind(fd.value(), (struct sockaddr*)&sa, sizeof(sa));
  }
}

int SocketBase::Connect(const fd_t fd, const Endpoint& peer) {
  if (peer.IsIPv4()) {
    const struct sockaddr_in sa = peer.BuildIPv4Sockaddr();
    return ::connect(fd.value(), (struct sockaddr*)&sa, sizeof(sa));
  } else {
    DCHECK(peer.IsIPv6());
    const struct sockaddr_in6 sa = peer.BuildIPv6Sockaddr();
    return ::connect(fd.value(), (struct sockaddr*)&sa, sizeof(sa));
  }
}

}  // namespace peregrine::internal
