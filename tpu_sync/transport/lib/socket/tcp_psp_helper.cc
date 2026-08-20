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

#include "tpu_sync/transport/lib/socket/tcp_psp_helper.h"

#include <arpa/inet.h>
#include <chrono>  // NOLINT
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace tpu_raiden::transport::lib {

namespace {

}  // namespace


absl::StatusOr<PspPeerKey> RegisterPspPeerKey(
    int server_fd, uint32_t client_spi, absl::string_view client_key) {
  if (server_fd < 0) {
    return absl::InvalidArgumentError("Invalid server file descriptor");
  }
  if (client_spi == 0) {
    return absl::InvalidArgumentError("Invalid client SPI: 0 is reserved");
  }
  if (client_key.size() != 16) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid client key size: ", client_key.size(),
                     " (expected 16 bytes)"));
  }

  return absl::UnimplementedError("PSP-TCP is disabled.");
}

bool PspEnabled(int client_fd) {
  if (client_fd < 0) {
    return false;
  }

  return false;
}

absl::Status TcpPspConnect(
    int sock_fd, const struct sockaddr* addr, socklen_t addrlen,
    std::shared_ptr<grpc::Channel> channel) {
  if (sock_fd < 0 || addr == nullptr || addrlen == 0) {
    return absl::InvalidArgumentError("Invalid arguments for TcpPspConnect");
  }
  if (channel == nullptr) {
    return absl::InvalidArgumentError(
        "A valid gRPC Channel is required for TCP-over-PSP peer handshake");
  }

  return absl::UnimplementedError("PSP-TCP is disabled.");
}

}  // namespace tpu_raiden::transport::lib
