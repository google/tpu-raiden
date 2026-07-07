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

#include "tpu_raiden/rpc/rpc_utils.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tpu_raiden/core/status_macros.h"
#include "tpu_raiden/transport/raw_buffer_transport.h"

namespace tpu_raiden {
namespace rpc {

absl::StatusOr<int> SimpleConnect(absl::string_view peer) {
  std::string host;
  std::string port_str;

  if (!peer.empty() && peer.front() == '[') {
    size_t closing_bracket = peer.find(']');
    if (closing_bracket == absl::string_view::npos ||
        closing_bracket + 1 >= peer.size() ||
        peer[closing_bracket + 1] != ':') {
      return absl::InvalidArgumentError(
          "Invalid IPv6 peer bracket string format");
    }
    host = std::string(peer.substr(1, closing_bracket - 1));
    port_str = std::string(peer.substr(closing_bracket + 2));
  } else {
    size_t colon = peer.rfind(':');
    if (colon == absl::string_view::npos) {
      return absl::InvalidArgumentError(
          "Invalid peer string format, missing port");
    }
    host = std::string(peer.substr(0, colon));
    port_str = std::string(peer.substr(colon + 1));
  }

  struct addrinfo hints;
  struct addrinfo* result = nullptr;
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  int ret = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result);
  if (ret != 0 || result == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "getaddrinfo failed for host ", host, ": ", gai_strerror(ret)));
  }

  int sock_fd = -1;
  struct addrinfo* rp;
  for (rp = result; rp != nullptr; rp = rp->ai_next) {
    sock_fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (sock_fd < 0) continue;

    if (connect(sock_fd, rp->ai_addr, rp->ai_addrlen) != -1) {
      break;  // Success
    }
    close(sock_fd);
    sock_fd = -1;
  }

  freeaddrinfo(result);

  if (sock_fd < 0) {
    return absl::InternalError(absl::StrCat("Failed to connect to ", peer));
  }

  return sock_fd;
}

absl::Status SendRpcSync(absl::string_view address,
                         const rpc::ControlRequest& req,
                         rpc::ControlResponse& resp) {
  ASSIGN_OR_RETURN(int fd, SimpleConnect(address));

  std::string payload;
  if (!req.SerializeToString(&payload)) {
    close(fd);
    return absl::InternalError("Failed to serialize ControlRequest");
  }

  uint32_t net_len = htonl(payload.size());
  absl::Status status =
      transport::RawBufferTransport::WriteExact(fd, &net_len, sizeof(net_len));
  if (!status.ok()) {
    close(fd);
    return status;
  }

  status = transport::RawBufferTransport::WriteExact(fd, payload.data(),
                                                     payload.size());
  if (!status.ok()) {
    close(fd);
    return status;
  }

  uint32_t resp_net_len = 0;
  status = transport::RawBufferTransport::ReadExact(fd, &resp_net_len,
                                                    sizeof(resp_net_len));
  if (!status.ok()) {
    close(fd);
    return status;
  }

  uint32_t resp_len = ntohl(resp_net_len);
  std::vector<char> resp_buffer(resp_len);
  status = transport::RawBufferTransport::ReadExact(fd, resp_buffer.data(),
                                                    resp_len);
  if (!status.ok()) {
    close(fd);
    return status;
  }

  close(fd);

  if (!resp.ParseFromArray(resp_buffer.data(), resp_len)) {
    return absl::InternalError("Failed to parse ControlResponse");
  }

  return absl::OkStatus();
}

}  // namespace rpc
}  // namespace tpu_raiden
