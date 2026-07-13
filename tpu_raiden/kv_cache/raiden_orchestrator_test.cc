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

#include "tpu_raiden/kv_cache/raiden_orchestrator.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#ifndef HAVE_NET_UTIL_PORTS
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <cstdlib>
#include "absl/log/check.h"
namespace net_util {
inline int PickUnusedPortOrDie() {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  CHECK_GE(fd, 0);
  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(0);
  CHECK_EQ(bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
  socklen_t len = sizeof(addr);
  CHECK_EQ(getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len), 0);
  int port = ntohs(addr.sin_port);
  close(fd);
  return port;
}
}  // namespace net_util
#endif

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {


using ::tpu_raiden::rpc::ControlRequest;
using ::tpu_raiden::rpc::ControlResponse;

// Helper to send a simple RPC to a Custom TCP server (Length + Proto)
absl::StatusOr<ControlResponse> SendRawRpc(const std::string& address,
                                           const ControlRequest& req) {
  size_t colon = address.rfind(':');
  if (colon == std::string::npos)
    return absl::InvalidArgumentError("Invalid address");
  std::string host = address.substr(0, colon);
  std::string port_str = address.substr(colon + 1);

  struct addrinfo hints;
  struct addrinfo* result = nullptr;
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  int ret = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result);
  if (ret != 0 || result == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("getaddrinfo failed: ", gai_strerror(ret)));
  }

  int sock = -1;
  struct addrinfo* rp;
  for (rp = result; rp != nullptr; rp = rp->ai_next) {
    sock = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (sock < 0) continue;

    if (connect(sock, rp->ai_addr, rp->ai_addrlen) != -1) {
      break;  // Success
    }
    close(sock);
    sock = -1;
  }

  freeaddrinfo(result);

  if (sock < 0) {
    return absl::InternalError("Connection Failed");
  }

  std::string payload;
  if (!req.SerializeToString(&payload)) {
    close(sock);
    return absl::InternalError("Failed to serialize request");
  }

  uint32_t net_len = htonl(payload.size());
  if (send(sock, &net_len, sizeof(net_len), 0) != sizeof(net_len)) {
    close(sock);
    return absl::InternalError("Failed to send length");
  }

  if (send(sock, payload.data(), payload.size(), 0) != payload.size()) {
    close(sock);
    return absl::InternalError("Failed to send payload");
  }

  uint32_t resp_net_len = 0;
  if (recv(sock, &resp_net_len, sizeof(resp_net_len), 0) !=
      sizeof(resp_net_len)) {
    close(sock);
    return absl::InternalError("Failed to receive length");
  }

  uint32_t resp_len = ntohl(resp_net_len);
  std::vector<char> resp_buffer(resp_len);
  size_t total_recv = 0;
  while (total_recv < resp_len) {
    ssize_t n =
        recv(sock, resp_buffer.data() + total_recv, resp_len - total_recv, 0);
    if (n <= 0) {
      close(sock);
      return absl::InternalError("Failed to receive payload");
    }
    total_recv += n;
  }

  ControlResponse resp;
  if (!resp.ParseFromString(absl::string_view(resp_buffer.data(), resp_len))) {
    close(sock);
    return absl::InternalError("Failed to parse response");
  }

  close(sock);
  return resp;
}

TEST(RaidenOrchestratorTest, RegistrationAndResolutionIsolated) {
  int port = net_util::PickUnusedPortOrDie();
  RaidenOrchestrator orchestrator(port);
  // Orchestrator starts in constructor.

  std::string orch_addr = absl::StrCat("127.0.0.1:", port);

  // 1. Register a unit
  ControlRequest reg_req;
  reg_req.set_command(ControlRequest::COMMAND_REGISTER_WORK_UNIT);
  auto* reg_data = reg_req.mutable_register_work_unit_request();
  auto* raiden_id = reg_data->mutable_unit();
  raiden_id->set_job_name("test_job");
  raiden_id->set_job_replica_id("0");
  raiden_id->set_data_name("test_data");
  raiden_id->set_data_replica_idx(0);
  reg_data->set_control_plane_rpc_address("10.0.0.1:5000");

  auto reg_resp_or = SendRawRpc(orch_addr, reg_req);
  ASSERT_OK(reg_resp_or);
  EXPECT_TRUE(reg_resp_or->success());

  // 2. Resolve the unit
  ControlRequest res_req;
  res_req.set_command(ControlRequest::COMMAND_RESOLVE_CONTROLLER);
  auto* target_unit = res_req.mutable_target_unit();
  target_unit->set_job_name("test_job");
  target_unit->set_job_replica_id("0");
  target_unit->set_data_name("test_data");
  target_unit->set_data_replica_idx(0);

  auto res_resp_or = SendRawRpc(orch_addr, res_req);
  ASSERT_OK(res_resp_or);
  EXPECT_TRUE(res_resp_or->success());
  EXPECT_EQ(res_resp_or->response_data(), "10.0.0.1:5000");

  // 3. Resolve non-existent unit
  target_unit->set_data_name("non_existent");
  auto res_resp_fail = SendRawRpc(orch_addr, res_req);
  ASSERT_OK(res_resp_fail);
  EXPECT_FALSE(res_resp_fail->success());
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
