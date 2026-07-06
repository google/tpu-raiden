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
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"
#include "tpu_raiden/transport/raw_buffer_transport.h"

namespace tpu_raiden {
namespace kv_cache {

using ::tpu_raiden::rpc::ControlRequest;
using ::tpu_raiden::rpc::ControlResponse;

RaidenOrchestrator::RaidenOrchestrator(int port, const std::string& bind_ip)
    : port_(port) {
  bool is_ipv4 = !bind_ip.empty() && absl::StrContains(bind_ip, '.');
  int family = is_ipv4 ? AF_INET : AF_INET6;

  server_fd_ = socket(family, SOCK_STREAM, 0);
  if (server_fd_ < 0) {
    LOG(FATAL) << "Failed to create RaidenOrchestrator socket: "
               << std::strerror(errno);
  }

  int opt = 1;
  if (setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
    LOG(WARNING) << "setsockopt SO_REUSEADDR failed";
  }

  if (family == AF_INET) {
    struct sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port_);
    if (!bind_ip.empty()) {
      inet_pton(AF_INET, bind_ip.c_str(), &address.sin_addr);
    } else {
      address.sin_addr.s_addr = INADDR_ANY;
    }
    if (bind(server_fd_, reinterpret_cast<sockaddr*>(&address),
             sizeof(address)) < 0) {
      LOG(FATAL) << "RaidenOrchestrator bind failed on port " << port_ << ": "
                 << std::strerror(errno);
    }
  } else {
    struct sockaddr_in6 address{};
    address.sin6_family = AF_INET6;
    address.sin6_port = htons(port_);
    if (!bind_ip.empty()) {
      inet_pton(AF_INET6, bind_ip.c_str(), &address.sin6_addr);
    } else {
      address.sin6_addr = in6addr_any;
    }
    if (bind(server_fd_, reinterpret_cast<sockaddr*>(&address),
             sizeof(address)) < 0) {
      LOG(FATAL) << "RaidenOrchestrator bind failed on port " << port_ << ": "
                 << std::strerror(errno);
    }
  }

  if (listen(server_fd_, 128) < 0) {
    LOG(FATAL) << "RaidenOrchestrator listen failed: " << std::strerror(errno);
  }

  // Retrieve actual port if 0 was passed
  struct sockaddr_storage ss;
  socklen_t addr_len = sizeof(ss);
  if (getsockname(server_fd_, reinterpret_cast<sockaddr*>(&ss), &addr_len) ==
      0) {
    if (ss.ss_family == AF_INET) {
      struct sockaddr_in* sin = reinterpret_cast<struct sockaddr_in*>(&ss);
      port_ = ntohs(sin->sin_port);
    } else if (ss.ss_family == AF_INET6) {
      struct sockaddr_in6* sin6 = reinterpret_cast<struct sockaddr_in6*>(&ss);
      port_ = ntohs(sin6->sin6_port);
    }
  }

  LOG(INFO) << "RaidenOrchestrator actively listening on port: " << port_;

  listener_thread_ = std::thread(&RaidenOrchestrator::ListenerLoop, this);
}

RaidenOrchestrator::~RaidenOrchestrator() {
  stopping_ = true;
  if (server_fd_ >= 0) {
    // Wake up listener thread by connecting to self
    int sock = socket(AF_INET6, SOCK_STREAM, 0);
    if (sock >= 0) {
      sockaddr_in6 serv_addr{};
      serv_addr.sin6_family = AF_INET6;
      serv_addr.sin6_port = htons(port_);
      inet_pton(AF_INET6, "::1", &serv_addr.sin6_addr);
      connect(sock, reinterpret_cast<sockaddr*>(&serv_addr), sizeof(serv_addr));
      close(sock);
    }
    close(server_fd_);
  }

  if (listener_thread_.joinable()) {
    listener_thread_.join();
  }

  for (auto& t : worker_threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
}

absl::Status RaidenOrchestrator::RegisterController(
    const RaidenId& raiden_id, const std::string& address) {
  absl::MutexLock lock(&mutex_);
  registry_[raiden_id] = address;
  LOG(INFO) << "Registered controller for " << raiden_id.job_name << ":"
            << raiden_id.job_replica_id << " at " << address;
  return absl::OkStatus();
}

absl::StatusOr<std::string> RaidenOrchestrator::ResolveController(
    const RaidenId& raiden_id) {
  absl::MutexLock lock(&mutex_);
  auto it = registry_.find(raiden_id);
  if (it == registry_.end()) {
    return absl::NotFoundError("Controller not found for specified RaidenId");
  }
  return it->second;
}

void RaidenOrchestrator::ListenerLoop() {
  while (!stopping_) {
    sockaddr_in6 client_addr{};
    socklen_t client_len = sizeof(client_addr);
    int client_fd = accept(
        server_fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
    if (client_fd < 0) {
      if (stopping_) break;
      continue;
    }

    worker_threads_.push_back(
        std::thread(&RaidenOrchestrator::ConnectionWorker, this, client_fd));
  }
}

void RaidenOrchestrator::ConnectionWorker(int client_fd) {
  uint32_t net_len = 0;
  if (!transport::RawBufferTransport::ReadExact(client_fd, &net_len,
                                                sizeof(net_len))
           .ok()) {
    close(client_fd);
    return;
  }
  uint32_t payload_len = ntohl(net_len);

  std::vector<char> buffer(payload_len);
  if (!transport::RawBufferTransport::ReadExact(client_fd, buffer.data(),
                                                payload_len)
           .ok()) {
    close(client_fd);
    return;
  }

  ControlRequest req;
  if (!req.ParseFromString(absl::string_view(buffer.data(), buffer.size()))) {
    LOG(ERROR) << "Failed to parse ControlRequest Protobuf";
    close(client_fd);
    return;
  }

  ControlResponse resp;
  resp.set_success(true);
  resp.set_message("SUCCESS");

  if (req.command() == ControlRequest::COMMAND_REGISTER_WORK_UNIT) {
    if (req.has_register_work_unit_request()) {
      const auto& reg_req = req.register_work_unit_request();
      RaidenId id{.job_name = reg_req.unit().job_name(),
                  .job_replica_id = reg_req.unit().job_replica_id(),
                  .data_name = reg_req.unit().data_name(),
                  .data_replica_idx = reg_req.unit().data_replica_idx()};

      // In Controller->Orchestrator context, control_plane_rpc_address is the
      // address of the registering controller.
      std::string addr = reg_req.control_plane_rpc_address();
      if (addr.empty()) {
        resp.set_success(false);
        resp.set_message(
            "Missing control_plane_rpc_address (Controller Address)");
      } else {
        absl::Status status = RegisterController(id, addr);
        if (!status.ok()) {
          resp.set_success(false);
          resp.set_message(std::string(status.message()));
        }
      }
    } else {
      resp.set_success(false);
      resp.set_message("Missing register_work_unit_request");
    }
  } else if (req.command() == ControlRequest::COMMAND_RESOLVE_CONTROLLER) {
    if (req.has_target_unit()) {
      const auto& target = req.target_unit();
      RaidenId id{.job_name = target.job_name(),
                  .job_replica_id = target.job_replica_id(),
                  .data_name = target.data_name(),
                  .data_replica_idx = target.data_replica_idx()};

      auto addr_or = ResolveController(id);
      if (addr_or.ok()) {
        resp.set_response_data(addr_or.value());
      } else {
        resp.set_success(false);
        resp.set_message(std::string(addr_or.status().message()));
      }
    } else {
      resp.set_success(false);
      resp.set_message("Missing target_unit for resolution");
    }
  } else {
    resp.set_success(false);
    resp.set_message("Unsupported command");
  }

  std::string resp_str;
  if (resp.SerializeToString(&resp_str)) {
    uint32_t resp_net_len = htonl(resp_str.size());
    transport::RawBufferTransport::WriteExact(client_fd, &resp_net_len,
                                              sizeof(resp_net_len))
        .IgnoreError();
    transport::RawBufferTransport::WriteExact(client_fd, resp_str.data(),
                                              resp_str.size())
        .IgnoreError();
  }

  close(client_fd);
}

}  // namespace kv_cache
}  // namespace tpu_raiden
