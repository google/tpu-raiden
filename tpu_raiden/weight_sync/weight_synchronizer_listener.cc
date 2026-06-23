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

#include "tpu_raiden/weight_sync/weight_synchronizer_listener.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "absl/log/log.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"
#include "tpu_raiden/weight_sync/weight_synchronizer_base.h"

namespace tpu_raiden {
namespace weight_sync {

WeightSynchronizerListener::WeightSynchronizerListener(
    WeightSynchronizerBase* engine, int listener_port)
    : engine_(engine), listener_port_(listener_port) {
  server_fd_ = socket(AF_INET6, SOCK_STREAM, 0);
  if (server_fd_ < 0) {
    LOG(FATAL) << "Failed to create C++ Listener socket: "
               << std::strerror(errno);
  }

  int opt = 1;
  if (setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
    LOG(WARNING) << "setsockopt SO_REUSEADDR failed";
  }

  sockaddr_in6 address{};
  address.sin6_family = AF_INET6;
  address.sin6_addr = in6addr_any;
  address.sin6_port = htons(listener_port_);

  if (bind(server_fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) <
      0) {
    LOG(FATAL) << "C++ Listener bind failed on port " << listener_port_ << ": "
               << std::strerror(errno);
  }

  if (listen(server_fd_, 128) < 0) {
    LOG(FATAL) << "C++ Listener listen failed: " << std::strerror(errno);
  }

  socklen_t addr_len = sizeof(address);
  if (getsockname(server_fd_, reinterpret_cast<sockaddr*>(&address),
                  &addr_len) == 0) {
    listener_port_ = ntohs(address.sin6_port);
  }

  LOG(INFO) << "Native C++ WeightSynchronizerListener actively listening "
               "on port: "
            << listener_port_;

  listener_thread_ =
      std::thread(&WeightSynchronizerListener::ListenerLoop, this);
}

WeightSynchronizerListener::~WeightSynchronizerListener() {
  stopping_ = true;
  if (server_fd_ >= 0) {
    int sock = socket(AF_INET6, SOCK_STREAM, 0);
    if (sock >= 0) {
      sockaddr_in6 serv_addr{};
      serv_addr.sin6_family = AF_INET6;
      serv_addr.sin6_port = htons(listener_port_);
      serv_addr.sin6_addr = in6addr_loopback;
      if (connect(sock, reinterpret_cast<sockaddr*>(&serv_addr),
                  sizeof(serv_addr)) == 0) {
        tpu_raiden::rpc::ControlRequest req;
        req.set_command(tpu_raiden::rpc::ControlRequest::COMMAND_SHUTDOWN);
        std::string payload;
        if (req.SerializeToString(&payload)) {
          uint32_t net_len = htonl(payload.size());
          write(sock, &net_len, sizeof(net_len));
          write(sock, payload.data(), payload.size());
        }
      }
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

void WeightSynchronizerListener::ListenerLoop() {
  while (!stopping_) {
    sockaddr_in6 client_addr{};
    socklen_t client_len = sizeof(client_addr);
    int client_fd = accept(
        server_fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
    if (client_fd < 0) {
      if (stopping_) break;
      continue;
    }

    worker_threads_.push_back(std::thread(
        &WeightSynchronizerListener::ConnectionWorker, this, client_fd));
  }
}

void WeightSynchronizerListener::ConnectionWorker(int client_fd) {
  uint32_t net_len = 0;
  if (read(client_fd, &net_len, sizeof(net_len)) != sizeof(net_len)) {
    close(client_fd);
    return;
  }
  uint32_t payload_len = ntohl(net_len);

  std::vector<char> buffer(payload_len);
  size_t total_read = 0;
  while (total_read < payload_len) {
    ssize_t n =
        read(client_fd, buffer.data() + total_read, payload_len - total_read);
    if (n <= 0) {
      close(client_fd);
      return;
    }
    total_read += n;
  }

  tpu_raiden::rpc::ControlRequest req;
  if (!req.ParseFromString(absl::string_view(buffer.data(), buffer.size()))) {
    LOG(ERROR) << "Failed to parse ControlRequest Protobuf";
    close(client_fd);
    return;
  }

  tpu_raiden::rpc::ControlResponse resp;
  resp.set_success(true);
  resp.set_message("SUCCESS");

  if (req.command() ==
      tpu_raiden::rpc::ControlRequest::COMMAND_START_TRANSFER) {
    if (req.has_start_transfer_request()) {
      const auto& start_req = req.start_transfer_request();
      if (start_req.is_sender()) {
        LOG(INFO) << "C++ Listener received START_TRANSFER (Sender)";
        if (!start_req.shard_push_schedules().empty()) {
          LOG(INFO) << "C++ Listener executing PushWeightsResharded";
          absl::Status status = engine_->PushWeightsResharded(start_req);
          if (!status.ok()) {
            resp.set_success(false);
            resp.set_message(std::string(status.message()));
            LOG(ERROR) << "PushWeightsResharded native execution failed: "
                       << status;
          }
        } else {
          std::vector<std::string> peers(req.peers().begin(),
                                         req.peers().end());
          LOG(INFO) << "C++ Listener executing PushWeights to " << peers.size()
                    << " peers";
          if (!peers.empty()) {
            absl::Status status = engine_->PushWeights(peers);
            if (!status.ok()) {
              resp.set_success(false);
              resp.set_message(std::string(status.message()));
              LOG(ERROR) << "PushWeights native execution failed: " << status;
            }
          }
        }
      } else {
        LOG(INFO) << "C++ Listener received START_TRANSFER (Receiver) - no-op "
                     "for weight sync";
      }
    } else {
      resp.set_success(false);
      resp.set_message("Missing start_transfer_request");
      LOG(ERROR) << "Missing start_transfer_request in START_TRANSFER command";
    }
  } else if (req.command() ==
             tpu_raiden::rpc::ControlRequest::COMMAND_SHUTDOWN) {
    LOG(INFO) << "C++ Listener received SHUTDOWN command. Initiating "
                 "clean exit.";
    stopping_ = true;
  } else {
    resp.set_success(false);
    resp.set_message("COMMAND_UNSPECIFIED");
    LOG(WARNING) << "C++ Listener received unknown or unspecified "
                    "Protobuf command";
  }

  std::string resp_str;
  if (resp.SerializeToString(&resp_str)) {
    uint32_t resp_net_len = htonl(resp_str.size());
    write(client_fd, &resp_net_len, sizeof(resp_net_len));
    write(client_fd, resp_str.data(), resp_str.size());
  }
  close(client_fd);
}

}  // namespace weight_sync
}  // namespace tpu_raiden
