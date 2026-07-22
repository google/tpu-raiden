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

#include "tpu_raiden/kv_cache/kv_cache_listener.h"

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
#include "absl/status/status.h"
#include "tpu_raiden/kv_cache/kv_cache_manager_base.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

using ::tpu_raiden::rpc::ControlRequest;
using ::tpu_raiden::rpc::ControlResponse;

namespace {

bool HasPoolReshardFields(
    const tpu_raiden::rpc::StartTransferRequest& request) {
  // Treat either field as opting into the pool executor. This intentionally
  // sends partially populated pool plans to PoolReshardPush/RegisterRecv so
  // their validation fails closed instead of silently taking the legacy path.
  return request.expected_pushes_per_pool() != 0 ||
         !request.transfer_pool_indices().empty();
}

}  // namespace

KVCacheListener::KVCacheListener(KVCacheManagerBase* engine,
                                 int listener_port)
    : engine_(engine), listener_port_(listener_port) {
  server_fd_ = socket(AF_INET6, SOCK_STREAM, 0);
  if (server_fd_ < 0) {
    LOG(FATAL) << "Failed to create C++ KVCacheListener socket: "
               << std::strerror(errno);
  }

  int opt = 1;
  if (setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
    LOG(WARNING) << "setsockopt SO_REUSEADDR failed";
  }

  sockaddr_in6 address{
      .sin6_family = AF_INET6,
      .sin6_port = htons(listener_port_),
      .sin6_addr = in6addr_any,
  };

  if (bind(server_fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) <
      0) {
    LOG(FATAL) << "C++ KVCacheListener bind failed on port "
               << listener_port_ << ": " << std::strerror(errno);
  }

  if (listen(server_fd_, 128) < 0) {
    LOG(FATAL) << "C++ KVCacheListener listen failed: "
               << std::strerror(errno);
  }

  socklen_t addr_len = sizeof(address);
  if (getsockname(server_fd_, reinterpret_cast<sockaddr*>(&address),
                  &addr_len) == 0) {
    listener_port_ = ntohs(address.sin6_port);
  }

  LOG(INFO) << "Native C++ KVCacheListener actively listening on port: "
            << listener_port_;

  listener_thread_ = std::thread(&KVCacheListener::ListenerLoop, this);
}

KVCacheListener::~KVCacheListener() {
  stopping_ = true;
  if (server_fd_ >= 0) {
    int sock = socket(AF_INET6, SOCK_STREAM, 0);
    if (sock >= 0) {
      sockaddr_in6 serv_addr{};
      serv_addr.sin6_family = AF_INET6;
      serv_addr.sin6_port = htons(listener_port_);
      inet_pton(AF_INET6, "::1", &serv_addr.sin6_addr);
      if (connect(sock, reinterpret_cast<sockaddr*>(&serv_addr),
                  sizeof(serv_addr)) == 0) {
        ControlRequest req;
        req.set_command(ControlRequest::COMMAND_SHUTDOWN);
        std::string payload;
        if (req.SerializeToString(&payload)) {
          uint32_t net_len = htonl(payload.size());
          send(sock, &net_len, sizeof(net_len), MSG_NOSIGNAL);
          send(sock, payload.data(), payload.size(), MSG_NOSIGNAL);
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

void KVCacheListener::ListenerLoop() {
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
        std::thread(&KVCacheListener::ConnectionWorker, this, client_fd));
  }
}

void KVCacheListener::ConnectionWorker(int client_fd) {
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

  ControlRequest req;
  if (!req.ParseFromString(absl::string_view(buffer.data(), buffer.size()))) {
    LOG(ERROR) << "Failed to parse ControlRequest Protobuf";
    close(client_fd);
    return;
  }

  ControlResponse resp;
  resp.set_success(true);
  resp.set_message("SUCCESS");

  if (req.command() == ControlRequest::COMMAND_START_TRANSFER) {
    if (req.has_start_transfer_request()) {
      const auto& start_req = req.start_transfer_request();
      const bool is_pool_reshard = HasPoolReshardFields(start_req);
      if (is_pool_reshard && start_req.is_sender()) {
        LOG(INFO) << "C++ KVCacheListener received pool START_TRANSFER "
                     "(Sender)";
        const std::vector<int64_t> src_block_ids(
            start_req.src_block_ids().begin(), start_req.src_block_ids().end());
        absl::Status status = engine_->PoolReshardPush(start_req, src_block_ids,
                                                       start_req.parallelism());
        if (!status.ok()) {
          resp.set_success(false);
          resp.set_message(std::string(status.message()));
          LOG(ERROR) << "PoolReshardPush native execution failed: " << status;
        }
      } else if (is_pool_reshard) {
        LOG(INFO) << "C++ KVCacheListener received pool START_TRANSFER "
                     "(Receiver)";
        const std::vector<int64_t> chip_block_ids(
            start_req.dst_device_block_ids().begin(),
            start_req.dst_device_block_ids().end());
        absl::Status status =
            engine_->PoolReshardRegisterRecv(start_req, chip_block_ids);
        if (!status.ok()) {
          resp.set_success(false);
          resp.set_message(std::string(status.message()));
          LOG(ERROR) << "PoolReshardRegisterRecv native execution failed: "
                     << status;
        }
      } else if (start_req.is_sender()) {
        // Preserve the pre-pool controller protocol for existing callers.
        LOG(INFO) << "C++ KVCacheListener received legacy START_TRANSFER "
                     "(Sender)";
        absl::Status status = engine_->PushKVCacheResharded(start_req);
        if (!status.ok()) {
          resp.set_success(false);
          resp.set_message(std::string(status.message()));
          LOG(ERROR) << "PushKVCacheResharded native execution failed: "
                     << status;
        }
      } else {
        LOG(INFO) << "C++ KVCacheListener received START_TRANSFER (Receiver), "
                     "registering expected buffers for uuid "
                  << start_req.uuid()
                  << ", expected blocks: " << start_req.expected_block_count();
        absl::Status status = engine_->RegisterActivePlan(
            start_req.uuid(), start_req, /*is_sender=*/false);
        if (!status.ok()) {
          resp.set_success(false);
          resp.set_message(std::string(status.message()));
          LOG(ERROR) << "RegisterActivePlan native execution failed: "
                     << status;
        }
      }
    } else {
      resp.set_success(false);
      resp.set_message("Missing start_transfer_request");
      LOG(ERROR) << "Missing start_transfer_request in START_TRANSFER command";
    }
  } else if (req.command() == ControlRequest::COMMAND_SHUTDOWN) {
    LOG(INFO) << "C++ KVCacheListener received SHUTDOWN command. Initiating clean exit.";
    absl::Status status = engine_->WaitForPendingWork();
    if (!status.ok()) {
      LOG(ERROR) << "WaitForPendingWork failed during shutdown: " << status;
    }
    stopping_ = true;
  } else {
    resp.set_success(false);
    resp.set_message("COMMAND_UNSPECIFIED");
    LOG(WARNING) << "C++ KVCacheListener received unknown or unspecified Protobuf command";
  }

  std::string resp_str;
  if (resp.SerializeToString(&resp_str)) {
    uint32_t resp_net_len = htonl(resp_str.size());
    // The peer may intentionally close after sending a one-way shutdown
    // request. Suppress SIGPIPE so listener teardown cannot terminate the
    // hosting process while this worker races to send its acknowledgement.
    send(client_fd, &resp_net_len, sizeof(resp_net_len), MSG_NOSIGNAL);
    send(client_fd, resp_str.data(), resp_str.size(), MSG_NOSIGNAL);
  }
  close(client_fd);
}

}  // namespace kv_cache
}  // namespace tpu_raiden
