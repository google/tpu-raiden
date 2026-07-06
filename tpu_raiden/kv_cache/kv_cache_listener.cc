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
#include <chrono>  // NOLINT
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/kv_cache/kv_cache_manager_base.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"
#include "tpu_raiden/rpc/rpc_utils.h"

namespace tpu_raiden {
namespace kv_cache {

using ::tpu_raiden::rpc::ControlRequest;
using ::tpu_raiden::rpc::ControlResponse;

KVCacheListener::KVCacheListener(KVCacheManagerBase* engine, int listener_port,
                                 int controller_port)
    : engine_(engine),
      listener_port_(listener_port),
      controller_port_(controller_port) {
  if (engine_ && controller_port_ > 0) {
    engine_->SetBlocksReceivedCallback(
        [this](const std::vector<int>& block_ids, uint64_t uuid) {
          ControlRequest req;
          req.set_command(ControlRequest::COMMAND_TRANSFER_COMPLETED);
          req.set_transfer_uuid(static_cast<int64_t>(uuid));
          for (int id : block_ids) {
            req.add_completed_block_ids(id);
          }

          std::string controller_addr =
              absl::StrCat("localhost:", controller_port_);
          ControlResponse resp;
          absl::Status status = rpc::SendRpcSync(controller_addr, req, resp);
          if (!status.ok()) {
            LOG(WARNING) << "Failed to notify controller at " << controller_addr
                         << ": " << status;
          } else if (!resp.success()) {
            LOG(WARNING) << "Controller failed to process completion: "
                         << resp.message();
          }
        });
  }

  server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd_ < 0) {
    LOG(FATAL) << "Failed to create C++ KVCacheListener socket: "
               << std::strerror(errno);
  }

  int opt = 1;
  if (setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
    LOG(WARNING) << "setsockopt SO_REUSEADDR failed";
  }

  sockaddr_in address;
  address.sin_family = AF_INET;
  address.sin_port = htons(listener_port_);
  address.sin_addr.s_addr = INADDR_ANY;

  // Retry on EADDRINUSE (a listener from an earlier transfer/test may still be
  // releasing the port), then fail GRACEFULLY rather than LOG(FATAL): aborting
  // the whole process on one port collision would kill every other in-process
  // user (e.g. every other test in a suite). On give-up we mark the listener
  // inactive so callers see is_active()==false instead of a crash.
  constexpr int kMaxBindAttempts = 100;  // ~10s at 100ms
  int bind_attempts = 0;
  while (bind(server_fd_, reinterpret_cast<sockaddr*>(&address),
              sizeof(address)) < 0) {
    if (errno == EADDRINUSE && ++bind_attempts < kMaxBindAttempts) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }
    LOG(ERROR) << "C++ KVCacheListener bind failed on port " << listener_port_
               << " after " << bind_attempts
               << " attempts: " << std::strerror(errno);
    close(server_fd_);
    server_fd_ = -1;
    stopping_ = true;  // is_active() -> false
    return;
  }

  if (listen(server_fd_, 128) < 0) {
    LOG(ERROR) << "C++ KVCacheListener listen failed on port " << listener_port_
               << ": " << std::strerror(errno);
    close(server_fd_);
    server_fd_ = -1;
    stopping_ = true;
    return;
  }

  socklen_t addr_len = sizeof(address);
  if (getsockname(server_fd_, reinterpret_cast<sockaddr*>(&address),
                  &addr_len) == 0) {
    listener_port_ = ntohs(address.sin_port);
  }

  LOG(INFO) << "Native C++ KVCacheListener actively listening on port: "
            << listener_port_;

  listener_thread_ = std::thread(&KVCacheListener::ListenerLoop, this);
}

KVCacheListener::~KVCacheListener() {
  stopping_ = true;
  if (server_fd_ >= 0) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock >= 0) {
      sockaddr_in serv_addr{};
      serv_addr.sin_family = AF_INET;
      serv_addr.sin_port = htons(listener_port_);
      inet_pton(AF_INET, "localhost", &serv_addr.sin_addr);
      if (connect(sock, reinterpret_cast<sockaddr*>(&serv_addr),
                  sizeof(serv_addr)) == 0) {
        ControlRequest req;
        req.set_command(ControlRequest::COMMAND_SHUTDOWN);
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
      if (start_req.is_sender()) {
        LOG(INFO) << "C++ KVCacheListener received START_TRANSFER (Sender)";
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
          LOG(ERROR) << "RegisterRecv native execution failed: " << status;
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
  } else if (req.command() == ControlRequest::COMMAND_GET_ENDPOINTS) {
    LOG(INFO) << "C++ KVCacheListener received GET_ENDPOINTS command.";
    std::string ip = engine_->local_ip();
    int port = engine_->local_port().value_or(0);
    std::string endpoint = absl::StrContains(ip, ':')
                               ? absl::StrCat("[", ip, "]:", port)
                               : absl::StrCat(ip, ":", port);
    for (size_t i = 0; i < engine_->num_shards(); ++i) {
      resp.add_endpoints(endpoint);
    }
  } else if (req.command() == ControlRequest::COMMAND_EXECUTE_LOAD) {
    if (req.has_load_request()) {
      const auto& load_req = req.load_request();
      uint64_t uuid = load_req.uuid();
      LOG(INFO) << "C++ KVCacheListener received EXECUTE_LOAD, uuid=" << uuid;

      std::vector<raiden::PjRtCopyFuture> shard_futures;
      absl::Status load_status = absl::OkStatus();
      std::vector<int> all_dst_block_ids;

      for (const auto& [local_sh_idx, schedule] :
           load_req.shard_load_schedules()) {
        std::vector<int64_t> src_block_ids;
        std::vector<int64_t> dst_block_ids;
        std::vector<int64_t> copy_sizes;

        for (const auto& entry : schedule.entries()) {
          src_block_ids.push_back(entry.src_block_id());
          dst_block_ids.push_back(entry.dst_block_id());
          copy_sizes.push_back(1);
          all_dst_block_ids.push_back(entry.dst_block_id());
        }

        if (src_block_ids.empty()) continue;

        auto fut_or = engine_->H2d(
            src_block_ids, dst_block_ids, copy_sizes,
            /*slot_idx=*/std::nullopt,
            /*target_layer_idx=*/std::nullopt,
            /*target_shard_idx=*/static_cast<size_t>(local_sh_idx));
        if (!fut_or.ok()) {
          load_status = fut_or.status();
          LOG(ERROR) << "Failed to trigger H2D for shard " << local_sh_idx
                     << ": " << load_status;
          break;
        }
        shard_futures.push_back(std::move(fut_or.value()));
      }

      if (load_status.ok() && !shard_futures.empty()) {
        raiden::PjRtCopyFuture joined_future =
            raiden::JoinPjRtCopyFutures(absl::MakeSpan(shard_futures));

        absl::flat_hash_set<int> unique_dst_ids(all_dst_block_ids.begin(),
                                                all_dst_block_ids.end());
        std::vector<int> deduped_dst_ids(unique_dst_ids.begin(),
                                         unique_dst_ids.end());

        joined_future.OnReady([port = controller_port_,
                               deduped_dst_ids = std::move(deduped_dst_ids),
                               uuid](auto res_status) {
          ControlRequest comp_req;
          comp_req.set_command(ControlRequest::COMMAND_LOAD_COMPLETED);
          comp_req.mutable_load_request()->set_uuid(uuid);
          for (int id : deduped_dst_ids) {
            comp_req.add_completed_block_ids(id);
          }
          if (res_status.ok()) {
            comp_req.set_success(true);
          } else {
            comp_req.set_success(false);
            comp_req.set_error_message(
                std::string(res_status.status().message()));
          }

          std::string controller_addr = absl::StrCat("localhost:", port);
          ControlResponse comp_resp;
          absl::Status rpc_status =
              rpc::SendRpcSync(controller_addr, comp_req, comp_resp);
          if (!rpc_status.ok()) {
            LOG(WARNING) << "Failed to notify controller of load completion at "
                         << controller_addr << ": " << rpc_status;
          } else if (!comp_resp.success()) {
            LOG(WARNING) << "Controller failed to process load completion: "
                         << comp_resp.message();
          }
        });
      } else if (!load_status.ok()) {
        resp.set_success(false);
        resp.set_message("Failed to trigger H2D: " +
                         std::string(load_status.message()));
      }
    } else {
      resp.set_success(false);
      resp.set_message("Missing load_request");
    }
  } else if (req.command() == ControlRequest::COMMAND_EXECUTE_SAVE) {
    if (req.has_save_request()) {
      const auto& save_req = req.save_request();
      uint64_t uuid = save_req.uuid();
      LOG(INFO) << "C++ KVCacheListener received EXECUTE_SAVE, uuid=" << uuid;

      std::vector<int64_t> src_block_ids;
      std::vector<int64_t> dst_block_ids;
      std::vector<int64_t> copy_sizes;

      if (!save_req.shard_save_schedules().empty()) {
        const auto& first_schedule =
            save_req.shard_save_schedules().begin()->second;
        for (const auto& entry : first_schedule.entries()) {
          src_block_ids.push_back(entry.src_block_id());
          dst_block_ids.push_back(entry.dst_block_id());
          copy_sizes.push_back(1);
        }
      }

      // Central allocation: if the store assigned all dst HOST slots (>=0), D2H
      // straight into them instead of auto-allocating on the worker. Keeps the
      // id identical across every sub-manager. Empty/-1 => legacy auto-alloc.
      bool central_dst = !dst_block_ids.empty();
      for (int64_t d : dst_block_ids) {
        if (d < 0) { central_dst = false; break; }
      }

      if (src_block_ids.empty()) {
        resp.set_success(true);
        resp.set_message("No blocks to save");
      } else {
        std::vector<int> allocated_ids;
        absl::StatusOr<raiden::PjRtCopyFuture> res_or;
        if (central_dst) {
          allocated_ids.assign(dst_block_ids.begin(), dst_block_ids.end());
          VLOG(2) << "Save central D2h into store dst slots (n="
                  << dst_block_ids.size() << ")";
          res_or = engine_->D2h(src_block_ids, dst_block_ids, copy_sizes);
        } else {
          auto aa_or = engine_->D2hAutoAllocate(src_block_ids, copy_sizes);
          if (aa_or.ok()) {
            allocated_ids = aa_or.value().first;
            res_or = std::move(aa_or.value().second);
          } else {
            res_or = aa_or.status();
          }
        }
        if (!res_or.ok()) {
          resp.set_success(false);
          resp.set_message("Failed to trigger D2H: " +
                           std::string(res_or.status().message()));
        } else {
          auto future = std::move(res_or).value();

          future.OnReady(
              [port = controller_port_,
               allocated_ids = std::move(allocated_ids),
               uuid](auto res_status) {
                ControlRequest comp_req;
                comp_req.set_command(ControlRequest::COMMAND_SAVE_COMPLETED);
                comp_req.mutable_save_request()->set_uuid(uuid);
                for (int id : allocated_ids) {
                  comp_req.add_completed_block_ids(id);
                }
                if (res_status.ok()) {
                  comp_req.set_success(true);
                } else {
                  comp_req.set_success(false);
                  comp_req.set_error_message(
                      std::string(res_status.status().message()));
                }

                std::string controller_addr = absl::StrCat("localhost:", port);
                ControlResponse comp_resp;
                absl::Status rpc_status =
                    rpc::SendRpcSync(controller_addr, comp_req, comp_resp);
                if (!rpc_status.ok()) {
                  LOG(WARNING)
                      << "Failed to notify controller of save completion at "
                      << controller_addr << ": " << rpc_status;
                } else if (!comp_resp.success()) {
                  LOG(WARNING)
                      << "Controller failed to process save completion: "
                      << comp_resp.message();
                }
              });
        }
      }
    } else {
      resp.set_success(false);
      resp.set_message("Missing save_request");
    }
  } else if (req.command() == ControlRequest::COMMAND_EXECUTE_EVICT) {
    if (req.has_evict_request()) {
      const auto& evict_req = req.evict_request();
      uint64_t uuid = evict_req.uuid();
      LOG(INFO) << "C++ KVCacheListener received EXECUTE_EVICT, uuid=" << uuid;

      std::vector<int> host_block_ids;

      if (!evict_req.shard_evict_schedules().empty()) {
        const auto& first_schedule =
            evict_req.shard_evict_schedules().begin()->second;
        for (const auto& entry : first_schedule.entries()) {
          host_block_ids.push_back(entry.host_block_id());
        }
      }

      if (host_block_ids.empty()) {
        resp.set_success(true);
        resp.set_message("No blocks to evict");
      } else {
        absl::Status status = engine_->UnlockBlocks(host_block_ids);

        ControlRequest comp_req;
        comp_req.set_command(ControlRequest::COMMAND_EVICT_COMPLETED);
        comp_req.mutable_evict_request()->set_uuid(uuid);
        if (status.ok()) {
          comp_req.set_success(true);
        } else {
          comp_req.set_success(false);
          comp_req.set_error_message(std::string(status.message()));
        }

        std::string controller_addr =
            absl::StrCat("localhost:", controller_port_);
        ControlResponse comp_resp;
        absl::Status rpc_status =
            rpc::SendRpcSync(controller_addr, comp_req, comp_resp);
        if (!rpc_status.ok()) {
          LOG(WARNING) << "Failed to notify controller of evict completion at "
                       << controller_addr << ": " << rpc_status;
        } else if (!comp_resp.success()) {
          LOG(WARNING) << "Controller failed to process evict completion: "
                       << comp_resp.message();
        }
        resp.set_success(status.ok());
        if (!status.ok()) {
          resp.set_message("Failed to unlock blocks: " +
                           std::string(status.message()));
        }
      }
    } else {
      resp.set_success(false);
      resp.set_message("Missing evict_request");
    }
  } else {
    resp.set_success(false);
    resp.set_message("COMMAND_UNSPECIFIED");
    LOG(WARNING) << "C++ KVCacheListener received unknown or unspecified Protobuf command";
  }

  std::string resp_str;
  if (resp.SerializeToString(&resp_str)) {
    uint32_t resp_net_len = htonl(resp_str.size());
    write(client_fd, &resp_net_len, sizeof(resp_net_len));
    write(client_fd, resp_str.data(), resp_str.size());
  }
  close(client_fd);
}

}  // namespace kv_cache
}  // namespace tpu_raiden
