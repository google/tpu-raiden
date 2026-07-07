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

#include "tpu_raiden/kv_cache/raiden_controller_embedded.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_raiden/core/status_macros.h"
#include "tpu_raiden/core/tpu_utils.h"
#include "tpu_raiden/kv_cache/kv_cache_store.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"
#include "tpu_raiden/rpc/rpc_utils.h"
#include "tpu_raiden/transport/raw_buffer_transport.h"

namespace tpu_raiden {
namespace kv_cache {

using ::tpu_raiden::rpc::ControlRequest;
using ::tpu_raiden::rpc::ControlResponse;
using ::tpu_raiden::rpc::SendRpcSync;

RaidenControllerEmbedded::RaidenControllerEmbedded(
    KVCacheStore* store, int port, const std::string& orchestrator_address,
    int local_worker_port,
    const std::vector<std::string>& local_worker_data_addresses,
    size_t bytes_per_block, size_t num_shards, size_t num_listeners)
    : store_(store),

      port_(port),
      orchestrator_address_(orchestrator_address),
      local_worker_port_(local_worker_port),
      local_worker_data_addresses_(local_worker_data_addresses),
      bytes_per_block_(bytes_per_block),
      num_shards_(num_shards),
      num_listeners_(num_listeners) {}

RaidenControllerEmbedded::~RaidenControllerEmbedded() { Stop(); }

absl::Status RaidenControllerEmbedded::Start() {
  server_fd_ = socket(AF_INET6, SOCK_STREAM, 0);
  if (server_fd_ < 0) {
    return absl::InternalError(std::string("Failed to create socket: ") +
                               std::strerror(errno));
  }

  int opt = 1;
  setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in6 address{
      .sin6_family = AF_INET6,
      .sin6_port = htons(port_),
      .sin6_addr = in6addr_any,
  };

  if (bind(server_fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) <
      0) {
    close(server_fd_);
    server_fd_ = -1;
    return absl::InternalError(std::string("Bind failed: ") +
                               std::strerror(errno));
  }

  if (listen(server_fd_, 128) < 0) {
    close(server_fd_);
    server_fd_ = -1;
    return absl::InternalError(std::string("Listen failed: ") +
                               std::strerror(errno));
  }

  socklen_t addr_len = sizeof(address);
  if (getsockname(server_fd_, reinterpret_cast<sockaddr*>(&address),
                  &addr_len) == 0) {
    port_ = ntohs(address.sin6_port);
  }

  // Determine self IP address properly.
  std::vector<HostNicAddress> host_nics = GetLocalHostNicAddresses();
  std::string ctrl_ip = "localhost";
  for (const auto& nic : host_nics) {
    if (nic.classification == NicClassification::kControlPlane) {
      ctrl_ip = nic.ip_address;
      break;
    }
  }
  if (ctrl_ip == "localhost" && !host_nics.empty()) {
    ctrl_ip = host_nics[0].ip_address;
  }

  self_address_ = absl::StrContains(ctrl_ip, ':')
                      ? absl::StrCat("[", ctrl_ip, "]:", port_)
                      : absl::StrCat(ctrl_ip, ":", port_);

  LOG(INFO) << "RaidenControllerEmbedded listening on port: " << port_
            << " with address: " << self_address_;

  // Query Local Worker for its data endpoints
  local_worker_data_addresses_.clear();
  listener_shard_counts_.assign(num_listeners_, 0);
  for (size_t i = 0; i < num_listeners_; ++i) {
    int port = local_worker_port_ + i;
    std::string worker_addr = absl::StrCat("localhost:", port);
    ControlRequest req;
    req.set_command(ControlRequest::COMMAND_GET_ENDPOINTS);
    ControlResponse resp;
    absl::Status status = SendRpcSync(worker_addr, req, resp);
    if (status.ok() && resp.success()) {
      listener_shard_counts_[i] = resp.endpoints_size();
      for (const auto& addr : resp.endpoints()) {
        local_worker_data_addresses_.push_back(addr);
        LOG(INFO) << "Auto-discovered worker data address from listener " << i
                  << ": " << addr;
      }
    } else {
      LOG(WARNING) << "Failed to query worker endpoints from listener " << i
                   << ": " << status << " " << resp.message();
    }
  }

  absl::Status status = RegisterWithOrchestrator();

  if (!status.ok()) {
    LOG(ERROR) << "Failed to register with orchestrator: " << status;
    // Might want to fail start if registration fails.
  }

  listener_thread_ = std::thread(&RaidenControllerEmbedded::ListenerLoop, this);
  poller_thread_ =
      std::thread(&RaidenControllerEmbedded::WorkQueuePollerLoop, this);
  load_poller_thread_ =
      std::thread(&RaidenControllerEmbedded::LoadWorkQueuePollerLoop, this);
  save_poller_thread_ =
      std::thread(&RaidenControllerEmbedded::SaveWorkQueuePollerLoop, this);
  evict_poller_thread_ =
      std::thread(&RaidenControllerEmbedded::EvictWorkQueuePollerLoop, this);

  return absl::OkStatus();
}

void RaidenControllerEmbedded::Stop() {
  if (stopping_.exchange(true)) return;

  // Wake up poller threads if blocked on queues
  if (store_) {
    FetchRequest dummy;
    store_->PushFetchWork(dummy);
    LoadRequest dummy_load;
    store_->PushLoadWork(dummy_load);
    SaveRequest dummy_save;
    store_->PushSaveWork(dummy_save);
    EvictRequest dummy_evict;
    store_->PushEvictWork(dummy_evict);
  }

  if (server_fd_ >= 0) {
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
    server_fd_ = -1;
  }

  if (listener_thread_.joinable()) listener_thread_.join();
  if (poller_thread_.joinable()) poller_thread_.join();
  if (load_poller_thread_.joinable()) load_poller_thread_.join();
  if (save_poller_thread_.joinable()) save_poller_thread_.join();
  if (evict_poller_thread_.joinable()) evict_poller_thread_.join();

  for (auto& t : worker_threads_) {
    if (t.joinable()) t.join();
  }
}

absl::Status RaidenControllerEmbedded::RegisterWithOrchestrator() {
  if (orchestrator_address_.empty()) {
    return absl::InvalidArgumentError("Orchestrator address is empty");
  }

  ControlRequest req;
  req.set_command(ControlRequest::COMMAND_REGISTER_WORK_UNIT);
  auto* reg_req = req.mutable_register_work_unit_request();

  const auto& self_id = store_->raiden_id();
  auto* unit = reg_req->mutable_unit();
  unit->set_job_name(self_id.job_name);
  unit->set_job_replica_id(self_id.job_replica_id);
  unit->set_data_name(self_id.data_name);
  unit->set_data_replica_idx(self_id.data_replica_idx);

  reg_req->set_control_plane_rpc_address(self_address_);

  ControlResponse resp;
  RETURN_IF_ERROR(SendRpcSync(orchestrator_address_, req, resp));

  if (!resp.success()) {
    return absl::InternalError(
        absl::StrCat("Registration failed: ", resp.message()));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::string> RaidenControllerEmbedded::ResolveRemoteController(
    const RaidenId& remote_id) {
  if (orchestrator_address_.empty()) {
    return absl::InvalidArgumentError("Orchestrator address is empty");
  }

  ControlRequest req;
  req.set_command(ControlRequest::COMMAND_RESOLVE_CONTROLLER);
  auto* target = req.mutable_target_unit();
  target->set_job_name(remote_id.job_name);
  target->set_job_replica_id(remote_id.job_replica_id);
  target->set_data_name(remote_id.data_name);
  target->set_data_replica_idx(remote_id.data_replica_idx);

  ControlResponse resp;
  RETURN_IF_ERROR(SendRpcSync(orchestrator_address_, req, resp));

  if (!resp.success()) {
    return absl::NotFoundError(
        absl::StrCat("Resolution failed: ", resp.message()));
  }

  return resp.response_data();
}

absl::StatusOr<tpu_raiden::rpc::ControlResponse>
RaidenControllerEmbedded::NegotiateFetch(const std::string& remote_addr,
                                         const FetchRequestItem& request,
                                         uint64_t uuid) {
  if (request.block_hashes.empty()) {
    tpu_raiden::rpc::ControlResponse empty_resp;
    return empty_resp;
  }

  ControlRequest req;
  req.set_command(ControlRequest::COMMAND_NEGOTIATE_FETCH);
  auto* neg_req = req.mutable_fetch_negotiation_request();

  const auto& src_id = request.src_raiden_id;
  auto* src_unit = neg_req->mutable_src_unit();
  src_unit->set_job_name(src_id.job_name);
  src_unit->set_job_replica_id(src_id.job_replica_id);
  src_unit->set_data_name(src_id.data_name);
  src_unit->set_data_replica_idx(src_id.data_replica_idx);

  const auto& self_id = store_->raiden_id();
  auto* dst_unit = neg_req->mutable_dst_unit();
  dst_unit->set_job_name(self_id.job_name);
  dst_unit->set_job_replica_id(self_id.job_replica_id);
  dst_unit->set_data_name(self_id.data_name);
  dst_unit->set_data_replica_idx(self_id.data_replica_idx);

  for (const auto& addr : local_worker_data_addresses_) {
    neg_req->add_dst_worker_data_addresses(addr);
  }

  for (size_t i = 0; i < request.block_hashes.size(); ++i) {
    neg_req->add_block_hashes(request.block_hashes[i]);
    neg_req->add_dst_block_ids(request.dst_block_ids[i]);
  }
  neg_req->set_uuid(uuid);

  ControlResponse resp;
  RETURN_IF_ERROR(SendRpcSync(remote_addr, req, resp));

  if (!resp.success()) {
    return absl::InternalError(
        absl::StrCat("Negotiation failed: ", resp.message()));
  }

  return resp;
}

void RaidenControllerEmbedded::WorkQueuePollerLoop() {
  while (!stopping_) {
    FetchRequest req;
    if (!store_->PopFetchWork(req)) {
      if (stopping_) break;
      continue;
    }

    if (req.empty()) continue;

    for (const auto& batch_item : req) {
      auto addr_or = ResolveRemoteController(batch_item.src_raiden_id);
      if (!addr_or.ok()) {
        LOG(ERROR) << "Failed to resolve remote controller for "
                   << batch_item.src_raiden_id.job_name << ": "
                   << addr_or.status();
        // TODO: Push failure to CompletionQueue
        continue;
      }

      // Record pending fetches before negotiating
      {
        absl::MutexLock lock(pending_mu_);
        for (size_t i = 0; i < batch_item.block_hashes.size(); ++i) {
          pending_fetches_[batch_item.dst_block_ids[i]] =
              batch_item.block_hashes[i];
        }
      }

      uint64_t uuid = absl::ToUnixMicros(absl::Now());
      auto neg_resp_or = NegotiateFetch(addr_or.value(), batch_item, uuid);

      if (!neg_resp_or.ok()) {
        LOG(ERROR) << "Negotiation failed with " << addr_or.value() << ": "
                   << neg_resp_or.status();
        // TODO: Push failure to CompletionQueue
        continue;
      }

      ControlRequest base_start_req;
      base_start_req.set_command(ControlRequest::COMMAND_START_TRANSFER);
      auto* base_start_transfer =
          base_start_req.mutable_start_transfer_request();
      if (neg_resp_or.value().has_start_transfer_request()) {
        *base_start_transfer = neg_resp_or.value().start_transfer_request();
      }
      base_start_transfer->set_is_sender(false);
      base_start_transfer->set_use_block_chunks(false);
      base_start_transfer->set_uuid(uuid);
      base_start_transfer->set_expected_block_count(
          batch_item.block_hashes.size());
      base_start_transfer->set_dst_mem_type(tpu_raiden::rpc::MEMORY_TYPE_DRAM);

      // Broadcast START_TRANSFER (is_sender=False) to all local listeners
      size_t global_shard_offset = 0;
      for (size_t i = 0; i < num_listeners_; ++i) {
        size_t num_local_shards = listener_shard_counts_[i];
        if (num_local_shards == 0) continue;

        ControlRequest start_req_i = base_start_req;
        auto* start_transfer_i = start_req_i.mutable_start_transfer_request();

        if (start_transfer_i->shard_push_schedules().size() > 0) {
          start_transfer_i->clear_shard_push_schedules();
          auto& schedules_i = *start_transfer_i->mutable_shard_push_schedules();
          for (size_t local_sh = 0; local_sh < num_local_shards; ++local_sh) {
            size_t global_sh = global_shard_offset + local_sh;
            auto it = base_start_transfer->shard_push_schedules().find(
                static_cast<int32_t>(global_sh));
            if (it != base_start_transfer->shard_push_schedules().end()) {
              schedules_i[static_cast<int32_t>(local_sh)] = it->second;
            }
          }
        }

        int port = local_worker_port_ + i;
        std::string worker_addr = absl::StrCat("localhost:", port);

        ControlResponse start_resp;
        absl::Status start_status =
            SendRpcSync(worker_addr, start_req_i, start_resp);
        if (!start_status.ok() || !start_resp.success()) {
          LOG(WARNING) << "Failed to broadcast START_TRANSFER to listener " << i
                       << ": " << start_status << " " << start_resp.message();
        } else {
          VLOG(1) << "Broadcasted START_TRANSFER to listener " << i;
        }

        global_shard_offset += num_local_shards;
      }

      if (neg_resp_or.value().has_start_transfer_request()) {
        ControlRequest exec_req;
        exec_req.set_command(ControlRequest::COMMAND_EXECUTE_FETCH);
        *exec_req.mutable_start_transfer_request() =
            neg_resp_or.value().start_transfer_request();
        exec_req.mutable_start_transfer_request()->set_is_sender(true);
        ControlResponse exec_resp;
        absl::Status exec_status =
            SendRpcSync(addr_or.value(), exec_req, exec_resp);
        if (!exec_status.ok() || !exec_resp.success()) {
          LOG(ERROR) << "Execute fetch failed: " << exec_status << " "
                     << exec_resp.message();
        }
      }
    }
  }
}

void RaidenControllerEmbedded::ListenerLoop() {
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
        &RaidenControllerEmbedded::ConnectionWorker, this, client_fd));
  }
}

void RaidenControllerEmbedded::ConnectionWorker(int client_fd) {
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

  if (req.command() == ControlRequest::COMMAND_NEGOTIATE_FETCH) {
    if (req.has_fetch_negotiation_request()) {
      const auto& neg_req = req.fetch_negotiation_request();

      std::vector<std::string> hashes(neg_req.block_hashes().begin(),
                                      neg_req.block_hashes().end());
      auto lookup_res_or = store_->Lookup(hashes, /*enable_global=*/false);

      if (!lookup_res_or.ok()) {
        resp.set_success(false);
        resp.set_message(std::string(lookup_res_or.status().message()));
      } else {
        const auto& lookup_res = lookup_res_or.value();
        auto* neg_resp = resp.mutable_fetch_negotiation_response();

        std::set<std::string> found_set;
        for (const auto& pair : lookup_res) {
          neg_resp->add_found_block_hashes(pair.first);
          found_set.insert(pair.first);
        }
        for (const auto& hash : hashes) {
          if (found_set.find(hash) == found_set.end()) {
            neg_resp->add_missing_block_hashes(hash);
          }
        }

        // Trigger PUSH via local worker if all/any blocks found.
        if (!lookup_res.empty()) {
          absl::flat_hash_map<std::string, int> hash_to_dst_id;
          for (int i = 0; i < neg_req.block_hashes_size(); ++i) {
            if (i < neg_req.dst_block_ids_size()) {
              hash_to_dst_id[neg_req.block_hashes(i)] =
                  neg_req.dst_block_ids(i);
            }
          }

          ControlRequest start_req;
          start_req.set_command(ControlRequest::COMMAND_START_TRANSFER);
          auto* start_transfer = start_req.mutable_start_transfer_request();
          start_transfer->set_is_sender(true);
          start_transfer->set_use_block_chunks(false);
          start_transfer->set_src_mem_type(tpu_raiden::rpc::MEMORY_TYPE_DRAM);
          uint64_t uuid = neg_req.uuid();
          start_transfer->set_uuid(uuid);

          auto& schedules = *start_transfer->mutable_shard_push_schedules();

          for (const auto& pair : lookup_res) {
            const std::string& hash = pair.first;
            const auto& local_block = pair.second;
            int src_block_id = local_block.host_block_id;
            auto it = hash_to_dst_id.find(hash);
            if (it == hash_to_dst_id.end()) {
              LOG(WARNING)
                  << "Found hash in lookup but not in negotiation request: "
                  << hash;
              continue;
            }
            int dst_block_id = it->second;

            absl::flat_hash_map<std::string, int> peer_to_local_shard;
            for (size_t sh = 0; sh < num_shards_; ++sh) {
              auto* entry = schedules[sh].add_entries();
              std::string peer_addr;
              if (sh < neg_req.dst_worker_data_addresses_size()) {
                peer_addr = neg_req.dst_worker_data_addresses(sh);
                entry->set_dst_peer(peer_addr);
              } else {
                LOG(ERROR) << "Missing peer address for shard " << sh;
                continue;
              }
              int local_dst_sh = peer_to_local_shard[peer_addr]++;
              entry->set_src_block_id(src_block_id);
              entry->set_dst_block_id(dst_block_id);
              entry->set_dst_shard_idx(local_dst_sh);
              entry->set_size_bytes(bytes_per_block_);
              entry->set_count(1);
              entry->set_src_offset_bytes(0);
              entry->set_dst_offset_bytes(0);
              entry->set_src_stride_bytes(0);
              entry->set_dst_stride_bytes(0);
            }
          }

          if (resp.success()) {
            *resp.mutable_start_transfer_request() =
                start_req.start_transfer_request();
          }
        }
      }
    } else {
      resp.set_success(false);
      resp.set_message("Missing fetch_negotiation_request");
    }
  } else if (req.command() == ControlRequest::COMMAND_EXECUTE_FETCH) {
    if (!req.has_start_transfer_request()) {
      resp.set_success(false);
      resp.set_message("Missing start_transfer_request");
    } else {
      const auto& start_req = req;
      size_t global_shard_offset = 0;
      for (size_t i = 0; i < num_listeners_; ++i) {
        size_t num_local_shards = listener_shard_counts_[i];
        if (num_local_shards == 0) continue;

        ControlRequest start_req_i = start_req;
        start_req_i.set_command(ControlRequest::COMMAND_START_TRANSFER);
        auto* start_transfer_i = start_req_i.mutable_start_transfer_request();
        start_transfer_i->clear_shard_push_schedules();
        auto& schedules_i = *start_transfer_i->mutable_shard_push_schedules();

        bool has_schedule = false;
        for (size_t local_sh = 0; local_sh < num_local_shards; ++local_sh) {
          size_t global_sh = global_shard_offset + local_sh;
          auto it =
              start_req.start_transfer_request().shard_push_schedules().find(
                  static_cast<int32_t>(global_sh));
          if (it !=
              start_req.start_transfer_request().shard_push_schedules().end()) {
            schedules_i[static_cast<int32_t>(local_sh)] = it->second;
            has_schedule = true;
          }
        }

        if (!has_schedule) {
          global_shard_offset += num_local_shards;
          continue;  // Skip if this listener has no work in this request
        }

        int port = local_worker_port_ + i;
        std::string worker_addr = absl::StrCat("localhost:", port);
        ControlResponse worker_resp;
        absl::Status worker_status =
            SendRpcSync(worker_addr, start_req_i, worker_resp);

        if (!worker_status.ok()) {
          LOG(ERROR) << "Failed to trigger local worker PUSH for listener " << i
                     << ": " << worker_status;
          resp.set_success(false);
          resp.set_message("Failed to trigger local worker PUSH: " +
                           std::string(worker_status.message()));
          break;
        } else if (!worker_resp.success()) {
          LOG(ERROR) << "Local worker PUSH failed for listener " << i << ": "
                     << worker_resp.message();
          resp.set_success(false);
          resp.set_message("Local worker PUSH failed: " +
                           worker_resp.message());
          break;
        }

        global_shard_offset += num_local_shards;
      }
    }
  } else if (req.command() == ControlRequest::COMMAND_TRANSFER_COMPLETED) {
    VLOG(1) << "Received COMMAND_TRANSFER_COMPLETED";
    FetchCompletion completion;
    {
      absl::MutexLock lock(pending_mu_);
      for (int32_t block_id : req.completed_block_ids()) {
        auto it = pending_fetches_.find(block_id);
        if (it != pending_fetches_.end()) {
          block_completion_counts_[block_id]++;
          if (block_completion_counts_[block_id] >= num_listeners_) {
            FetchCompletionItem item;
            item.block_hash = it->second;
            item.host_block_id = block_id;
            item.success = true;
            completion.push_back(item);
            pending_fetches_.erase(it);
            block_completion_counts_.erase(block_id);
            VLOG(1) << "Fetch completed for block_id " << block_id
                    << " across all listeners";
          } else {
            VLOG(1) << "Fetch completed for block_id " << block_id << " on "
                    << block_completion_counts_[block_id] << "/"
                    << num_listeners_ << " listeners";
          }
        } else {
          LOG(WARNING) << "Received completion for unknown block_id: "
                       << block_id;
        }
      }
    }

    if (!completion.empty()) {
      store_->PushFetchCompletion(std::move(completion));
    }
    resp.set_success(true);
  } else if (req.command() == ControlRequest::COMMAND_LOAD_COMPLETED) {
    VLOG(1) << "Received COMMAND_LOAD_COMPLETED";
    uint64_t load_id = req.load_request().uuid();
    bool success = req.success();
    std::string err_msg = req.error_message();
    HandleLoadCompletionFromWorker(load_id, success, err_msg);
    resp.set_success(true);
  } else if (req.command() == ControlRequest::COMMAND_SAVE_COMPLETED) {
    VLOG(1) << "Received COMMAND_SAVE_COMPLETED";
    uint64_t save_id = req.save_request().uuid();
    bool success = req.success();
    std::string err_msg = req.error_message();
    std::vector<int> completed_ids(req.completed_block_ids().begin(),
                                   req.completed_block_ids().end());
    HandleSaveCompletionFromWorker(save_id, success, err_msg, completed_ids);
    resp.set_success(true);
  } else if (req.command() == ControlRequest::COMMAND_EVICT_COMPLETED) {
    VLOG(1) << "Received COMMAND_EVICT_COMPLETED";
    uint64_t evict_id = req.evict_request().uuid();
    bool success = req.success();
    std::string err_msg = req.error_message();
    HandleEvictCompletionFromWorker(evict_id, success, err_msg);
    resp.set_success(true);
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

void RaidenControllerEmbedded::LoadWorkQueuePollerLoop() {
  while (!stopping_) {
    LoadRequest req;
    if (!store_->PopLoadWork(req)) {
      if (stopping_) break;
      continue;
    }

    if (req.block_hashes.empty()) continue;
    uint64_t load_id = req.load_id;

    std::vector<ControlRequest> listener_reqs(num_listeners_);
    for (size_t l = 0; l < num_listeners_; ++l) {
      listener_reqs[l].set_command(ControlRequest::COMMAND_EXECUTE_LOAD);
      listener_reqs[l].mutable_load_request()->set_uuid(load_id);
    }

    size_t num_blocks = req.block_hashes.size();

    for (size_t b = 0; b < num_blocks; ++b) {
      std::string hash = req.block_hashes[b];
      int src_id = req.src_block_ids[b];
      int dst_id = req.dst_block_ids[b];

      size_t global_shard_offset = 0;
      for (size_t l = 0; l < num_listeners_; ++l) {
        size_t num_local_shards = listener_shard_counts_[l];
        auto* shard_schedules = listener_reqs[l]
                                    .mutable_load_request()
                                    ->mutable_shard_load_schedules();

        for (size_t local_sh = 0; local_sh < num_local_shards; ++local_sh) {
          auto& schedule = (*shard_schedules)[static_cast<int32_t>(local_sh)];
          auto* entry = schedule.add_entries();
          entry->set_src_block_id(src_id);
          entry->set_dst_block_id(dst_id);
        }
        global_shard_offset += num_local_shards;
      }
    }

    ControllerLoadState load_state;
    load_state.load_id = load_id;
    for (size_t i = 0; i < req.block_hashes.size(); ++i) {
      LoadCompletionItem c_item;
      c_item.block_hash = req.block_hashes[i];
      c_item.dst_hbm_block_id = req.dst_block_ids[i];
      c_item.success = true;
      load_state.items.push_back(c_item);
    }

    size_t active_listeners = 0;
    for (size_t l = 0; l < num_listeners_; ++l) {
      if (listener_shard_counts_[l] == 0) continue;
      active_listeners++;
    }
    load_state.expected_workers = active_listeners;

    {
      absl::MutexLock lock(&pending_mu_);
      active_loads_[load_id] = std::move(load_state);
    }

    for (size_t l = 0; l < num_listeners_; ++l) {
      if (listener_shard_counts_[l] == 0) continue;

      int port = local_worker_port_ + l;
      std::string worker_addr = absl::StrCat("localhost:", port);

      ControlResponse resp;
      absl::Status status = SendRpcSync(worker_addr, listener_reqs[l], resp);
      if (!status.ok() || !resp.success()) {
        LOG(ERROR) << "Failed to send COMMAND_EXECUTE_LOAD to listener " << l
                   << ": " << status << " " << resp.message();
        HandleLoadCompletionFromWorker(
            load_id, /*success=*/false,
            absl::StrCat("Failed to send RPC: ", status.message(), " ",
                         resp.message()));
      } else {
        VLOG(1) << "Sent COMMAND_EXECUTE_LOAD to listener " << l
                << " for load_id " << load_id;
      }
    }
  }
}

void RaidenControllerEmbedded::HandleLoadCompletionFromWorker(
    uint64_t load_id, bool success, const std::string& error_message) {
  LoadCompletion completion;
  bool push_completion = false;

  {
    absl::MutexLock lock(&pending_mu_);
    auto it = active_loads_.find(load_id);
    if (it != active_loads_.end()) {
      auto& state = it->second;
      state.completed_workers++;
      if (!success) {
        for (auto& item : state.items) {
          item.success = false;
          item.error_message += error_message + ";";
        }
      }

      if (state.completed_workers >= state.expected_workers) {
        completion.load_id = load_id;
        completion.items = std::move(state.items);
        active_loads_.erase(it);
        push_completion = true;
      }
    }
  }

  if (push_completion) {
    store_->PushLoadCompletion(std::move(completion));
    VLOG(1) << "Load fully completed for load_id " << load_id
            << ", pushing completion to store";
  }
}

void RaidenControllerEmbedded::SaveWorkQueuePollerLoop() {
  while (!stopping_) {
    SaveRequest req;
    if (!store_->PopSaveWork(req)) {
      if (stopping_) break;
      continue;
    }

    if (req.block_hashes.empty()) continue;
    uint64_t save_id = req.save_id;

    std::vector<ControlRequest> listener_reqs(num_listeners_);
    for (size_t l = 0; l < num_listeners_; ++l) {
      listener_reqs[l].set_command(ControlRequest::COMMAND_EXECUTE_SAVE);
      listener_reqs[l].mutable_save_request()->set_uuid(save_id);
    }

    size_t num_blocks = req.block_hashes.size();

    for (size_t b = 0; b < num_blocks; ++b) {
      std::string hash = req.block_hashes[b];
      int src_id = req.src_block_ids[b];

      size_t global_shard_offset = 0;
      for (size_t l = 0; l < num_listeners_; ++l) {
        size_t num_local_shards = listener_shard_counts_[l];
        auto* shard_schedules = listener_reqs[l]
                                    .mutable_save_request()
                                    ->mutable_shard_save_schedules();

        for (size_t local_sh = 0; local_sh < num_local_shards; ++local_sh) {
          auto& schedule = (*shard_schedules)[static_cast<int32_t>(local_sh)];
          auto* entry = schedule.add_entries();
          entry->set_src_block_id(src_id);
        }
        global_shard_offset += num_local_shards;
      }
    }

    ControllerSaveState save_state;
    save_state.save_id = save_id;
    for (size_t i = 0; i < req.block_hashes.size(); ++i) {
      SaveCompletionItem c_item;
      c_item.block_hash = req.block_hashes[i];
      c_item.dst_host_block_id = -1;
      c_item.success = true;
      save_state.items.push_back(c_item);
    }

    size_t active_listeners = 0;
    for (size_t l = 0; l < num_listeners_; ++l) {
      if (listener_shard_counts_[l] == 0) continue;
      active_listeners++;
    }
    save_state.expected_workers = active_listeners;

    {
      absl::MutexLock lock(&pending_mu_);
      active_saves_[save_id] = std::move(save_state);
    }

    for (size_t l = 0; l < num_listeners_; ++l) {
      if (listener_shard_counts_[l] == 0) continue;

      int port = local_worker_port_ + l;
      std::string worker_addr = absl::StrCat("localhost:", port);

      ControlResponse resp;
      absl::Status status = SendRpcSync(worker_addr, listener_reqs[l], resp);
      if (!status.ok() || !resp.success()) {
        LOG(ERROR) << "Failed to send COMMAND_EXECUTE_SAVE to listener " << l
                   << ": " << status << " " << resp.message();
        HandleSaveCompletionFromWorker(
            save_id, /*success=*/false,
            absl::StrCat("Failed to send RPC: ", status.message(), " ",
                         resp.message()));
      } else {
        VLOG(1) << "Sent COMMAND_EXECUTE_SAVE to listener " << l
                << " for save_id " << save_id;
      }
    }
  }
}

void RaidenControllerEmbedded::HandleSaveCompletionFromWorker(
    uint64_t save_id, bool success, const std::string& error_message,
    const std::vector<int>& completed_block_ids) {
  SaveCompletion completion;
  bool push_completion = false;

  {
    absl::MutexLock lock(&pending_mu_);
    auto it = active_saves_.find(save_id);
    if (it != active_saves_.end()) {
      auto& state = it->second;
      state.completed_workers++;
      if (!success) {
        for (auto& item : state.items) {
          item.success = false;
          item.error_message += error_message + ";";
        }
      } else {
        for (size_t i = 0; i < state.items.size(); ++i) {
          if (i < completed_block_ids.size()) {
            state.items[i].dst_host_block_id = completed_block_ids[i];
          }
        }
      }

      if (state.completed_workers >= state.expected_workers) {
        completion.save_id = save_id;
        completion.items = std::move(state.items);
        active_saves_.erase(it);
        push_completion = true;
      }
    }
  }

  if (push_completion) {
    store_->PushSaveCompletion(std::move(completion));
    VLOG(1) << "Save fully completed for save_id " << save_id
            << ", pushing completion to store";
  }
}

void RaidenControllerEmbedded::EvictWorkQueuePollerLoop() {
  while (!stopping_) {
    EvictRequest req;
    if (!store_->PopEvictWork(req)) {
      if (stopping_) break;
      continue;
    }

    if (req.block_hashes.empty()) continue;
    uint64_t evict_id = req.evict_id;

    std::vector<ControlRequest> listener_reqs(num_listeners_);
    for (size_t l = 0; l < num_listeners_; ++l) {
      listener_reqs[l].set_command(ControlRequest::COMMAND_EXECUTE_EVICT);
      listener_reqs[l].mutable_evict_request()->set_uuid(evict_id);
    }

    size_t num_blocks = req.block_hashes.size();

    for (size_t b = 0; b < num_blocks; ++b) {
      std::string hash = req.block_hashes[b];
      int host_id = req.host_block_ids[b];

      size_t global_shard_offset = 0;
      for (size_t l = 0; l < num_listeners_; ++l) {
        size_t num_local_shards = listener_shard_counts_[l];
        auto* shard_schedules = listener_reqs[l]
                                    .mutable_evict_request()
                                    ->mutable_shard_evict_schedules();

        for (size_t local_sh = 0; local_sh < num_local_shards; ++local_sh) {
          auto& schedule = (*shard_schedules)[static_cast<int32_t>(local_sh)];
          auto* entry = schedule.add_entries();
          entry->set_host_block_id(host_id);
        }
        global_shard_offset += num_local_shards;
      }
    }

    ControllerEvictState evict_state;
    evict_state.evict_id = evict_id;
    for (size_t i = 0; i < req.block_hashes.size(); ++i) {
      EvictCompletionItem c_item;
      c_item.block_hash = req.block_hashes[i];
      c_item.success = true;
      evict_state.items.push_back(c_item);
    }

    size_t active_listeners = 0;
    for (size_t l = 0; l < num_listeners_; ++l) {
      if (listener_shard_counts_[l] == 0) continue;
      active_listeners++;
    }
    evict_state.expected_workers = active_listeners;

    {
      absl::MutexLock lock(&pending_mu_);
      active_evicts_[evict_id] = std::move(evict_state);
    }

    for (size_t l = 0; l < num_listeners_; ++l) {
      if (listener_shard_counts_[l] == 0) continue;

      int port = local_worker_port_ + l;
      std::string worker_addr = absl::StrCat("localhost:", port);

      ControlResponse resp;
      absl::Status status = SendRpcSync(worker_addr, listener_reqs[l], resp);
      if (!status.ok() || !resp.success()) {
        LOG(ERROR) << "Failed to send COMMAND_EXECUTE_EVICT to listener " << l
                   << ": " << status << " " << resp.message();
        HandleEvictCompletionFromWorker(
            evict_id, /*success=*/false,
            absl::StrCat("Failed to send RPC: ", status.message(), " ",
                         resp.message()));
      } else {
        VLOG(1) << "Sent COMMAND_EXECUTE_EVICT to listener " << l
                << " for evict_id " << evict_id;
      }
    }
  }
}

void RaidenControllerEmbedded::HandleEvictCompletionFromWorker(
    uint64_t evict_id, bool success, const std::string& error_message) {
  EvictCompletion completion;
  bool push_completion = false;

  {
    absl::MutexLock lock(&pending_mu_);
    auto it = active_evicts_.find(evict_id);
    if (it != active_evicts_.end()) {
      auto& state = it->second;
      state.completed_workers++;
      if (!success) {
        for (auto& item : state.items) {
          item.success = false;
          item.error_message += error_message + ";";
        }
      }

      if (state.completed_workers >= state.expected_workers) {
        completion.evict_id = evict_id;
        completion.items = std::move(state.items);
        active_evicts_.erase(it);
        push_completion = true;
      }
    }
  }

  if (push_completion) {
    store_->PushEvictCompletion(std::move(completion));
    VLOG(1) << "Evict fully completed for evict_id " << evict_id
            << ", pushing completion to store";
  }
}

}  // namespace kv_cache
}  // namespace tpu_raiden
