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

#include "kv_cache/disagg_kv_cache_manager_base.h"

#include <chrono>  // NOLINT
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "core/raw_transfer_core.h"
#include "third_party/zeromq/include/zmq.hpp"

namespace tpu_raiden {
namespace kv_cache {

namespace {
zmq::context_t& GetGlobalZmqContext() {
  static auto* context = new zmq::context_t(1);
  return *context;
}
}  // namespace

DisaggKVCacheManagerBase::~DisaggKVCacheManagerBase() { Stop(); }

absl::Status DisaggKVCacheManagerBase::Start() {
  absl::MutexLock lock(&running_mutex_);
  if (running_) {
    return absl::OkStatus();
  }

  try {
    zmq_listener_socket_ =
        std::make_unique<zmq::socket_t>(GetGlobalZmqContext(), ZMQ_REP);
    zmq_listener_socket_->set(zmq::sockopt::linger, 0);
    zmq_listener_socket_->bind("tcp://*:*");

    char addr[1024];
    size_t size = sizeof(addr);
    zmq_listener_socket_->getsockopt(ZMQ_LAST_ENDPOINT, addr, &size);
    std::string addr_str(addr);
    size_t last_colon = addr_str.find_last_of(':');
    if (last_colon != std::string::npos) {
      zmq_control_port_ = std::stoi(addr_str.substr(last_colon + 1));
    } else {
      return absl::InternalError("Failed to get ZMQ ephemeral port");
    }
  } catch (const zmq::error_t& e) {
    return absl::InternalError(
        absl::StrCat("Failed to initialize ZMQ: ", e.what()));
  }

  running_ = true;

  orchestration_thread_ =
      std::thread(&DisaggKVCacheManagerBase::OrchestrationLoop, this);
  local_transfer_thread_ =
      std::thread(&DisaggKVCacheManagerBase::LocalTransferLoop, this);
  for (int i = 0; i < parallelism_; ++i) {
    h2h_transfer_threads_.push_back(
        std::thread(&DisaggKVCacheManagerBase::H2hTransferLoop, this));
  }
  listener_thread_ = std::thread(&DisaggKVCacheManagerBase::ListenerLoop, this);

  LOG(INFO) << "DisaggKVCacheManagerBase started. ZMQ port: "
            << zmq_control_port_;
  return absl::OkStatus();
}

void DisaggKVCacheManagerBase::Stop() {
  LOG(INFO) << "[Stop] Stop() started";
  {
    absl::MutexLock lock(&running_mutex_);
    if (!running_) {
      LOG(INFO) << "[Stop] Already stopped or not running";
      return;
    }
    running_ = false;
  }

  // Shutdown queues to unblock threads
  local_work_queue_.Shutdown();
  h2h_work_queue_.Shutdown();
  orchestrator_queue_.Shutdown();

  LOG(INFO) << "[Stop] Joining orchestration thread...";
  if (orchestration_thread_.joinable()) orchestration_thread_.join();
  LOG(INFO) << "[Stop] Joining local transfer thread...";
  if (local_transfer_thread_.joinable()) local_transfer_thread_.join();
  LOG(INFO) << "[Stop] Joining H2H transfer threads...";
  for (auto& thread : h2h_transfer_threads_) {
    if (thread.joinable()) thread.join();
  }
  h2h_transfer_threads_.clear();
  LOG(INFO) << "[Stop] Joining listener thread...";
  if (listener_thread_.joinable()) listener_thread_.join();

  // Safely destroy socket after the listener thread is dead
  LOG(INFO) << "[Stop] Resetting zmq_listener_socket_...";
  zmq_listener_socket_.reset();

  LOG(INFO) << "DisaggKVCacheManagerBase stopped.";
}

absl::Status DisaggKVCacheManagerBase::SubmitRequest(
    DisaggTransferRequest request) {
  absl::MutexLock lock(&running_mutex_);
  if (!running_) {
    return absl::FailedPreconditionError("Manager is not running");
  }
  int64_t req_id = request.request_id;
  LOG(INFO) << "[Orchestrator] Submitting request " << req_id
            << " (type: " << static_cast<int>(request.type) << ")";
  orchestrator_queue_.Push({Event::Type::kExternalRequest,
                            req_id,
                            absl::OkStatus(),
                            std::move(request),
                            {},
                            ""});
  return absl::OkStatus();
}

void DisaggKVCacheManagerBase::RegisterPeer(const std::string& name,
                                            const std::string& ip, int zmq_port,
                                            int transport_port) {
  absl::MutexLock lock(&peer_map_mutex_);
  peers_[name] = {ip, zmq_port, transport_port};
  LOG(INFO) << "Manually registered peer: " << name << " at " << ip
            << " (ZMQ:" << zmq_port << ", Transport:" << transport_port << ")";
}

absl::StatusOr<std::string> DisaggKVCacheManagerBase::SendZmqMessage(
    const std::string& peer, const std::string& message) {
  PeerInfo peer_info;
  {
    absl::MutexLock lock(&peer_map_mutex_);
    auto it = peers_.find(peer);
    if (it == peers_.end()) {
      return absl::NotFoundError(absl::StrCat("Peer ", peer, " not found"));
    }
    peer_info = it->second;
  }

  std::string endpoint =
      absl::StrCat("tcp://", peer_info.ip, ":", peer_info.control_port);
  LOG(INFO) << "[ZMQ Client] Sending message to peer " << peer << " at "
            << endpoint << ": " << message;
  try {
    zmq::socket_t socket(GetGlobalZmqContext(), ZMQ_REQ);
    socket.set(zmq::sockopt::linger, 0);
    socket.connect(endpoint);

    zmq::message_t req(message.size());
    std::memcpy(req.data(), message.data(), message.size());
    socket.send(req, zmq::send_flags::none);

    LOG(INFO) << "[ZMQ Client] Message sent, waiting for reply...";
    zmq::message_t resp;
    auto recv_res = socket.recv(resp, zmq::recv_flags::none);
    if (!recv_res.has_value()) {
      return absl::InternalError("ZMQ recv failed or interrupted");
    }

    std::string resp_str(static_cast<char*>(resp.data()), resp.size());
    LOG(INFO) << "[ZMQ Client] Received reply: " << resp_str;
    if (resp_str != "OK" && resp_str.rfind("CONNECT_RESP", 0) != 0) {
      return absl::InternalError(
          absl::StrCat("Peer returned error: ", resp_str));
    }
    return resp_str;
  } catch (const zmq::error_t& e) {
    LOG(ERROR) << "[ZMQ Client] ZMQ error: " << e.what();
    return absl::InternalError(absl::StrCat("ZMQ error: ", e.what()));
  }
}

absl::Status DisaggKVCacheManagerBase::SendZmqReply(const std::string& reply) {
  LOG(INFO) << "[ZMQ Server] Sending reply: " << reply;
  try {
    zmq_listener_socket_->send(zmq::message_t(reply.data(), reply.size()),
                               zmq::send_flags::none);
  } catch (const zmq::error_t& e) {
    return absl::InternalError(
        absl::StrCat("Failed to send ZMQ reply: ", e.what()));
  }
  return absl::OkStatus();
}

void DisaggKVCacheManagerBase::InvokeCallback(
    std::function<void(absl::Status)> callback, absl::Status status) {
  if (callback) {
    callback(status);
  }
}

void DisaggKVCacheManagerBase::OrchestrationLoop() {
  absl::flat_hash_map<int64_t, DisaggTransferRequest> active_requests;

  Event event;
  while (orchestrator_queue_.Pop(event)) {
    LOG(INFO) << "[Orchestrator] Popped event type: "
              << static_cast<int>(event.type) << " for request "
              << event.request_id;
    switch (event.type) {
      case Event::Type::kExternalRequest: {
        int64_t req_id = event.request_id;
        active_requests[req_id] = std::move(event.request);
        const auto& req = active_requests[req_id];

        if (req.type == DisaggTransferRequest::Type::kPrefillD2H) {
          LOG(INFO) << "[Orchestrator] Request " << req_id
                    << ": Triggering Prefill D2H";
          local_work_queue_.Push(req);
        } else if (req.type == DisaggTransferRequest::Type::kDecodeH2D) {
          // Check if we already have peer notification with block IDs
          auto it = active_requests.find(req_id);
          if (it != active_requests.end() && !it->second.block_ids.empty()) {
            LOG(INFO) << "[Orchestrator] Request " << req_id
                      << ": Peer notification already arrived. Triggering H2D.";
            auto& existing = it->second;
            existing.src_offsets.clear();
            for (int bid : existing.block_ids) {
              existing.src_offsets.push_back(bid * block_size_);
            }
            existing.dst_offsets = req.dst_offsets;
            existing.sizes = req.sizes;
            existing.callback = std::move(req.callback);
            local_work_queue_.Push(existing);
          } else {
            LOG(INFO) << "[Orchestrator] Request " << req_id
                      << ": Waiting for peer notification to trigger H2D.";
          }
        } else if (req.type == DisaggTransferRequest::Type::kH2HWrite ||
                   req.type == DisaggTransferRequest::Type::kH2HRead) {
          LOG(INFO) << "[Orchestrator] Request " << req_id
                    << ": Triggering H2H transfer";
          h2h_work_queue_.Push(req);
        }
        break;
      }
      case Event::Type::kLocalComplete: {
        auto it = active_requests.find(event.request_id);
        if (it == active_requests.end()) {
          LOG(ERROR) << "[Orchestrator] Unknown request ID completed locally: "
                     << event.request_id;
          continue;
        }
        auto& req = it->second;
        LOG(INFO) << "[Orchestrator] Request " << event.request_id
                  << ": Local transfer completed with status: " << event.status;
        if (!event.status.ok()) {
          InvokeCallback(req.callback, event.status);
          active_requests.erase(it);
          continue;
        }

        if (req.type == DisaggTransferRequest::Type::kPrefillD2H) {
          LOG(INFO) << "[Orchestrator] Request " << event.request_id
                    << ": D2H complete, triggering H2H Write";
          req.type = DisaggTransferRequest::Type::kH2HWrite;
          req.block_ids.clear();
          for (int64_t offset : req.dst_offsets) {
            req.block_ids.push_back(offset / block_size_);
          }
          h2h_work_queue_.Push(req);
        } else if (req.type == DisaggTransferRequest::Type::kDecodeH2D) {
          LOG(INFO) << "[Orchestrator] Request " << event.request_id
                    << ": H2D complete, request fully done!";
          // Unlock host blocks
          if (block_manager_) {
            absl::Status unlock_status = block_manager_->Unlock(req.block_ids);
            if (!unlock_status.ok()) {
              LOG(ERROR) << "[Orchestrator] Failed to unlock blocks on Decode: "
                         << unlock_status;
            }
          }
          InvokeCallback(req.callback, absl::OkStatus());
          active_requests.erase(it);
        }
        break;
      }
      case Event::Type::kH2hComplete: {
        auto it = active_requests.find(event.request_id);
        if (it == active_requests.end()) {
          LOG(ERROR) << "[Orchestrator] Unknown request ID completed H2H: "
                     << event.request_id;
          continue;
        }
        auto& req = it->second;
        LOG(INFO) << "[Orchestrator] Request " << event.request_id
                  << ": H2H transfer completed with status: " << event.status;
        if (!event.status.ok()) {
          InvokeCallback(req.callback, event.status);
          active_requests.erase(it);
          continue;
        }

        if (req.type == DisaggTransferRequest::Type::kH2HWrite) {
          LOG(INFO) << "[Orchestrator] Request " << event.request_id
                    << ": H2H Write complete, sending ZMQ notification to "
                    << req.peer;
          // Notify peer via ZMQ
          std::string msg =
              absl::StrCat("NOTIFY_COMPLETE:", req.request_id, ":");
          for (size_t i = 0; i < req.block_ids.size(); ++i) {
            absl::StrAppend(&msg, req.block_ids[i]);
            if (i + 1 < req.block_ids.size()) absl::StrAppend(&msg, ",");
          }

          absl::Status mq_status = SendZmqMessage(req.peer, msg).status();
          if (!mq_status.ok()) {
            LOG(ERROR) << "[Orchestrator] Failed to send ZMQ notification to "
                       << req.peer << ": " << mq_status;
            InvokeCallback(req.callback, mq_status);
          } else {
            LOG(INFO) << "[Orchestrator] ZMQ notification successfully sent "
                         "and acked by "
                      << req.peer;
            InvokeCallback(req.callback, absl::OkStatus());
          }
          active_requests.erase(it);
        } else if (req.type == DisaggTransferRequest::Type::kH2HRead) {
          LOG(INFO) << "[Orchestrator] Request " << event.request_id
                    << ": H2H Read complete, triggering local H2D";
          req.type = DisaggTransferRequest::Type::kDecodeH2D;
          local_work_queue_.Push(req);
        }
        break;
      }
      case Event::Type::kPeerNotification: {
        int64_t req_id = event.request_id;
        LOG(INFO) << "[Orchestrator] Received peer notification for request "
                  << req_id << " with block IDs: ";
        for (int bid : event.block_ids) LOG(INFO) << "  " << bid;

        auto it = active_requests.find(req_id);
        if (it != active_requests.end()) {
          auto& existing = it->second;
          existing.block_ids = std::move(event.block_ids);
          if (existing.type == DisaggTransferRequest::Type::kDecodeH2D &&
              !existing.dst_offsets.empty()) {
            LOG(INFO) << "[Orchestrator] Request " << req_id
                      << ": Target offsets already present. Triggering H2D.";
            existing.src_offsets.clear();
            for (int bid : existing.block_ids) {
              existing.src_offsets.push_back(bid * block_size_);
            }
            local_work_queue_.Push(existing);
          } else {
            LOG(INFO) << "[Orchestrator] Request " << req_id
                      << ": Target offsets not present yet.";
          }
        } else {
          LOG(INFO) << "[Orchestrator] Request " << req_id
                    << ": JAX request not arrived yet. Storing notification.";
          // Notification arrived before JAX request, store it
          DisaggTransferRequest pending_req;
          pending_req.request_id = req_id;
          pending_req.type = DisaggTransferRequest::Type::kDecodeH2D;
          pending_req.block_ids = std::move(event.block_ids);
          active_requests[req_id] = std::move(pending_req);
        }
        break;
      }
    }
  }
  LOG(INFO) << "[Orchestrator] OrchestrationLoop exiting";
}

void DisaggKVCacheManagerBase::LocalTransferLoop() {
  DisaggTransferRequest req;
  while (local_work_queue_.Pop(req)) {
    if (req.type == DisaggTransferRequest::Type::kPrefillD2H) {
      auto future_or = D2h(req.src_offsets, req.dst_offsets, req.sizes);
      if (!future_or.ok()) {
        orchestrator_queue_.Push({Event::Type::kLocalComplete,
                                  req.request_id,
                                  future_or.status(),
                                  req,
                                  {},
                                  ""});
      } else {
        future_or.value().OnReady([this, req](absl::Status status) {
          orchestrator_queue_.Push({Event::Type::kLocalComplete,
                                    req.request_id,
                                    status,
                                    req,
                                    {},
                                    ""});
        });
      }
    } else if (req.type == DisaggTransferRequest::Type::kDecodeH2D) {
      auto future_or = H2d(req.src_offsets, req.dst_offsets, req.sizes);
      if (!future_or.ok()) {
        orchestrator_queue_.Push({Event::Type::kLocalComplete,
                                  req.request_id,
                                  future_or.status(),
                                  req,
                                  {},
                                  ""});
      } else {
        future_or.value().OnReady([this, req](absl::Status status) {
          orchestrator_queue_.Push({Event::Type::kLocalComplete,
                                    req.request_id,
                                    status,
                                    req,
                                    {},
                                    ""});
        });
      }
    }
  }
  LOG(INFO) << "[Local] LocalTransferLoop exiting";
}

void DisaggKVCacheManagerBase::H2hTransferLoop() {
  DisaggTransferRequest req;
  while (h2h_work_queue_.Pop(req)) {
    absl::Status status;
    if (req.type == DisaggTransferRequest::Type::kH2HWrite) {
      // Resolve peer to get IP:Port
      PeerInfo peer_info;
      {
        absl::MutexLock lock(&peer_map_mutex_);
        auto it = peers_.find(req.peer);
        if (it == peers_.end()) {
          status = absl::NotFoundError(
              absl::StrCat("Peer ", req.peer, " not found"));
        } else {
          peer_info = it->second;
        }
      }

      if (status.ok()) {
        std::string peer_endpoint =
            absl::StrCat(peer_info.ip, ":", peer_info.transport_port);
        // H2hWriteDirect is blocking
        auto res_or =
            H2hWriteDirect(peer_endpoint, req.block_ids, req.entity_id);
        if (!res_or.ok()) {
          status = res_or.status();
        } else {
          // Update request with allocated block IDs on receiver
          req.block_ids = res_or.value();
          status = absl::OkStatus();
        }
      }
    } else if (req.type == DisaggTransferRequest::Type::kH2HRead) {
      // Handle H2H Read if needed (similar to Write but calls H2hReadDirect)
      status = absl::UnimplementedError("H2H Read is not implemented yet");
    }
    orchestrator_queue_.Push(
        {Event::Type::kH2hComplete, req.request_id, status, req, {}, ""});
  }
  LOG(INFO) << "[H2H] H2hTransferLoop exiting";
}

void DisaggKVCacheManagerBase::ListenerLoop() {
  zmq::pollitem_t items[] = {
      {static_cast<void*>(*zmq_listener_socket_), 0, ZMQ_POLLIN, 0}};

  while (true) {
    {
      absl::MutexLock lock(&running_mutex_);
      if (!running_) break;
    }
    try {
      // Poll with 100ms timeout to allow clean shutdown check
      int rc = zmq::poll(items, 1, std::chrono::milliseconds(100));
      if (rc < 0) {
        LOG(ERROR) << "[ZMQ Server] Poll error";
        continue;
      }
      if (rc == 0) {
        // Timeout, re-evaluate loop condition (running_)
        continue;
      }

      if (items[0].revents & ZMQ_POLLIN) {
        zmq::message_t request;
        auto recv_res =
            zmq_listener_socket_->recv(request, zmq::recv_flags::none);
        if (!recv_res.has_value() || *recv_res == 0) {
          absl::MutexLock lock(&running_mutex_);
          if (!running_) break;
          continue;
        }

        std::string msg_str(static_cast<char*>(request.data()), request.size());
        LOG(INFO) << "Received ZMQ control message: " << msg_str;

        std::vector<std::string> parts = absl::StrSplit(msg_str, ':');
        if (parts.empty()) {
          absl::Status s = SendZmqReply("ERROR:Empty message");
          if (!s.ok()) LOG(ERROR) << s;
          continue;
        }

        std::string cmd = parts[0];
        if (cmd == "CONNECT_REQ") {
          if (parts.size() != 5) {
            absl::Status s = SendZmqReply("ERROR:Invalid CONNECT_REQ");
            if (!s.ok()) LOG(ERROR) << s;
            continue;
          }
          std::string peer_name = parts[1];
          std::string peer_ip = parts[2];
          int peer_zmq_port = std::stoi(parts[3]);
          int peer_transport_port = std::stoi(parts[4]);

          {
            absl::MutexLock lock(&peer_map_mutex_);
            peers_[peer_name] = {peer_ip, peer_zmq_port, peer_transport_port};
          }

          LOG(INFO) << "Registered peer (incoming CONNECT_REQ): " << peer_name
                    << " at " << peer_ip << " (ZMQ:" << peer_zmq_port
                    << ", Transport:" << peer_transport_port << ")";

          int my_transport_port = local_port().value_or(0);
          std::string reply =
              absl::StrCat("CONNECT_RESP:ok:", my_transport_port);
          absl::Status s = SendZmqReply(reply);
          if (!s.ok()) LOG(ERROR) << s;

        } else if (cmd == "NOTIFY_COMPLETE") {
          if (parts.size() != 3) {
            absl::Status s = SendZmqReply("ERROR:Invalid NOTIFY_COMPLETE");
            if (!s.ok()) LOG(ERROR) << s;
            continue;
          }
          int64_t req_id = std::stoll(parts[1]);
          std::vector<std::string> block_strs = absl::StrSplit(parts[2], ',');
          std::vector<int> block_ids;
          for (const auto& s : block_strs) {
            if (!s.empty()) {
              block_ids.push_back(std::stoi(s));
            }
          }

          Event event;
          event.type = Event::Type::kPeerNotification;
          event.request_id = req_id;
          event.block_ids = std::move(block_ids);
          orchestrator_queue_.Push(std::move(event));

          absl::Status s = SendZmqReply("OK");
          if (!s.ok()) LOG(ERROR) << s;
        } else {
          absl::Status s = SendZmqReply("ERROR:Unknown command");
          if (!s.ok()) LOG(ERROR) << s;
        }
      }
    } catch (const zmq::error_t& e) {
      absl::MutexLock lock(&running_mutex_);
      if (e.num() == ETERM || !running_) {
        break;
      }
      LOG(ERROR) << "ZMQ Listener error: " << e.what();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
  LOG(INFO) << "[ZMQ Server] ListenerLoop exiting";
}

}  // namespace kv_cache
}  // namespace tpu_raiden
