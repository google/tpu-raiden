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
#include <optional>
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

// Expands the parallel chunk arrays (src/dst offsets + sizes) of a D2H/H2D
// request into unit-block arrays where every size == block_size. A chunk whose
// size is n*block_size becomes n consecutive single-block entries, emitted in
// chunk order so the prefill and decode sides stay aligned block-for-block.
// src is expanded in lockstep with dst when present; an empty src (a DecodeH2D
// whose source offsets are filled later from the peer's block ids) is left
// empty. After this every downstream stage (staging block-id derivation, H2H
// transport, NOTIFY, H2D src reconstruction, Unlock) sees exactly one block per
// entry, which is the invariant the rest of the orchestrator already assumes.
// Requires sizes[i] > 0 and sizes[i] % block_size == 0.
absl::Status ExpandChunksToBlocks(DisaggTransferRequest& req,
                                  int64_t block_size) {
  if (block_size <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("block_size must be positive, got ", block_size));
  }
  if (req.dst_offsets.size() != req.sizes.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "dst_offsets and sizes must have equal length, got ",
        req.dst_offsets.size(), " vs ", req.sizes.size()));
  }
  const bool have_src = !req.src_offsets.empty();
  if (have_src && req.src_offsets.size() != req.dst_offsets.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "src_offsets and dst_offsets must have equal length, got ",
        req.src_offsets.size(), " vs ", req.dst_offsets.size()));
  }
  std::vector<int64_t> esrc, edst, esize;
  for (size_t i = 0; i < req.dst_offsets.size(); ++i) {
    const int64_t size = req.sizes[i];
    if (size <= 0 || size % block_size != 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "sizes[", i, "]=", size,
          " must be a positive multiple of block_size=", block_size));
    }
    const int64_t n = size / block_size;
    for (int64_t b = 0; b < n; ++b) {
      if (have_src) esrc.push_back(req.src_offsets[i] + b * block_size);
      edst.push_back(req.dst_offsets[i] + b * block_size);
      esize.push_back(block_size);
    }
  }
  if (have_src) req.src_offsets = std::move(esrc);
  req.dst_offsets = std::move(edst);
  req.sizes = std::move(esize);
  return absl::OkStatus();
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
  for (int i = 0; i < worker_parallelism_; ++i) {
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
  uint64_t uuid = request.uuid;
  LOG(INFO) << "[Orchestrator] Submitting request " << uuid
            << " (type: " << static_cast<int>(request.type) << ")";
  orchestrator_queue_.Push({Event::Type::kExternalRequest,
                            uuid,
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
    std::function<void(std::optional<std::string>)> callback,
    absl::Status status) {
  if (callback) {
    callback(status.ok() ? std::nullopt
                         : std::optional<std::string>(
                               std::string(status.message())));
  }
}

void DisaggKVCacheManagerBase::OrchestrationLoop() {
  absl::flat_hash_map<uint64_t, DisaggTransferRequest> active_requests;

  Event event;
  while (orchestrator_queue_.Pop(event)) {
    LOG(INFO) << "[Orchestrator] Popped event type: "
              << static_cast<int>(event.type) << " for request "
              << event.uuid;
    switch (event.type) {
      case Event::Type::kExternalRequest: {
        uint64_t uuid = event.uuid;
        active_requests[uuid] = std::move(event.request);
        auto& req = active_requests[uuid];

        // Normalize the chunk plan into per-block form so all downstream stages
        // see one block per entry. Two cases for D2H/H2D requests:
        //   - await_pull producer (kPrefillD2H + empty dst_offsets): the manager
        //     owns the staging, so AUTO-ALLOCATE the host blocks and synthesize
        //     dst_offsets from them.
        //   - everything else (caller-supplied dst_offsets): just expand
        //     multi-block chunks (sizes[i] a multiple of block_size).
        // Raw H2H requests address blocks directly and are left untouched.
        if (req.type == DisaggTransferRequest::Type::kPrefillD2H ||
            req.type == DisaggTransferRequest::Type::kDecodeH2D) {
          absl::Status norm_status;
          if (req.type == DisaggTransferRequest::Type::kPrefillD2H &&
              req.dst_offsets.empty()) {
            norm_status = AutoAllocateStaging(req);
          } else {
            norm_status = ExpandChunksToBlocks(req, block_size_);
          }
          if (!norm_status.ok()) {
            LOG(ERROR) << "[Orchestrator] Request " << uuid
                       << ": invalid transfer plan: " << norm_status;
            InvokeCallback(req.callback, norm_status);
            active_requests.erase(uuid);
            break;
          }
        }

        if (req.type == DisaggTransferRequest::Type::kPrefillD2H) {
          LOG(INFO) << "[Orchestrator] Request " << uuid
                    << ": Triggering Prefill D2H";
          local_work_queue_.Push(req);
        } else if (req.type == DisaggTransferRequest::Type::kDecodeH2D) {
          // The decode is the puller: it sets up first and waits for the
          // producer's NOTIFY_READY (kPullReady) to trigger the H2H Read.
          LOG(INFO) << "[Orchestrator] Request " << uuid
                    << ": Waiting for NOTIFY_READY to trigger the pull.";
        } else if (req.type == DisaggTransferRequest::Type::kH2HRead) {
          LOG(INFO) << "[Orchestrator] Request " << uuid
                    << ": Triggering H2H transfer";
          h2h_work_queue_.Push(req);
        }
        break;
      }
      case Event::Type::kLocalComplete: {
        auto it = active_requests.find(event.uuid);
        if (it == active_requests.end()) {
          LOG(ERROR) << "[Orchestrator] Unknown request ID completed locally: "
                     << event.uuid;
          continue;
        }
        auto& req = it->second;
        LOG(INFO) << "[Orchestrator] Request " << event.uuid
                  << ": Local transfer completed with status: " << event.status;
        if (!event.status.ok()) {
          InvokeCallback(req.callback, event.status);
          active_requests.erase(it);
          continue;
        }

        if (req.type == DisaggTransferRequest::Type::kPrefillD2H) {
          // Our local staging block ids for this request.
          req.block_ids.clear();
          for (int64_t offset : req.dst_offsets) {
            req.block_ids.push_back(offset / block_size_);
          }
          // Don't transfer; tell the decode the data is staged and where to read
          // it. Keep the request pending until PULL_COMPLETE so the staging
          // buffer isn't reused before the decode has pulled.
          LOG(INFO) << "[Orchestrator] Request " << event.uuid
                    << ": D2H complete, sending NOTIFY_READY to " << req.peer;
          // Wire format: NOTIFY_READY:<uuid>:<staging_block_ids>:<src_offsets>
          // The trailing src_offsets (our expanded source region) lets the
          // consumer validate it is pulling the region its pull() asked for.
          std::string msg = absl::StrCat("NOTIFY_READY:", req.uuid, ":");
          for (size_t i = 0; i < req.block_ids.size(); ++i) {
            absl::StrAppend(&msg, req.block_ids[i]);
            if (i + 1 < req.block_ids.size()) absl::StrAppend(&msg, ",");
          }
          absl::StrAppend(&msg, ":");
          for (size_t i = 0; i < req.src_offsets.size(); ++i) {
            absl::StrAppend(&msg, req.src_offsets[i]);
            if (i + 1 < req.src_offsets.size()) absl::StrAppend(&msg, ",");
          }
          absl::Status mq_status = SendZmqMessage(req.peer, msg).status();
          if (!mq_status.ok()) {
            LOG(ERROR) << "[Orchestrator] Failed to send NOTIFY_READY to "
                       << req.peer << ": " << mq_status;
            InvokeCallback(req.callback, mq_status);
            active_requests.erase(it);
          }
          // else: leave pending; completion happens on kPullComplete.
        } else if (req.type == DisaggTransferRequest::Type::kDecodeH2D) {
          LOG(INFO) << "[Orchestrator] Request " << event.uuid
                    << ": H2D complete, request fully done!";
          // Ack the prefill so it can release its staging buffer.
          {
            std::string msg = absl::StrCat("PULL_COMPLETE:", req.uuid, ":");
            absl::Status mq_status = SendZmqMessage(req.peer, msg).status();
            if (!mq_status.ok()) {
              LOG(ERROR) << "[Orchestrator] Failed to send PULL_COMPLETE to "
                         << req.peer << ": " << mq_status;
            }
          }
          // Unlock host blocks. Must hold block_manager_mutex_ because the
          // transport's receiver threads concurrently Allocate() on it.
          if (block_manager_) {
            absl::Status unlock_status;
            {
              absl::MutexLock lock(&block_manager_mutex_);
              unlock_status = block_manager_->Unlock(req.block_ids);
            }
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
        auto it = active_requests.find(event.uuid);
        if (it == active_requests.end()) {
          LOG(ERROR) << "[Orchestrator] Unknown request ID completed H2H: "
                     << event.uuid;
          continue;
        }
        auto& req = it->second;
        LOG(INFO) << "[Orchestrator] Request " << event.uuid
                  << ": H2H transfer completed with status: " << event.status;
        if (!event.status.ok()) {
          InvokeCallback(req.callback, event.status);
          active_requests.erase(it);
          continue;
        }

        if (req.type == DisaggTransferRequest::Type::kH2HRead) {
          LOG(INFO) << "[Orchestrator] Request " << event.uuid
                    << ": H2H Read complete, triggering local H2D";
          // The pull returned the decode-LOCAL block ids where the data landed
          // (active_requests still holds the remote/prefill ids we pulled from),
          // so refresh and derive the H2D source offsets from them.
          req.block_ids = event.request.block_ids;
          req.src_offsets.clear();
          for (int bid : req.block_ids) {
            req.src_offsets.push_back(bid * block_size_);
          }
          req.type = DisaggTransferRequest::Type::kDecodeH2D;
          local_work_queue_.Push(req);
        }
        break;
      }
      case Event::Type::kPullReady: {
        // PULL (decode side): the prefill staged the data and told us its
        // remote block ids. Pull them via H2H Read. Assumes the DECODE_H2D
        // request was already submitted (the decode is the puller; it sets up
        // first), so it is present in active_requests with peer + dst_offsets.
        uint64_t uuid = event.uuid;
        auto it = active_requests.find(uuid);
        if (it == active_requests.end()) {
          LOG(ERROR) << "[Orchestrator] kPullReady for unknown request "
                     << uuid << " (decode request not submitted yet?)";
          break;
        }
        auto& existing = it->second;
        // Identity validation: the source region the producer staged (carried in
        // NOTIFY_READY) must match the source region this consumer's pull()
        // asked for. Both are expanded per-block, so a direct compare suffices.
        // Only enforced when the consumer actually supplied src_offsets (the
        // public pull() always does); the low-level path may omit them and opt
        // out of the check.
        if (!existing.src_offsets.empty() &&
            existing.src_offsets != event.src_offsets) {
          absl::Status mismatch = absl::InvalidArgumentError(absl::StrCat(
              "pull(uuid=", uuid, ") source region does not match the "
              "producer's await_pull (consumer ", existing.src_offsets.size(),
              " blocks vs producer ", event.src_offsets.size(), ")"));
          LOG(ERROR) << "[Orchestrator] " << mismatch;
          InvokeCallback(existing.callback, mismatch);
          active_requests.erase(it);
          break;
        }
        existing.block_ids = std::move(event.block_ids);  // remote (prefill) ids
        existing.type = DisaggTransferRequest::Type::kH2HRead;
        LOG(INFO) << "[Orchestrator] Request " << uuid
                  << ": peer ready, triggering H2H Read (pull) from "
                  << existing.peer;
        h2h_work_queue_.Push(existing);
        break;
      }
      case Event::Type::kPullComplete: {
        // PULL (prefill side): the decode finished pulling and acked. The
        // staging buffer can be released; complete the request.
        uint64_t uuid = event.uuid;
        auto it = active_requests.find(uuid);
        if (it == active_requests.end()) {
          LOG(ERROR) << "[Orchestrator] kPullComplete for unknown request "
                     << uuid;
          break;
        }
        LOG(INFO) << "[Orchestrator] Request " << uuid
                  << ": pull complete ack received, request fully done!";
        // Release the producer-side staging blocks we allocated in
        // AutoAllocateStaging (await_pull). Gated on owns_staging so the
        // low-level path (caller-supplied, un-locked staging) is never
        // unlocked. No block_manager_mutex_ needed here (unlike the decode-side
        // Unlock): on the producer the transport only READS its host blocks to
        // serve the pull and never calls AllocateBlocks, so the orchestrator is
        // the sole thread touching block_manager_ for this manager.
        if (it->second.owns_staging && block_manager_) {
          absl::Status unlock_status =
              block_manager_->Unlock(it->second.block_ids);
          if (!unlock_status.ok()) {
            LOG(ERROR) << "[Orchestrator] Failed to unlock staging blocks on "
                          "Prefill: "
                       << unlock_status;
          }
        }
        InvokeCallback(it->second.callback, absl::OkStatus());
        active_requests.erase(it);
        break;
      }
    }
  }
  LOG(INFO) << "[Orchestrator] OrchestrationLoop exiting";
}

absl::Status DisaggKVCacheManagerBase::AutoAllocateStaging(
    DisaggTransferRequest& req) {
  if (block_size_ <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("block_size must be positive, got ", block_size_));
  }
  if (req.src_offsets.size() != req.sizes.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "src_offsets and sizes must have equal length, got ",
        req.src_offsets.size(), " vs ", req.sizes.size()));
  }
  int64_t total_blocks = 0;
  for (size_t i = 0; i < req.sizes.size(); ++i) {
    if (req.sizes[i] <= 0 || req.sizes[i] % block_size_ != 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "sizes[", i, "]=", req.sizes[i],
          " must be a positive multiple of block_size=", block_size_));
    }
    total_blocks += req.sizes[i] / block_size_;
  }

  // Allocate (and lock) the host staging blocks. AllocateBlocks takes the
  // block_manager_mutex_ internally and releases it on return; the block-level
  // lock (lock=true) is what persists, holding the staging until the matching
  // Unlock on kPullComplete (gated by owns_staging) so it is not reused before
  // the decode has pulled.
  std::vector<int> staging_ids;
  {
    auto ids_or = AllocateBlocks(total_blocks, req.entity_id);
    if (!ids_or.ok()) return ids_or.status();
    staging_ids = std::move(ids_or.value());
  }

  std::vector<int64_t> esrc, edst, esize;
  esrc.reserve(total_blocks);
  edst.reserve(total_blocks);
  esize.reserve(total_blocks);
  size_t idx = 0;
  for (size_t i = 0; i < req.sizes.size(); ++i) {
    const int64_t n = req.sizes[i] / block_size_;
    for (int64_t b = 0; b < n; ++b) {
      esrc.push_back(req.src_offsets[i] + b * block_size_);
      edst.push_back(static_cast<int64_t>(staging_ids[idx]) * block_size_);
      esize.push_back(block_size_);
      ++idx;
    }
  }
  req.src_offsets = std::move(esrc);
  req.dst_offsets = std::move(edst);
  req.sizes = std::move(esize);
  req.owns_staging = true;
  return absl::OkStatus();
}

void DisaggKVCacheManagerBase::LocalTransferLoop() {
  DisaggTransferRequest req;
  while (local_work_queue_.Pop(req)) {
    if (req.type == DisaggTransferRequest::Type::kPrefillD2H) {
      auto future_or = D2h(req.src_offsets, req.dst_offsets, req.sizes);
      if (!future_or.ok()) {
        orchestrator_queue_.Push({Event::Type::kLocalComplete,
                                  req.uuid,
                                  future_or.status(),
                                  req,
                                  {},
                                  ""});
      } else {
        future_or.value().OnReady([this, req](absl::Status status) {
          orchestrator_queue_.Push({Event::Type::kLocalComplete,
                                    req.uuid,
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
                                  req.uuid,
                                  future_or.status(),
                                  req,
                                  {},
                                  ""});
      } else {
        future_or.value().OnReady([this, req](absl::Status status) {
          orchestrator_queue_.Push({Event::Type::kLocalComplete,
                                    req.uuid,
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

    // The H2H Read (pull) targets the producer peer; resolve it.
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
    std::string peer_endpoint =
        absl::StrCat(peer_info.ip, ":", peer_info.transport_port);

    if (status.ok() && req.type == DisaggTransferRequest::Type::kH2HRead) {
      // PULL: read the peer's blocks (req.block_ids = remote ids) into our host
      // buffer. Returns the LOCAL block ids the data was placed into.
      auto res_or = H2hReadDirect(peer_endpoint, req.block_ids, req.entity_id);
      if (!res_or.ok()) {
        status = res_or.status();
      } else {
        req.block_ids = res_or.value();
      }
    }
    orchestrator_queue_.Push(
        {Event::Type::kH2hComplete, req.uuid, status, req, {}, ""});
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

        } else if (cmd == "NOTIFY_READY") {
          // PULL: prefill staged data and is advertising its remote block ids
          // (and its source region for validation) for the decode to pull.
          // Wire format: NOTIFY_READY:<uuid>:<block_ids>:<src_offsets>
          if (parts.size() != 4) {
            absl::Status s = SendZmqReply("ERROR:Invalid NOTIFY_READY");
            if (!s.ok()) LOG(ERROR) << s;
            continue;
          }
          uint64_t uuid = std::stoull(parts[1]);
          std::vector<int> block_ids;
          for (const std::string& s :
               std::vector<std::string>(absl::StrSplit(parts[2], ','))) {
            if (!s.empty()) block_ids.push_back(std::stoi(s));
          }
          std::vector<int64_t> src_offsets;
          for (const std::string& s :
               std::vector<std::string>(absl::StrSplit(parts[3], ','))) {
            if (!s.empty()) src_offsets.push_back(std::stoll(s));
          }

          Event event;
          event.type = Event::Type::kPullReady;
          event.uuid = uuid;
          event.block_ids = std::move(block_ids);
          event.src_offsets = std::move(src_offsets);
          orchestrator_queue_.Push(std::move(event));

          absl::Status s = SendZmqReply("OK");
          if (!s.ok()) LOG(ERROR) << s;
        } else if (cmd == "PULL_COMPLETE") {
          // PULL: decode finished pulling; ack back to the prefill.
          if (parts.size() < 2) {
            absl::Status s = SendZmqReply("ERROR:Invalid PULL_COMPLETE");
            if (!s.ok()) LOG(ERROR) << s;
            continue;
          }
          Event event;
          event.type = Event::Type::kPullComplete;
          event.uuid = std::stoull(parts[1]);
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
