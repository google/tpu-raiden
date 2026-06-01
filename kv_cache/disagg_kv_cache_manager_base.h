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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_DISAGG_KV_CACHE_MANAGER_BASE_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_DISAGG_KV_CACHE_MANAGER_BASE_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "kv_cache/kv_cache_manager_base.h"
#include "third_party/zeromq/include/zmq.hpp"

namespace tpu_raiden {
namespace kv_cache {

struct DisaggTransferRequest {
  // PULL-only transfer. kPrefillD2H (producer stages) and kDecodeH2D (consumer
  // loads) are the public request types; kH2HRead is the internal H2H Read stage
  // the orchestrator drives between them. The producer (kPrefillD2H) and
  // consumer (kDecodeH2D) of one transfer must share the same `uuid` and each
  // specify the other engine as `peer` (registered on both sides).
  enum class Type { kPrefillD2H, kDecodeH2D, kH2HRead };

  // uuid: globally-unique key that pairs a producer's await_pull with the
  //   matching consumer's pull during the handshake. Both sides must pass the
  //   SAME uuid; it is the correlation id used for active_requests, the ZMQ
  //   NOTIFY_READY / PULL_COMPLETE messages, and event routing.
  // req_id: opaque caller bookkeeping tag (e.g. an inference request id) carried
  //   through unchanged; NOT used for matching and never crosses the wire (each
  //   side tags its own).
  uint64_t uuid = 0;
  int64_t req_id = 0;
  Type type;

  // For local D2H/H2D
  std::vector<int64_t> src_offsets;
  std::vector<int64_t> dst_offsets;
  std::vector<int64_t> sizes;

  // For H2H
  std::string peer;
  std::vector<int> block_ids;
  int64_t entity_id = 0;

  // Internal (not exposed to callers): true when the manager allocated the
  // producer-side host staging blocks for this request (the await_pull path,
  // signalled by an empty dst_offsets on a pull kPrefillD2H). Drives the
  // staging Unlock on kPullComplete so caller-supplied (un-locked) blocks on
  // the low-level path are never unlocked.
  bool owns_staging = false;

  // Completion callback. The argument is std::nullopt on success, or the error
  // message on failure (NOT absl::Status: the nanobind absl::Status caster
  // raises on a non-OK status instead of passing it, which would prevent the
  // Python callback from ever observing an error -- see InvokeCallback).
  std::function<void(std::optional<std::string>)> callback;
};

template <typename T>
class ThreadSafeQueue {
 public:
  void Push(T value) {
    absl::MutexLock lock(&mutex_);
    queue_.push(std::move(value));
  }

  bool Pop(T& value) {
    absl::MutexLock lock(&mutex_);
    mutex_.Await(absl::Condition(this, &ThreadSafeQueue::CanPop));
    if (shutdown_ && queue_.empty()) {
      return false;
    }
    value = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  void Shutdown() {
    absl::MutexLock lock(&mutex_);
    shutdown_ = true;
  }

 private:
  bool CanPop() const ABSL_SHARED_LOCKS_REQUIRED(mutex_) {
    return !queue_.empty() || shutdown_;
  }

  std::queue<T> queue_ ABSL_GUARDED_BY(mutex_);
  mutable absl::Mutex mutex_;
  bool shutdown_ ABSL_GUARDED_BY(mutex_) = false;
};

class DisaggKVCacheManagerBase : public KVCacheManagerBase {
 public:
  using KVCacheManagerBase::KVCacheManagerBase;

  ~DisaggKVCacheManagerBase() override;

  absl::Status Start();
  void Stop();

  absl::Status SubmitRequest(DisaggTransferRequest request);

  void RegisterPeer(const std::string& name, const std::string& ip,
                    int zmq_port, int transport_port);

  int zmq_control_port() const { return zmq_control_port_; }

 protected:
  struct Event {
    enum class Type {
      kExternalRequest,
      kLocalComplete,
      kH2hComplete,
      kPullReady,    // prefill staged data, decode may pull (NOTIFY_READY)
      kPullComplete  // decode finished pulling (PULL_COMPLETE)
    };
    Type type;
    uint64_t uuid;
    absl::Status status;

    // For ExternalRequest
    DisaggTransferRequest request;

    // For PullReady: the producer's advertised staging block ids.
    std::vector<int> block_ids;
    std::string peer_name;
    // For PullReady: the producer's (expanded) src_offsets, carried in
    // NOTIFY_READY so the consumer can validate it is pulling the source region
    // its pull() asked for (identity check against the matching await_pull).
    std::vector<int64_t> src_offsets;
  };

  // Thread loops
  void OrchestrationLoop();
  void LocalTransferLoop();
  void H2hTransferLoop();
  void ListenerLoop();

  // Producer-side (await_pull) staging auto-allocation. Given a request that
  // carries only src_offsets + sizes (device source), allocates the host
  // staging blocks via AllocateBlocks and rewrites the request into per-block
  // form: src_offsets expanded in lockstep, dst_offsets = allocated_id *
  // block_size_, sizes = block_size_, owns_staging = true. Mirrors
  // KVCacheManagerBase::D2hAutoAllocate but feeds the disagg orchestration.
  absl::Status AutoAllocateStaging(DisaggTransferRequest& req);

  // ZMQ helper
  absl::StatusOr<std::string> SendZmqMessage(const std::string& peer,
                                             const std::string& message);
  absl::Status SendZmqReply(const std::string& reply);

  // Callback invoker to protect GIL in subclasses (JAX). Translates `status`
  // into the callback's argument: std::nullopt on success, the error message
  // on failure.
  virtual void InvokeCallback(
      std::function<void(std::optional<std::string>)> callback,
      absl::Status status);

  // Queues
  ThreadSafeQueue<Event> orchestrator_queue_;
  ThreadSafeQueue<DisaggTransferRequest> local_work_queue_;
  ThreadSafeQueue<DisaggTransferRequest> h2h_work_queue_;

  std::thread orchestration_thread_;
  std::thread local_transfer_thread_;
  std::vector<std::thread> h2h_transfer_threads_;
  std::thread listener_thread_;

  // Number of H2hTransferLoop worker threads draining h2h_work_queue_, i.e. how
  // many H2H transfers run concurrently. Distinct from transport_parallelism_
  // (the per-transfer TCP stream count). Set by subclasses before Start().
  int worker_parallelism_ = 1;

  bool running_ ABSL_GUARDED_BY(running_mutex_) = false;
  absl::Mutex running_mutex_;

  // ZMQ Control Plane members
  std::unique_ptr<zmq::socket_t> zmq_listener_socket_;
  int zmq_control_port_ = 0;

  struct PeerInfo {
    std::string ip;
    int control_port;
    int transport_port;
  };
  mutable absl::Mutex peer_map_mutex_;
  absl::flat_hash_map<std::string, PeerInfo> peers_
      ABSL_GUARDED_BY(peer_map_mutex_);
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_DISAGG_KV_CACHE_MANAGER_BASE_H_
