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

#ifndef THIRD_PARTY_TPU_RAIDEN_WEIGHT_SYNC_WEIGHT_SYNCHRONIZER_CONTROL_SERVICE_H_
#define THIRD_PARTY_TPU_RAIDEN_WEIGHT_SYNC_WEIGHT_SYNCHRONIZER_CONTROL_SERVICE_H_

#include <atomic>
#include <string>
#include <thread>  // NOLINT
#include <vector>

namespace tpu_raiden {
namespace weight_sync {

class WeightSynchronizerBase;

// TCP Socket Server Daemon that runs natively in C++ to accept Control-Plane
// management RPC commands (like PushWeights and Shutdown) directly from the
// RL Coordinator or Controller task, bypassing Python servicer overhead.
class WeightSynchronizerControlService final {
 public:
  WeightSynchronizerControlService(WeightSynchronizerBase* engine,
                                   int control_port);
  ~WeightSynchronizerControlService();

  WeightSynchronizerControlService(const WeightSynchronizerControlService&) =
      delete;
  WeightSynchronizerControlService& operator=(
      const WeightSynchronizerControlService&) = delete;

  int control_port() const { return control_port_; }
  bool is_active() const { return !stopping_; }

 private:
  void ListenerLoop();
  void ConnectionWorker(int client_fd);

  WeightSynchronizerBase* engine_;
  int control_port_;
  int server_fd_ = -1;
  std::atomic<bool> stopping_{false};

  std::thread listener_thread_;
  std::vector<std::thread> worker_threads_;
};

}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_WEIGHT_SYNC_WEIGHT_SYNCHRONIZER_CONTROL_SERVICE_H_
