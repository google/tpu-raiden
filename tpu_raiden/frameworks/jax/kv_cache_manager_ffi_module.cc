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

#include <set>

#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "tpu_raiden/frameworks/jax/kv_cache_manager.h"
#include "tpu_raiden/frameworks/jax/kv_cache_manager_ffi.h"

namespace nb = nanobind;

NB_MODULE(_kv_cache_manager_ffi, m) {
  m.def("destroy_kv_cache", []() {
    std::set<tpu_raiden::KVCacheManagerWithTransfer*> unique_managers;
    for (int i = 0; i < 32; ++i) {
      if (tpu_raiden::kv_cache::g_kv_cache_managers[i] != nullptr) {
        // Unregister host pointers from the executor for this device before
        // deleting the manager
        auto platform_or =
            stream_executor::PlatformManager::PlatformWithName("TPU");
        if (!platform_or.ok()) {
          platform_or =
              stream_executor::PlatformManager::PlatformWithName("Deepsea");
        }
        if (!platform_or.ok()) {
          platform_or =
              stream_executor::PlatformManager::PlatformWithName("Host");
        }
        if (platform_or.ok()) {
          auto executor_or = platform_or.value()->ExecutorForDevice(i);
          if (executor_or.ok()) {
            auto* executor = executor_or.value();
            auto* manager = tpu_raiden::kv_cache::g_kv_cache_managers[i];
            size_t num_layers = manager->num_layers();
            size_t parallelism = manager->num_shards();
            for (size_t layer = 0; layer < num_layers; ++layer) {
              for (size_t shard = 0; shard < parallelism; ++shard) {
                uint8_t* ptr =
                    const_cast<uint8_t*>(manager->GetHostPointer(layer, shard));
                if (ptr != nullptr) {
                  (void)executor->HostMemoryUnregister(ptr);
                }
              }
            }
          }
        }
        unique_managers.insert(tpu_raiden::kv_cache::g_kv_cache_managers[i]);
        tpu_raiden::kv_cache::g_kv_cache_managers[i] = nullptr;
      }
    }
    for (auto* mgr : unique_managers) {
      mgr->BlockUntilFfiD2hDone();
      delete mgr;
    }
    {
      absl::MutexLock lock(tpu_raiden::kv_cache::g_manager_mutex);
      tpu_raiden::kv_cache::g_kv_cache_managers_map.clear();
    }
  });

  m.def("sync_copies", &tpu_raiden::kv_cache::SyncCopies);
}
