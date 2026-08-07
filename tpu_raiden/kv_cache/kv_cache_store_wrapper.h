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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_WRAPPER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_WRAPPER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "tpu_raiden/kv_cache/kv_cache_metadata_shm.h"
#include "tpu_raiden/kv_cache/kv_cache_store.h"
#include "tpu_raiden/kv_cache/raiden_id.h"

namespace tpu_raiden {
namespace kv_cache {

// Constructs the KVCacheStore behind the framework Python bindings (the jax
// and torch modules both bind this class) and owns it together with its
// crash-recovery state.
//
// When the KV pool lives in shared memory (RAIDEN_SHM_KEY), the
// crash-persistent KVCacheMetadata table is kept there too — the data and
// its metadata are always shm-backed together, never separately — and the
// LRU cache is rebuilt from a surviving table on restart. Any failure in
// that wiring degrades to serving without recovery; it never blocks
// construction.
class KVCacheStoreWrapper {
 public:
  explicit KVCacheStoreWrapper(size_t lru_capacity,
                               std::string global_registry_address = "",
                               RaidenId raiden_id = {}, int num_shards = 0,
                               int64_t shard_size_bytes = 0,
                               std::string raiden_orchestrator_address = "",
                               std::string store_server_ip = "",
                               int raiden_controller_port = 0,
                               int expected_worker_count = 0);

  KVCacheStore* operator->() { return controller_.get(); }
  KVCacheStore& operator*() { return *controller_; }

 private:
  std::unique_ptr<KVCacheMetadataShmRegion> metadata_region_;
  std::unique_ptr<KVCacheStore> controller_;
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_WRAPPER_H_
