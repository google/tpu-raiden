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

#include "tpu_raiden/kv_cache/kv_cache_store_wrapper.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tpu_raiden/kv_cache/kv_cache_metadata.h"
#include "tpu_raiden/kv_cache/kv_cache_metadata_shm.h"
#include "tpu_raiden/kv_cache/kv_cache_store.h"
#include "tpu_raiden/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_raiden/kv_cache/raiden_id.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

// Names the metadata segment after the KV pool segments: RAIDEN_SHM_KEY
// plus the server-name suffix (but no per-device suffix — the table spans
// the store, not one device).
std::string MetadataShmKey() {
  const char* shm_key = std::getenv("RAIDEN_SHM_KEY");
  if (shm_key == nullptr || std::strlen(shm_key) == 0) {
    return "";
  }
  std::string key = absl::StrCat(shm_key, "_metadata");
  const char* server_name = std::getenv("RAIDEN_SHM_SERVER_NAME");
  if (server_name != nullptr && std::strlen(server_name) > 0) {
    absl::StrAppend(&key, "_", server_name);
  }
  return key;
}

}  // namespace

KVCacheStoreWrapper::KVCacheStoreWrapper(
    size_t lru_capacity, std::string global_registry_address,
    RaidenId raiden_id, int num_shards, int64_t shard_size_bytes,
    std::string raiden_orchestrator_address, std::string store_server_ip,
    int raiden_controller_port, int expected_worker_count) {
  std::optional<KVCacheMetadata> metadata;
  if (num_shards > 0) {
    std::string shm_key = MetadataShmKey();
    if (!shm_key.empty()) {
      const char* model_uid = std::getenv("RAIDEN_SHM_MODEL_UID");
      auto region_or = KVCacheMetadataShmRegion::AttachOrFormat(
          shm_key, static_cast<int>(lru_capacity),
          model_uid != nullptr ? model_uid : "default_model");
      if (region_or.ok()) {
        metadata_region_ = *std::move(region_or);
        metadata = metadata_region_->metadata();
      } else {
        LOG(WARNING) << "KV metadata table unavailable, serving without "
                        "crash recovery: "
                     << region_or.status().message();
      }
    }
  }

  // Routed through Create() (not the raw constructor) so a misconfigured
  // caller -- e.g. a missing store_server_ip -- gets a Python exception
  // instead of aborting the process.
  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = lru_capacity;
  auto store_or = KVCacheStore::Create(
      config, /*capacity=*/lru_capacity, global_registry_address, raiden_id,
      num_shards, shard_size_bytes, raiden_orchestrator_address,
      store_server_ip, raiden_controller_port, metadata, expected_worker_count);
  if (!store_or.ok()) {
    throw std::invalid_argument(std::string(store_or.status().message()));
  }
  controller_ = *std::move(store_or);

  if (metadata_region_ != nullptr && metadata_region_->warm()) {
    auto recovered_or = controller_->RecoverFromLocalManifest();
    if (recovered_or.ok()) {
      LOG(INFO) << "Recovered " << *recovered_or
                << " blocks from the local KV metadata table";
    } else {
      LOG(WARNING) << "KV metadata recovery failed, falling back to a cold "
                      "start: "
                   << recovered_or.status().message();
      absl::Status reformat_status = metadata_region_->Reformat();
      if (!reformat_status.ok()) {
        LOG(WARNING) << "Failed to reformat the KV metadata table: "
                     << reformat_status.message();
      }
    }
  }
}

}  // namespace kv_cache
}  // namespace tpu_raiden
