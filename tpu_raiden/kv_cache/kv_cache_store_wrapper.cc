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
#include "absl/strings/str_cat.h"
#include "tpu_raiden/core/controller/raiden_controller.h"
#include "tpu_raiden/kv_cache/kv_cache_metadata.h"
#include "tpu_raiden/kv_cache/kv_cache_metadata_shm.h"
#include "tpu_raiden/kv_cache/kv_cache_store.h"
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
    int raiden_controller_port) {
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

  controller_ = std::make_unique<KVCacheStore>(
      lru_capacity, global_registry_address, std::move(raiden_id), num_shards,
      shard_size_bytes, raiden_orchestrator_address, store_server_ip,
      raiden_controller_port, std::move(metadata));

  // A controller whose server failed to bind (e.g. the requested port is in
  // use) is silently unreachable: workers cannot register and every
  // controller-driven transfer would fail. Surface that to the caller here
  // instead of letting serving come up looking healthy.
  if (controller_->raiden_controller() != nullptr) {
    const absl::Status& server_status =
        controller_->raiden_controller()->controller_server_status();
    if (!server_status.ok()) {
      throw std::runtime_error(absl::StrCat(
          "KVCacheStore raiden controller failed to bind its server (",
          "requested port ", raiden_controller_port,
          "): ", server_status.ToString()));
    }
  }

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
