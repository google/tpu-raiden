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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RAIDEN_ID_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RAIDEN_ID_H_

#include <ostream>
#include <string>

#include "absl/hash/hash.h"

namespace tpu_raiden {
namespace kv_cache {

// Represents a microservice slice identifier / entity address hosting a replica
// of a Key-Value cache block.
struct RaidenId {
  std::string job_name;
  std::string job_replica_id;
  std::string data_name;
  int data_replica_idx = 0;

  bool operator==(const RaidenId& other) const {
    return job_name == other.job_name &&
           job_replica_id == other.job_replica_id &&
           data_name == other.data_name &&
           data_replica_idx == other.data_replica_idx;
  }
};

inline std::ostream& operator<<(std::ostream& os, const RaidenId& id) {
  os << "RaidenId{" << id.job_name << ", " << id.job_replica_id << ", "
     << id.data_name << ", " << id.data_replica_idx << "}";
  return os;
}

struct RaidenIdHash {
  size_t operator()(const RaidenId& id) const {
    return absl::HashOf(id.job_name, id.job_replica_id, id.data_name,
                        id.data_replica_idx);
  }
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RAIDEN_ID_H_
