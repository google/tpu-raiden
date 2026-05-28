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

#ifndef THIRD_PARTY_TPU_RAIDEN_API_JAX_KV_CACHE_MANAGER_FFI_H_
#define THIRD_PARTY_TPU_RAIDEN_API_JAX_KV_CACHE_MANAGER_FFI_H_

namespace tpu_raiden {
namespace kv_cache {
namespace jax {
class KVCacheManager;
}  // namespace jax

// Global registry map for distributed JAX meshes multi-device support
extern jax::KVCacheManager* g_kv_cache_managers[32];

void SyncCopies();

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_API_JAX_KV_CACHE_MANAGER_FFI_H_
