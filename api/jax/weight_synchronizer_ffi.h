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

#ifndef THIRD_PARTY_TPU_RAIDEN_API_JAX_WEIGHT_SYNCHRONIZER_FFI_H_
#define THIRD_PARTY_TPU_RAIDEN_API_JAX_WEIGHT_SYNCHRONIZER_FFI_H_

namespace tpu_raiden {
namespace jax {
class WeightSynchronizer;
}  // namespace jax

namespace weight_sync {

// Global registry map for distributed JAX meshes multi-device support
extern jax::WeightSynchronizer* g_weight_synchronizers[32];

}  // namespace weight_sync
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_API_JAX_WEIGHT_SYNCHRONIZER_FFI_H_
