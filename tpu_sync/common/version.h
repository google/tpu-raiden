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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_VERSION_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_VERSION_H_

#include <cstdint>

namespace tpu_sync {

// Current version of the TPU Sync protocol/runtime.
// Dummy change to test Kokoro triggering.
inline constexpr uint32_t kCurrentVersion = 1;

// Minimum version supported by the current TPU Sync build.
inline constexpr uint32_t kSupportedMinVersion = 1;

// Returns true if the given version is supported by the current build.
constexpr bool IsVersionSupported(uint32_t version) {
  return version >= kSupportedMinVersion;
}

}  // namespace tpu_sync

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_COMMON_VERSION_H_
