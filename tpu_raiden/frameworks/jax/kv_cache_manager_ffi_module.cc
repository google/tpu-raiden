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

#include "tpu_raiden/frameworks/jax/kv_cache_manager.h"
#include "tpu_raiden/frameworks/jax/kv_cache_manager_ffi.h"

namespace nb = nanobind;

NB_MODULE(_kv_cache_manager_ffi, m) {
  m.def("destroy_kv_cache", []() {
    for (int i = 0; i < 32; ++i) {
      if (tpu_raiden::kv_cache::g_kv_cache_managers[i] != nullptr) {
        delete tpu_raiden::kv_cache::g_kv_cache_managers[i];
        tpu_raiden::kv_cache::g_kv_cache_managers[i] = nullptr;
      }
    }
  });

  m.def("sync_copies", &tpu_raiden::kv_cache::SyncCopies);
}
