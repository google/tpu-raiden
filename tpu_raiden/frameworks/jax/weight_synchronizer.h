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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_FRAMEWORKS_JAX_WEIGHT_SYNCHRONIZER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_FRAMEWORKS_JAX_WEIGHT_SYNCHRONIZER_H_

#include <optional>
#include <vector>

#ifndef WITHOUT_PYTHON
#include <nanobind/nanobind.h>
#endif
#include "tpu_raiden/weight_sync/weight_synchronizer_base.h"

namespace xla {
class PjRtBuffer;
}  // namespace xla

#ifndef WITHOUT_PYTHON
namespace nb = nanobind;
#endif

namespace tpu_raiden {
namespace jax {

#ifndef WITHOUT_PYTHON
struct UnpackedWeights {
  std::vector<std::vector<xla::PjRtBuffer*>> layer_buffers;
  nanobind::list jax_arrays;
};
#endif

class WeightSynchronizer : public weight_sync::WeightSynchronizerBase {
 public:
  using WeightSynchronizerBase::WeightSynchronizerBase;

#ifndef WITHOUT_PYTHON
  WeightSynchronizer(nanobind::list jax_arrays,
                     std::optional<int> local_port = std::nullopt,
                     int parallelism = 1, bool unsafe_skip_buffer_lock = false,
                     std::optional<int> control_port = std::nullopt);
#endif

  ~WeightSynchronizer() override;

 private:
#ifndef WITHOUT_PYTHON
  WeightSynchronizer(UnpackedWeights&& weights, std::optional<int> local_port,
                     int parallelism, bool unsafe_skip_buffer_lock,
                     std::optional<int> control_port);
  std::optional<nanobind::list> jax_arrays_;
#endif
};

}  // namespace jax
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_FRAMEWORKS_JAX_WEIGHT_SYNCHRONIZER_H_
