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

#ifndef THIRD_PARTY_TPU_RAIDEN_API_TORCH_WEIGHT_SYNCHRONIZER_H_
#define THIRD_PARTY_TPU_RAIDEN_API_TORCH_WEIGHT_SYNCHRONIZER_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "xla/pjrt/pjrt_client.h"
#include "weight_sync/weight_synchronizer_base.h"

namespace tpu_raiden {
namespace torch {

class WeightSynchronizer : public weight_sync::WeightSynchronizerBase {
 public:
  // PyTorch sharded constructor E2E
  WeightSynchronizer(const std::vector<std::vector<at::Tensor>>& device_tensors,
                     std::optional<int> local_port = std::nullopt,
                     int parallelism = 1);

  ~WeightSynchronizer() override;
};

}  // namespace torch
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_API_TORCH_WEIGHT_SYNCHRONIZER_H_
