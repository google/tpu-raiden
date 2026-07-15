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

#include "tpu_raiden/transport/lib/util/util.h"

#include "absl/log/check.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include "absl/types/span.h"
#include "tpu_raiden/transport/lib/base/types.h"

namespace tpu_raiden::transport::lib {

void RandomNonZero(absl::Span<Byte> data) {
  absl::BitGen bitgen;
  RandomNonZero(bitgen, data);
}

void RandomNonZero(absl::BitGenRef bitgen, absl::Span<Byte> data) {
  for (int i = 0; i < data.size(); ++i) {
    data[i] = Random<Byte>(bitgen, 0x01, 0xff);
    DCHECK_NE(data[i], 0);
  }
}

}  // namespace tpu_raiden::transport::lib
