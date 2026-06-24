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

// Copyright 2026 The Tensor Transporter Authors.
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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_RAIDEN_FUTURE_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_RAIDEN_FUTURE_H_

#include "absl/status/status.h"
#include "tpu_raiden/core/raw_transfer_core.h"

namespace tpu_raiden {

// A platform-agnostic wrapper around raiden::PjRtCopyFuture to expose a clean,
// unified future interface to both JAX and PyTorch Python bindings.
struct RaidenFuture {
  raiden::PjRtCopyFuture future;

  absl::Status Await() { return future.Await(); }

  bool IsReady() const { return future.IsReady(); }

  // Non-blocking error probe; see raiden::PjRtCopyFuture::PollError. Call after
  // IsReady() is true to tell a successful completion from a failed one without
  // blocking the polling thread.
  absl::Status PollError() { return future.PollError(); }
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_RAIDEN_FUTURE_H_
