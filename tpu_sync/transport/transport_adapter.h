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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_TRANSPORT_ADAPTER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_TRANSPORT_ADAPTER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace tpu_raiden {
namespace transport {

// `Handle` uniquely identifies a transport request.
using Handle = uint32_t;

struct Request {
  uint8_t socket_opcode;
  uint8_t* laddr;
  uint8_t* raddr;
  size_t len;

  // Block & Layer identification
  int layer_idx;
  int src_block_id;
  int dst_block_id;

  // Transfer metadata
  uint64_t uuid;
  int parallelism;
  uint8_t major_order;
};

// Abstract transport adapter interface for Post and Poll operations.
class TransportAdapter {
 public:
  using Handle = ::tpu_raiden::transport::Handle;

  // Transport operation status.
  enum class Status : int {
    kNotFound = 2,
    kInProgress = 1,
    kSuccess = 0,
    kFailure = -1,
  };

  virtual ~TransportAdapter() = default;

  virtual absl::StatusOr<Handle> Post(
      absl::string_view peer, absl::Span<const Request> requests,
      std::vector<int>* allocated_ids = nullptr) = 0;

  virtual absl::StatusOr<Status> Poll(Handle handle) = 0;
};

}  // namespace transport
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_TRANSPORT_ADAPTER_H_
