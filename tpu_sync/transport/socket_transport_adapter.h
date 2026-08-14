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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_SOCKET_TRANSPORT_ADAPTER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_SOCKET_TRANSPORT_ADAPTER_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/transport/lib/raw_buffer_transport.h"
#include "tpu_sync/transport/transport_adapter.h"

namespace tpu_raiden {
namespace transport {

// TCP Socket implementation of TransportAdapter.
class SocketTransportAdapter : public TransportAdapter {
 public:
  explicit SocketTransportAdapter(lib::RawBufferTransport* raw_transport);
  ~SocketTransportAdapter() override = default;

  absl::StatusOr<Handle> Post(
      absl::string_view peer, absl::Span<const Request> requests,
      std::vector<int>* allocated_ids = nullptr) override;

  absl::StatusOr<Status> Poll(Handle handle) override;

 private:
  // Block-level Socket Operations (Op 1, 6, 2).
  absl::Status ProcessBlockPush(absl::string_view peer,
                                absl::Span<const Request> requests,
                                std::vector<int>* allocated_ids = nullptr);

  absl::Status ProcessBlockPull(absl::string_view peer,
                                absl::Span<const Request> requests);

  // Raw Buffer Socket Operations (Op 5, 7, 3).
  absl::Status ProcessBufferPush(absl::string_view peer,
                                 absl::Span<const Request> requests);

  absl::Status ProcessBufferPull(absl::string_view peer,
                                 absl::Span<const Request> requests);

  lib::RawBufferTransport* raw_transport_;
};

}  // namespace transport
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_SOCKET_TRANSPORT_ADAPTER_H_
