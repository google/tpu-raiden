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

#include "tpu_sync/transport/socket_transport_adapter.h"

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/transport/lib/raw_buffer_transport.h"
#include "tpu_sync/transport/transport_adapter.h"

namespace tpu_raiden {
namespace transport {

SocketTransportAdapter::SocketTransportAdapter(
    lib::RawBufferTransport* raw_transport)
    : raw_transport_(raw_transport) {}

absl::StatusOr<TransportAdapter::Handle> SocketTransportAdapter::Post(
    absl::string_view peer, absl::Span<const Request> requests,
    std::vector<int>* allocated_ids) {
  // TODO(swasthi): Dispatch requests to push or pull handler based on opcode:
  //   - Op 1/6: ProcessBlockPush()
  //   - Op 2:   ProcessBlockPull()
  //   - Op 5/7: ProcessBufferPush()
  //   - Op 3:   ProcessBufferPull()
  return absl::UnimplementedError(
      "SocketTransportAdapter::Post not implemented yet");
}

absl::StatusOr<TransportAdapter::Status> SocketTransportAdapter::Poll(
    Handle handle) {
  return absl::UnimplementedError(
      "SocketTransportAdapter::Poll not implemented yet");
}

absl::Status SocketTransportAdapter::ProcessBlockPush(
    absl::string_view peer, absl::Span<const Request> requests,
    std::vector<int>* allocated_ids) {
  // TODO(swasthi): Send block push requests (Op 1/6) over sockets (migrating
  // socket I/O logic from BlockTransport::H2hWriteWorker).
  return absl::UnimplementedError(
      "SocketTransportAdapter::ProcessBlockPush not implemented yet");
}

absl::Status SocketTransportAdapter::ProcessBlockPull(
    absl::string_view peer, absl::Span<const Request> requests) {
  // TODO(swasthi): Receive block pull requests (Op 2) over sockets (migrating
  // socket I/O logic from BlockTransport::H2hReadWorker).
  return absl::UnimplementedError(
      "SocketTransportAdapter::ProcessBlockPull not implemented yet");
}

absl::Status SocketTransportAdapter::ProcessBufferPush(
    absl::string_view peer, absl::Span<const Request> requests) {
  // TODO(swasthi): Send raw buffer push requests (Op 5/7) over sockets
  // (delegating to raw_transport_->PushBuffer and PushBuffers).
  return absl::UnimplementedError(
      "SocketTransportAdapter::ProcessBufferPush not implemented yet");
}

absl::Status SocketTransportAdapter::ProcessBufferPull(
    absl::string_view peer, absl::Span<const Request> requests) {
  // TODO(swasthi): Send raw buffer pull requests (Op 3) over sockets
  // (delegating to raw_transport_->PullBuffer).
  return absl::UnimplementedError(
      "SocketTransportAdapter::ProcessBufferPull not implemented yet");
}

}  // namespace transport
}  // namespace tpu_raiden
