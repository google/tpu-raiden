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

#include "tpu_raiden/transport/peregrine/src/internal/socket/acceptor.h"

#include <atomic>
#include <memory>
#include <string_view>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/endpoint.h"
#include "tpu_raiden/transport/peregrine/src/internal/base/types.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_tcp.h"
#include "tpu_raiden/transport/peregrine/src/internal/socket/socket_util.h"

namespace peregrine::internal {

namespace {
constexpr std::string_view kAcceptor = "tcp acceptor ";
}  // namespace

std::unique_ptr<TcpAcceptor> TcpAcceptor::Create(const Endpoint& local) {
  DCHECK(local.IsValid());

  const int family = local.GetIpAddr().AddressFamily();
  std::unique_ptr<TcpSocket> socket = TcpSocket::Create(family);
  if ABSL_PREDICT_FALSE (socket == nullptr) {
    return nullptr;
  }

  DCHECK(socket->IsBlocking());
  if ABSL_PREDICT_FALSE (!socket->Listen(local)) {
    return nullptr;
  }

  LOG(INFO) << kAcceptor << "created, " << *socket;
  return absl::WrapUnique(new TcpAcceptor(std::move(socket)));
}

void TcpAcceptor::Start(AcceptCallback accept) {
  DCHECK_NE(accept, nullptr);
  LOG(INFO) << kAcceptor << "starting, " << *listener_;

  const int family = listener_->family();
  while (!stop_.load(std::memory_order_relaxed)) {
    DCHECK(listener_->IsBlocking());
    const fd_t fd = listener_->Accept();
    if ABSL_PREDICT_FALSE (fd.value() < 0) {
      if (IsShutdown(fd.value())) return;
      // TODO(yongx): Handle errors.
      DCHECK_EQ(fd.value(), -1);
      continue;
    }
    std::unique_ptr<TcpSocket> socket = TcpSocket::Create(fd, family);
    DCHECK(socket->IsBlocking());
    DCHECK(socket->IsConnected());
    LOG(INFO) << kAcceptor << "made " << *socket;
    accept(std::move(socket));
  }
}

void TcpAcceptor::Stop() {
  stop_.store(true, std::memory_order_relaxed);
  DCHECK(invariant());
  listener_->Shutdown();  // unblocks Accept()
  LOG(INFO) << kAcceptor << "stopped, " << *listener_;
}

}  // namespace peregrine::internal
