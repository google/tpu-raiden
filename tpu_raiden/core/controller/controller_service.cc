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

#include "tpu_raiden/core/controller/controller_service.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_raiden/core/buffer.h"
#include "tpu_raiden/core/controller/worker_registry.h"
#include "tpu_raiden/core/raiden_transfer_endpoint.h"
#include "tpu_raiden/proto/controller_service.pb.h"

namespace tpu_raiden {
namespace core {
namespace controller {

RaidenControllerServiceImpl::RaidenControllerServiceImpl(
    std::shared_ptr<WorkerRegistry> worker_registry)
    : worker_registry_(worker_registry ? std::move(worker_registry)
                                       : std::make_shared<WorkerRegistry>()) {}

grpc::Status RaidenControllerServiceImpl::RegisterWorker(
    grpc::ServerContext* context,
    const ::tpu_raiden::tpu_raiden::proto::RegisterWorkerRequest* request,
    ::tpu_raiden::tpu_raiden::proto::RegisterWorkerResponse* response) {
  std::vector<::tpu_raiden::RaidenTransferEndpoint> eps;
  eps.reserve(request->raiden_transfer_endpoints_size());
  for (const auto& desc_proto : request->raiden_transfer_endpoints()) {
    ::tpu_raiden::RaidenTransferEndpoint ep;
    ep.endpoint = desc_proto.endpoint();
    ep.shards.assign(desc_proto.shards().begin(), desc_proto.shards().end());
    eps.push_back(std::move(ep));
  }

  WorkerRegistration reg = {
      .worker_id = request->worker_id(),
      .raiden_worker_endpoint = request->raiden_worker_endpoint(),
      .raiden_transfer_endpoints = std::move(eps),
      .node_id = request->node_id(),
  };

  std::shared_ptr<WorkerRegistry> registry;
  {
    absl::MutexLock lock(mutex_);
    registry = worker_registry_;
  }

  absl::Status status = registry->RegisterWorker(reg);
  if (!status.ok()) {
    response->set_success(false);
    response->set_error_message(std::string(status.message()));
    return grpc::Status::OK;
  }

  response->set_success(true);
  return grpc::Status::OK;
}

grpc::Status RaidenControllerServiceImpl::ReadRemote(
    grpc::ServerContext* context,
    const ::tpu_raiden::tpu_raiden::proto::ReadRemoteRequest* request,
    ::tpu_raiden::tpu_raiden::proto::ReadRemoteResponse* response) {
  std::vector<Buffer> src_buffers;
  src_buffers.reserve(request->src_buffers_size());
  for (const auto& buf_proto : request->src_buffers()) {
    src_buffers.push_back(Buffer::FromProto(buf_proto));
  }

  std::vector<Buffer> dst_buffers;
  dst_buffers.reserve(request->dst_buffers_size());
  for (const auto& buf_proto : request->dst_buffers()) {
    dst_buffers.push_back(Buffer::FromProto(buf_proto));
  }

  std::shared_ptr<const TransferBuffersCallback> cb;
  {
    absl::MutexLock lock(mutex_);
    cb = transfer_buffers_cb_;
  }

  if (!cb) {
    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                        "TransferBuffers callback is not registered");
  }

  tsl::Future<> future = (*cb)(src_buffers, dst_buffers);
  absl::Status status = future.Await();
  if (!status.ok()) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        std::string(status.message()));
  }

  return grpc::Status::OK;
}

void RaidenControllerServiceImpl::SetWorkerRegistry(
    std::shared_ptr<WorkerRegistry> worker_registry) {
  if (worker_registry) {
    absl::MutexLock lock(mutex_);
    worker_registry_ = std::move(worker_registry);
  }
}

std::shared_ptr<WorkerRegistry> RaidenControllerServiceImpl::worker_registry()
    const {
  absl::MutexLock lock(mutex_);
  return worker_registry_;
}

}  // namespace controller
}  // namespace core
}  // namespace tpu_raiden
