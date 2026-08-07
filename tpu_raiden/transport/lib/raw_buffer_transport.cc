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

#include "tpu_raiden/transport/lib/raw_buffer_transport.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

#ifndef IOV_MAX
#define IOV_MAX 1024
#endif
#include "tpu_raiden/core/status_macros.h"
#include "tpu_raiden/transport/lib/chunk.h"
#include "tpu_raiden/transport/lib/chunk_serializer.h"
#include "tpu_raiden/transport/lib/conn/pool.h"
#include "tpu_raiden/transport/lib/raw_buffer_transport_delegate.h"
#include "tpu_raiden/transport/lib/socket/util.h"
#include "tpu_raiden/transport/peregrine/src/api/socket_util.h"

namespace tpu_raiden::transport::lib {

using ::peregrine::ReadExact;
using ::peregrine::ReadVExact;
using ::peregrine::WriteExact;
using ::peregrine::WriteVExact;

namespace {
absl::Status InternalError(absl::string_view msg, int _errno) {
  return absl::InternalError(absl::StrCat(msg, ": ", std::strerror(_errno)));
}

absl::StatusOr<std::pair<int, int>> CreateTcpIPv4Socket(const int port) {
  const int fd = socket(AF_INET, SOCK_STREAM, /*protocol=*/0);
  if (fd < 0) {
    const int last_errno = errno;
    return InternalError("tcp/ipv4 create socket", last_errno);
  }
  int opt = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
    const int last_errno = errno;
    close(fd);
    return InternalError("tcp/ipv4 setsockopt SO_REUSEADDR", last_errno);
  }
  struct sockaddr_in sa;
  socklen_t len = sizeof(sa);
  std::memset(&sa, 0, sizeof(sa));
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = INADDR_ANY;
  sa.sin_port = htons(port);
  if (bind(fd, reinterpret_cast<struct sockaddr*>(&sa), sizeof(sa)) < 0) {
    const int last_errno = errno;
    close(fd);
    return InternalError("tcp/ipv4 bind", last_errno);
  }
  if (getsockname(fd, reinterpret_cast<struct sockaddr*>(&sa), &len) < 0) {
    const int last_errno = errno;
    close(fd);
    return InternalError("tcp/ipv4 getsockname", last_errno);
  }
  DCHECK_GE(fd, 0);
  return std::make_pair(fd, ntohs(sa.sin_port));
}

absl::StatusOr<std::pair<int, int>> CreateTcpIPv6Socket(const int port) {
  const int fd = socket(AF_INET6, SOCK_STREAM, /*protocol=*/0);
  if (fd < 0) {
    const int last_errno = errno;
    return InternalError("tcp/ipv6 create socket", last_errno);
  }
  int opt = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
    const int last_errno = errno;
    close(fd);
    return InternalError("tcp/ipv6 setsockopt SO_REUSEADDR", last_errno);
  }
  int v6only = 0;
  if (setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &v6only, sizeof(v6only)) < 0) {
    LOG(WARNING) << "Failed to set IPV6_V6ONLY=0: " << std::strerror(errno);
  }
  struct sockaddr_in6 sa;
  socklen_t len = sizeof(sa);
  std::memset(&sa, 0, sizeof(sa));
  sa.sin6_family = AF_INET6;
  sa.sin6_addr = in6addr_any;
  sa.sin6_port = htons(port);
  if (bind(fd, reinterpret_cast<struct sockaddr*>(&sa), sizeof(sa)) < 0) {
    const int last_errno = errno;
    close(fd);
    return InternalError("tcp/ipv6 bind", last_errno);
  }
  if (getsockname(fd, reinterpret_cast<struct sockaddr*>(&sa), &len) < 0) {
    const int last_errno = errno;
    close(fd);
    return InternalError("tcp/ipv6 getsockname", last_errno);
  }
  DCHECK_GE(fd, 0);
  return std::make_pair(fd, ntohs(sa.sin6_port));
}

absl::StatusOr<std::pair<int, int>> CreateSocket(const int port) {
  const auto fd_port = CreateTcpIPv6Socket(port);
  return fd_port.ok() ? fd_port : CreateTcpIPv4Socket(port);
}

inline bool IsSocketValid(int fd) { return fcntl(fd, F_GETFD) >= 0; }
}  // namespace

RawBufferTransport::RawBufferTransport(
    RawBufferTransportDelegate* delegate, int local_port,
    const std::vector<std::string>& local_ips,
    CustomRequestHandler custom_request_handler)
    : raw_delegate_(delegate),
      custom_request_handler_(std::move(custom_request_handler)),
      local_port_(local_port),
      bound_ip_(local_ips.empty() ? "127.0.0.1" : local_ips[0]),
      local_ips_(local_ips) {
  // 1. Setup server listening socket.
  const absl::StatusOr<std::pair<int, int>> fd_port = CreateSocket(local_port_);
  if (!fd_port.ok()) {
    throw std::runtime_error(
        absl::StrCat("Failed to create server tcp listening socket: ",
                     fd_port.status().message()));
  }

  server_fd_ = fd_port->first;
  local_port_ = fd_port->second;
  DCHECK_GE(server_fd_, 0);
  DCHECK(1 <= local_port_ && local_port_ <= 65535);
  DCHECK(IsSocketValid(server_fd_));

  if (listen(server_fd_, 128) < 0) {
    LOG(FATAL) << "Failed to listen on server socket: " << std::strerror(errno);
  }

  // 2. Start listener
  listener_thread_ = std::thread(&RawBufferTransport::ListenerLoop, this);
}

RawBufferTransport::~RawBufferTransport() {
  stopping_ = true;

  // 1. Listener side:
  // 1.1 Shutdown on all active sockets to unblock the threads.
  if (server_fd_ >= 0) {
    DCHECK(IsSocketValid(server_fd_));
    shutdown(server_fd_, SHUT_RDWR);
  }
  {
    absl::MutexLock _(mu_);
    for (int fd : active_client_fds_) {
      shutdown(fd, SHUT_RDWR);
    }
  }
  // 1.2 Join all threads.
  if (listener_thread_.joinable()) {
    listener_thread_.join();
  }
  for (auto& t : worker_threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
  {
    // Each worker thread should have closed its own client_fd.
    absl::MutexLock _(mu_);
    DCHECK(active_client_fds_.empty());
  }

  // 2. Connector side:
  // Close all pooled file descriptors _after_ all the threads are joined.
  conn_pool_.Close();
}


absl::Status RawBufferTransport::ProcessPeerRequest(int client_fd) {
  char header_buf[kChunkHeaderSize];
  RETURN_IF_ERROR(ReadExact(client_fd, header_buf, sizeof(header_buf)));
  ASSIGN_OR_RETURN(const ChunkHeader header,
                   DeserializeChunkHeader(
                       absl::string_view(header_buf, sizeof(header_buf))));

  if (header.op == kOpBufferPush) {  // peer push request
    const uint32_t dst_offset = header.remote_id;
    const uint32_t dst_shard_idx = header.local_id;
    const uint32_t size_bytes = header.count_or_size;
    const uint16_t buf_id = header.buffer_id;

    uint8_t* const base_host_ptr =
        raw_delegate_->GetHostPointer(buf_id, dst_shard_idx);
    const size_t host_size = raw_delegate_->GetHostSize(buf_id, dst_shard_idx);
    if (base_host_ptr == nullptr || dst_offset + size_bytes > host_size) {
      return absl::InvalidArgumentError("Destination out of bounds");
    }
    uint8_t* const dest_ptr = base_host_ptr + dst_offset;
    RETURN_IF_ERROR(ReadExact(client_fd, dest_ptr, size_bytes));

    bool trigger_h2d = false;
    if (header.uuid > 0) {
      absl::MutexLock lock(raw_progress_mu_);
      auto& prog = raw_progress_[header.uuid];
      prog.completed_chunks++;
      VLOG(1) << "Received chunk for uuid=" << header.uuid
              << " shard=" << dst_shard_idx << " offset=" << dst_offset
              << " size=" << size_bytes << " progress=" << prog.completed_chunks
              << "/"
              << (prog.expected_chunks.has_value()
                      ? std::to_string(*prog.expected_chunks)
                      : "unknown");
      if (prog.expected_chunks.has_value() &&
          prog.completed_chunks == *prog.expected_chunks) {
        raw_progress_.erase(header.uuid);
        trigger_h2d = true;
        VLOG(1) << "Triggering H2D for uuid=" << header.uuid;
      }
    }

    if (trigger_h2d) {
      RETURN_IF_ERROR(raw_delegate_->OnDataReceived());
    }

    const uint8_t ack = 1;
    RETURN_IF_ERROR(WriteExact(client_fd, &ack, 1));
    return absl::OkStatus();

  } else if (header.op == kOpBufferPull) {  // peer pull request
    const uint32_t src_offset = header.remote_id;
    const uint32_t src_shard_idx = header.local_id;
    const uint32_t size_bytes = header.count_or_size;
    const uint16_t buf_id = header.buffer_id;

    uint8_t* const base_host_ptr =
        raw_delegate_->GetHostPointer(buf_id, src_shard_idx);
    const size_t host_size = raw_delegate_->GetHostSize(buf_id, src_shard_idx);
    if (base_host_ptr == nullptr || src_offset + size_bytes > host_size) {
      return absl::InvalidArgumentError("Source out of bounds");
    }
    uint8_t* const src_ptr = base_host_ptr + src_offset;
    RETURN_IF_ERROR(WriteExact(client_fd, src_ptr, size_bytes));
    return absl::OkStatus();

  } else if (header.op == kOpBufferPushBatched) {  // peer batched push request
    const uint32_t batch_size = header.count_or_size;
    if (batch_size > IOV_MAX) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Batch size ", batch_size, " exceeds IOV_MAX (", IOV_MAX, ")"));
    }
    const size_t meta_size = GetChunkMetadataSize(header.version);
    std::vector<char> meta_buf(meta_size * batch_size);
    RETURN_IF_ERROR(ReadExact(client_fd, meta_buf.data(), meta_buf.size()));

    std::vector<ChunkMetadata> metadata(batch_size);
    for (uint32_t i = 0; i < batch_size; ++i) {
      absl::string_view item_bytes(meta_buf.data() + i * meta_size, meta_size);
      ASSIGN_OR_RETURN(metadata[i],
                       DeserializeChunkMetadata(item_bytes, header.version));
    }

    std::vector<struct iovec> iovs;
    iovs.reserve(batch_size);

    for (uint32_t i = 0; i < batch_size; ++i) {
      const auto& meta = metadata[i];
      uint8_t* const base_host_ptr =
          raw_delegate_->GetHostPointer(meta.layer_idx, meta.dst_shard_idx);
      const size_t host_size =
          raw_delegate_->GetHostSize(meta.layer_idx, meta.dst_shard_idx);
      if (base_host_ptr == nullptr ||
          meta.dst_offset_bytes + meta.size_bytes > host_size) {
        return absl::InvalidArgumentError(
            "Destination out of bounds in batched push");
      }
      struct iovec iov;
      iov.iov_base = base_host_ptr + meta.dst_offset_bytes;
      iov.iov_len = meta.size_bytes;
      iovs.push_back(iov);
    }

    // Read all payloads directly using ReadVExact.
    RETURN_IF_ERROR(ReadVExact(client_fd, iovs));

    bool trigger_h2d = false;
    if (header.uuid > 0) {
      absl::MutexLock lock(raw_progress_mu_);
      auto& prog = raw_progress_[header.uuid];
      prog.completed_chunks += batch_size;
      VLOG(1) << "Received batched chunks for uuid=" << header.uuid
              << " batch_size=" << batch_size
              << " progress=" << prog.completed_chunks << "/"
              << (prog.expected_chunks.has_value()
                      ? std::to_string(*prog.expected_chunks)
                      : "unknown");
      if (prog.expected_chunks.has_value() &&
          prog.completed_chunks == *prog.expected_chunks) {
        raw_progress_.erase(header.uuid);
        trigger_h2d = true;
        VLOG(1) << "Triggering H2D for uuid=" << header.uuid;
      }
    }

    if (trigger_h2d) {
      RETURN_IF_ERROR(raw_delegate_->OnDataReceived());
    }

    const uint8_t ack = 1;
    RETURN_IF_ERROR(WriteExact(client_fd, &ack, 1));
    return absl::OkStatus();

  } else {
    if (custom_request_handler_) {
      return custom_request_handler_(client_fd, header);
    }
    return absl::UnimplementedError(
        absl::StrCat("Unsupported raw transport op code: ", header.op));
  }
}

void RawBufferTransport::ConnectionWorker(int client_fd) {
  DCHECK_GE(client_fd, 0);
  while (!stopping_) {
    struct pollfd pfd;
    pfd.fd = client_fd;
    pfd.events = POLLIN;
    int ret = poll(&pfd, 1, 50);
    if (ret < 0) {
      // EINTR (interrupted by a signal) and EAGAIN (transient kernel resource
      // pressure) are benign: retry the poll rather than tearing down a healthy
      // connection. Only a genuine error closes the connection.
      if (errno == EINTR || errno == EAGAIN) continue;
      break;
    }
    if (ret == 0) continue;

    if (!ProcessPeerRequest(client_fd).ok()) {
      break;
    }
  }

  DCHECK_GE(client_fd, 0);
  {
    absl::MutexLock _( mu_ );
    active_client_fds_.erase(client_fd);
  }
  close(client_fd);
}

void RawBufferTransport::ListenerLoop() {
  while (!stopping_) {
    DCHECK(IsSocketValid(server_fd_));
    struct pollfd pfd;
    pfd.fd = server_fd_;
    pfd.events = POLLIN;
    int ret = poll(&pfd, 1, 50);
    if (ret <= 0) {
      if (stopping_) break;
      continue;
    }

    struct sockaddr_in6 client_addr;
    socklen_t clilen = sizeof(client_addr);
    int client_fd = accept(
        server_fd_, reinterpret_cast<struct sockaddr*>(&client_addr), &clilen);
    if (client_fd < 0) {
      if (stopping_) break;
      continue;
    }

    int opt = 1;
    setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    int buf_opt = 16 * 1024 * 1024;  // 16MB
    setsockopt(client_fd, SOL_SOCKET, SO_SNDBUF, &buf_opt, sizeof(buf_opt));
    setsockopt(client_fd, SOL_SOCKET, SO_RCVBUF, &buf_opt, sizeof(buf_opt));

    DCHECK_GE(client_fd, 0);
    {
      absl::MutexLock _( mu_ );
      active_client_fds_.insert(client_fd);
    }
    worker_threads_.push_back(
        std::thread([this, client_fd]() { ConnectionWorker(client_fd); }));
  }

  DCHECK(IsSocketValid(server_fd_));
  close(server_fd_);
  server_fd_ = -1;
  DCHECK(!IsSocketValid(server_fd_));
}

absl::Status RawBufferTransport::PullBuffer(
    absl::string_view peer, size_t buffer_id, size_t src_shard_idx,
    size_t src_offset_bytes, size_t dst_shard_idx, size_t dst_offset_bytes,
    size_t size_bytes) {
  if (peer.empty()) {
    return absl::InvalidArgumentError("Source peer address cannot be empty");
  }

  const size_t host_size = raw_delegate_->GetHostSize(buffer_id, dst_shard_idx);
  if (dst_offset_bytes + size_bytes > host_size) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Destination offset out of bounds. Offset: ", dst_offset_bytes,
        ", Size: ", size_bytes, ", Shard Host Size: ", host_size));
  }

  ASSIGN_OR_RETURN(const int fd, conn_pool_.Borrow(peer));
  bool ok_to_pool = false;
  auto fd_cleaner =
      absl::MakeCleanup([&] { conn_pool_.Return(ok_to_pool, fd, peer); });

  const ChunkHeader header = {
      .version = 1,
      .op = kOpBufferPull,
      .buffer_id = static_cast<uint16_t>(buffer_id),
      .remote_id = static_cast<uint32_t>(src_offset_bytes),
      .local_id = static_cast<uint32_t>(src_shard_idx),
      .count_or_size = static_cast<uint32_t>(size_bytes),
  };
  const std::string serialized_header = SerializeChunkHeader(header);
  RETURN_IF_ERROR(
      WriteExact(fd, serialized_header.data(), serialized_header.size()));

  uint8_t* dest_ptr = raw_delegate_->GetHostPointer(buffer_id, dst_shard_idx) +
                      dst_offset_bytes;
  RETURN_IF_ERROR(ReadExact(fd, dest_ptr, size_bytes));

  ok_to_pool = true;
  return absl::OkStatus();
}

absl::Status RawBufferTransport::RegisterExpectedChunks(
    uint64_t uuid, uint32_t expected_chunks) {
  if (expected_chunks == 0) {
    return absl::InvalidArgumentError("expected_chunks must be positive");
  }

  bool trigger_h2d = false;
  {
    absl::MutexLock lock(raw_progress_mu_);
    auto& prog = raw_progress_[uuid];
    prog.expected_chunks = expected_chunks;
    VLOG(1) << "RegisterExpectedChunks: uuid=" << uuid
            << " expected_chunks=" << expected_chunks
            << " completed_chunks=" << prog.completed_chunks;
    if (prog.completed_chunks == expected_chunks) {
      raw_progress_.erase(uuid);
      trigger_h2d = true;
      VLOG(1) << "RegisterExpectedChunks triggering H2D for uuid=" << uuid;
    }
  }

  if (trigger_h2d) {
    RETURN_IF_ERROR(raw_delegate_->OnDataReceived());
  }

  return absl::OkStatus();
}

absl::Status RawBufferTransport::PushBuffer(absl::string_view peer,
                                            size_t buffer_id,
                                            size_t dst_shard_idx,
                                            size_t dst_offset_bytes,
                                            const uint8_t* data_ptr,
                                            size_t size_bytes, uint64_t uuid) {
  if (peer.empty()) {
    return absl::InvalidArgumentError(
        "Destination peer address cannot be empty");
  }

  ASSIGN_OR_RETURN(const int fd, conn_pool_.Borrow(peer));
  bool ok_to_pool = false;
  auto fd_cleaner =
      absl::MakeCleanup([&] { conn_pool_.Return(ok_to_pool, fd, peer); });

  const ChunkHeader header = {
      .version = 1,
      .op = kOpBufferPush,
      .buffer_id = static_cast<uint16_t>(buffer_id),
      .remote_id = static_cast<uint32_t>(dst_offset_bytes),
      .local_id = static_cast<uint32_t>(dst_shard_idx),
      .count_or_size = static_cast<uint32_t>(size_bytes),
      .uuid = uuid,
  };

  VLOG(1) << "Pushing chunk to peer=" << peer << " uuid=" << uuid
          << " dst_shard=" << dst_shard_idx
          << " dst_offset=" << dst_offset_bytes << " size=" << size_bytes;

  const std::string serialized_header = SerializeChunkHeader(header);
  RETURN_IF_ERROR(
      WriteExact(fd, serialized_header.data(), serialized_header.size()));
  RETURN_IF_ERROR(WriteExact(fd, data_ptr, size_bytes));

  uint8_t ack = 0;
  RETURN_IF_ERROR(ReadExact(fd, &ack, 1));
  if (ack != 1) {
    return absl::InternalError("PushBuffer verification failed");
  }

  ok_to_pool = true;
  return absl::OkStatus();
}

absl::Status RawBufferTransport::PushBuffers(
    const std::vector<BufferPushTask>& tasks, int parallelism, uint64_t uuid) {
  std::vector<BufferPushTask> sorted_tasks = tasks;
  std::stable_sort(sorted_tasks.begin(), sorted_tasks.end(),
                   [](const BufferPushTask& a, const BufferPushTask& b) {
                     return a.peer < b.peer;
                   });

  struct BatchInfo {
    std::string peer;
    size_t start_idx;
    size_t count;
  };

  std::vector<BatchInfo> batches;
  for (size_t i = 0; i < sorted_tasks.size();) {
    std::string peer = sorted_tasks[i].peer;
    size_t peer_start = i;
    size_t peer_task_count = 0;
    while (i < sorted_tasks.size() && sorted_tasks[i].peer == peer) {
      peer_task_count++;
      i++;
    }

    const size_t effective_parallelism = std::max(1, parallelism);
    const size_t target_batch_size =
        (peer_task_count + effective_parallelism - 1) / effective_parallelism;
    const size_t batch_size =
        std::clamp(target_batch_size, size_t{1}, static_cast<size_t>(IOV_MAX));

    size_t tasks_added = 0;
    while (tasks_added < peer_task_count) {
      size_t count = std::min(batch_size, peer_task_count - tasks_added);
      batches.push_back({peer, peer_start + tasks_added, count});
      tasks_added += count;
    }
  }

  if (batches.empty()) {
    return absl::OkStatus();
  }

  if (parallelism <= 1 || batches.size() == 1) {
    for (const auto& batch : batches) {
      RETURN_IF_ERROR(PushBatch(batch.peer, sorted_tasks, batch.start_idx,
                                batch.count, uuid));
    }
    return absl::OkStatus();
  }

  // Parallel execution using standard threads.
  std::vector<std::thread> threads;
  std::atomic<size_t> next_batch_idx(0);
  std::vector<absl::Status> statuses(batches.size(), absl::OkStatus());

  int num_threads = std::min(static_cast<size_t>(parallelism), batches.size());
  threads.reserve(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    threads.push_back(std::thread([&]() {
      while (true) {
        size_t idx = next_batch_idx.fetch_add(1);
        if (idx >= batches.size()) break;
        const auto& batch = batches[idx];
        statuses[idx] = PushBatch(batch.peer, sorted_tasks, batch.start_idx,
                                  batch.count, uuid);
      }
    }));
  }

  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  for (const auto& s : statuses) {
    if (!s.ok()) return s;
  }
  return absl::OkStatus();
}

absl::Status RawBufferTransport::PushBatch(
    absl::string_view peer, const std::vector<BufferPushTask>& tasks,
    size_t start_idx, size_t batch_size, uint64_t uuid) {
  if (peer.empty()) {
    return absl::InvalidArgumentError(
        "Destination peer address cannot be empty");
  }

  ASSIGN_OR_RETURN(const int fd, conn_pool_.Borrow(peer));
  bool ok_to_pool = false;
  auto fd_cleaner =
      absl::MakeCleanup([&] { conn_pool_.Return(ok_to_pool, fd, peer); });

  const ChunkHeader header = {
      .version = 1,
      .op = kOpBufferPushBatched,
      .buffer_id = 0,
      .remote_id = 0,
      .local_id = 0,
      .count_or_size = static_cast<uint32_t>(batch_size),
      .uuid = uuid,
  };

  VLOG(1) << "Pushing batch to peer=" << peer << " uuid=" << uuid
          << " batch_size=" << batch_size;

  const std::string serialized_header = SerializeChunkHeader(header);
  RETURN_IF_ERROR(
      WriteExact(fd, serialized_header.data(), serialized_header.size()));

  std::string serialized_metadata;
  serialized_metadata.reserve(GetChunkMetadataSize(header.version) *
                              batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    const auto& task = tasks[start_idx + i];
    ChunkMetadata meta = {
        .layer_idx = static_cast<uint32_t>(task.buffer_id),
        .dst_shard_idx = static_cast<uint32_t>(task.dst_shard_idx),
        .dst_offset_bytes = static_cast<uint32_t>(task.dst_offset_bytes),
        .size_bytes = static_cast<uint32_t>(task.size_bytes),
    };
    serialized_metadata.append(SerializeChunkMetadata(meta));
  }
  RETURN_IF_ERROR(
      WriteExact(fd, serialized_metadata.data(), serialized_metadata.size()));

  std::vector<struct iovec> iovs;
  iovs.reserve(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    const auto& task = tasks[start_idx + i];
    struct iovec iov;
    iov.iov_base = const_cast<uint8_t*>(task.data_ptr);
    iov.iov_len = task.size_bytes;
    iovs.push_back(iov);
  }

  RETURN_IF_ERROR(WriteVExact(fd, iovs));

  uint8_t ack = 0;
  RETURN_IF_ERROR(ReadExact(fd, &ack, 1));
  if (ack != 1) {
    return absl::InternalError("PushBatch verification failed");
  }

  ok_to_pool = true;
  return absl::OkStatus();
}

void RawBufferTransport::ForgetPushProgress(uint64_t uuid) {
  absl::MutexLock lock(raw_progress_mu_);
  raw_progress_.erase(uuid);
}

}  // namespace tpu_raiden::transport::lib
