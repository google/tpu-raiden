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

#include "kv_cache/kv_cache_manager.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/status_casters.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "raiden_lib/raw_transfer/raw_transfer_impl.h"

namespace nb = nanobind;

namespace tpu_raiden {
namespace kv_cache {

namespace {

absl::Status WriteExact(int fd, const void* buffer, size_t length) {
  const uint8_t* ptr = static_cast<const uint8_t*>(buffer);
  size_t remaining = length;
  while (remaining > 0) {
    ssize_t written = write(fd, ptr, remaining);
    if (written < 0) {
      if (errno == EINTR) continue;
      return absl::InternalError(
          absl::StrCat("Socket write failed: ", std::strerror(errno)));
    }
    if (written == 0) {
      return absl::InternalError("Socket closed unexpectedly during write");
    }
    ptr += written;
    remaining -= written;
  }
  return absl::OkStatus();
}

absl::Status ReadExact(int fd, void* buffer, size_t length) {
  uint8_t* ptr = static_cast<uint8_t*>(buffer);
  size_t remaining = length;
  while (remaining > 0) {
    ssize_t bytes_read = read(fd, ptr, remaining);
    if (bytes_read < 0) {
      if (errno == EINTR) continue;
      return absl::InternalError(
          absl::StrCat("Socket read failed: ", std::strerror(errno)));
    }
    if (bytes_read == 0) {
      return absl::InternalError("Socket closed unexpectedly during read");
    }
    ptr += bytes_read;
    remaining -= bytes_read;
  }
  return absl::OkStatus();
}

absl::StatusOr<int> ConnectToPeer(const std::string& peer) {
  std::string host;
  std::string port_str;

  if (!peer.empty() && peer.front() == '[') {
    size_t closing_bracket = peer.find(']');
    if (closing_bracket == std::string::npos ||
        closing_bracket + 1 >= peer.size() ||
        peer[closing_bracket + 1] != ':') {
      return absl::InvalidArgumentError(
          "Invalid IPv6 peer bracket string format");
    }
    host = peer.substr(1, closing_bracket - 1);
    port_str = peer.substr(closing_bracket + 2);
  } else {
    std::vector<std::string> parts = absl::StrSplit(peer, ':');
    if (parts.size() != 2) {
      return absl::InvalidArgumentError("Invalid peer string format");
    }
    host = parts[0];
    port_str = parts[1];
  }

  struct addrinfo hints;
  struct addrinfo* result = nullptr;
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  int ret = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result);
  if (ret != 0 || result == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "getaddrinfo failed for host ", host, ": ", gai_strerror(ret)));
  }

  int sock_fd =
      socket(result->ai_family, result->ai_socktype, result->ai_protocol);
  if (sock_fd < 0) {
    freeaddrinfo(result);
    return absl::InternalError(
        absl::StrCat("Socket creation failed: ", std::strerror(errno)));
  }

  int opt = 1;
  setsockopt(sock_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
  int buf_opt = 16 * 1024 * 1024;  // 16MB
  setsockopt(sock_fd, SOL_SOCKET, SO_SNDBUF, &buf_opt, sizeof(buf_opt));
  setsockopt(sock_fd, SOL_SOCKET, SO_RCVBUF, &buf_opt, sizeof(buf_opt));

  if (connect(sock_fd, result->ai_addr, result->ai_addrlen) < 0) {
    close(sock_fd);
    freeaddrinfo(result);
    return absl::UnavailableError(
        absl::StrCat("Connect failed: ", std::strerror(errno)));
  }

  freeaddrinfo(result);
  return sock_fd;
}

}  // namespace

struct alignas(8) BlockPacketHeader {
  uint8_t op;
  uint32_t remote_block_id;
  uint32_t local_block_id;
  uint32_t num_blocks;
};

struct KVCacheManager::BlockTransportServer {
  explicit BlockTransportServer(KVCacheManager* parent, int port)
      : parent_(parent), local_port_(port) {
    server_fd_ = socket(AF_INET6, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
      LOG(FATAL) << "Failed to create server socket: " << std::strerror(errno);
    }
    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in6 serv_addr;
    std::memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin6_family = AF_INET6;
    serv_addr.sin6_addr = in6addr_any;
    serv_addr.sin6_port = htons(local_port_);

    if (bind(server_fd_, reinterpret_cast<struct sockaddr*>(&serv_addr),
             sizeof(serv_addr)) < 0) {
      LOG(FATAL) << "Failed to bind server socket to port " << local_port_
                 << ": " << std::strerror(errno);
    }

    socklen_t addr_len = sizeof(serv_addr);
    if (getsockname(server_fd_, reinterpret_cast<struct sockaddr*>(&serv_addr),
                    &addr_len) == 0) {
      local_port_ = ntohs(serv_addr.sin6_port);
    }

    if (listen(server_fd_, 128) < 0) {
      LOG(FATAL) << "Failed to listen on server socket: "
                 << std::strerror(errno);
    }
    listener_thread_ = std::thread(&BlockTransportServer::ListenerLoop, this);
  }

  ~BlockTransportServer() {
    stopping_ = true;
    if (server_fd_ >= 0) {
      shutdown(server_fd_, SHUT_RDWR);
      close(server_fd_);
    }
    if (listener_thread_.joinable()) {
      listener_thread_.join();
    }
    for (auto& t : worker_threads_) {
      if (t.joinable()) {
        t.join();
      }
    }
    absl::MutexLock _(&conn_mu_);
    for (const auto& [peer, fd] : connection_pool_) {
      close(fd);
    }
  }

  absl::StatusOr<int> GetOrCreateConnection(const std::string& peer) {
    {
      absl::MutexLock _(&conn_mu_);
      auto it = connection_pool_.find(peer);
      if (it != connection_pool_.end()) {
        return it->second;
      }
    }
    std::vector<std::string> parts = absl::StrSplit(peer, ':');
    if (parts.size() != 2) {
      return absl::InvalidArgumentError("Invalid peer string format");
    }
    std::string host = parts[0];
    std::string port_str = parts[1];

    struct addrinfo hints;
    struct addrinfo* result = nullptr;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    int ret = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result);
    if (ret != 0 || result == nullptr) {
      return absl::InvalidArgumentError(absl::StrCat(
          "getaddrinfo failed for host ", host, ": ", gai_strerror(ret)));
    }

    int sock_fd =
        socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (sock_fd < 0) {
      freeaddrinfo(result);
      return absl::InternalError(
          absl::StrCat("Socket creation failed: ", std::strerror(errno)));
    }

    int opt = 1;
    setsockopt(sock_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    int buf_opt = 16 * 1024 * 1024;  // 16MB
    setsockopt(sock_fd, SOL_SOCKET, SO_SNDBUF, &buf_opt, sizeof(buf_opt));
    setsockopt(sock_fd, SOL_SOCKET, SO_RCVBUF, &buf_opt, sizeof(buf_opt));

    if (connect(sock_fd, result->ai_addr, result->ai_addrlen) < 0) {
      close(sock_fd);
      freeaddrinfo(result);
      return absl::UnavailableError(
          absl::StrCat("Connect failed: ", std::strerror(errno)));
    }

    freeaddrinfo(result);

    absl::MutexLock _(&conn_mu_);
    connection_pool_[peer] = sock_fd;
    return sock_fd;
  }

  void ListenerLoop() {
    while (!stopping_) {
      struct pollfd pfd;
      pfd.fd = server_fd_;
      pfd.events = POLLIN;
      int ret = poll(&pfd, 1, 50);
      if (ret <= 0) continue;

      struct sockaddr_in client_addr;
      socklen_t clilen = sizeof(client_addr);
      int client_fd =
          accept(server_fd_, reinterpret_cast<struct sockaddr*>(&client_addr),
                 &clilen);
      if (client_fd < 0) continue;

      int opt = 1;
      setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
      int buf_opt = 16 * 1024 * 1024;  // 16MB
      setsockopt(client_fd, SOL_SOCKET, SO_SNDBUF, &buf_opt, sizeof(buf_opt));
      setsockopt(client_fd, SOL_SOCKET, SO_RCVBUF, &buf_opt, sizeof(buf_opt));
      worker_threads_.push_back(
          std::thread([this, client_fd]() { ConnectionWorker(client_fd); }));
    }
  }

  absl::Status ProcessSingleRequest(int client_fd) {
    BlockPacketHeader header;
    TF_RETURN_IF_ERROR(ReadExact(client_fd, &header, sizeof(header)));

    size_t bytes_per_block = parent_->block_size_ * parent_->slice_byte_size_;

    if (header.op == 1) {
      // Push transfer (incoming write). Dynamically allocate blocks locally.
      size_t local_blocks = header.num_blocks / parent_->shard_factor_;
      TF_ASSIGN_OR_RETURN(
          std::vector<int> allocated_ids,
          parent_->block_manager_->Allocate(local_blocks, /*entity_id=*/0,
                                            /*lock=*/true));

      // Send back acknowledgment header containing assigned block IDs.
      TF_RETURN_IF_ERROR(WriteExact(client_fd, allocated_ids.data(),
                                    allocated_ids.size() * sizeof(int)));

      // Read payload blocks across all layers and shards symmetrically.
      for (size_t l = 0; l < parent_->num_layers_; ++l) {
        const auto& layer_info = parent_->layers_[l];
        for (size_t sh = 0; sh < parent_->num_shards_; ++sh) {
          const auto& shard_info = layer_info.shards[sh];
          uint8_t* base_host_ptr = const_cast<uint8_t*>(shard_info.host_ptr);

          for (int k = 0; k < local_blocks; ++k) {
            int assigned_id = allocated_ids[k];
            uint8_t* dest_ptr = base_host_ptr + assigned_id * bytes_per_block;
            TF_RETURN_IF_ERROR(ReadExact(client_fd, dest_ptr, bytes_per_block));
          }
        }
      }

      // Send final completion acknowledgment byte to unblock client Post loop.
      uint8_t ack = 1;
      TF_RETURN_IF_ERROR(WriteExact(client_fd, &ack, 1));
    } else if (header.op == 2) {
      // Pull transfer (incoming read request).
      // Read data from requested local blocks and push back response writes.
      BlockPacketHeader resp_header;
      resp_header.op = 1;
      resp_header.remote_block_id = header.local_block_id;
      resp_header.local_block_id = 0;
      resp_header.num_blocks = header.num_blocks;

      TF_RETURN_IF_ERROR(
          WriteExact(client_fd, &resp_header, sizeof(resp_header)));

      size_t local_blocks = header.num_blocks / parent_->shard_factor_;
      for (size_t l = 0; l < parent_->num_layers_; ++l) {
        const auto& layer_info = parent_->layers_[l];
        for (size_t sh = 0; sh < parent_->num_shards_; ++sh) {
          const auto& shard_info = layer_info.shards[sh];
          const uint8_t* base_host_ptr = shard_info.host_ptr;

          for (int k = 0; k < local_blocks; ++k) {
            int read_id = header.remote_block_id + k;
            const uint8_t* src_ptr = base_host_ptr + read_id * bytes_per_block;
            TF_RETURN_IF_ERROR(WriteExact(client_fd, src_ptr, bytes_per_block));
          }
        }
      }
    }
    return absl::OkStatus();
  }

  void ConnectionWorker(int client_fd) {
    while (!stopping_) {
      struct pollfd pfd;
      pfd.fd = client_fd;
      pfd.events = POLLIN;
      int ret = poll(&pfd, 1, 50);
      if (ret < 0) break;
      if (ret == 0) continue;

      if (!ProcessSingleRequest(client_fd).ok()) {
        break;
      }
    }
    close(client_fd);
  }

  KVCacheManager* parent_;
  int local_port_;
  int server_fd_ = -1;
  std::atomic<bool> stopping_{false};

  absl::Mutex conn_mu_;
  absl::flat_hash_map<std::string, int> connection_pool_
      ABSL_GUARDED_BY(conn_mu_);

  std::thread listener_thread_;
  std::vector<std::thread> worker_threads_;
};

KVCacheManager::KVCacheManager(nb::list device_arrays, int block_size,
                               std::optional<int> local_port,
                               int host_blocks_to_allocate,
                               bool unsafe_skip_buffer_lock, int parallelism)
    : device_arrays_(std::move(device_arrays)),
      parallelism_(parallelism) {
  num_layers_ = nb::len(*device_arrays_);

  if (num_layers_ == 0) {
    return;
  }

  nb::object first_dst_arr = (*device_arrays_)[0];
  nb::tuple global_shape = nb::cast<nb::tuple>(first_dst_arr.attr("shape"));
  int64_t global_major_dim = nb::cast<int64_t>(global_shape[0]);

  nb::object addressable_shards = first_dst_arr.attr("addressable_shards");
  num_shards_ = nb::len(addressable_shards);

  if (num_shards_ == 0) {
    return;
  }

  nb::object first_shard_data = addressable_shards[0].attr("data");
  xla::PjRtBuffer* first_buffer =
      jax::GetPjrtBufferFromPyObject(first_shard_data.ptr());
  const xla::Shape& shape = first_buffer->on_device_shape();

  auto status_or_dst_size = first_buffer->GetOnDeviceSizeInBytes();
  if (!status_or_dst_size.ok()) {
    throw std::runtime_error("Failed to get destination buffer size");
  }
  physical_size_ = status_or_dst_size.value();

  extension_ = raiden::GetRawBufferExtension(first_buffer, &c_api_);
  xla::PjRtCApiBuffer* first_capi_buffer =
      dynamic_cast<xla::PjRtCApiBuffer*>(first_buffer);

  if (first_capi_buffer) {
    if (!extension_) {
      throw std::runtime_error(
          "RawBuffer extension not found in PjRtCApiClient");
    }
    is_common_buffer_ = false;
  } else {
    is_common_buffer_ =
        (dynamic_cast<xla::CommonPjRtBuffer*>(first_buffer) != nullptr);
    if (!is_common_buffer_) {
      throw std::runtime_error("Unsupported PjRtBuffer type");
    }
  }

  slice_byte_size_ = raiden::GetMajorSliceByteSize(first_buffer);

  if (shape.dimensions_size() > 0) {
    major_dim_size_ = shape.dimensions(0);
    shard_factor_ = std::max<size_t>(1, global_major_dim / major_dim_size_);
    block_size_ = block_size;
    if (block_size_ <= 0) {
      throw std::invalid_argument("Block size must be greater than 0");
    }
    int total_blocks = major_dim_size_ / block_size_;
    block_manager_ = std::make_unique<LogicalBlockManager>(total_blocks);
  }

  // Extract all layer and shard pointers
  layers_.reserve(num_layers_);
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    nb::object dst = (*device_arrays_)[layer_idx];
    xla::ifrt::Array* dst_ifrt_array = jax::GetIfrtArrayFromPyObject(dst.ptr());
    auto* dst_compat_arr =
        llvm::dyn_cast_or_null<xla::ifrt::PjRtCompatibleArray>(dst_ifrt_array);

    if (dst_compat_arr == nullptr) {
      throw std::runtime_error("Not a PjRt compatible array");
    }

    auto dst_buffers = dst_compat_arr->pjrt_buffers();
    if (dst_buffers.size() != num_shards_) {
      throw std::runtime_error("Number of shards mismatch across layers");
    }

    LayerInfo layer_info;
    layer_info.shards.reserve(num_shards_);

    for (size_t i = 0; i < num_shards_; ++i) {
      xla::PjRtBuffer* dst_buffer = dst_buffers[i].get();
      ShardBufferInfo shard_info;

      shard_info.device_size = dst_buffer->GetOnDeviceSizeInBytes().value();
      if (shard_info.device_size < physical_size_) {
        throw std::runtime_error(
            "Device buffer shard size smaller than expected physical size");
      }

      int num_host_blocks = host_blocks_to_allocate;
      if (num_host_blocks < 0) {
        throw std::invalid_argument(
            "host_blocks_to_allocate must be non-negative");
      }
      size_t alloc_size = num_host_blocks * block_size_ * slice_byte_size_;
      void* ptr = nullptr;
      if (alloc_size > 0) {
        if (posix_memalign(&ptr, 64, alloc_size) != 0) {
          throw std::runtime_error("Failed to allocate host buffer");
        }
        std::memset(ptr, 0, alloc_size);
      }
      shard_info.owned_host_buffer =
          std::unique_ptr<uint8_t[], void (*)(void*)>(
              static_cast<uint8_t*>(ptr), [](void* p) { free(p); });
      shard_info.host_ptr = shard_info.owned_host_buffer.get();
      shard_info.host_size = alloc_size;

      auto status_or_hold = raiden::BufferHoldAndAlias::Acquire(
          dst_buffer, c_api_, extension_, unsafe_skip_buffer_lock);
      if (!status_or_hold.ok()) {
        throw std::runtime_error(
            std::string("Failed to acquire hold/alias: ") +
            std::string(status_or_hold.status().message()));
      }
      static_cast<raiden::BufferHoldAndAlias&>(shard_info) =
          std::move(status_or_hold.value());

      layer_info.shards.push_back(std::move(shard_info));
    }
    layers_.push_back(std::move(layer_info));
  }

  if (local_port.has_value()) {
    server_ = std::make_unique<BlockTransportServer>(this, local_port.value());
  }
}

KVCacheManager::KVCacheManager(size_t num_layers, size_t num_shards,
                               size_t slice_byte_size, int block_size,
                               std::optional<int> local_port,
                               std::optional<int> host_blocks_to_allocate,
                               int parallelism)
    : num_layers_(num_layers),
      num_shards_(num_shards),
      slice_byte_size_(slice_byte_size),
      block_size_(block_size),
      parallelism_(parallelism) {
  shard_factor_ = 1;

  if (block_size_ <= 0) {
    throw std::invalid_argument("Block size must be greater than 0");
  }

  int total_blocks = host_blocks_to_allocate.value_or(0);
  block_manager_ = std::make_unique<LogicalBlockManager>(total_blocks);

  layers_.reserve(num_layers_);
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    LayerInfo layer_info;
    layer_info.shards.reserve(num_shards_);

    for (size_t i = 0; i < num_shards_; ++i) {
      ShardBufferInfo shard_info;

      int num_host_blocks = host_blocks_to_allocate.value_or(0);
      size_t alloc_size = num_host_blocks * block_size_ * slice_byte_size_;
      void* ptr = nullptr;
      if (alloc_size > 0) {
        if (posix_memalign(&ptr, 64, alloc_size) != 0) {
          throw std::runtime_error("Failed to allocate host buffer");
        }
        std::memset(ptr, 0, alloc_size);
      }
      shard_info.owned_host_buffer =
          std::unique_ptr<uint8_t[], void (*)(void*)>(
              static_cast<uint8_t*>(ptr), [](void* p) { free(p); });
      shard_info.host_ptr = shard_info.owned_host_buffer.get();
      shard_info.host_size = alloc_size;

      shard_info.buffer = nullptr;

      layer_info.shards.push_back(std::move(shard_info));
    }
    layers_.push_back(std::move(layer_info));
  }

  if (local_port.has_value()) {
    server_ = std::make_unique<BlockTransportServer>(this, local_port.value());
  }
}

const uint8_t* KVCacheManager::GetHostPointer(size_t layer_idx,
                                              size_t shard_idx) const {
  if (layer_idx >= num_layers_ || shard_idx >= num_shards_) {
    return nullptr;
  }
  return layers_[layer_idx].shards[shard_idx].host_ptr;
}

KVCacheManager::~KVCacheManager() = default;

absl::StatusOr<raiden::PjRtCopyFuture> KVCacheManager::H2d(
    nb::list src_offsets_major_dim, nb::list dst_offsets_major_dim,
    nb::list copy_sizes_major_dim) {
  bool is_partial = (src_offsets_major_dim.size() > 0);
  if (is_partial) {
    if (src_offsets_major_dim.size() != dst_offsets_major_dim.size() ||
        src_offsets_major_dim.size() != copy_sizes_major_dim.size()) {
      return absl::InvalidArgumentError(
          "Lengths of offset and size lists must match");
    }
  }

  raiden::PjRtCopyFuture acc({});
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    const auto& layer_info = layers_[layer_idx];
    for (size_t i = 0; i < num_shards_; ++i) {
      const auto& shard_info = layer_info.shards[i];

      std::vector<xla::Future<>> shard_futures;
      if (!is_partial) {
        xla::Future<> future;
        {
          nb::gil_scoped_release release;
          future = shard_info.CopyRawHostToDevice(shard_info.host_ptr, 0,
                                                  physical_size_);
        }
        shard_futures.push_back(std::move(future));
      } else {
        for (size_t j = 0; j < src_offsets_major_dim.size(); ++j) {
          int64_t src_major_dim_offset =
              nb::cast<int64_t>(src_offsets_major_dim[j]);
          int64_t dst_major_dim_offset =
              nb::cast<int64_t>(dst_offsets_major_dim[j]);
          int64_t major_dim_size = nb::cast<int64_t>(copy_sizes_major_dim[j]);

          int64_t src_offset = src_major_dim_offset * slice_byte_size_;
          int64_t dst_offset = dst_major_dim_offset * slice_byte_size_;
          int64_t size_to_copy = major_dim_size * slice_byte_size_;

          if (src_offset + size_to_copy > shard_info.host_size) {
            return absl::InvalidArgumentError(
                "Copy range exceeds source host buffer size");
          }
          if (dst_offset + size_to_copy > shard_info.device_size) {
            return absl::InvalidArgumentError(
                "Copy range exceeds destination device buffer size");
          }

          const uint8_t* src_ptr = shard_info.host_ptr + src_offset;

          xla::Future<> future;
          {
            nb::gil_scoped_release release;
            future = shard_info.CopyRawHostToDevice(src_ptr, dst_offset,
                                                    size_to_copy);
          }
          shard_futures.push_back(std::move(future));
        }
      }
      acc.Append(std::move(shard_futures), shard_info);
    }
  }

  return acc;
}

absl::StatusOr<raiden::PjRtCopyFuture> KVCacheManager::DispatchD2hChunks(
    const std::vector<int64_t>& src_offsets,
    const std::vector<int64_t>& dst_offsets,
    const std::vector<int64_t>& copy_sizes, int64_t device_id) {
  bool is_partial = !src_offsets.empty();
  raiden::PjRtCopyFuture acc({});

  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    const auto& layer_info = layers_[layer_idx];
    for (size_t i = 0; i < num_shards_; ++i) {
      if (device_id >= 0 && static_cast<int64_t>(i) != device_id) {
        continue;
      }
      const auto& shard_info = layer_info.shards[i];
      uint8_t* dst_host_ptr = const_cast<uint8_t*>(shard_info.host_ptr);

      std::vector<xla::Future<>> shard_futures;
      if (!is_partial) {
        xla::Future<> future =
            shard_info.CopyRawDeviceToHost(dst_host_ptr, 0, physical_size_);
        shard_futures.push_back(std::move(future));
      } else {
        shard_futures.reserve(src_offsets.size());
        for (size_t j = 0; j < src_offsets.size(); ++j) {
          int64_t src_offset = src_offsets[j] * slice_byte_size_;
          int64_t dst_offset = dst_offsets[j] * slice_byte_size_;
          int64_t size_to_copy = copy_sizes[j] * slice_byte_size_;

          if (src_offset + size_to_copy > shard_info.device_size) {
            LOG(ERROR) << "Copy range check failed: src_offset=" << src_offset
                       << ", size_to_copy=" << size_to_copy
                       << ", device_size=" << shard_info.device_size
                       << ", slice_byte_size_=" << slice_byte_size_;
            return absl::InvalidArgumentError(
                "Copy range exceeds source device buffer size");
          }
          if (dst_offset + size_to_copy > shard_info.host_size) {
            return absl::InvalidArgumentError(
                "Copy range exceeds destination host buffer size");
          }

          uint8_t* dst_ptr = dst_host_ptr + dst_offset;

          xla::Future<> future =
              shard_info.CopyRawDeviceToHost(dst_ptr, src_offset, size_to_copy);
          shard_futures.push_back(std::move(future));
        }
      }
      acc.Append(std::move(shard_futures), shard_info);
    }
  }

  return acc;
}

absl::StatusOr<raiden::PjRtCopyFuture> KVCacheManager::D2h(
    nb::list src_offsets_major_dim, nb::list dst_offsets_major_dim,
    nb::list copy_sizes_major_dim) {
  size_t num_chunks = src_offsets_major_dim.size();
  if (num_chunks != dst_offsets_major_dim.size() ||
      num_chunks != copy_sizes_major_dim.size()) {
    return absl::InvalidArgumentError(
        "Lengths of offset and size lists must match");
  }

  std::vector<int64_t> src_offsets;
  std::vector<int64_t> dst_offsets;
  std::vector<int64_t> copy_sizes;
  src_offsets.reserve(num_chunks);
  dst_offsets.reserve(num_chunks);
  copy_sizes.reserve(num_chunks);

  for (size_t j = 0; j < num_chunks; ++j) {
    src_offsets.push_back(nb::cast<int64_t>(src_offsets_major_dim[j]));
    dst_offsets.push_back(nb::cast<int64_t>(dst_offsets_major_dim[j]));
    copy_sizes.push_back(nb::cast<int64_t>(copy_sizes_major_dim[j]));
  }

  return DispatchD2hChunks(src_offsets, dst_offsets, copy_sizes);
}

absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
KVCacheManager::D2hAutoAllocate(nb::list src_offsets_major_dim,
                                nb::list copy_sizes_major_dim,
                                int64_t entity_id) {
  size_t num_chunks = src_offsets_major_dim.size();
  if (num_chunks != copy_sizes_major_dim.size()) {
    return absl::InvalidArgumentError(
        "Lengths of offset and size lists must match");
  }

  if (!block_manager_) {
    return absl::InternalError("Block manager is not initialized");
  }

  int total_blocks_to_allocate = 0;
  std::vector<int> blocks_per_chunk;
  blocks_per_chunk.reserve(num_chunks);

  for (size_t j = 0; j < num_chunks; ++j) {
    int64_t copy_size = nb::cast<int64_t>(copy_sizes_major_dim[j]);
    if (copy_size % block_size_ != 0) {
      return absl::InvalidArgumentError(
          "Copy size must be a multiple of block size");
    }
    int needed = copy_size / block_size_;
    total_blocks_to_allocate += needed;
    blocks_per_chunk.push_back(needed);
  }

  TF_ASSIGN_OR_RETURN(std::vector<int> allocated_block_ids,
                      block_manager_->Allocate(total_blocks_to_allocate,
                                               entity_id, /*lock=*/true));

  std::vector<int64_t> flat_src_offsets;
  std::vector<int64_t> flat_dst_offsets;
  std::vector<int64_t> flat_copy_sizes;
  flat_src_offsets.reserve(total_blocks_to_allocate);
  flat_dst_offsets.reserve(total_blocks_to_allocate);
  flat_copy_sizes.reserve(total_blocks_to_allocate);

  size_t block_id_idx = 0;
  for (size_t j = 0; j < num_chunks; ++j) {
    int64_t src_major_dim_offset = nb::cast<int64_t>(src_offsets_major_dim[j]);
    int needed = blocks_per_chunk[j];

    for (int k = 0; k < needed; ++k) {
      int assigned_block_id = allocated_block_ids[block_id_idx++];
      flat_src_offsets.push_back(src_major_dim_offset + k * block_size_);
      flat_dst_offsets.push_back(assigned_block_id * block_size_);
      flat_copy_sizes.push_back(block_size_);
    }
  }

  TF_ASSIGN_OR_RETURN(
      raiden::PjRtCopyFuture future,
      DispatchD2hChunks(flat_src_offsets, flat_dst_offsets, flat_copy_sizes));

  return std::make_pair(allocated_block_ids, std::move(future));
}

std::optional<int> KVCacheManager::local_port() const {
  if (server_) return server_->local_port_;
  return std::nullopt;
}

absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
KVCacheManager::H2hWrite(std::string peer, nb::list src_block_ids,
                         int64_t entity_id) {
  size_t num_blocks = nb::len(src_block_ids);
  if (num_blocks == 0) {
    return absl::InvalidArgumentError("Block list cannot be empty");
  }

  int P = parallelism_;
  if (static_cast<int>(num_blocks) < P) P = num_blocks;

  if (num_blocks % P != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Block count (", num_blocks, ") must be fully divisible by parallelism (", P, ")"));
  }

  size_t blocks_per_stream = num_blocks / P;
  std::vector<int> allocated_ids(num_blocks, 0);
  std::vector<std::thread> threads;
  std::vector<absl::Status> statuses(P, absl::OkStatus());

  threads.reserve(P);
  for (int i = 0; i < P; ++i) {
    threads.push_back(std::thread(
        &KVCacheManager::H2hWriteWorker, this, i, peer, blocks_per_stream,
        std::ref(src_block_ids), std::ref(allocated_ids), std::ref(statuses)));
  }

  for (auto& t : threads) {
    if (t.joinable()) t.join();
  }

  // Check thread execution statuses
  for (int i = 0; i < P; ++i) {
    if (!statuses[i].ok()) return statuses[i];
  }

  return std::make_pair(allocated_ids, raiden::PjRtCopyFuture({}));
}

void KVCacheManager::H2hWriteWorker(int stream_idx, const std::string& peer,
                                    size_t blocks_per_stream,
                                    const nb::list& src_block_ids,
                                    std::vector<int>& allocated_ids,
                                    std::vector<absl::Status>& statuses) {
  size_t offset = stream_idx * blocks_per_stream;
  auto status_or_fd = ConnectToPeer(peer);
  if (!status_or_fd.ok()) {
    statuses[stream_idx] = status_or_fd.status();
    return;
  }
  int fd = status_or_fd.value();
  auto fd_cleaner = absl::MakeCleanup([fd] { close(fd); });

  BlockPacketHeader header;
  header.op = 1;  // Push
  header.remote_block_id = 0;
  header.local_block_id = 0;
  header.num_blocks = blocks_per_stream;

  absl::Status s = WriteExact(fd, &header, sizeof(header));
  if (!s.ok()) {
    statuses[stream_idx] = s;
    return;
  }

  std::vector<int> stream_allocated_ids(blocks_per_stream, 0);
  s = ReadExact(fd, stream_allocated_ids.data(),
                blocks_per_stream * sizeof(int));
  if (!s.ok()) {
    statuses[stream_idx] = s;
    return;
  }

  // Copy assigned remote block IDs to the shared global allocated_ids vector
  for (size_t k = 0; k < blocks_per_stream; ++k) {
    allocated_ids[offset + k] = stream_allocated_ids[k];
  }

  size_t bytes_per_block = block_size_ * slice_byte_size_;

  // Stream payload blocks cleanly across all layers and shards.
  for (size_t l = 0; l < num_layers_; ++l) {
    const auto& layer_info = layers_[l];
    for (size_t sh = 0; sh < num_shards_; ++sh) {
      const auto& shard_info = layer_info.shards[sh];
      const uint8_t* base_host_ptr = shard_info.host_ptr;

      for (size_t k = 0; k < blocks_per_stream; ++k) {
        int src_id = nb::cast<int>(src_block_ids[offset + k]);
        const uint8_t* src_ptr = base_host_ptr + src_id * bytes_per_block;
        s = WriteExact(fd, src_ptr, bytes_per_block);
        if (!s.ok()) {
          statuses[stream_idx] = s;
          return;
        }
      }
    }
  }

  // Read final completion acknowledgment byte
  uint8_t ack = 0;
  s = ReadExact(fd, &ack, 1);
  if (!s.ok()) {
    statuses[stream_idx] = s;
    return;
  }
}

absl::StatusOr<std::pair<std::vector<int>, raiden::PjRtCopyFuture>>
KVCacheManager::H2hRead(std::string peer, nb::list src_block_ids,
                        int64_t entity_id) {
  size_t num_blocks = nb::len(src_block_ids);
  if (num_blocks == 0) {
    return absl::InvalidArgumentError("Block list cannot be empty");
  }

  size_t local_blocks = num_blocks / shard_factor_;
  TF_ASSIGN_OR_RETURN(std::vector<int> allocated_ids,
                      block_manager_->Allocate(local_blocks, entity_id, true));

  int P = parallelism_;
  if (static_cast<int>(local_blocks) < P) P = local_blocks;

  if (local_blocks % P != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Local block count (", local_blocks, ") must be fully divisible by parallelism (", P, ")"));
  }

  size_t blocks_per_stream = local_blocks / P;
  size_t remote_blocks_per_stream = num_blocks / P;
  int base_remote_id = nb::cast<int>(src_block_ids[0]);

  std::vector<std::thread> threads;
  std::vector<absl::Status> statuses(P, absl::OkStatus());

  threads.reserve(P);
  for (int i = 0; i < P; ++i) {
    threads.push_back(std::thread(&KVCacheManager::H2hReadWorker, this, i, peer,
                                  blocks_per_stream, remote_blocks_per_stream,
                                  base_remote_id, std::ref(allocated_ids),
                                  std::ref(statuses)));
  }

  for (auto& t : threads) {
    if (t.joinable()) t.join();
  }

  // Check thread execution statuses
  for (int i = 0; i < P; ++i) {
    if (!statuses[i].ok()) return statuses[i];
  }

  return std::make_pair(allocated_ids, raiden::PjRtCopyFuture({}));
}

void KVCacheManager::H2hReadWorker(int stream_idx, const std::string& peer,
                                   size_t blocks_per_stream,
                                   size_t remote_blocks_per_stream,
                                   int base_remote_id,
                                   const std::vector<int>& allocated_ids,
                                   std::vector<absl::Status>& statuses) {
  size_t offset = stream_idx * blocks_per_stream;
  size_t remote_offset = stream_idx * remote_blocks_per_stream;

  auto status_or_fd = ConnectToPeer(peer);
  if (!status_or_fd.ok()) {
    statuses[stream_idx] = status_or_fd.status();
    return;
  }
  int fd = status_or_fd.value();
  auto fd_cleaner = absl::MakeCleanup([fd] { close(fd); });

  BlockPacketHeader header;
  header.op = 2;  // Pull request
  header.remote_block_id = base_remote_id + remote_offset;
  header.local_block_id = allocated_ids[offset];
  header.num_blocks = remote_blocks_per_stream;

  absl::Status s = WriteExact(fd, &header, sizeof(header));
  if (!s.ok()) {
    statuses[stream_idx] = s;
    return;
  }

  // Read back response push packet header.
  BlockPacketHeader resp_header;
  s = ReadExact(fd, &resp_header, sizeof(resp_header));
  if (!s.ok()) {
    statuses[stream_idx] = s;
    return;
  }

  size_t bytes_per_block = block_size_ * slice_byte_size_;

  // Read payload blocks streaming back directly into locally assigned memory
  // structures.
  for (size_t l = 0; l < num_layers_; ++l) {
    const auto& layer_info = layers_[l];
    for (size_t sh = 0; sh < num_shards_; ++sh) {
      const auto& shard_info = layer_info.shards[sh];
      uint8_t* base_host_ptr = const_cast<uint8_t*>(shard_info.host_ptr);

      for (size_t k = 0; k < blocks_per_stream; ++k) {
        int dst_id = allocated_ids[offset + k];
        uint8_t* dest_ptr = base_host_ptr + dst_id * bytes_per_block;
        s = ReadExact(fd, dest_ptr, bytes_per_block);
        if (!s.ok()) {
          statuses[stream_idx] = s;
          return;
        }
      }
    }
  }
}

absl::Status KVCacheManager::H2dDirect(
    stream_executor::Stream* stream,
    const std::vector<uint8_t*>& device_buffers,
    const std::vector<int64_t>& src_offsets,
    const std::vector<int64_t>& dst_offsets,
    const std::vector<int64_t>& copy_sizes) {
  if (src_offsets.size() != dst_offsets.size() ||
      src_offsets.size() != copy_sizes.size()) {
    return absl::InvalidArgumentError(
        "Lengths of offset and size vectors must match");
  }
  if (device_buffers.size() != num_layers_) {
    return absl::InvalidArgumentError(
        "Number of device buffers must match layer count");
  }

  int64_t block_byte_size = block_size_ * slice_byte_size_;
  int64_t num_chunks = src_offsets.size();

  for (size_t l = 0; l < num_layers_; ++l) {
    const auto& layer_info = layers_[l];
    const auto& shard_info = layer_info.shards[0];
    const uint8_t* h_base = shard_info.host_ptr;
    uint8_t* d_base = device_buffers[l];

    for (int64_t i = 0; i < num_chunks; ++i) {
      int64_t copy_size = copy_sizes[i];
      if (copy_size == 0) continue;

      int64_t s_offset = src_offsets[i] * block_byte_size;
      int64_t d_offset = dst_offsets[i] * block_byte_size;
      size_t size_bytes = copy_size * block_byte_size;

      const uint8_t* src_ptr = h_base + s_offset;
      uint8_t* dst_ptr = d_base + d_offset;

      stream_executor::DeviceAddressBase device_addr(dst_ptr, size_bytes);
      TF_RETURN_IF_ERROR(stream->Memcpy(&device_addr, src_ptr, size_bytes));
    }
  }
  return absl::OkStatus();
}

absl::Status KVCacheManager::D2hDirect(
    stream_executor::Stream* stream,
    const std::vector<uint8_t*>& device_buffers,
    const std::vector<int64_t>& src_offsets,
    const std::vector<int64_t>& dst_offsets,
    const std::vector<int64_t>& copy_sizes) {
  if (src_offsets.size() != dst_offsets.size() ||
      src_offsets.size() != copy_sizes.size()) {
    return absl::InvalidArgumentError(
        "Lengths of offset and size vectors must match");
  }
  if (device_buffers.size() != num_layers_) {
    return absl::InvalidArgumentError(
        "Number of device buffers must match layer count");
  }

  int64_t block_byte_size = block_size_ * slice_byte_size_;
  int64_t num_chunks = src_offsets.size();

  for (size_t l = 0; l < num_layers_; ++l) {
    const auto& layer_info = layers_[l];
    const auto& shard_info = layer_info.shards[0];
    uint8_t* h_base = const_cast<uint8_t*>(shard_info.host_ptr);
    const uint8_t* d_base = device_buffers[l];

    for (int64_t i = 0; i < num_chunks; ++i) {
      int64_t copy_size = copy_sizes[i];
      if (copy_size == 0) continue;

      int64_t s_offset = src_offsets[i] * block_byte_size;
      int64_t d_offset = dst_offsets[i] * block_byte_size;
      size_t size_bytes = copy_size * block_byte_size;

      const uint8_t* src_ptr = d_base + s_offset;
      uint8_t* dst_ptr = h_base + d_offset;

      stream_executor::DeviceAddressBase src_addr(const_cast<uint8_t*>(src_ptr),
                                                  size_bytes);
      TF_RETURN_IF_ERROR(stream->Memcpy(dst_ptr, src_addr, size_bytes));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int>> KVCacheManager::H2hWriteDirect(
    const std::string& peer, const std::vector<int32_t>& src_block_ids,
    int64_t entity_id) {
  size_t num_blocks = src_block_ids.size();
  if (num_blocks == 0) {
    return absl::InvalidArgumentError("Block list cannot be empty");
  }

  int P = parallelism_;
  if (static_cast<int>(num_blocks) < P) P = num_blocks;

  if (num_blocks % P != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Block count (", num_blocks,
                     ") must be fully divisible by parallelism (", P, ")"));
  }

  nb::list src_block_ids_list;
  for (int id : src_block_ids) {
    src_block_ids_list.append(id);
  }

  size_t blocks_per_stream = num_blocks / P;
  std::vector<int> allocated_ids(num_blocks, 0);
  std::vector<std::thread> threads;
  std::vector<absl::Status> statuses(P, absl::OkStatus());

  threads.reserve(P);
  for (int i = 0; i < P; ++i) {
    threads.push_back(std::thread(&KVCacheManager::H2hWriteWorker, this, i,
                                  peer, blocks_per_stream,
                                  std::ref(src_block_ids_list),
                                  std::ref(allocated_ids), std::ref(statuses)));
  }

  for (auto& t : threads) {
    if (t.joinable()) t.join();
  }

  for (int i = 0; i < P; ++i) {
    if (!statuses[i].ok()) return statuses[i];
  }

  return allocated_ids;
}

absl::StatusOr<std::vector<int>> KVCacheManager::H2hReadDirect(
    const std::string& peer, const std::vector<int32_t>& src_block_ids,
    int64_t entity_id) {
  size_t num_blocks = src_block_ids.size();
  if (num_blocks == 0) {
    return absl::InvalidArgumentError("Block list cannot be empty");
  }

  size_t local_blocks = num_blocks / shard_factor_;
  TF_ASSIGN_OR_RETURN(std::vector<int> allocated_ids,
                      block_manager_->Allocate(local_blocks, entity_id, true));

  int P = parallelism_;
  if (static_cast<int>(local_blocks) < P) P = local_blocks;

  if (local_blocks % P != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Local block count (", local_blocks,
                     ") must be fully divisible by parallelism (", P, ")"));
  }

  nb::list src_block_ids_list;
  for (int id : src_block_ids) {
    src_block_ids_list.append(id);
  }

  size_t blocks_per_stream = local_blocks / P;
  size_t remote_blocks_per_stream = num_blocks / P;
  int base_remote_id = src_block_ids[0];

  std::vector<std::thread> threads;
  std::vector<absl::Status> statuses(P, absl::OkStatus());

  threads.reserve(P);
  for (int i = 0; i < P; ++i) {
    threads.push_back(std::thread(&KVCacheManager::H2hReadWorker, this, i, peer,
                                  blocks_per_stream, remote_blocks_per_stream,
                                  base_remote_id, std::ref(allocated_ids),
                                  std::ref(statuses)));
  }

  for (auto& t : threads) {
    if (t.joinable()) t.join();
  }

  for (int i = 0; i < P; ++i) {
    if (!statuses[i].ok()) return statuses[i];
  }

  return allocated_ids;
}

absl::StatusOr<raiden::PjRtCopyFuture> KVCacheManager::H2dDirect(
    const std::vector<int64_t>& src_offsets,
    const std::vector<int64_t>& dst_offsets,
    const std::vector<int64_t>& copy_sizes, int64_t device_id) {
  bool is_partial = !src_offsets.empty();
  if (is_partial) {
    if (src_offsets.size() != dst_offsets.size() ||
        src_offsets.size() != copy_sizes.size()) {
      return absl::InvalidArgumentError(
          "Lengths of offset and size vectors must match");
    }
  }

  raiden::PjRtCopyFuture acc({});
  for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
    const auto& layer_info = layers_[layer_idx];
    for (size_t i = 0; i < num_shards_; ++i) {
      if (device_id >= 0 && static_cast<int64_t>(i) != device_id) {
        continue;
      }
      const auto& shard_info = layer_info.shards[i];

      std::vector<xla::Future<>> shard_futures;
      if (!is_partial) {
        xla::Future<> future = shard_info.CopyRawHostToDevice(
            shard_info.host_ptr, 0, physical_size_);
        shard_futures.push_back(std::move(future));
      } else {
        for (size_t j = 0; j < src_offsets.size(); ++j) {
          int64_t src_major_dim_offset = src_offsets[j];
          int64_t dst_major_dim_offset = dst_offsets[j];
          int64_t major_dim_size = copy_sizes[j];

          int64_t src_offset = src_major_dim_offset * slice_byte_size_;
          int64_t dst_offset = dst_major_dim_offset * slice_byte_size_;
          int64_t size_to_copy = major_dim_size * slice_byte_size_;

          if (src_offset + size_to_copy > shard_info.host_size) {
            return absl::InvalidArgumentError(
                "Copy range exceeds source host buffer size");
          }
          if (dst_offset + size_to_copy > shard_info.device_size) {
            return absl::InvalidArgumentError(
                "Copy range exceeds destination device buffer size");
          }

          const uint8_t* src_ptr = shard_info.host_ptr + src_offset;

          xla::Future<> future =
              shard_info.CopyRawHostToDevice(src_ptr, dst_offset, size_to_copy);
          shard_futures.push_back(std::move(future));
        }
      }
      acc.Append(std::move(shard_futures), shard_info);
    }
  }

  return acc;
}

absl::StatusOr<raiden::PjRtCopyFuture> KVCacheManager::D2hDirect(
    const std::vector<int64_t>& src_offsets,
    const std::vector<int64_t>& dst_offsets,
    const std::vector<int64_t>& copy_sizes, int64_t device_id) {
  return DispatchD2hChunks(src_offsets, dst_offsets, copy_sizes, device_id);
}

}  // namespace kv_cache
}  // namespace tpu_raiden

NB_MODULE(kv_cache_manager, m) {
  nb::module_::import_(
      "google3.third_party.tpu_raiden.raiden_lib.raw_transfer.jax.raw_transfer");
  nb::class_<tpu_raiden::kv_cache::KVCacheManager>(m, "KVCacheManager")
      .def(nb::init<nb::list, int, std::optional<int>, int, bool, int>(),
           nb::arg("device_arrays"), nb::arg("block_size") = 1,
           nb::arg("local_port") = nb::none(),
           nb::arg("host_blocks_to_allocate") = 64,
           nb::arg("unsafe_skip_buffer_lock") = false,
           nb::arg("parallelism") = 1)
      .def(
          "h2d",
          [](tpu_raiden::kv_cache::KVCacheManager& self, nb::list src_offsets,
             nb::list dst_offsets, nb::list copy_sizes) {
            return xla::ValueOrThrow(
                self.H2d(src_offsets, dst_offsets, copy_sizes));
          },
          nb::arg("src_offsets_major_dim") = nb::list(),
          nb::arg("dst_offsets_major_dim") = nb::list(),
          nb::arg("copy_sizes_major_dim") = nb::list())
      .def(
          "d2h",
          [](tpu_raiden::kv_cache::KVCacheManager& self, nb::list src_offsets,
             nb::list dst_offsets, nb::list copy_sizes) {
            return xla::ValueOrThrow(
                self.D2h(src_offsets, dst_offsets, copy_sizes));
          },
          nb::arg("src_offsets_major_dim") = nb::list(),
          nb::arg("dst_offsets_major_dim") = nb::list(),
          nb::arg("copy_sizes_major_dim") = nb::list())
      .def(
          "d2h_auto_allocate",
          [](tpu_raiden::kv_cache::KVCacheManager& self, nb::list src_offsets,
             nb::list copy_sizes, int64_t entity_id) {
            auto result = xla::ValueOrThrow(
                self.D2hAutoAllocate(src_offsets, copy_sizes, entity_id));
            nb::list block_ids_list;
            for (int id : result.first) {
              block_ids_list.append(id);
            }
            return nb::make_tuple(block_ids_list, std::move(result.second));
          },
          nb::arg("src_offsets_major_dim") = nb::list(),
          nb::arg("copy_sizes_major_dim") = nb::list(),
          nb::arg("entity_id") = 0)
      .def(
          "h2h_write",
          [](tpu_raiden::kv_cache::KVCacheManager& self, std::string peer,
             nb::list src_block_ids, int64_t entity_id) {
            auto result = xla::ValueOrThrow(
                self.H2hWrite(peer, src_block_ids, entity_id));
            nb::list ids_list;
            for (int id : result.first) ids_list.append(id);
            return nb::make_tuple(ids_list, std::move(result.second));
          },
          nb::arg("peer"), nb::arg("src_block_ids"), nb::arg("entity_id") = 0)
      .def(
          "h2h_read",
          [](tpu_raiden::kv_cache::KVCacheManager& self, std::string peer,
             nb::list src_block_ids, int64_t entity_id) {
            auto result =
                xla::ValueOrThrow(self.H2hRead(peer, src_block_ids, entity_id));
            nb::list ids_list;
            for (int id : result.first) ids_list.append(id);
            return nb::make_tuple(ids_list, std::move(result.second));
          },
          nb::arg("peer"), nb::arg("src_block_ids"), nb::arg("entity_id") = 0)
      .def("local_port", &tpu_raiden::kv_cache::KVCacheManager::local_port);
}
