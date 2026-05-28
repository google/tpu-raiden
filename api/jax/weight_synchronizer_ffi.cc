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

#include "api/jax/weight_synchronizer_ffi.h"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/ffi/api/ffi.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "api/jax/weight_synchronizer.h"

namespace tpu_raiden {
namespace weight_sync {

jax::WeightSynchronizer* g_weight_synchronizers[32] = {nullptr};
std::unique_ptr<stream_executor::Stream> g_streams[32] = {nullptr};

namespace {

// FFI Init custom call implementation for WeightSynchronizer (Host CPU
// Executed)
static xla::ffi::Error TriggerWeightSynchronizerInitImpl(
    xla::ffi::AnyBuffer x, xla::ffi::AnyBuffer shard_idx_buf,
    int64_t slice_byte_size, int32_t local_port, int32_t parallelism,
    int32_t num_layers, xla::ffi::Result<xla::ffi::AnyBuffer> out) {
  (void)x;
  if (shard_idx_buf.untyped_data() == nullptr) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInvalidArgument,
                           "shard_idx_buf has null untyped data.");
  }
  int32_t shard_idx =
      *reinterpret_cast<const int32_t*>(shard_idx_buf.untyped_data());
  if (shard_idx < 0 || shard_idx >= 32) {
    return xla::ffi::Error(
        xla::ffi::ErrorCode::kInvalidArgument,
        "shard_idx out of bounds [0, 32): " + std::to_string(shard_idx));
  }

  if (g_weight_synchronizers[shard_idx] == nullptr) {
    VLOG(1)
        << "[TPU Worker FFI] >>> WS LAZY INITIALIZATION TRIGGERED <<< Shard: "
        << shard_idx;

    g_weight_synchronizers[shard_idx] = new tpu_raiden::jax::WeightSynchronizer(
        static_cast<size_t>(num_layers), static_cast<size_t>(parallelism),
        static_cast<size_t>(slice_byte_size), std::make_optional(local_port),
        std::nullopt,  // host_blocks_to_allocate
        parallelism);

    // Allocate the StreamExecutor Stream once per shard, and cache E2E!
    int64_t dev_id = static_cast<int64_t>(shard_idx);
    auto platform_or =
        stream_executor::PlatformManager::PlatformWithName("TPU");
    if (!platform_or.ok()) {
      platform_or =
          stream_executor::PlatformManager::PlatformWithName("Deepsea");
    }
    if (!platform_or.ok()) {
      platform_or = stream_executor::PlatformManager::PlatformWithName("Host");
    }
    if (!platform_or.ok()) {
      std::cerr << "[C++ FFI] Failed to resolve platform, skipping stream init "
                   "for CPU test."
                << std::endl;
    } else {
      auto platform = platform_or.value();

      auto executor_or = platform->ExecutorForDevice(dev_id);
      if (!executor_or.ok()) {
        return xla::ffi::Error(
            xla::ffi::ErrorCode::kInternal,
            "Failed to retrieve StreamExecutor for device: " +
                std::string(executor_or.status().message()));
      }
      auto executor = executor_or.value();

      auto stream_or = executor->CreateStream();
      if (!stream_or.ok()) {
        return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                               "Failed to allocate execution Stream: " +
                                   std::string(stream_or.status().message()));
      }
      g_streams[shard_idx] = std::move(stream_or.value());
    }
  }

  // Get port
  auto port_opt = g_weight_synchronizers[shard_idx]->local_port();
  if (!port_opt.has_value()) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                           "WS has no local port assigned.");
  }
  int32_t port = port_opt.value();

  // Get IP
  uint8_t ipv6[16] = {0};

  struct ifaddrs *ifaddr, *ifa;
  bool found = false;
  if (getifaddrs(&ifaddr) == 0) {
    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
      if (ifa->ifa_addr == nullptr) continue;
      if (ifa->ifa_addr->sa_family == AF_INET6) {
        struct sockaddr_in6* ipv6_addr =
            reinterpret_cast<struct sockaddr_in6*>(ifa->ifa_addr);
        if (std::strcmp(ifa->ifa_name, "lo") != 0) {
          std::memcpy(ipv6, &ipv6_addr->sin6_addr, 16);
          found = true;
          break;
        }
      }
    }
    if (!found) {
      for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr) continue;
        if (ifa->ifa_addr->sa_family == AF_INET) {
          struct sockaddr_in* ipv4 =
              reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr);
          if (std::strcmp(ifa->ifa_name, "lo") != 0) {
            ipv6[10] = 0xff;
            ipv6[11] = 0xff;
            std::memcpy(ipv6 + 12, &ipv4->sin_addr.s_addr, 4);
            found = true;
            break;
          }
        }
      }
    }
    freeifaddrs(ifaddr);
  }

  if (out->element_count() < 5) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInvalidArgument,
                           "Output buffer too small for IPv6 and port.");
  }
  int32_t* out_ptr = reinterpret_cast<int32_t*>(out->untyped_data());
  std::memcpy(out_ptr, ipv6, 16);
  out_ptr[4] = port;

  return xla::ffi::Error::Success();
}

// FFI execution handler for WeightSynchronizer D2H (Host CPU Executed)
static xla::ffi::Error TriggerWeightSynchronizerD2hImpl(
    xla::ffi::AnyBuffer anchor, xla::ffi::AnyBuffer shard_idx_buf,
    xla::ffi::Result<xla::ffi::AnyBuffer> out) {
  (void)out;

  const int32_t* p_shard_idx =
      reinterpret_cast<const int32_t*>(shard_idx_buf.untyped_data());
  int32_t shard_idx = *p_shard_idx;

  if (shard_idx < 0 || shard_idx >= 32) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kInvalidArgument,
                           "Invalid shard_idx");
  }

  if (g_weight_synchronizers[shard_idx] == nullptr) {
    return xla::ffi::Error(xla::ffi::ErrorCode::kFailedPrecondition,
                           "WeightSynchronizer not initialized for shard.");
  }

  size_t size = g_weight_synchronizers[shard_idx]->slice_byte_size();
  uint8_t* dst_host_ptr = const_cast<uint8_t*>(
      g_weight_synchronizers[shard_idx]->GetHostBufferPtr(0, 0));
  const uint8_t* src_device_ptr =
      reinterpret_cast<const uint8_t*>(anchor.untyped_data());

  if (g_streams[shard_idx] == nullptr) {
    std::memcpy(dst_host_ptr, src_device_ptr, size);
    return xla::ffi::Error::Success();
  }

  stream_executor::Stream* stream = g_streams[shard_idx].get();

  stream_executor::DeviceAddressBase device_src(
      const_cast<uint8_t*>(src_device_ptr), size);

  absl::Status status = stream->Memcpy(dst_host_ptr, device_src, size);
  if (!status.ok()) {
    return xla::ffi::Error(
        xla::ffi::ErrorCode::kInternal,
        "D2H Memcpy failed: " + std::string(status.message()));
  }

  absl::Status sync_status = stream->BlockHostUntilDone();
  if (!sync_status.ok()) {
    return xla::ffi::Error(
        xla::ffi::ErrorCode::kInternal,
        "Stream sync failed: " + std::string(sync_status.message()));
  }

  return xla::ffi::Error::Success();
}

// FFI execution handler for Resharding (Host CPU Executed)
static xla::ffi::Error TriggerExecuteReshardingImpl(
    xla::ffi::AnyBuffer anchor, xla::ffi::AnyBuffer shard_idx_buf,
    xla::ffi::AnyBuffer src_ips, xla::ffi::AnyBuffer src_ports,
    xla::ffi::AnyBuffer src_offsets, xla::ffi::AnyBuffer dst_offsets,
    xla::ffi::AnyBuffer sizes, xla::ffi::AnyBuffer dst_shard_indices,
    xla::ffi::Result<xla::ffi::AnyBuffer> out) {
  (void)anchor;
  (void)out;

  if (shard_idx_buf.untyped_data() == nullptr ||
      src_ips.untyped_data() == nullptr ||
      src_ports.untyped_data() == nullptr ||
      src_offsets.untyped_data() == nullptr ||
      dst_offsets.untyped_data() == nullptr ||
      sizes.untyped_data() == nullptr ||
      dst_shard_indices.untyped_data() == nullptr) {
    return xla::ffi::Error(
        xla::ffi::ErrorCode::kInvalidArgument,
        "One or more metadata/buffer arguments have null untyped data.");
  }

  int32_t current_device_id =
      *reinterpret_cast<const int32_t*>(shard_idx_buf.untyped_data());

  int64_t num_chunks = src_offsets.element_count();
  const uint32_t* p_src_ips =
      reinterpret_cast<const uint32_t*>(src_ips.untyped_data());
  const int32_t* p_src_ports =
      reinterpret_cast<const int32_t*>(src_ports.untyped_data());
  const int32_t* p_src_offsets =
      reinterpret_cast<const int32_t*>(src_offsets.untyped_data());
  const int32_t* p_dst_offsets =
      reinterpret_cast<const int32_t*>(dst_offsets.untyped_data());
  const int32_t* p_sizes =
      reinterpret_cast<const int32_t*>(sizes.untyped_data());
  const int32_t* p_dst_shard_indices =
      reinterpret_cast<const int32_t*>(dst_shard_indices.untyped_data());

  std::memcpy(out->untyped_data(), anchor.untyped_data(),
              anchor.element_count() * 4);

  for (int64_t i = 0; i < num_chunks; ++i) {
    int32_t dst_device_id = p_dst_shard_indices[i];
    if (dst_device_id < 0 || dst_device_id >= 32) {
      return xla::ffi::Error(xla::ffi::ErrorCode::kInvalidArgument,
                             "Invalid dst_device_id");
    }
    if (dst_device_id != current_device_id) {
      continue;  // Skip if it's not for this device!
    }

    if (g_weight_synchronizers[dst_device_id] == nullptr) {
      continue;
    }

    const uint8_t* ipv6_bytes =
        reinterpret_cast<const uint8_t*>(&p_src_ips[i * 4]);
    char ip_str[INET6_ADDRSTRLEN];
    inet_ntop(AF_INET6, ipv6_bytes, ip_str, INET6_ADDRSTRLEN);
    std::string source_addr;

    bool is_mapped = true;
    for (int j = 0; j < 10; ++j) {
      if (ipv6_bytes[j] != 0) {
        is_mapped = false;
        break;
      }
    }
    if (is_mapped && ipv6_bytes[10] == 0xff && ipv6_bytes[11] == 0xff) {
      struct in_addr addr;
      std::memcpy(&addr.s_addr, ipv6_bytes + 12, 4);
      source_addr = absl::StrCat(inet_ntoa(addr), ":", p_src_ports[i]);
    } else {
      source_addr = absl::StrCat("[", ip_str, "]:", p_src_ports[i]);
    }

    size_t src_offset = p_src_offsets[i];
    size_t dst_offset = p_dst_offsets[i];
    size_t size = p_sizes[i];

    // Execute pull
    size_t scratch_pad_offset =
        g_weight_synchronizers[dst_device_id]->slice_byte_size();
    absl::Status s = g_weight_synchronizers[dst_device_id]->PullWeightsChunk(
        source_addr, 0, src_offset, 0, scratch_pad_offset, size);
    if (!s.ok()) {
      return xla::ffi::Error(xla::ffi::ErrorCode::kInternal,
                             "Pull failed: " + std::string(s.message()));
    }

    const uint8_t* src_host_ptr =
        g_weight_synchronizers[dst_device_id]->GetHostBufferPtr(0, 0) +
        scratch_pad_offset;

    // Execute H2D using stream memcpy directly!
    uint8_t* d_base = reinterpret_cast<uint8_t*>(out->untyped_data());
    stream_executor::DeviceAddressBase device_dst(d_base + dst_offset, size);

    if (g_streams[dst_device_id] == nullptr) {
      std::memcpy(d_base + dst_offset, src_host_ptr, size);
      continue;
    }

    stream_executor::Stream* stream = g_streams[dst_device_id].get();

    absl::Status status = stream->Memcpy(&device_dst, src_host_ptr, size);
    if (!status.ok()) {
      return xla::ffi::Error(
          xla::ffi::ErrorCode::kInternal,
          "H2D Memcpy failed: " + std::string(status.message()));
    }

    absl::Status sync_status = stream->BlockHostUntilDone();
    if (!sync_status.ok()) {
      return xla::ffi::Error(
          xla::ffi::ErrorCode::kInternal,
          "Stream sync failed: " + std::string(sync_status.message()));
    }
  }

  return xla::ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER(
    kWSInit, TriggerWeightSynchronizerInitImpl,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()  // anchor JAX input array (Arg 0)
        .Arg<xla::ffi::AnyBuffer>()  // shard_idx JAX input array (Arg 1)
        .Attr<int64_t>("slice_byte_size")
        .Attr<int32_t>("local_port")
        .Attr<int32_t>("parallelism")
        .Attr<int32_t>("num_layers")
        .Ret<xla::ffi::AnyBuffer>());  // return result buffer

XLA_FFI_DEFINE_HANDLER(
    kExecuteResharding, TriggerExecuteReshardingImpl,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::AnyBuffer>()  // anchor
        .Arg<xla::ffi::AnyBuffer>()  // shard_idx_buf
        .Arg<xla::ffi::AnyBuffer>()  // src_ips
        .Arg<xla::ffi::AnyBuffer>()  // src_ports
        .Arg<xla::ffi::AnyBuffer>()  // src_offsets
        .Arg<xla::ffi::AnyBuffer>()  // dst_offsets
        .Arg<xla::ffi::AnyBuffer>()  // sizes
        .Arg<xla::ffi::AnyBuffer>()  // dst_shard_indices
        .Ret<xla::ffi::AnyBuffer>()  // result (aliased to anchor)
);

XLA_FFI_DEFINE_HANDLER(kWeightSynchronizerD2h, TriggerWeightSynchronizerD2hImpl,
                       xla::ffi::Ffi::Bind()
                           .Arg<xla::ffi::AnyBuffer>()  // anchor
                           .Arg<xla::ffi::AnyBuffer>()  // shard_idx_buf
                           .Ret<xla::ffi::AnyBuffer>()  // dummy result
);

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "weight_synchronizer_d2h",
                         "Host", kWeightSynchronizerD2h);
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "weight_synchronizer_d2h",
                         "TPU", kWeightSynchronizerD2h);

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "init_weight_synchronizer",
                         "Host", kWSInit);
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "init_weight_synchronizer",
                         "TPU", kWSInit);

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "execute_resharding", "Host",
                         kExecuteResharding);
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "execute_resharding", "TPU",
                         kExecuteResharding);

}  // namespace weight_sync
}  // namespace tpu_raiden
