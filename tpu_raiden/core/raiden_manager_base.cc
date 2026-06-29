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

#include "tpu_raiden/core/raiden_manager_base.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/future.h"
#include "xla/pjrt/pjrt_client.h"
#include "tpu_raiden/core/tpu_utils.h"
#include "tpu_raiden/transport/block_transport.h"

namespace tpu_raiden {

xla::Future<> ReturnFuture(const absl::Status& status) {
  return xla::Future<>(status);
}

void RaidenManagerBase::DetectAndAssignNumaNode(
    const std::vector<std::vector<xla::PjRtBuffer*>>& layer_buffers) {
  std::vector<int> unique_numa_nodes;
  for (const auto& layer : layer_buffers) {
    for (xla::PjRtBuffer* buf : layer) {
      if (buf && buf->device()) {
        int node = GetPjRtDeviceNumaNode(buf->device());
        if (node >= 0) {
          bool found = false;
          for (int n : unique_numa_nodes) {
            if (n == node) {
              found = true;
              break;
            }
          }
          if (!found) unique_numa_nodes.push_back(node);
        }
      }
    }
  }
  if (!unique_numa_nodes.empty()) {
    assigned_numa_node_ = unique_numa_nodes[0];
    if (unique_numa_nodes.size() > 1) {
      LOG(WARNING) << "Incoming PJRT buffers are associated with more than one "
                      "NUMA node ("
                   << unique_numa_nodes[0] << " vs " << unique_numa_nodes[1]
                   << "). Picking the first detected NUMA node: "
                   << unique_numa_nodes[0];
    }
  }
  InitTransportServer();
}

RaidenManagerBase::RaidenManagerBase(size_t num_layers, size_t num_shards,
                                     size_t slice_byte_size,
                                     std::optional<int> local_port,
                                     int parallelism,
                                     std::optional<std::string> bind_ip)
    : num_layers_(num_layers),
      num_shards_(num_shards),
      slice_byte_size_(slice_byte_size),
      parallelism_(parallelism),
      local_port_cfg_(local_port.value_or(0)),
      bind_ip_cfg_(bind_ip) {
  shard_factor_ = 1;
}

RaidenManagerBase::~RaidenManagerBase() {
  if (server_) {
    server_.reset();
  }
}

void RaidenManagerBase::InitTransportServer() {
  absl::MutexLock lock(&server_init_mu_);
  if (server_) return;
  std::string bind_ip = "127.0.0.1";
  if (bind_ip_cfg_.has_value()) {
    bind_ip = *bind_ip_cfg_;
  } else {
    std::vector<HostNicAddress> host_nics = GetLocalHostNicAddresses();
    const char* exclude_ctrl_env = std::getenv("EXCLUDE_CONTROL_INTERFACE");
    if (exclude_ctrl_env != nullptr && exclude_ctrl_env[0] != '\0') {
      std::vector<HostNicAddress> filtered_nics;
      for (const auto& nic : host_nics) {
        if (nic.interface_name != exclude_ctrl_env) {
          filtered_nics.push_back(nic);
        }
      }
      host_nics = std::move(filtered_nics);
    }
    if (!host_nics.empty()) {
      int target_numa = assigned_numa_node_.value_or(-1);
      std::cerr << "InitTransportServer: target_numa=" << target_numa
                << std::endl;
      for (const auto& nic : host_nics) {
        std::cerr << "  NIC: name=" << nic.interface_name
                  << " ip=" << nic.ip_address << " numa=" << nic.numa_node
                  << std::endl;
      }
      std::vector<HostNicAddress>::const_iterator it = host_nics.end();
      if (target_numa == 1) {
        // Prefer secondary data plane NICs (dcn1 on Borg, ens6 on GCP VM, eth2,
        // eth1 fallback)
        it = std::find_if(
            host_nics.begin(), host_nics.end(),
            [](const HostNicAddress& n) { return n.interface_name == "dcn1"; });
        if (it == host_nics.end()) {
          it = std::find_if(host_nics.begin(), host_nics.end(),
                            [](const HostNicAddress& n) {
                              return n.interface_name == "ens6";
                            });
        }
        if (it == host_nics.end()) {
          it = std::find_if(host_nics.begin(), host_nics.end(),
                            [](const HostNicAddress& n) {
                              return n.interface_name == "eth2";
                            });
        }
        if (it == host_nics.end()) {
          it = std::find_if(host_nics.begin(), host_nics.end(),
                            [](const HostNicAddress& n) {
                              return n.interface_name == "eth1";
                            });
        }
      } else {
        // target_numa <= 0 (primary). Prefer control plane/primary NICs (eth1
        // on Borg, ens5 on GCP VM, eth0)
        it = std::find_if(
            host_nics.begin(), host_nics.end(),
            [](const HostNicAddress& n) { return n.interface_name == "eth1"; });
        if (it == host_nics.end()) {
          it = std::find_if(host_nics.begin(), host_nics.end(),
                            [](const HostNicAddress& n) {
                              return n.interface_name == "ens5";
                            });
        }
        if (it == host_nics.end()) {
          it = std::find_if(host_nics.begin(), host_nics.end(),
                            [](const HostNicAddress& n) {
                              return n.interface_name == "eth0";
                            });
        }
      }

      if (it != host_nics.end()) {
        bind_ip = it->ip_address;
        std::cerr << "InitTransportServer: Binding to NIC ("
                  << it->interface_name << "): " << bind_ip << std::endl;
      } else {
        bind_ip = host_nics[0].ip_address;
        std::cerr << "InitTransportServer: Fallback bind to first NIC: "
                  << bind_ip << std::endl;
      }
    }
  }
  server_ = std::make_unique<tpu_raiden::transport::BlockTransport>(
      this, local_port_cfg_, true, bind_ip);
}

std::optional<int> RaidenManagerBase::local_port() const {
  const_cast<RaidenManagerBase*>(this)->InitTransportServer();
  absl::MutexLock lock(&server_init_mu_);
  if (server_) return server_->local_port();
  return std::nullopt;
}

std::string RaidenManagerBase::local_ip() const {
  const_cast<RaidenManagerBase*>(this)->InitTransportServer();
  absl::MutexLock lock(&server_init_mu_);
  if (server_) return server_->bound_ip();
  return "127.0.0.1";
}

uint8_t* RaidenManagerBase::GetHostPointer(size_t layer_idx, size_t shard_idx) {
  if (layer_idx >= layers_.size() ||
      shard_idx >= layers_[layer_idx].shards.size()) {
    return nullptr;
  }
  return const_cast<uint8_t*>(layers_[layer_idx].shards[shard_idx].host_ptr);
}

size_t RaidenManagerBase::GetHostSize(size_t layer_idx, size_t shard_idx) {
  if (layer_idx >= layers_.size() ||
      shard_idx >= layers_[layer_idx].shards.size()) {
    return 0;
  }
  return layers_[layer_idx].shards[shard_idx].host_size;
}

const uint8_t* RaidenManagerBase::GetHostPointer(size_t layer_idx,
                                                 size_t shard_idx) const {
  if (layer_idx >= layers_.size() ||
      shard_idx >= layers_[layer_idx].shards.size()) {
    return nullptr;
  }
  return layers_[layer_idx].shards[shard_idx].host_ptr;
}

void RaidenManagerBase::SetExternalHostPointers(
    const std::vector<const uint8_t*>& host_ptrs,
    const std::vector<size_t>& host_sizes) {
  size_t idx = 0;
  for (size_t l = 0; l < layers_.size(); ++l) {
    for (size_t sh = 0; sh < layers_[l].shards.size(); ++sh) {
      if (idx < host_ptrs.size() && idx < host_sizes.size()) {
        layers_[l].shards[sh].host_ptr = host_ptrs[idx];
        layers_[l].shards[sh].host_size = host_sizes[idx];
        idx++;
      }
    }
  }
}

absl::StatusOr<std::vector<int>> RaidenManagerBase::H2hWriteDirect(
    absl::string_view peer, const std::vector<int>& src_block_ids,
    const std::vector<int>& dst_block_ids, uint64_t uuid, int layer_idx) {
  InitTransportServer();
  absl::MutexLock lock(&server_init_mu_);
  if (!server_) {
    return absl::FailedPreconditionError("Transport server is not running");
  }
  return server_->Push(peer, src_block_ids, dst_block_ids, parallelism_,
                       tpu_raiden::transport::MajorOrder::kLayerMajor, uuid,
                       layer_idx);
}

absl::StatusOr<std::vector<int>> RaidenManagerBase::H2hReadDirect(
    absl::string_view peer, const std::vector<int>& src_block_ids) {
  InitTransportServer();
  absl::MutexLock lock(&server_init_mu_);
  if (!server_) {
    return absl::FailedPreconditionError("Transport server is not running");
  }
  return server_->Pull(peer, src_block_ids, {}, {}, parallelism_);
}

absl::Status RaidenManagerBase::PullWeightsChunk(
    absl::string_view source, size_t src_shard_idx, size_t src_offset_bytes,
    size_t dst_shard_idx, size_t dst_offset_bytes, size_t size_bytes) {
  InitTransportServer();
  absl::MutexLock lock(&server_init_mu_);
  if (!server_) {
    return absl::FailedPreconditionError("Transport server is not running");
  }
  return server_->PullBuffer(source, /*buffer_id=*/0, src_shard_idx,
                             src_offset_bytes, dst_shard_idx, dst_offset_bytes,
                             size_bytes);
}

absl::Status RaidenManagerBase::PushWeightsChunk(absl::string_view peer,
                                                 size_t dst_shard_idx,
                                                 size_t dst_offset_bytes,
                                                 const uint8_t* data_ptr,
                                                 size_t size_bytes) {
  InitTransportServer();
  absl::MutexLock lock(&server_init_mu_);
  if (!server_) {
    return absl::FailedPreconditionError("Transport server is not running");
  }
  return server_->PushBuffer(peer, /*buffer_id=*/0, dst_shard_idx,
                             dst_offset_bytes, data_ptr, size_bytes);
}

size_t RaidenManagerBase::bytes_per_block() const { return slice_byte_size_; }

}  // namespace tpu_raiden
