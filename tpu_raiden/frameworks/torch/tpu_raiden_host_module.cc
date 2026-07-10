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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tpu_raiden/core/kv_cache_manager_with_transfer.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"
#include "tpu_raiden/transport/block_transport.h"

namespace nb = nanobind;

namespace tpu_raiden {
namespace {

std::string FormatAddressWithPort(absl::string_view ip, int port) {
  if (ip.find(':') != absl::string_view::npos && !ip.empty() &&
      ip.front() != '[') {
    return absl::StrCat("[", ip, "]:", port);
  }
  return absl::StrCat(ip, ":", port);
}

class HostKVCacheManager : public KVCacheManagerWithTransfer {
 public:
  HostKVCacheManager(size_t num_layers, size_t num_shards,
                     size_t slice_byte_size, int64_t node_id,
                     std::optional<int> local_port,
                     std::optional<int> host_blocks_to_allocate,
                     int parallelism)
      : KVCacheManagerWithTransfer(
            num_layers, num_shards, slice_byte_size, local_port,
            host_blocks_to_allocate, parallelism, node_id,
            /*local_control_port=*/-1,
            /*max_blocks=*/host_blocks_to_allocate.value_or(0),
            /*num_slots=*/0, /*timeout_s=*/120.0) {}

  std::string transfer_address() const {
    std::optional<int> port = local_port();
    if (!port.has_value()) return "";
    return FormatAddressWithPort(local_ip(), *port);
  }

  absl::Status PushRegisteredPlan(uint64_t uuid, const std::string& peer,
                                  const std::vector<int>& src_block_ids,
                                  const std::vector<int>& dst_block_ids,
                                  int layer_idx, int parallelism) {
    if (peer.empty()) {
      return absl::InvalidArgumentError("peer must not be empty");
    }
    if (src_block_ids.empty()) {
      return absl::InvalidArgumentError("src_block_ids must not be empty");
    }
    if (src_block_ids.size() != dst_block_ids.size()) {
      return absl::InvalidArgumentError(
          "src_block_ids and dst_block_ids must have the same length");
    }
    InitTransportServer();
    // Copy the transport pointer and release server_init_mu_ before the
    // blocking Push: holding the lock across it serializes concurrent pushes
    // from the same manager (one per destination peer).
    tpu_raiden::transport::BlockTransport* transport = nullptr;
    {
      absl::MutexLock lock(server_init_mu_);
      transport = server_.get();
    }
    if (!transport) {
      return absl::FailedPreconditionError("Transport server is not running");
    }
    auto status_or = transport->Push(
        peer, src_block_ids, dst_block_ids, parallelism,
        tpu_raiden::transport::MajorOrder::kLayerMajor, uuid, layer_idx);
    if (!status_or.ok()) {
      return status_or.status();
    }
    return absl::OkStatus();
  }

  absl::StatusOr<std::string> ReadBlockBytes(size_t layer_idx, int block_id,
                                             size_t shard_idx = 0) {
    if (block_id < 0) {
      return absl::InvalidArgumentError("block_id must be non-negative");
    }
    const size_t block_bytes = bytes_per_block();
    const size_t host_size = GetHostSize(layer_idx, shard_idx);
    const uint8_t* base = GetHostPointer(layer_idx, shard_idx);
    if (base == nullptr) {
      return absl::OutOfRangeError("layer or shard index out of range");
    }
    const size_t block = static_cast<size_t>(block_id);
    if (block_bytes == 0 || block > host_size / block_bytes ||
        block * block_bytes + block_bytes > host_size) {
      return absl::OutOfRangeError("block range exceeds host buffer");
    }
    const char* ptr = reinterpret_cast<const char*>(base + block * block_bytes);
    return std::string(ptr, block_bytes);
  }

  absl::Status WriteBlockBytes(size_t layer_idx, int block_id,
                               absl::string_view payload,
                               size_t shard_idx = 0) {
    if (block_id < 0) {
      return absl::InvalidArgumentError("block_id must be non-negative");
    }
    const size_t block_bytes = bytes_per_block();
    if (payload.size() != block_bytes) {
      return absl::InvalidArgumentError(
          absl::StrCat("payload size must equal block size: got ",
                       payload.size(), ", expected ", block_bytes));
    }
    const size_t host_size = GetHostSize(layer_idx, shard_idx);
    uint8_t* base = GetHostPointer(layer_idx, shard_idx);
    if (base == nullptr) {
      return absl::OutOfRangeError("layer or shard index out of range");
    }
    const size_t block = static_cast<size_t>(block_id);
    if (block_bytes == 0 || block > host_size / block_bytes ||
        block * block_bytes + block_bytes > host_size) {
      return absl::OutOfRangeError("block range exceeds host buffer");
    }
    std::memcpy(base + block * block_bytes, payload.data(), block_bytes);
    return absl::OkStatus();
  }
};

void ThrowIfError(const absl::Status& status, absl::string_view context) {
  if (!status.ok()) {
    throw std::runtime_error(absl::StrCat(context, ": ", status.message()));
  }
}

}  // namespace

NB_MODULE(_tpu_raiden_host, m) {
  nb::class_<HostKVCacheManager>(m, "KVCacheManager")
      .def(nb::init<size_t, size_t, size_t, int64_t, std::optional<int>,
                    std::optional<int>, int>(),
           nb::arg("num_layers"), nb::arg("num_shards"),
           nb::arg("slice_byte_size"), nb::arg("node_id"),
           nb::arg("local_port") = nb::none(),
           nb::arg("host_blocks_to_allocate") = nb::none(),
           nb::arg("parallelism") = 1)
      .def("node_id", &HostKVCacheManager::node_id)
      .def_prop_ro("local_port", &HostKVCacheManager::local_port)
      .def_prop_ro("num_layers", &HostKVCacheManager::num_layers)
      .def_prop_ro("num_shards", &HostKVCacheManager::num_shards)
      .def_prop_ro("slice_byte_size", &HostKVCacheManager::slice_byte_size)
      .def_prop_ro("transfer_address", &HostKVCacheManager::transfer_address)
      .def("get_local_endpoints",
           [](const HostKVCacheManager& self) {
             auto eps = self.get_local_endpoints();
             nb::list py_eps;
             for (const auto& ep : eps) {
               nb::dict d;
               d["endpoint"] = ep.endpoint;
               d["shards"] = ep.shards;
               py_eps.append(d);
             }
             return py_eps;
           })
      .def(
          "register_active_plan",
          [](HostKVCacheManager& self, uint64_t uuid,
             const nb::bytes& request_bytes, bool is_sender) {
            tpu_raiden::rpc::StartTransferRequest request;
            if (!request.ParseFromArray(request_bytes.c_str(),
                                        request_bytes.size())) {
              throw std::runtime_error(
                  "KVCacheManager register_active_plan failed: invalid "
                  "StartTransferRequest bytes");
            }
            ThrowIfError(self.RegisterActivePlan(uuid, request, is_sender),
                         "KVCacheManager register_active_plan failed");
          },
          nb::arg("uuid"), nb::arg("request_bytes"), nb::arg("is_sender"))
      .def(
          "unregister_active_plan",
          [](HostKVCacheManager& self, uint64_t uuid) {
            ThrowIfError(self.UnregisterActivePlan(uuid),
                         "KVCacheManager unregister_active_plan failed");
          },
          nb::arg("uuid"))
      .def(
          "push_registered_plan",
          [](HostKVCacheManager& self, uint64_t uuid, const std::string& peer,
             const std::vector<int>& src_block_ids,
             const std::vector<int>& dst_block_ids, int layer_idx,
             int parallelism) {
            ThrowIfError(
                self.PushRegisteredPlan(uuid, peer, src_block_ids,
                                        dst_block_ids, layer_idx, parallelism),
                "KVCacheManager push_registered_plan failed");
          },
          nb::arg("uuid"), nb::arg("peer"), nb::arg("src_block_ids"),
          nb::arg("dst_block_ids"), nb::arg("layer_idx") = -1,
          nb::arg("parallelism") = 1, nb::call_guard<nb::gil_scoped_release>())
      .def(
          "read_block_bytes",
          [](HostKVCacheManager& self, size_t layer_idx, int block_id) {
            auto status_or = self.ReadBlockBytes(layer_idx, block_id);
            if (!status_or.ok()) {
              throw std::runtime_error(
                  absl::StrCat("KVCacheManager read_block_bytes failed: ",
                               status_or.status().message()));
            }
            const std::string& data = status_or.value();
            return nb::bytes(data.data(), data.size());
          },
          nb::arg("layer_idx"), nb::arg("block_id"))
      .def(
          "write_block_bytes",
          [](HostKVCacheManager& self, size_t layer_idx, int block_id,
             const nb::bytes& payload) {
            std::string payload_str(payload.c_str(), payload.size());
            ThrowIfError(self.WriteBlockBytes(layer_idx, block_id, payload_str),
                         "KVCacheManager write_block_bytes failed");
          },
          nb::arg("layer_idx"), nb::arg("block_id"), nb::arg("payload"));
}

}  // namespace tpu_raiden
