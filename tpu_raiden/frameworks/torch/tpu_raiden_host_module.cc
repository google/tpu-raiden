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
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/frameworks/torch/pool_layout_nanobind.h"
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

// Test/library role configuration of KVCacheManagerWithTransfer; not part of
// the TPU serving path (scaffolding ledger A.4 classification; exported to
// the third-party raiden library surface, 539d7ea).
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

};

// Distinct C++ wrapper so the host and torch extension modules can each
// register a Python RaidenFuture without colliding in nanobind's global type
// registry when both modules are imported in one process.
struct HostRaidenFuture {
  explicit HostRaidenFuture(raiden::PjRtCopyFuture value)
      : future(std::move(value)) {}

  absl::Status Await() { return future.Await(); }
  bool IsReady() const { return future.IsReady(); }
  absl::Status PollError() { return future.PollError(); }

  raiden::PjRtCopyFuture future;
};

void ThrowIfError(const absl::Status& status, absl::string_view context) {
  if (!status.ok()) {
    throw std::runtime_error(absl::StrCat(context, ": ", status.message()));
  }
}

}  // namespace

NB_MODULE(_tpu_raiden_host, m) {
  nb::class_<HostRaidenFuture>(m, "RaidenFuture")
      .def("Await",
           [](HostRaidenFuture& self) {
             nb::gil_scoped_release release;
             ThrowIfError(self.Await(), "Async copy failed");
           })
      .def("wait",
           [](HostRaidenFuture& self) {
             nb::gil_scoped_release release;
             ThrowIfError(self.Await(), "Async copy failed");
           })
      .def("IsReady", &HostRaidenFuture::IsReady)
      .def("is_ready", &HostRaidenFuture::IsReady)
      .def("ok", [](HostRaidenFuture& self) { return self.PollError().ok(); })
      .def("error_message", [](HostRaidenFuture& self) {
        absl::Status status = self.PollError();
        return status.ok() ? std::string() : std::string(status.message());
      });

  auto manager_cls = nb::class_<HostKVCacheManager>(m, "KVCacheManager");
  manager_cls
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
          nb::arg("uuid"));
  tpu_raiden::torch_bindings::BindPoolApi<HostRaidenFuture>(manager_cls);
}

}  // namespace tpu_raiden
