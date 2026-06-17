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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_TPU_UTILS_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_TPU_UTILS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_client.h"

namespace tpu_raiden {

// Returns a default local IP by scanning network interfaces (excluding
// loopback). Returns "127.0.0.1" as a fallback.
std::string GetLocalIp();

// Discovers physical NICs, maps their local IPs to their NUMA nodes.
// Returns a map from IP address string to NUMA node ID.
absl::flat_hash_map<std::string, int> DiscoverNumaNicIPs(
    const std::string& sysfs_root = "");

// Detects the correct physical NIC IP to bind to based on the NUMA node of PJRT
// buffers.
absl::StatusOr<std::string> DetectLocalIpFromPjRtBuffers(
    const std::vector<std::vector<xla::PjRtBuffer*>>& layer_buffers);

// Splits a "host:port" or "[ipv6]:port" address into host and port.
absl::Status SplitHostPort(const std::string& address, std::string& host,
                           int& port);

struct TpuPciDevice {
  std::string bdf;        // BDF address, e.g. "0000:16:00.0"
  std::string device_id;  // Device ID, e.g. "0x0075"
  int numa_node = -1;     // NUMA node, e.g. 0
};

// Sets the memory policy for the current thread using set_mempolicy syscall.
// Returns 0 on success, or a negative error code on failure.
int64_t SetThreadMempolicy(int mode, int node = -1);

// Returns the CPU core IDs belonging to a given NUMA node.
// Parses /sys/devices/system/node/node<N>/cpulist.
std::vector<int> GetNumaNodeCpuCores(int numa_node);

// Pins the current thread to the given CPU cores.
// Returns 0 on success, or a negative error code on failure.
int PinCurrentThreadToCores(const std::vector<int>& cores);

// Returns the next CPU core for the given NUMA node in a round-robin fashion.
// Useful for distributing worker threads evenly across the node's cores.
int GetNextRoundRobinCore(int numa_node);

// Pins the current thread to the given NUMA node and binds its memory
// allocations. Returns 0 on success, or a negative error code on failure.
int PinCurrentThreadToNumaNode(int node);

// Scans the PCI bus and returns all detected TPU PCI devices, sorted by BDF.
const std::vector<TpuPciDevice>& GetTpuPciDevices();

// Returns the NUMA node for a given TPU chip index.
// Maps the chip's physical index to the sorted PCI devices.
// Returns -1 if the node cannot be determined.
int GetTpuDeviceNumaNode(int chip_idx);

// Returns the NUMA node for a given PjRtDevice.
// Maps the device's local_hardware_id to the sorted PCI devices.
// Returns -1 if the node cannot be determined.
int GetPjRtDeviceNumaNode(const xla::PjRtDevice* device);

// Prints the detected TPU hardware topology to std::cout.
void PrintTpuHardwareTopology();

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_TPU_UTILS_H_
