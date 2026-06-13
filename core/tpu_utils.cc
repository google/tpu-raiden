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

#include "core/tpu_utils.h"

#include <dirent.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "xla/pjrt/pjrt_client.h"

namespace tpu_raiden {

int64_t SetThreadMempolicy(int mode, int node) {
  constexpr int kMpolDefault = 0;
  constexpr int kMpolBind = 2;
#ifdef __NR_set_mempolicy
  int64_t res;
  if (mode == kMpolDefault || node < 0) {
    res = static_cast<int64_t>(
        syscall(__NR_set_mempolicy, kMpolDefault, nullptr, 0));
  } else {
    uint64_t mask = 1ULL << node;
    res = static_cast<int64_t>(
        syscall(__NR_set_mempolicy, kMpolBind, &mask, sizeof(mask) * 8));
  }
  if (res < 0) {
    std::cerr << "[ERROR] SetThreadMempolicy(mode=" << mode << ", node=" << node
              << ") failed: " << std::strerror(errno) << " (errno=" << errno
              << ")" << std::endl;
  }
  return res;
#else
  return -1;  // Syscall not available
#endif
}

std::vector<TpuPciDevice> GetTpuPciDevices() {
  std::vector<TpuPciDevice> tpu_devices;
  std::string pci_dir = "/sys/bus/pci/devices";
  DIR* dir = opendir(pci_dir.c_str());
  if (dir == nullptr) {
    return tpu_devices;
  }

  // Known Google TPU Device IDs (from util/platforminfo/pci_ids.h)
  const std::vector<std::string> kTpuDeviceIds = {
      // TPU v4 (Pufferfish)
      "0x0056",  // PUFFYLITE_CHIP
      "0x005e",  // PUFFERFISH_CHIP

      // TPU v5e (Viperfish/Viperlite)
      "0x0062",  // VIPERFISH_CHIP
      "0x0063",  // VIPERLITE_CHIP

      // TPU v6e (Ghostlite)
      "0x006e",  // GHOSTLITE_CHIP_PF (Application)
      "0x006f",  // GHOSTLITE_CHIP_VF (Application)
      "0x0070",  // GHOSTLITE_CHIP_MANAGEMENT_PF
      "0x0071",  // GHOSTLITE_CHIP_MANAGEMENT_VF

      // TPU v7/v7x (Ghostfish)
      "0x0075",  // GHOSTFISH_CHIP_APPLICATION_PF
      "0x0076",  // GHOSTFISH_CHIP_APPLICATION_VF
      "0x0077",  // GHOSTFISH_CHIP_MANAGEMENT_PF
      "0x0078",  // GHOSTFISH_CHIP_MANAGEMENT_VF
  };

  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string dev_name = entry->d_name;
    if (dev_name == "." || dev_name == "..") continue;

    std::string dev_path = pci_dir + "/" + dev_name;
    std::string vendor_path = dev_path + "/vendor";
    std::string device_path = dev_path + "/device";
    std::string numa_path = dev_path + "/numa_node";

    // Read vendor
    std::ifstream vendor_file(vendor_path);
    std::string vendor_id;
    if (!(vendor_file >> vendor_id)) continue;

    // Google Vendor ID is 0x1ae0
    if (vendor_id != "0x1ae0") continue;

    // Read device ID
    std::ifstream device_file(device_path);
    std::string device_id;
    if (!(device_file >> device_id)) continue;

    // Check if it is a known TPU device
    bool is_tpu = false;
    for (const auto& id : kTpuDeviceIds) {
      if (device_id == id) {
        is_tpu = true;
        break;
      }
    }
    if (!is_tpu) continue;

    // Read NUMA node
    std::ifstream numa_file(numa_path);
    int node = -1;
    if (numa_file >> node) {
      TpuPciDevice dev;
      dev.bdf = dev_name;
      dev.device_id = device_id;
      dev.numa_node = node;
      tpu_devices.push_back(dev);
    }
  }
  closedir(dir);

  // Sort by BDF to ensure consistent ordering
  std::sort(tpu_devices.begin(), tpu_devices.end(),
            [](const TpuPciDevice& a, const TpuPciDevice& b) {
              return a.bdf < b.bdf;
            });

  return tpu_devices;
}

int GetPjRtDeviceNumaNode(const xla::PjRtDevice* device) {
  if (device == nullptr) return -1;

  int chip_idx = device->local_hardware_id().value();

  auto pci_devices = GetTpuPciDevices();
  if (pci_devices.empty()) {
    return -1;
  }

  // Find unique buses (domain:bus) and their NUMA nodes
  std::vector<std::pair<std::string, int>> unique_chips;
  for (const auto& dev : pci_devices) {
    size_t last_colon = dev.bdf.find_last_of(':');
    if (last_colon == std::string::npos) continue;
    std::string domain_bus = dev.bdf.substr(0, last_colon);

    bool found = false;
    for (const auto& chip : unique_chips) {
      if (chip.first == domain_bus) {
        found = true;
        break;
      }
    }
    if (!found) {
      unique_chips.push_back({domain_bus, dev.numa_node});
    }
  }

  if (unique_chips.empty()) return -1;

  // Sort unique chips by BDF to ensure consistent mapping
  std::sort(unique_chips.begin(), unique_chips.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  if (device->client() == nullptr) return -1;

  int local_device_count = device->client()->addressable_devices().size();
  int num_physical_chips = unique_chips.size();

  if (local_device_count <= 0 || num_physical_chips <= 0) return -1;

  int devices_per_chip = local_device_count / num_physical_chips;
  if (devices_per_chip <= 0) devices_per_chip = 1;

  int physical_chip_idx = chip_idx / devices_per_chip;
  if (physical_chip_idx >= num_physical_chips) {
    physical_chip_idx = num_physical_chips - 1;
  }

  return unique_chips[physical_chip_idx].second;
}

void PrintTpuHardwareTopology() {
  std::cout << "[INFO] Querying TPU NUMA topology via PCI sysfs:" << std::endl;
  auto devices = GetTpuPciDevices();
  for (const auto& dev : devices) {
    std::cout << "  TPU Device " << dev.bdf << " (Device ID: " << dev.device_id
              << ") is attached to NUMA Node " << dev.numa_node << std::endl;
  }
  if (devices.empty()) {
    std::cout << "  No Google TPU PCI devices found in sysfs." << std::endl;
  }
}

}  // namespace tpu_raiden
