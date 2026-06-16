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

#include <arpa/inet.h>
#include <dirent.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sched.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/logging.h"

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
    LOG(ERROR) << "SetThreadMempolicy(mode=" << mode << ", node=" << node
               << ") failed: " << std::strerror(errno) << " (errno=" << errno
               << ")";
  }
  return res;
#else
  return -1;  // Syscall not available
#endif
}

std::vector<int> GetNumaNodeCpuCores(int numa_node) {
  std::vector<int> cores;
  std::string path =
      "/sys/devices/system/node/node" + std::to_string(numa_node) + "/cpulist";
  std::ifstream file(path);
  if (!file.is_open()) {
    LOG(ERROR) << "Failed to open " << path;
    return cores;
  }
  std::string line;
  if (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string part;
    while (std::getline(ss, part, ',')) {
      size_t dash = part.find('-');
      if (dash == std::string::npos) {
        try {
          cores.push_back(std::stoi(part));
        } catch (...) {
        }
      } else {
        try {
          int start = std::stoi(part.substr(0, dash));
          int end = std::stoi(part.substr(dash + 1));
          for (int c = start; c <= end; ++c) {
            cores.push_back(c);
          }
        } catch (...) {
        }
      }
    }
  }
  return cores;
}

int PinCurrentThreadToCores(const std::vector<int>& cores) {
  if (cores.empty()) return -1;
#ifdef __linux__
  cpu_set_t allowed_set;
  CPU_ZERO(&allowed_set);
  if (sched_getaffinity(0, sizeof(cpu_set_t), &allowed_set) != 0) {
    LOG(ERROR) << "sched_getaffinity failed: " << std::strerror(errno);
    // Fallback: assume all cores are allowed if we can't query
    for (int i = 0; i < CPU_SETSIZE; ++i) {
      CPU_SET(i, &allowed_set);
    }
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  bool has_cores = false;
  for (int core : cores) {
    if (core >= 0 && core < CPU_SETSIZE && CPU_ISSET(core, &allowed_set)) {
      CPU_SET(core, &cpuset);
      has_cores = true;
    }
  }
  if (!has_cores) {
    LOG(WARNING) << "No allowed cores found for pinning on this NUMA node";
    return -1;
  }
  int res = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  if (res != 0) {
    LOG(ERROR) << "pthread_setaffinity_np failed: " << std::strerror(res)
               << " (rc=" << res << ")";
    return -res;
  }
  return 0;
#else
  return -1;
#endif
}

int PinCurrentThreadToNumaNode(int node) {
  if (node < 0) return -1;
  std::vector<int> cores = GetNumaNodeCpuCores(node);
  if (cores.empty()) {
    LOG(WARNING) << "No CPU cores found for NUMA node " << node;
    return -2;
  }
  int rc = PinCurrentThreadToCores(cores);
  if (rc != 0) return rc;
  constexpr int kMpolBind = 2;
  int64_t mem_rc = SetThreadMempolicy(kMpolBind, node);
  if (mem_rc < 0) return static_cast<int>(mem_rc);
  return 0;
}

const std::vector<TpuPciDevice>& GetTpuPciDevices() {
  static const std::vector<TpuPciDevice>* cached_devices = []() {
    auto* devices = new std::vector<TpuPciDevice>();
    std::string pci_dir = "/sys/bus/pci/devices";
    DIR* dir = opendir(pci_dir.c_str());
    if (dir == nullptr) {
      return devices;
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
        devices->push_back(dev);
      }
    }
    closedir(dir);

    // Sort by BDF to ensure consistent ordering
    std::sort(devices->begin(), devices->end(),
              [](const TpuPciDevice& a, const TpuPciDevice& b) {
                return a.bdf < b.bdf;
              });

    return devices;
  }();

  return *cached_devices;
}

int GetPjRtDeviceNumaNode(const xla::PjRtDevice* device) {
  if (device == nullptr) return -1;

  int chip_idx = device->local_hardware_id().value();

  const auto& pci_devices = GetTpuPciDevices();
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
  LOG(INFO) << "Querying TPU NUMA topology via PCI sysfs:";
  const auto& devices = GetTpuPciDevices();
  for (const auto& dev : devices) {
    LOG(INFO) << "  TPU Device " << dev.bdf << " (Device ID: " << dev.device_id
              << ") is attached to NUMA Node " << dev.numa_node;
  }
  if (devices.empty()) {
    LOG(INFO) << "  No Google TPU PCI devices found in sysfs.";
  }
}

bool IsPhysicalInterface(const std::string& iface,
                         const std::string& sysfs_root) {
  std::string device_path = sysfs_root + "/sys/class/net/" + iface + "/device";
  struct stat st;
  if (lstat(device_path.c_str(), &st) != 0) {
    return false;
  }
  return S_ISLNK(st.st_mode);
}

std::string GetInterfaceBdf(const std::string& iface,
                            const std::string& sysfs_root) {
  std::string device_path = sysfs_root + "/sys/class/net/" + iface + "/device";
  char buf[PATH_MAX];
  ssize_t len = readlink(device_path.c_str(), buf, sizeof(buf) - 1);
  if (len != -1) {
    buf[len] = '\0';
    std::string target(buf);
    size_t last_slash = target.find_last_of('/');
    if (last_slash != std::string::npos) {
      return target.substr(last_slash + 1);
    }
  }
  return "";
}

int GetInterfaceNumaNode(const std::string& iface,
                         const std::string& sysfs_root) {
  std::string numa_path =
      sysfs_root + "/sys/class/net/" + iface + "/device/numa_node";
  std::ifstream file(numa_path);
  int node = -1;
  if (file >> node) {
    return node;
  }
  return -1;
}

absl::flat_hash_map<std::string, int> DiscoverNumaNicIPs(
    const std::string& sysfs_root) {
  absl::flat_hash_map<std::string, int> ip_to_numa;

  struct PhysIfaceInfo {
    std::string name;
    std::string bdf;
    int detected_numa = -1;
    int assigned_numa = -1;
  };
  std::vector<PhysIfaceInfo> phys_ifaces;

  std::string net_dir = sysfs_root + "/sys/class/net";
  DIR* dir = opendir(net_dir.c_str());
  if (dir == nullptr) {
    LOG(ERROR) << "Failed to open " << net_dir;
    return ip_to_numa;
  }
  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string name = entry->d_name;
    if (name == "." || name == "..") continue;
    if (name == "lo") continue;
    if (name.rfind("veth", 0) == 0 || name.rfind("docker", 0) == 0 ||
        name.rfind("br-", 0) == 0)
      continue;

    if (IsPhysicalInterface(name, sysfs_root)) {
      PhysIfaceInfo info;
      info.name = name;
      info.bdf = GetInterfaceBdf(name, sysfs_root);
      info.detected_numa = GetInterfaceNumaNode(name, sysfs_root);
      phys_ifaces.push_back(info);
    }
  }
  closedir(dir);

  if (phys_ifaces.empty()) {
    LOG(WARNING) << "No physical network interfaces found.";
    return ip_to_numa;
  }

  std::sort(phys_ifaces.begin(), phys_ifaces.end(),
            [](const PhysIfaceInfo& a, const PhysIfaceInfo& b) {
              return a.bdf < b.bdf;
            });

  std::vector<int> preferred_numa_nodes;
  const char* env_val = std::getenv("WORKLOAD_NIC_PREFERRED_NUMA");
  if (env_val != nullptr && std::strlen(env_val) > 0) {
    std::vector<std::string> parts = absl::StrSplit(env_val, ',');
    for (const auto& part : parts) {
      int node;
      if (absl::SimpleAtoi(part, &node)) {
        preferred_numa_nodes.push_back(node);
      }
    }
  }

  for (size_t i = 0; i < phys_ifaces.size(); ++i) {
    if (i < preferred_numa_nodes.size()) {
      phys_ifaces[i].assigned_numa = preferred_numa_nodes[i];
    } else if (phys_ifaces[i].detected_numa != -1) {
      phys_ifaces[i].assigned_numa = phys_ifaces[i].detected_numa;
    } else {
      phys_ifaces[i].assigned_numa = (i % 2 == 0) ? 0 : 1;
    }
    LOG(INFO) << "Interface " << phys_ifaces[i].name
              << " (BDF: " << phys_ifaces[i].bdf
              << ", detected NUMA: " << phys_ifaces[i].detected_numa
              << ") assigned to NUMA " << phys_ifaces[i].assigned_numa;
  }

  struct ifaddrs* ifaddr = nullptr;
  if (getifaddrs(&ifaddr) == -1) {
    LOG(ERROR) << "getifaddrs failed: " << std::strerror(errno);
    return ip_to_numa;
  }

  for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) continue;
    if (ifa->ifa_addr->sa_family == AF_INET) {
      std::string name(ifa->ifa_name);
      auto it = std::find_if(
          phys_ifaces.begin(), phys_ifaces.end(),
          [&name](const PhysIfaceInfo& info) { return info.name == name; });
      if (it != phys_ifaces.end()) {
        char host[NI_MAXHOST];
        int s = getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in), host,
                            NI_MAXHOST, nullptr, 0, NI_NUMERICHOST);
        if (s == 0) {
          ip_to_numa[host] = it->assigned_numa;
          LOG(INFO) << "Mapped IP " << host << " (interface " << name
                    << ") to NUMA " << it->assigned_numa;
        }
      }
    }
  }
  freeifaddrs(ifaddr);

  return ip_to_numa;
}

std::string GetLocalIp() {
  struct ifaddrs* ifaddr = nullptr;
  if (getifaddrs(&ifaddr) == -1) {
    return "127.0.0.1";
  }

  std::string local_ip = "127.0.0.1";
  for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) continue;

    int family = ifa->ifa_addr->sa_family;
    // Skip loopback
    if (std::string(ifa->ifa_name) == "lo") continue;

    if (family == AF_INET) {
      char host[NI_MAXHOST];
      int s = getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in), host,
                          NI_MAXHOST, nullptr, 0, NI_NUMERICHOST);
      if (s == 0) {
        local_ip = host;
        break;  // Pick first non-loopback IPv4
      }
    }
  }

  freeifaddrs(ifaddr);
  return local_ip;
}

absl::Status SplitHostPort(const std::string& address, std::string& host,
                           int& port) {
  if (address.empty()) {
    return absl::InvalidArgumentError("Address is empty");
  }
  if (address.front() == '[') {
    size_t closing_bracket = address.find(']');
    if (closing_bracket == std::string::npos ||
        closing_bracket + 1 >= address.size() ||
        address[closing_bracket + 1] != ':') {
      return absl::InvalidArgumentError("Invalid IPv6 address format");
    }
    host = address.substr(1, closing_bracket - 1);
    std::string port_str = address.substr(closing_bracket + 2);
    if (!absl::SimpleAtoi(port_str, &port)) {
      return absl::InvalidArgumentError("Invalid port number");
    }
  } else {
    std::vector<std::string> parts = absl::StrSplit(address, ':');
    if (parts.size() != 2) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid address format: ", address));
    }
    host = parts[0];
    if (!absl::SimpleAtoi(parts[1], &port)) {
      return absl::InvalidArgumentError("Invalid port number");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> DetectLocalIpFromPjRtBuffers(
    const std::vector<std::vector<xla::PjRtBuffer*>>& layer_buffers) {
  if (layer_buffers.empty() || layer_buffers[0].empty() ||
      layer_buffers[0][0] == nullptr) {
    return absl::NotFoundError("No PJRT buffers available to detect local IP.");
  }
  auto* device = layer_buffers[0][0]->device();
  if (device == nullptr) {
    return absl::NotFoundError("First PJRT buffer has no associated device.");
  }
  int numa_node = GetPjRtDeviceNumaNode(device);
  if (numa_node < 0) {
    return absl::NotFoundError(absl::StrCat(
        "Could not determine NUMA node for PJRT device: ", device->ToString()));
  }

  auto ip_to_numa = DiscoverNumaNicIPs();
  for (const auto& [ip, node] : ip_to_numa) {
    if (node == numa_node) {
      VLOG(1) << "DetectLocalIpFromPjRtBuffers: Mapped NUMA node " << numa_node
              << " to physical NIC IP " << ip;
      return ip;
    }
  }

  return absl::NotFoundError(
      absl::StrCat("No physical NIC found mapping to NUMA node ", numa_node));
}

}  // namespace tpu_raiden
