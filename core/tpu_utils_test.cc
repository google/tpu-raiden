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

#include <ifaddrs.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <fstream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "core/tpu_pjrt_manager.h"

namespace tpu_raiden {
namespace {

TEST(TpuUtilsTest, GetTpuPciDevicesTest) {
  const auto& pci_devices = GetTpuPciDevices();
  LOG(INFO) << "Detected " << pci_devices.size() << " TPU PCI devices:";
  for (const auto& dev : pci_devices) {
    LOG(INFO) << "  BDF: " << dev.bdf << ", Device ID: " << dev.device_id
              << ", NUMA Node: " << dev.numa_node;
  }
  // On a TPU VM, we expect to find at least one TPU PCI device.
  EXPECT_FALSE(pci_devices.empty());
}

TEST(TpuUtilsTest, GetPjRtDeviceNumaNodeTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto manager, TpuPjrtManager::GetDefault());
  auto all_devices = manager->client()->addressable_devices();

  LOG(INFO) << "Available PJRT devices: " << all_devices.size();

  // On ghostfish:4, we expect to see exactly 8 PJRT devices (4 chips * 2
  // devices/chip)
  if (all_devices.size() == 8) {
    LOG(INFO) << "Confirmed ghostfish:4 topology with 8 devices!";
  } else {
    LOG(WARNING) << "Expected 8 devices for ghostfish:4, but saw "
                 << all_devices.size() << " devices.";
  }

  int resolved_nodes = 0;
  for (auto* device : all_devices) {
    int numa_node = GetPjRtDeviceNumaNode(device);
    LOG(INFO) << "  PJRT Device " << device->DebugString()
              << " (local_hardware_id: " << device->local_hardware_id().value()
              << ") is mapped to NUMA Node: " << numa_node;

    // NUMA node should be valid (0 or 1 on Ghostfish)
    EXPECT_GE(numa_node, 0);
    EXPECT_LE(numa_node, 1);
    resolved_nodes++;
  }
  EXPECT_GT(resolved_nodes, 0);
}

TEST(TpuUtilsTest, PinCurrentThreadToNumaNodeTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto manager, TpuPjrtManager::GetDefault());
  auto all_devices = manager->client()->addressable_devices();
  ASSERT_FALSE(all_devices.empty());

  // Take the first local device and get its NUMA node
  auto* device = all_devices[0];
  int numa_node = GetPjRtDeviceNumaNode(device);
  LOG(INFO) << "Attempting to pin current thread to NUMA Node of device "
            << device->DebugString() << ": " << numa_node;

  if (numa_node >= 0) {
    int rc = PinCurrentThreadToNumaNode(numa_node);
    // Thread pinning and memory binding can fail in sandboxed environments
    // (like Forge). We tolerate EPERM (-1 from set_mempolicy) and EINVAL (-22
    // from pthread_setaffinity_np).
    EXPECT_TRUE(rc == 0 || rc == -1 || rc == -22)
        << "Unexpected failure pinning to NUMA node " << numa_node
        << ", rc=" << rc;
    if (rc == 0) {
      LOG(INFO) << "Successfully pinned thread to NUMA Node " << numa_node;
    }
  } else {
    LOG(WARNING)
        << "Could not resolve NUMA node for device, skipping pinning test.";
  }
}

void CreateDir(const std::string& path) { mkdir(path.c_str(), 0755); }

void WriteFile(const std::string& path, const std::string& content) {
  std::ofstream f(path);
  f << content;
}

TEST(TpuUtilsTest, DiscoverNumaNicIPsMocked) {
  std::string temp_dir = testing::TempDir();
  std::string mock_sysfs = temp_dir + "/mock_sysfs_" + std::to_string(getpid());

  std::string net_dir = mock_sysfs + "/sys/class/net";
  CreateDir(mock_sysfs);
  CreateDir(mock_sysfs + "/sys");
  CreateDir(mock_sysfs + "/sys/class");
  CreateDir(net_dir);

  struct ifaddrs* ifaddr = nullptr;
  std::string real_iface = "";
  std::string real_ip = "";
  if (getifaddrs(&ifaddr) != -1) {
    for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
      if (ifa->ifa_addr == nullptr) continue;
      if (ifa->ifa_addr->sa_family == AF_INET) {
        std::string name(ifa->ifa_name);
        if (name != "lo" && name.rfind("veth", 0) != 0 &&
            name.rfind("docker", 0) != 0) {
          char host[NI_MAXHOST];
          if (getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in), host,
                          NI_MAXHOST, nullptr, 0, NI_NUMERICHOST) == 0) {
            real_iface = name;
            real_ip = host;
            break;
          }
        }
      }
    }
    freeifaddrs(ifaddr);
  }

  if (real_iface.empty()) {
    LOG(WARNING) << "No suitable real network interface found for testing. "
                    "Skipping test.";
    return;
  }

  LOG(INFO) << "Using real interface " << real_iface << " (IP: " << real_ip
            << ") for mocked test.";

  std::string iface_dir = net_dir + "/" + real_iface;
  CreateDir(iface_dir);

  std::string device_target = "../../../devices/pci0000:00/0000:00:03.0";
  std::string device_link = iface_dir + "/device";
  symlink(device_target.c_str(), device_link.c_str());

  CreateDir(mock_sysfs + "/sys/devices");
  CreateDir(mock_sysfs + "/sys/devices/pci0000:00");
  CreateDir(mock_sysfs + "/sys/devices/pci0000:00/0000:00:03.0");
  WriteFile(mock_sysfs + "/sys/devices/pci0000:00/0000:00:03.0/numa_node",
            "1\n");

  auto ip_to_numa = DiscoverNumaNicIPs(mock_sysfs);

  EXPECT_FALSE(ip_to_numa.empty());
  auto it = ip_to_numa.find(real_ip);
  ASSERT_NE(it, ip_to_numa.end());
  EXPECT_EQ(it->second, 1);

  unlink(device_link.c_str());
}

}  // namespace
}  // namespace tpu_raiden
