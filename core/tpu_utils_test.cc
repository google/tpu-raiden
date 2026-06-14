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

#include <iostream>
#include <vector>

#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "core/tpu_pjrt_manager.h"

namespace tpu_raiden {
namespace {

TEST(TpuUtilsTest, GetTpuPciDevicesTest) {
  const auto& pci_devices = GetTpuPciDevices();
  std::cout << "[TEST] Detected " << pci_devices.size()
            << " TPU PCI devices:" << std::endl;
  for (const auto& dev : pci_devices) {
    std::cout << "  BDF: " << dev.bdf << ", Device ID: " << dev.device_id
              << ", NUMA Node: " << dev.numa_node << std::endl;
  }
  // On a TPU VM, we expect to find at least one TPU PCI device.
  EXPECT_FALSE(pci_devices.empty());
}

TEST(TpuUtilsTest, GetPjRtDeviceNumaNodeTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto manager, TpuPjrtManager::GetDefault());
  auto all_devices = manager->client()->addressable_devices();

  std::cout << "[TEST] Available PJRT devices: " << all_devices.size()
            << std::endl;

  // On ghostfish:4, we expect to see exactly 8 PJRT devices (4 chips * 2
  // devices/chip)
  if (all_devices.size() == 8) {
    std::cout << "[TEST] Confirmed ghostfish:4 topology with 8 devices!"
              << std::endl;
  } else {
    std::cout << "[WARNING] Expected 8 devices for ghostfish:4, but saw "
              << all_devices.size() << " devices." << std::endl;
  }

  int resolved_nodes = 0;
  for (auto* device : all_devices) {
    int numa_node = GetPjRtDeviceNumaNode(device);
    std::cout << "  PJRT Device " << device->DebugString()
              << " (local_hardware_id: " << device->local_hardware_id().value()
              << ") is mapped to NUMA Node: " << numa_node << std::endl;

    // NUMA node should be valid (0 or 1 on Ghostfish)
    EXPECT_GE(numa_node, 0);
    EXPECT_LE(numa_node, 1);
    resolved_nodes++;
  }
  EXPECT_GT(resolved_nodes, 0);
}

}  // namespace
}  // namespace tpu_raiden
