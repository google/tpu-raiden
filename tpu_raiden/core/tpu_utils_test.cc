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

#include "tpu_raiden/core/tpu_utils.h"

#include <vector>

#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tpu_raiden/core/tpu_pjrt_manager.h"

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

}  // namespace
}  // namespace tpu_raiden
