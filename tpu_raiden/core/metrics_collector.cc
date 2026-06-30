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

// Copyright 2026 Google LLC
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

#include "tpu_raiden/core/metrics_collector.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(c++17-feature-std-filesystem)
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace tpu_raiden {

namespace {
int64_t ToUs(absl::Duration d) { return absl::ToInt64Microseconds(d); }

NicBytes ReadNicStats(const std::string& sysfs_dir, const std::string& iface) {
  NicBytes stats;
  std::string rx_path =
      absl::StrCat(sysfs_dir, "/", iface, "/statistics/rx_bytes");
  std::string tx_path =
      absl::StrCat(sysfs_dir, "/", iface, "/statistics/tx_bytes");

  std::ifstream rx_file(rx_path);
  if (rx_file.is_open()) rx_file >> stats.rx_bytes;

  std::ifstream tx_file(tx_path);
  if (tx_file.is_open()) tx_file >> stats.tx_bytes;

  return stats;
}
}  // namespace

MetricsCollector::MetricsCollector(std::string sysfs_dir)
    : sysfs_dir_(std::move(sysfs_dir)) {
  for (size_t i = 0; i < kMaxMetrics; ++i) {
    metrics_valid_[i].store(false, std::memory_order_relaxed);
  }
}

MetricsCollector::~MetricsCollector() { WriteJsonReport(); }

void MetricsCollector::RecordStart(uint64_t uuid, const std::string& req_id,
                                   int64_t num_blocks, int64_t total_bytes) {
  size_t idx = uuid % kMaxMetrics;
  TransferMetrics& m = metrics_array_[idx];
  m.uuid = uuid;
  m.req_id = req_id;
  m.num_blocks = num_blocks;
  m.total_bytes = total_bytes;
  m.start_time = absl::Now();
  m.initial_nic_bytes = SnapshotAllNics();
  metrics_valid_[idx].store(true, std::memory_order_release);
}

void MetricsCollector::RecordFirstPacket(uint64_t uuid) {
  size_t idx = uuid % kMaxMetrics;
  if (!metrics_valid_[idx].load(std::memory_order_acquire)) return;
  metrics_array_[idx].first_packet_time = absl::Now();
}

void MetricsCollector::RecordLastPacket(uint64_t uuid) {
  size_t idx = uuid % kMaxMetrics;
  if (!metrics_valid_[idx].load(std::memory_order_acquire)) return;
  metrics_array_[idx].last_packet_time = absl::Now();
}

void MetricsCollector::RecordH2dEnqueue(uint64_t uuid) {
  size_t idx = uuid % kMaxMetrics;
  if (!metrics_valid_[idx].load(std::memory_order_acquire)) return;
  metrics_array_[idx].h2d_enqueue_time = absl::Now();
}

void MetricsCollector::RecordH2dComplete(uint64_t uuid) {
  size_t idx = uuid % kMaxMetrics;
  if (!metrics_valid_[idx].load(std::memory_order_acquire)) return;
  metrics_array_[idx].h2d_complete_time = absl::Now();
}

void MetricsCollector::RecordEnd(uint64_t uuid) {
  size_t idx = uuid % kMaxMetrics;
  if (!metrics_valid_[idx].load(std::memory_order_acquire)) return;
  metrics_array_[idx].end_time = absl::Now();
  metrics_array_[idx].final_nic_bytes = SnapshotAllNics();
}

absl::flat_hash_map<std::string, NicBytes> MetricsCollector::SnapshotAllNics() {
  absl::flat_hash_map<std::string, NicBytes> snapshot;
  if (!std::filesystem::exists(sysfs_dir_)) {
    return snapshot;
  }
  for (const auto& entry : std::filesystem::directory_iterator(sysfs_dir_)) {
    std::string iface = entry.path().filename().string();
    // Physical NICs: eth* (classic) or predictable names ens*/enp*/eno* (the
    // GCP TPU VMs name the primary NIC ens5, which the old eth-only filter
    // missed -> empty nic_wire_bandwidth_gbps). Skip loopback / docker /
    // virtual.
    if (absl::StartsWith(iface, "eth") || absl::StartsWith(iface, "ens") ||
        absl::StartsWith(iface, "enp") || absl::StartsWith(iface, "eno")) {
      snapshot[iface] = ReadNicStats(sysfs_dir_, iface);
    }
  }
  return snapshot;
}

void MetricsCollector::WriteJsonReport(const std::string& filepath) {
  std::string content = DumpMetricsToString();
  if (content.empty()) return;

  std::ofstream f(filepath);
  if (f.is_open()) {
    f << content;
  }
}

std::string MetricsCollector::DumpMetricsToString() {
  std::stringstream f;
  f << "[\n";
  bool first = true;
  for (size_t i = 0; i < kMaxMetrics; ++i) {
    if (!metrics_valid_[i].load(std::memory_order_acquire)) continue;
    const auto& m = metrics_array_[i];
    if (!first) f << ",\n";
    first = false;

    int64_t connection_setup_us = ToUs(m.first_packet_time - m.start_time);
    int64_t network_transit_us = ToUs(m.last_packet_time - m.first_packet_time);
    int64_t staging_queue_us = ToUs(m.h2d_enqueue_time - m.last_packet_time);
    int64_t h2d_copy_us = ToUs(m.h2d_complete_time - m.h2d_enqueue_time);
    int64_t e2e_duration_us = ToUs(m.end_time - m.start_time);

    double active_duration_s = e2e_duration_us / 1e6;

    f << absl::StrFormat(
        R"(  {
    "uuid": %u,
    "req_id": "%s",
    "num_blocks": %d,
    "total_bytes": %d,
    "connection_setup_us": %d,
    "network_transit_us": %d,
    "staging_queue_us": %d,
    "h2d_copy_us": %d,
    "e2e_duration_us": %d,
    "nic_wire_bandwidth_gbps": {
)",
        m.uuid, m.req_id, m.num_blocks, m.total_bytes, connection_setup_us,
        network_transit_us, staging_queue_us, h2d_copy_us, e2e_duration_us);

    bool first_nic = true;
    for (const auto& [iface, init_stats] : m.initial_nic_bytes) {
      auto final_stats_it = m.final_nic_bytes.find(iface);
      if (final_stats_it == m.final_nic_bytes.end()) continue;

      if (!first_nic) f << ",\n";
      first_nic = false;

      uint64_t rx_delta = final_stats_it->second.rx_bytes - init_stats.rx_bytes;
      double rx_gbps = active_duration_s > 0
                           ? (rx_delta * 8) / (active_duration_s * 1e9)
                           : 0.0;

      f << absl::StrFormat(R"(      "%s_rx": %.3f)", iface, rx_gbps);
    }
    f << "\n    }\n  }";
  }
  f << "\n]\n";
  return f.str();
}

}  // namespace tpu_raiden
