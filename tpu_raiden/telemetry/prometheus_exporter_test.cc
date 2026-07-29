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

#include "tpu_raiden/telemetry/prometheus_exporter.h"

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "tpu_raiden/telemetry/metrics_api.h"

namespace tpu_raiden::telemetry {
namespace {

using ::testing::HasSubstr;

TEST(PrometheusExporterTest, RecordAndExportFormat) {
  auto exporter = std::make_shared<PrometheusExporter>();
  RaidenMetricStore store;
  store.AddBackend(exporter);

  store.RecordCounter(kMetricKVCacheTransferBytes, 1024.0, {{"model", "llama"}});
  store.RecordGauge(kMetricKVCacheHitRatio, 0.95, {{"model", "llama"}});
  store.RecordHistogram(kMetricKVCacheTransferLatencyMs, 5.5, {{"model", "llama"}});

  std::string output = store.ExportPrometheusText();

  EXPECT_TRUE(absl::StrContains(output, "# TYPE tpu_raiden_kv_cache_transfer_bytes counter"));
  EXPECT_TRUE(absl::StrContains(output, "tpu_raiden_kv_cache_transfer_bytes{model=\"llama\"} 1024"));
  EXPECT_TRUE(absl::StrContains(output, "# TYPE tpu_raiden_kv_cache_hit_ratio gauge"));
  EXPECT_TRUE(absl::StrContains(output, "tpu_raiden_kv_cache_hit_ratio{model=\"llama\"} 0.95"));
  EXPECT_TRUE(absl::StrContains(output, "# TYPE tpu_raiden_kv_cache_transfer_latency_ms histogram"));
  EXPECT_TRUE(absl::StrContains(output, "tpu_raiden_kv_cache_transfer_latency_ms_sum{model=\"llama\"} 5.5"));
  EXPECT_TRUE(absl::StrContains(output, "tpu_raiden_kv_cache_transfer_latency_ms_count{model=\"llama\"} 1"));
}

TEST(PrometheusExporterTest, ResetAndGlobalExporter) {
  auto global_exporter = GetGlobalPrometheusExporter();
  global_exporter->Reset();
  global_exporter->RecordCounter(kMetricRPCRequestCount, 3.0);

  std::string text = global_exporter->GetPrometheusText();
  EXPECT_THAT(text, HasSubstr("tpu_raiden_rpc_request_count 3"));

  global_exporter->Reset();
  EXPECT_EQ(global_exporter->GetPrometheusText(), "");
}

}  // namespace
}  // namespace tpu_raiden::telemetry
