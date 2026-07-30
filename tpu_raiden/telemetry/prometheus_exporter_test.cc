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

  store.IncrementCounter(kMetricSentBytesTotal, 1024, {{"interface", "ICI"}});
  store.SetGauge(kMetricActiveTransfers, 5, {{"direction", "OUTBOUND"}});
  store.ObserveHistogram(kMetricTransferDurationSeconds, 0.0055,
                         {{"status", "OK"}});

  std::string output = store.GetPrometheusTextSnapshot();

  EXPECT_TRUE(
      absl::StrContains(output, "# TYPE tpu_raiden_sent_bytes_total counter"));
  EXPECT_TRUE(absl::StrContains(
      output, "tpu_raiden_sent_bytes_total{interface=\"ICI\"} 1024"));
  EXPECT_TRUE(
      absl::StrContains(output, "# TYPE tpu_raiden_active_transfers gauge"));
  EXPECT_TRUE(absl::StrContains(
      output, "tpu_raiden_active_transfers{direction=\"OUTBOUND\"} 5"));
  EXPECT_TRUE(absl::StrContains(
      output, "# TYPE tpu_raiden_transfer_duration_seconds histogram"));
  EXPECT_TRUE(absl::StrContains(
      output,
      "tpu_raiden_transfer_duration_seconds_sum{status=\"OK\"} 0.0055"));
  EXPECT_TRUE(absl::StrContains(
      output, "tpu_raiden_transfer_duration_seconds_count{status=\"OK\"} 1"));
}

TEST(PrometheusExporterTest, ResetAndGlobalExporter) {
  auto global_exporter = GetGlobalPrometheusExporter();
  global_exporter->Reset();
  global_exporter->IncrementCounter(kMetricReceivedBytesTotal, 3);

  std::string text = global_exporter->GetPrometheusTextSnapshot();
  EXPECT_THAT(text, HasSubstr("tpu_raiden_received_bytes_total 3"));

  global_exporter->Reset();
  EXPECT_EQ(global_exporter->GetPrometheusTextSnapshot(), "");
}

}  // namespace
}  // namespace tpu_raiden::telemetry
