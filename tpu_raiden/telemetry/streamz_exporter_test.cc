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

#include "tpu_raiden/telemetry/streamz_exporter.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "tpu_raiden/telemetry/telemetry_exporter.h"

namespace tpu_raiden::telemetry {
namespace {

TEST(StreamzExporterTest, DefaultPathNormalization) {
  StreamzExporter exporter;

  EXPECT_TRUE(exporter.RecordCounter("my_counter", 15.0).ok());
  EXPECT_EQ(exporter.GetCellValue("my_counter"), 15.0);
  EXPECT_EQ(exporter.GetCellValue("/tpu_raiden/my_counter"), 15.0);

  auto snapshots = exporter.GetSnapshot();
  ASSERT_EQ(snapshots.size(), 1);
  EXPECT_EQ(snapshots[0].metric_path, "/tpu_raiden/my_counter");
  EXPECT_EQ(snapshots[0].value, 15.0);
  EXPECT_EQ(snapshots[0].sample_count, 1);
}

TEST(StreamzExporterTest, CustomRootPrefixAndAbsolutePaths) {
  StreamzExporterOptions options;
  options.root_prefix = "/custom/prefix";
  StreamzExporter exporter(options);

  EXPECT_TRUE(exporter.RecordGauge("relative_gauge", 99.0).ok());
  EXPECT_TRUE(exporter.RecordGauge("/already/absolute/gauge", 100.0).ok());

  EXPECT_EQ(exporter.GetCellValue("relative_gauge"), 99.0);
  EXPECT_EQ(exporter.GetCellValue("/already/absolute/gauge"), 100.0);

  auto snapshots = exporter.GetSnapshot();
  EXPECT_EQ(snapshots.size(), 2);
}

TEST(StreamzExporterTest, StreamzSummaryGeneration) {
  StreamzExporter exporter;

  EXPECT_TRUE(exporter.RecordCounter("events", 5.0, {{"status", "ok"}}).ok());
  EXPECT_TRUE(exporter.RecordCounter("events", 2.0, {{"status", "ok"}}).ok());

  std::string summary = exporter.ExportStreamzSummary();
  EXPECT_TRUE(absl::StrContains(summary, "Streamz Telemetry Summary:"));
  EXPECT_TRUE(absl::StrContains(summary, "/tpu_raiden/events [counter] Value: 7 Samples: 2 {status=\"ok\"}"));
}

TEST(StreamzExporterTest, ClearAndEnablement) {
  StreamzExporter exporter;
  EXPECT_TRUE(exporter.RecordCounter("cnt", 1.0).ok());
  EXPECT_EQ(exporter.GetRecordCount(), 1);

  exporter.Clear();
  EXPECT_EQ(exporter.GetRecordCount(), 0);

  exporter.SetEnabled(false);
  EXPECT_TRUE(exporter.RecordCounter("cnt", 1.0).ok());
  EXPECT_EQ(exporter.GetRecordCount(), 0);
}

}  // namespace
}  // namespace tpu_raiden::telemetry
