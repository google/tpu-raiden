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

#include "tpu_raiden/core/telemetry.h"

#include <cstdlib>
#include <string>

#include "third_party/opentelemetry/cpp/api/include/opentelemetry/trace/provider.h"

namespace tpu_raiden {
namespace telemetry {

opentelemetry::nostd::shared_ptr<opentelemetry::trace::Tracer> GetTracer() {
  return opentelemetry::trace::Provider::GetTracerProvider()->GetTracer(
      "tpu_raiden");
}

int GetTraceLevel() {
  static int level = []() {
    const char* env_val = std::getenv("RAIDEN_TRACE_LEVEL");
    if (env_val == nullptr) {
      return 1;  // Default to COARSE
    }
    try {
      return std::stoi(env_val);
    } catch (...) {
      return 1;
    }
  }();
  return level;
}

}  // namespace telemetry
}  // namespace tpu_raiden
