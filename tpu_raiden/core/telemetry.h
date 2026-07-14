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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_TELEMETRY_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_TELEMETRY_H_

#include "third_party/opentelemetry/cpp/api/include/opentelemetry/trace/tracer.h"

namespace tpu_raiden {
namespace telemetry {

opentelemetry::nostd::shared_ptr<opentelemetry::trace::Tracer> GetTracer();

// Returns the trace level controlled by RAIDEN_TRACE_LEVEL environment
// variable. 0: DISABLED 1: COARSE (Default) 2: DETAILED 3: DEBUG
int GetTraceLevel();

}  // namespace telemetry
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_TELEMETRY_H_
