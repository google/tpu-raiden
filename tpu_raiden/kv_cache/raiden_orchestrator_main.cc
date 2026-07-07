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

#include <string>

#include "absl/flags/parse.h"
#include "absl/log/initialize.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "tpu_raiden/kv_cache/raiden_orchestrator.h"

ABSL_FLAG(int, port, 9999, "Port to listen on");
ABSL_FLAG(std::string, bind_ip, "", "IP to bind to");

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();

  int port = absl::GetFlag(FLAGS_port);
  std::string bind_ip = absl::GetFlag(FLAGS_bind_ip);

  LOG(INFO) << "Starting RaidenOrchestrator on port " << port;
  tpu_raiden::kv_cache::RaidenOrchestrator orchestrator(port, bind_ip);

  // Keep alive until terminated
  while (true) {
    pause();
  }

  return 0;
}
