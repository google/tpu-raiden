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

#include <grpcpp/grpcpp.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

#include "base/init_google.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/time/time.h"
#include "third_party/grpc/include/grpcpp/security/server_credentials.h"
#include "third_party/grpc/include/grpcpp/server.h"
#include "third_party/grpc/include/grpcpp/server_builder.h"
#include "kv_cache/global_registry/global_registry_server.h"

ABSL_FLAG(int32_t, port, 50051, "Port to listen on");
ABSL_FLAG(absl::Duration, default_ttl, absl::InfiniteDuration(),
          "Default TTL for registrations");
ABSL_FLAG(absl::Duration, cleanup_interval, absl::Seconds(300),
          "Interval for cleanup thread");

void RunServer() {
  std::string server_address =
      "[::]:" + std::to_string(absl::GetFlag(FLAGS_port));

  absl::Duration default_ttl = absl::GetFlag(FLAGS_default_ttl);
  absl::Duration cleanup_interval = absl::GetFlag(FLAGS_cleanup_interval);

  tpu_raiden::kv_cache::global_registry::GlobalRegistryServiceImpl service(
      default_ttl, cleanup_interval);

  grpc::ServerBuilder builder;
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(&service);
  // Finally assemble the server.
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  LOG(INFO) << "Server listening on " << server_address;

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
}

int main(int argc, char** argv) {
  InitGoogle(argv[0], &argc, &argv, true);
  RunServer();
  return 0;
}
