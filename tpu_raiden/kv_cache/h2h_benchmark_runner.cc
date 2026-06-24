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

//
// GKE Launch Instructions:
// 1. Cluster Setup: Ensure a GKE cluster with dual-NIC/TPU capabilities is
// available.
// 2. Build: Build the h2h_benchmark_runner binary.
// 3. Execution: Run the binary on two separate pods (sender and receiver).
//    - Receiver pod: run with --role=receiver and its local control IP.
//    - Sender pod: run with --role=sender, its local control IP, and pass
//      the receiver's IP to establish the connection.
//
// Example:
//  ${TARGET_ROOT}/bazel-bin/tpu_raiden/kv_cache/h2h_benchmark_runner \
//      --role=sender \
//      --peer_control_ip="$RECEIVER_DNS" \
//      --peer_control_port=9999 \
//      --data_interface="${DATA_INTERFACE-eth1}" \
//      --block_size=$BS_BYTES \
//      --parallelism=$P

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tpu_raiden/core/tpu_utils.h"
#include "tpu_raiden/kv_cache/kv_cache_manager_base.h"
#include "tpu_raiden/kv_cache/logical_block_manager.h"

ABSL_FLAG(std::string, role, "", "Role: sender or receiver");
ABSL_FLAG(std::string, control_ip, "127.0.0.1", "Local control IP");
ABSL_FLAG(std::string, control_interface, "",
          "Local control interface (e.g. eth0)");
ABSL_FLAG(std::string, data_ip, "127.0.0.1", "Local data IP");
ABSL_FLAG(std::string, data_interface, "",
          "Local data interfaces (e.g. eth1,eth2). "
          "Leave empty to auto-discover.");
ABSL_FLAG(std::string, peer_control_ip, "127.0.0.1", "Peer control IP");
ABSL_FLAG(int32_t, peer_control_port, 9999, "Peer control port");
ABSL_FLAG(int64_t, block_size, 1024 * 1024, "Block size in bytes");
ABSL_FLAG(int32_t, parallelism, 1, "Parallelism");
ABSL_FLAG(int32_t, numa_node, -1, "NUMA node to pin to");

namespace {

constexpr size_t kNumLayers = 32;
constexpr size_t kNumShards = 1;
constexpr int kNumBlocks = 8;

std::string get_ip_of_interface(const std::string& interface_name) {
  struct ifaddrs* ifaddr;
  if (getifaddrs(&ifaddr) == -1) {
    perror("getifaddrs");
    return "";
  }
  std::string ip = "";
  for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) continue;
    if (ifa->ifa_name == interface_name) {
      int family = ifa->ifa_addr->sa_family;
      if (family == AF_INET) {
        char host[NI_MAXHOST];
        int s = getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in), host,
                            NI_MAXHOST, nullptr, 0, NI_NUMERICHOST);
        if (s == 0) {
          ip = host;
          break;
        }
      } else if (family == AF_INET6) {
        char host[NI_MAXHOST];
        int s = getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in6), host,
                            NI_MAXHOST, nullptr, 0, NI_NUMERICHOST);
        if (s == 0) {
          ip = host;
          break;
        }
      }
    }
  }
  freeifaddrs(ifaddr);
  return ip;
}

// Dynamically discover all active data interfaces (excluding lo and eth0 unless
// on Borg)
std::vector<std::pair<std::string, std::string>> discover_data_interfaces() {
  std::vector<std::pair<std::string, std::string>> interfaces;
  struct ifaddrs* ifaddr;
  if (getifaddrs(&ifaddr) == -1) {
    perror("getifaddrs");
    return interfaces;
  }
  for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) continue;
    std::string name = ifa->ifa_name;

    // Filter out loopback
    if (name == "lo") continue;

    // Filter out virtual, non-routable interfaces to prevent cross-node
    // benchmark failures
    if (name == "docker0" || name == "nodelocaldns" ||
        absl::StartsWith(name, "veth") || absl::StartsWith(name, "lxc") ||
        absl::StartsWith(name, "cilium")) {
      continue;
    }

    int family = ifa->ifa_addr->sa_family;
    if (family == AF_INET) {
      char host[NI_MAXHOST];
      int s = getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in), host,
                          NI_MAXHOST, nullptr, 0, NI_NUMERICHOST);
      if (s == 0) {
        interfaces.push_back({name, host});
      }
    } else if (family == AF_INET6) {
      char host[NI_MAXHOST];
      int s = getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in6), host,
                          NI_MAXHOST, nullptr, 0, NI_NUMERICHOST);
      if (s == 0) {
        interfaces.push_back({name, host});
      }
    }
  }
  freeifaddrs(ifaddr);

  // Deduplicate interfaces
  std::vector<std::pair<std::string, std::string>> unique_interfaces;
  for (const auto& pair : interfaces) {
    bool found = false;
    for (const auto& upair : unique_interfaces) {
      if (upair.first == pair.first) {
        found = true;
        break;
      }
    }
    if (!found) {
      unique_interfaces.push_back(pair);
    }
  }
  return unique_interfaces;
}

// Precise subnet matching check (do the first three octets match? e.g. 10.10.64
// vs 10.10.0)
bool in_same_subnet(const std::string& ip1, const std::string& ip2) {
  size_t dot1 = ip1.find('.');
  if (dot1 == std::string::npos) return false;
  size_t dot2 = ip1.find('.', dot1 + 1);
  if (dot2 == std::string::npos) return false;
  size_t dot3 = ip1.find('.', dot2 + 1);
  if (dot3 == std::string::npos) return false;

  std::string prefix1 = ip1.substr(0, dot3);

  size_t p_dot1 = ip2.find('.');
  if (p_dot1 == std::string::npos) return false;
  size_t p_dot2 = ip2.find('.', p_dot1 + 1);
  if (p_dot2 == std::string::npos) return false;
  size_t p_dot3 = ip2.find('.', p_dot2 + 1);
  if (p_dot3 == std::string::npos) return false;

  std::string prefix2 = ip2.substr(0, p_dot3);

  return prefix1 == prefix2;
}

constexpr int kNumIterations = 50;
constexpr int kNumWarmup = 3;

bool send_all(int sock, const std::string& data) {
  size_t total_sent = 0;
  while (total_sent < data.size()) {
    ssize_t sent =
        send(sock, data.data() + total_sent, data.size() - total_sent, 0);
    if (sent <= 0) {
      perror("send failed");
      return false;
    }
    total_sent += sent;
  }
  return true;
}

std::string read_line(int sock) {
  std::string line;
  char c;
  while (recv(sock, &c, 1, 0) > 0) {
    if (c == '\n') break;
    line.push_back(c);
  }
  return line;
}

int start_control_server(const std::string& ip, int port) {
  bool is_ipv6 = (ip.find(':') != std::string::npos);
  int domain = is_ipv6 ? AF_INET6 : AF_INET;

  int server_fd = socket(domain, SOCK_STREAM, 0);
  if (server_fd < 0) {
    perror("socket failed");
    return -1;
  }
  int opt = 1;
  if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
    perror("setsockopt");
    close(server_fd);
    return -1;
  }

  if (is_ipv6) {
    struct sockaddr_in6 address;
    std::memset(&address, 0, sizeof(address));
    address.sin6_family = AF_INET6;
    if (inet_pton(AF_INET6, ip.c_str(), &address.sin6_addr) <= 0) {
      perror("invalid IPv6 address");
      close(server_fd);
      return -1;
    }
    address.sin6_port = htons(port);
    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
      perror("bind failed");
      close(server_fd);
      return -1;
    }
  } else {
    struct sockaddr_in address;
    std::memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    if (inet_pton(AF_INET, ip.c_str(), &address.sin_addr) <= 0) {
      perror("invalid IPv4 address");
      close(server_fd);
      return -1;
    }
    address.sin_port = htons(port);
    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
      perror("bind failed");
      close(server_fd);
      return -1;
    }
  }

  if (listen(server_fd, 1) < 0) {
    perror("listen");
    close(server_fd);
    return -1;
  }
  return server_fd;
}

int connect_to_control_server(const std::string& host, int port) {
  struct addrinfo hints, *result = nullptr;
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  std::string port_str = std::to_string(port);
  int s = -1;
  int retries = 30;  // 60 seconds total wait

  while (retries > 0) {
    s = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result);
    if (s == 0) {
      for (struct addrinfo* rp = result; rp != nullptr; rp = rp->ai_next) {
        int sock = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sock < 0) continue;

        if (connect(sock, rp->ai_addr, rp->ai_addrlen) == 0) {
          freeaddrinfo(result);
          return sock;
        }
        close(sock);
      }
      freeaddrinfo(result);
      result = nullptr;
    }

    std::cout << "Connection or resolution to control plane failed, retrying "
                 "in 2 seconds..."
              << std::endl;
    absl::SleepFor(absl::Seconds(2));
    retries--;
  }
  if (result) freeaddrinfo(result);
  return -1;
}

}  // namespace

int main(int argc, char* argv[]) {
  // Ignore SIGPIPE to prevent abrupt crashes when writing to broken sockets;
  // allow gRPC/sockets to return errors gracefully instead.
  std::signal(SIGPIPE, SIG_IGN);

  absl::ParseCommandLine(argc, argv);

  std::string role = absl::GetFlag(FLAGS_role);
  std::string control_ip = absl::GetFlag(FLAGS_control_ip);
  std::string control_interface = absl::GetFlag(FLAGS_control_interface);
  std::string data_ip = absl::GetFlag(FLAGS_data_ip);
  std::string data_interface = absl::GetFlag(FLAGS_data_interface);
  std::string peer_control_ip = absl::GetFlag(FLAGS_peer_control_ip);
  int peer_control_port = absl::GetFlag(FLAGS_peer_control_port);
  int64_t block_size = absl::GetFlag(FLAGS_block_size);
  int parallelism = absl::GetFlag(FLAGS_parallelism);
  int numa_node = absl::GetFlag(FLAGS_numa_node);

  if (role != "sender" && role != "receiver") {
    std::cerr << "Error: --role must be either 'sender' or 'receiver'"
              << std::endl;
    return 1;
  }

  if (role == "sender") {
    if (peer_control_ip == "auto" || peer_control_ip.empty()) {
      std::cerr << "Error: --peer_control_ip must be provided for sender."
                << std::endl;
      return 1;
    }
  }

  if (!control_interface.empty()) {
    control_ip = get_ip_of_interface(control_interface);
    if (control_ip.empty()) {
      std::cerr << "Failed to resolve IP for control interface: "
                << control_interface << std::endl;
      return 1;
    }
    std::cout << "Resolved control IP from interface " << control_interface
              << ": " << control_ip << std::endl;
  }

  // Resolve local data interfaces (use specified flag as override, fallback to
  // auto-discovery)
  std::vector<std::pair<std::string, std::string>> local_interfaces;
  if (!data_interface.empty()) {
    std::vector<std::string> interfaces = absl::StrSplit(
        data_interface, absl::ByAnyChar(" ,"), absl::SkipEmpty());
    for (const auto& iface : interfaces) {
      std::string resolved_ip = get_ip_of_interface(iface);
      if (!resolved_ip.empty()) {
        local_interfaces.push_back({iface, resolved_ip});
        std::cout << "Using specified data interface override: " << iface
                  << " (" << resolved_ip << ")" << std::endl;
      } else {
        std::cerr
            << "Error: Failed to resolve IP for specified data interface: "
            << iface << std::endl;
        return 1;
      }
    }
  } else {
    local_interfaces = discover_data_interfaces();
    if (local_interfaces.empty()) {
      std::cout
          << "No data interfaces discovered, falling back to default data IP."
          << std::endl;
      local_interfaces.push_back({"default", data_ip});
    }
  }

  std::cout << "Discovered " << local_interfaces.size()
            << " data interfaces:" << std::endl;
  for (const auto& iface : local_interfaces) {
    std::cout << "  - " << iface.first << ": " << iface.second << std::endl;
  }

  if (numa_node >= 0) {
    std::cout << "Pinning main thread to NUMA node " << numa_node << std::endl;
    tpu_raiden::PinCurrentThreadToNumaNode(numa_node);
  }

  if (role == "receiver") {
    // 1. Instantiate one KVCacheManagerBase per discovered interface
    std::vector<std::unique_ptr<tpu_raiden::kv_cache::KVCacheManagerBase>>
        managers;
    std::vector<std::vector<int>> allocated_block_ids;
    std::string endpoints_str = "";

    for (const auto& iface : local_interfaces) {
      std::cout << "Spawning Receiver Manager on " << iface.first << " ("
                << iface.second << ")..." << std::endl;
      auto manager = std::make_unique<tpu_raiden::kv_cache::KVCacheManagerBase>(
          kNumLayers, kNumShards, block_size,
          /*local_port=*/std::nullopt,
          /*host_blocks_to_allocate=*/kNumBlocks, parallelism,
          /*host_allocator=*/nullptr,
          /*bind_ip=*/iface.second);

      auto dst_block_ids_or =
          manager->block_manager()->Allocate(kNumBlocks, /*lock=*/true);
      if (!dst_block_ids_or.ok()) {
        std::cerr << "Failed to allocate receiver blocks: "
                  << dst_block_ids_or.status() << std::endl;
        return 1;
      }
      allocated_block_ids.push_back(dst_block_ids_or.value());

      // Zero-initialize receiver blocks
      for (size_t l = 0; l < kNumLayers; ++l) {
        uint8_t* receiver_base = manager->GetHostPointer(l, 0);
        for (int b = 0; b < kNumBlocks; ++b) {
          uint8_t* dst_ptr =
              receiver_base + dst_block_ids_or.value()[b] * block_size;
          std::memset(dst_ptr, 0, block_size);
        }
      }

      int port = *manager->local_port();
      if (!endpoints_str.empty()) endpoints_str += ",";
      endpoints_str +=
          absl::StrFormat("%s:%s:%d", iface.first, iface.second, port);
      managers.push_back(std::move(manager));
    }

    // 2. Start control server to handshake with Sender
    int server_fd = start_control_server(control_ip, peer_control_port);
    if (server_fd < 0) {
      std::cerr << "Failed to start control server" << std::endl;
      return 1;
    }

    int client_fd = -1;
    std::vector<int> active_indices;

    while (true) {
      std::cout << "Waiting for sender connection on control plane ("
                << control_ip << ":" << peer_control_port << ")..."
                << std::endl;
      struct sockaddr_storage client_addr;
      socklen_t client_len = sizeof(client_addr);
      client_fd =
          accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
      if (client_fd < 0) {
        perror("accept failed");
        close(server_fd);
        return 1;
      }
      std::cout << "Connection established on control plane. Verifying..."
                << std::endl;

      // Send endpoints to the connected client
      std::string msg = endpoints_str + "\n";
      if (!send_all(client_fd, msg)) {
        std::cerr << "Failed to send endpoints, closing connection."
                  << std::endl;
        close(client_fd);
        continue;
      }

      // Read active interface indices from connected client
      std::string active_indices_msg = read_line(client_fd);
      std::cout << "Received handshake message: " << active_indices_msg
                << std::endl;

      // Verify handshake signature
      std::string signature = "RAIDEN_SENDER_ACTIVE:";
      if (!absl::StartsWith(active_indices_msg, signature)) {
        std::cerr << "Warning: Invalid handshake signature from client, "
                     "closing connection (suspected scanner)."
                  << std::endl;
        close(client_fd);
        continue;
      }

      // Strip signature and parse active indices
      std::string indices_part = active_indices_msg.substr(signature.size());
      std::vector<std::string> active_tokens =
          absl::StrSplit(indices_part, ',');
      bool parse_ok = true;
      active_indices.clear();
      for (const auto& token : active_tokens) {
        if (!token.empty()) {
          int idx = 0;
          if (!absl::SimpleAtoi(token, &idx)) {
            std::cerr << "Warning: Failed to parse index '" << token
                      << "' as integer, invalid handshake." << std::endl;
            parse_ok = false;
            break;
          }
          active_indices.push_back(idx);
        }
      }

      if (!parse_ok) {
        close(client_fd);
        continue;
      }

      std::cout << "Handshake verified successfully. Sender is active."
                << std::endl;
      break;  // Exit loop and proceed to benchmark
    }

    std::cout << "Waiting for sender to complete benchmark..." << std::endl;
    char c;
    // Block until sender closes the connection
    ssize_t res = recv(client_fd, &c, 1, 0);
    if (res < 0) {
      perror("recv failed");
    }
    std::cout << "Sender disconnected. Benchmark finished." << std::endl;

    // 4. Verify data integrity across active managers only
    std::cout << "Verifying data integrity across active managers..."
              << std::endl;
    bool all_success = true;
    for (int m : active_indices) {
      std::cout << "Verifying manager " << m
                << " (interface: " << local_interfaces[m].first << ")..."
                << std::endl;
      bool success = true;
      const auto& manager = managers[m];
      const auto& dst_block_ids = allocated_block_ids[m];
      for (size_t l = 0; l < kNumLayers; ++l) {
        uint8_t* receiver_base = manager->GetHostPointer(l, 0);
        for (int b = 0; b < kNumBlocks; ++b) {
          uint8_t* dst_ptr = receiver_base + dst_block_ids[b] * block_size;
          for (size_t i = 0; i < block_size; ++i) {
            uint8_t expected = static_cast<uint8_t>((l + b + i + m) & 0xFF);
            if (dst_ptr[i] != expected) {
              std::cerr << "Data mismatch at manager " << m << ", layer " << l
                        << ", block " << b << ", byte " << i << ". Expected "
                        << (int)expected << ", got " << (int)dst_ptr[i]
                        << std::endl;
              success = false;
              break;
            }
          }
          if (!success) break;
        }
        if (!success) break;
      }
      if (!success) all_success = false;
    }

    if (all_success) {
      std::cout
          << "Data integrity verification PASSED across active interfaces!"
          << std::endl;
    } else {
      std::cout << "Data integrity verification FAILED!" << std::endl;
    }

    close(client_fd);
    close(server_fd);
  } else {
    // Sender
    std::cout << "Connecting to receiver control plane at " << peer_control_ip
              << ":" << peer_control_port << "..." << std::endl;
    int control_fd =
        connect_to_control_server(peer_control_ip, peer_control_port);
    if (control_fd < 0) {
      std::cerr << "Failed to connect to receiver control plane" << std::endl;
      return 1;
    }
    std::cout << "Connected to receiver control plane." << std::endl;

    // 1. Receive serialized endpoints from Receiver
    std::string peer_data_address = read_line(control_fd);
    std::cout << "Received peer data endpoints: " << peer_data_address
              << std::endl;

    // 2. Parse endpoints
    std::vector<std::string> endpoint_tokens =
        absl::StrSplit(peer_data_address, ',');
    struct RemoteEndpoint {
      std::string interface_name;
      std::string ip;
      int port;
    };
    std::vector<RemoteEndpoint> remote_endpoints;
    for (const auto& token : endpoint_tokens) {
      size_t first_colon = token.find(':');
      size_t last_colon = token.rfind(':');
      if (first_colon != std::string::npos && last_colon != std::string::npos &&
          first_colon < last_colon) {
        std::string interface_name = token.substr(0, first_colon);
        std::string ip =
            token.substr(first_colon + 1, last_colon - first_colon - 1);
        std::string port_str = token.substr(last_colon + 1);
        remote_endpoints.push_back({interface_name, ip, std::stoi(port_str)});
      }
    }

    // 3. Map remote endpoints to local interfaces and spawn managers
    std::vector<std::unique_ptr<tpu_raiden::kv_cache::KVCacheManagerBase>>
        managers;
    std::vector<std::string> mapped_peer_addresses;
    std::vector<std::vector<int>> allocated_block_ids;

    for (const auto& remote : remote_endpoints) {
      std::string local_ip = "";
      std::string local_iface = "";

      // Attempt to find a local interface in the same subnet
      for (const auto& local : local_interfaces) {
        if (in_same_subnet(local.second, remote.ip)) {
          local_ip = local.second;
          local_iface = local.first;
          break;
        }
      }

      if (local_ip.empty()) {
        std::cerr << "Warning: Could not find local interface in same subnet "
                     "as remote "
                  << remote.ip << " (" << remote.interface_name
                  << "). Falling back to first local interface." << std::endl;
        if (!local_interfaces.empty()) {
          local_ip = local_interfaces[0].second;
          local_iface = local_interfaces[0].first;
        } else {
          local_ip = data_ip;
        }
      }

      std::cout << "Mapping remote " << remote.interface_name << " ("
                << remote.ip << ":" << remote.port << ") to local "
                << local_iface << " (" << local_ip << ")" << std::endl;

      auto manager = std::make_unique<tpu_raiden::kv_cache::KVCacheManagerBase>(
          kNumLayers, kNumShards, block_size,
          /*local_port=*/std::nullopt,
          /*host_blocks_to_allocate=*/kNumBlocks, parallelism,
          /*host_allocator=*/nullptr,
          /*bind_ip=*/local_ip);

      auto src_block_ids_or =
          manager->block_manager()->Allocate(kNumBlocks, /*lock=*/true);
      if (!src_block_ids_or.ok()) {
        std::cerr << "Failed to allocate sender blocks: "
                  << src_block_ids_or.status() << std::endl;
        close(control_fd);
        return 1;
      }
      allocated_block_ids.push_back(src_block_ids_or.value());

      // Initialize sender blocks with pattern (unique per manager)
      int m_idx = managers.size();
      for (size_t l = 0; l < kNumLayers; ++l) {
        uint8_t* sender_base = manager->GetHostPointer(l, 0);
        for (int b = 0; b < kNumBlocks; ++b) {
          uint8_t* src_ptr =
              sender_base + src_block_ids_or.value()[b] * block_size;
          for (size_t i = 0; i < block_size; ++i) {
            src_ptr[i] = static_cast<uint8_t>((l + b + i + m_idx) & 0xFF);
          }
        }
      }

      std::string peer_addr = absl::StrFormat("%s:%d", remote.ip, remote.port);
      mapped_peer_addresses.push_back(peer_addr);
      managers.push_back(std::move(manager));
    }

    // 4. Run parallel warmup across all managers, and track successes
    std::cout << "Running parallel warmup across " << managers.size()
              << " interfaces..." << std::endl;
    std::vector<std::thread> threads;
    std::vector<bool> warmup_success(managers.size(), true);

    for (size_t m = 0; m < managers.size(); ++m) {
      threads.push_back(std::thread([&, m]() {
        std::vector<int> dst_block_ids(kNumBlocks);
        std::iota(dst_block_ids.begin(), dst_block_ids.end(), 0);
        for (int i = 0; i < kNumWarmup; ++i) {
          auto status_or = managers[m]->H2hWrite(
              mapped_peer_addresses[m], allocated_block_ids[m], dst_block_ids);
          if (!status_or.ok()) {
            std::cerr << "Warmup failed on manager " << m << " ("
                      << remote_endpoints[m].interface_name
                      << "): " << status_or.status() << std::endl;
            warmup_success[m] = false;
            break;
          }
        }
      }));
    }
    for (auto& t : threads) t.join();
    threads.clear();

    // Filter active (successful) managers
    std::vector<std::unique_ptr<tpu_raiden::kv_cache::KVCacheManagerBase>>
        active_managers;
    std::vector<std::string> active_peer_addresses;
    std::vector<std::vector<int>> active_allocated_block_ids;
    std::vector<std::string> active_interface_names;
    std::string active_indices_str = "";

    for (size_t m = 0; m < managers.size(); ++m) {
      if (warmup_success[m]) {
        active_managers.push_back(std::move(managers[m]));
        active_peer_addresses.push_back(mapped_peer_addresses[m]);
        active_allocated_block_ids.push_back(allocated_block_ids[m]);
        active_interface_names.push_back(remote_endpoints[m].interface_name);

        if (!active_indices_str.empty()) active_indices_str += ",";
        active_indices_str += std::to_string(m);
      }
    }
    active_indices_str += "\n";

    if (active_managers.empty()) {
      std::cerr << "Error: All managers failed warmup! Cannot run benchmark."
                << std::endl;
      close(control_fd);
      return 1;
    }

    std::cout << "Successfully warmed up " << active_managers.size() << " / "
              << managers.size() << " interfaces. Proceeding with benchmark."
              << std::endl;

    // Transmit active indices to Receiver with signature prefix
    std::string active_indices_msg =
        "RAIDEN_SENDER_ACTIVE:" + active_indices_str;
    if (!send_all(control_fd, active_indices_msg)) {
      std::cerr << "Failed to send active indices to receiver" << std::endl;
      close(control_fd);
      return 1;
    }

    // 5. Run parallel benchmark (timed collective iterations) across active
    // managers ONLY
    std::cout << "Running parallel benchmark (" << kNumIterations
              << " iterations) across " << active_managers.size()
              << " active interfaces..." << std::endl;
    std::vector<double> collective_latencies_ms;
    collective_latencies_ms.reserve(kNumIterations);

    for (int i = 0; i < kNumIterations; ++i) {
      absl::Time start = absl::Now();
      for (size_t m = 0; m < active_managers.size(); ++m) {
        threads.push_back(std::thread([&, m]() {
          std::vector<int> dst_block_ids(kNumBlocks);
          std::iota(dst_block_ids.begin(), dst_block_ids.end(), 0);
          auto status_or = active_managers[m]->H2hWrite(
              active_peer_addresses[m], active_allocated_block_ids[m],
              dst_block_ids);
          if (!status_or.ok()) {
            std::cerr << "Iteration " << i << " failed on active manager " << m
                      << ": " << status_or.status() << std::endl;
          }
        }));
      }
      for (auto& t : threads) t.join();
      threads.clear();
      absl::Time end = absl::Now();
      collective_latencies_ms.push_back(
          absl::ToDoubleMilliseconds(end - start));
    }

    // 6. Calculate and print collective statistics
    std::sort(collective_latencies_ms.begin(), collective_latencies_ms.end());
    double sum = 0;
    for (double val : collective_latencies_ms) {
      sum += val;
    }
    double mean_ms = sum / kNumIterations;
    double p50 =
        collective_latencies_ms[static_cast<size_t>(kNumIterations * 0.50)];
    double p90 =
        collective_latencies_ms[static_cast<size_t>(kNumIterations * 0.90)];
    double p99 =
        collective_latencies_ms[static_cast<size_t>(kNumIterations * 0.99)];

    // Combined throughput across active interfaces ONLY (telemetry correction!)
    size_t total_bytes = active_managers.size() * kNumLayers * kNumShards *
                         kNumBlocks * block_size;
    double throughput_gbs =
        (static_cast<double>(total_bytes) / 1e9) / (mean_ms / 1000.0);

    std::cout << "\n### Collective Benchmark Results (Sender) ###\n";
    std::cout << absl::StrFormat("Interfaces:  %d (active: %d)\n",
                                 managers.size(), active_managers.size());
    std::cout << absl::StrFormat("Block Size:  %d bytes\n", block_size);
    std::cout << absl::StrFormat("Parallelism: %d (streams per NIC)\n",
                                 parallelism);
    std::cout << absl::StrFormat("p50:   %8.3f ms\n", p50);
    std::cout << absl::StrFormat("p90:   %8.3f ms\n", p90);
    std::cout << absl::StrFormat("p99:   %8.3f ms\n", p99);
    std::cout << absl::StrFormat("Mean:  %8.3f ms\n", mean_ms);
    std::cout << absl::StrFormat("Throughput: %8.3f GB/s (collective)\n",
                                 throughput_gbs);

    std::cout << "Closing control connection..." << std::endl;
    close(control_fd);
  }

  return 0;
}
