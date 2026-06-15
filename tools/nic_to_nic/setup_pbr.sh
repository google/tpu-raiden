#!/bin/bash

# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# setup_pbr.sh
# Idempotent Policy-Based Routing (PBR) setup script for multi-NIC GKE Pods (Secondary NIC only)
set -ex

# 1. Resolve local IPs
IP_ETH1=$(ip -4 addr show dev eth1 | grep inet | awk '{print $2}' | cut -d'/' -f1)

echo "Resolved eth1 IP: $IP_ETH1"

if [ -z "$IP_ETH1" ]; then
  echo "ERROR: Failed to resolve local IP for eth1!"
  exit 1
fi

# 2. Query GCP Metadata Server for eth1 Gateway
GW_ETH1=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/1/gateway)

echo "Resolved eth1 GW: $GW_ETH1"

if [ -z "$GW_ETH1" ]; then
  echo "ERROR: Failed to resolve eth1 gateway from Metadata Server!"
  exit 1
fi

# 3. Flush custom routing table 101 (ignore error if table doesn't exist yet)
ip route flush table 101 2>/dev/null || true

# 4. Populate custom routing table 101
# Add default route via the eth1 gateway.
ip route add default via "$GW_ETH1" dev eth1 table 101

# 5. Add source-IP routing rule for eth1 (idempotent: delete first, then add)
ip rule del from "$IP_ETH1" lookup 101 2>/dev/null || true
ip rule add from "$IP_ETH1" lookup 101

# 6. Configure loose Reverse-Path Filtering (rp_filter=2)
sysctl -w net.ipv4.conf.eth0.rp_filter=2
sysctl -w net.ipv4.conf.eth1.rp_filter=2
sysctl -w net.ipv4.conf.all.rp_filter=2

echo "PBR configuration completed successfully for eth1!"
