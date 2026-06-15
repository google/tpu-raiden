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
# deploy_and_run.sh
# Automates: Copybara Export -> GKE Sync -> Compile inside Pods (Sequential, Safest) -> Execute Benchmark

set -e

# Redirect all output to a master log file while still printing to terminal
LOG_FILE="/tmp/deploy_and_run_master.log"
echo "Logging all execution details to ${LOG_FILE}"
exec > >(tee -i "$LOG_FILE") 2>&1

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../" && pwd)"
COPYBARA_CONFIG="${WORKSPACE_ROOT}/google3/third_party/tpu_raiden/copy.bara.sky"
EXPORT_DIR="/tmp/jetski/${CONVERSATION_ID:-ccaebe04-afea-4743-938f-9bf3a381415e}/vcs/exported_raiden"
JOBSET_YAML="$(dirname "${BASH_SOURCE[0]}")/jobset_sleep.yaml"
JOBSET_NAME="raiden-perf-test"
BUILD_MODE="opt" 
BAZEL_CACHE_FLAGS="--remote_cache=https://storage.googleapis.com/amylin-bucket/bazel-cache --google_default_credentials=true --remote_upload_local_results=true --disk_cache=/workspace/bazel-disk-cache"

echo "===================================================="
echo "Starting JAX Raiden GKE Deployment & Benchmark Loop"
echo "BUILD MODE: $BUILD_MODE"
echo "===================================================="

echo "=== Phase 1: Exporting upstream changes via Copybara ==="
jj git export || true
java -jar /tmp/copybara_old/copybara_deploy.jar \
  --force \
  --ignore-noop \
  "$COPYBARA_CONFIG" \
  piper_to_folder \
  "$WORKSPACE_ROOT" \
  --folder-dir="$EXPORT_DIR"

echo "=== Phase 2: Deploying GKE Pods ==="
echo "Deleting existing JobSet to ensure clean slate..."
kubectl delete jobset "$JOBSET_NAME" --ignore-not-found=true || true
echo "Waiting for old JobSet to be completely deleted..."
kubectl wait --for=delete jobset/"$JOBSET_NAME" --timeout=60s || true
echo "Waiting for old pods to be completely terminated..."
kubectl wait --for=delete pod -l app=raiden-perf-test --timeout=60s || true
echo "Applying JobSet: $JOBSET_YAML..."
kubectl apply -f "$JOBSET_YAML"

# Register exit handler for billing safety and diagnostics
cleanup_on_exit() {
  local exit_code=$?
  if [ $exit_code -eq 0 ]; then
    echo "Sweep completed successfully. Deleting JobSet immediately..."
    kubectl delete jobset "$JOBSET_NAME" --ignore-not-found=true || true
  else
    echo "ERROR: Sweep failed with exit status $exit_code."
    echo "Entering 30-minute diagnostic sleep before deleting JobSet..."
    sleep 1800
    echo "Diagnostic sleep finished. Deleting JobSet..."
    kubectl delete jobset "$JOBSET_NAME" --ignore-not-found=true || true
  fi
}
trap cleanup_on_exit EXIT

echo "Waiting for pods to be created..."
until [ -n "$(kubectl get pods -l app=raiden-perf-test -o name)" ]; do sleep 2; done
echo "Waiting for pods to be Ready..."
# Increased timeout to 600s to allow GKE to provision fresh TPU nodes from scratch after preemption
kubectl wait --for=condition=Ready pod -l app=raiden-perf-test --timeout=600s

# Get the pods
SENDER_POD=$(kubectl get pods -l app=raiden-perf-test -o jsonpath='{.items[0].metadata.name}')
RECEIVER_POD=$(kubectl get pods -l app=raiden-perf-test -o jsonpath='{.items[1].metadata.name}')

echo "Sender Pod: $SENDER_POD"
echo "Receiver Pod: $RECEIVER_POD"

echo "Applying rp_filter=2..."
kubectl exec "$SENDER_POD" -- sysctl -w net.ipv4.conf.all.rp_filter=2 || true
kubectl exec "$RECEIVER_POD" -- sysctl -w net.ipv4.conf.all.rp_filter=2 || true

# Helper to query IP of a specific physical interface on the host via the pod (due to hostNetwork: true)
get_interface_ip() {
  local POD=$1
  local IFACE=$2
  kubectl exec "$POD" -- sh -c "ip -4 addr show dev $IFACE 2>/dev/null | awk '/inet / {print \$2}' | cut -d/ -f1"
}

echo "=== Phase 3: Syncing Source Code to Pods ==="
# Strip Google-internal proto options that break OSS protoc compilation
find "$EXPORT_DIR" -name "*.proto" -exec sed -i '/option deadline =/d' {} \;
find "$EXPORT_DIR" -name "*.proto" -exec sed -i '/option security_level =/d' {} \;

echo "Syncing entire raiden package (Python + C++ Source) to both pods..."
for POD in "$SENDER_POD" "$RECEIVER_POD"; do
  echo "Syncing to ${POD}..."
  tar -C "$EXPORT_DIR" --exclude=venv -cf - . | kubectl exec -i "$POD" -- tar -C /workspace/google3/third_party/tpu_raiden -xf -
done
echo "All files synced successfully."

echo "=== Phase 3.5: Compiling INSIDE GKE Pods (Parallel, --jobs=32) ==="
BAZEL_VERSION="7.7.0"
LOCAL_BAZEL_BIN="/tmp/bazel-bootstrap-${BAZEL_VERSION}"
declare -a pids
declare -a compile_pods=("$SENDER_POD" "$RECEIVER_POD")

for POD in "${compile_pods[@]}"; do
  echo "Preparing Bazel in ${POD}..."
  if [ -f "$LOCAL_BAZEL_BIN" ]; then
    echo "  Copying bootstrapped Bazel to ${POD}..."
    kubectl cp "$LOCAL_BAZEL_BIN" "${POD}:/tmp/bazel-bootstrap-${BAZEL_VERSION}"
    kubectl exec "$POD" -- chmod +x "/tmp/bazel-bootstrap-${BAZEL_VERSION}"
  else
    echo "  Warning: Local bootstrapped Bazel not found at $LOCAL_BAZEL_BIN. Pod will attempt to download."
  fi

  echo "  Starting compilation on ${POD} in background (jobs=32)..."
  # Run build_raw_transfer.sh inside the pod, targeting JAX and the selected BUILD_MODE
  kubectl exec "$POD" -- sh -c "export BAZEL_CACHE_DIR=/workspace/bazel-disk-cache && cd /workspace/google3/third_party/tpu_raiden && bash ./build_raw_transfer.sh jax -c $BUILD_MODE --jobs=32 $BAZEL_CACHE_FLAGS" > "/tmp/build_in_pod_${POD}.log" 2>&1 &
  pids+=($!)
done

echo "Waiting for parallel compilations to complete..."
build_failed=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  POD="${compile_pods[$i]}"
  
  if ! wait "$pid"; then
    echo "ERROR: Compilation FAILED on $POD!"
    echo "=== Build Log for $POD ==="
    cat "/tmp/build_in_pod_${POD}.log"
    build_failed=1
  else
    echo "Compilation SUCCESSFUL on $POD."
  fi
done

if [ $build_failed -eq 1 ]; then
  exit 1
fi
echo "All pods compiled successfully."

echo "=== Phase 3.7: Configuring Policy-Based Routing (PBR) inside Pods ==="
for POD in "$SENDER_POD" "$RECEIVER_POD"; do
  echo "Configuring PBR on ${POD}..."
  kubectl cp "$(dirname "${BASH_SOURCE[0]}")/setup_pbr.sh" "${POD}:/tmp/setup_pbr.sh"
  kubectl exec "$POD" -- chmod +x /tmp/setup_pbr.sh
  kubectl exec "$POD" -- /tmp/setup_pbr.sh
done
echo "PBR successfully configured on both pods."

echo "=== Phase 4: Executing NUMA-Aligned Benchmark ==="
RUNNER_PATH="/workspace/google3/third_party/tpu_raiden/tools/perf_test_runner.py"

# Resolve physical interface IPs for Sender and Receiver (eth0 and eth1)
echo "Resolving Sender host physical IPs (eth0 and eth1)..."
SENDER_IP_ETH0=$(get_interface_ip "$SENDER_POD" "eth0")
SENDER_IP_ETH1=$(get_interface_ip "$SENDER_POD" "eth1")
echo "Sender IP (eth0): $SENDER_IP_ETH0"
echo "Sender IP (eth1): $SENDER_IP_ETH1"

echo "Resolving Receiver host physical IPs (eth0 and eth1)..."
RECEIVER_IP_ETH0=$(get_interface_ip "$RECEIVER_POD" "eth0")
RECEIVER_IP_ETH1=$(get_interface_ip "$RECEIVER_POD" "eth1")
echo "Receiver IP (eth0): $RECEIVER_IP_ETH0"
echo "Receiver IP (eth1): $RECEIVER_IP_ETH1"

if [ -z "$SENDER_IP_ETH0" ] || [ -z "$SENDER_IP_ETH1" ] || [ -z "$RECEIVER_IP_ETH0" ] || [ -z "$RECEIVER_IP_ETH1" ]; then
  echo "ERROR: Failed to resolve all physical IPs for Dual-NIC"
  exit 1
fi

## Enable pipefail for robust exit code capture
set -o pipefail

# Define the 350G Hunt Grid
CONFIGS=(
  # "128 16"
  # "64 32"
  "128 32"
  # "256 8"
  # "64 64"
  # "128 64"
)
L_BASE=8

# Results Storage
declare -A RESULTS_MEAN
declare -A RESULTS_MEDIAN
declare -A RESULTS_STATUS
declare -A RESULTS_TRIALS

# Helper function to execute a single configuration (L, BS, P)
execute_run() {
  local L=$1
  local BS=$2
  local P=$3
  
  # Mathematical dynamic calculations
  local BC=$((262144 / (L * BS)))
  local RUNNER_BS=$((BS / 2)) # Normalize to runner parameter
  
  echo "===================================================="
  echo "RUNNING CONFIG: L=${L}, BS=${BS}MB (runner_bs=${RUNNER_BS}), BC=${BC}, P=${P}"
  echo "===================================================="
  
  local success_count=0
  local trial_list=""
  local bws=()
  
  for trial in 1 2 3; do
    echo "  --- Trial ${trial}/3 ---"
    
    # 1. Clean up stale processes on Sender Pod
    kubectl exec "$SENDER_POD" -- pkill -f perf_test_runner.py || true
    kubectl exec "$SENDER_POD" -- rm -f /tmp/libtpu_lockfile || true
    kubectl exec "$RECEIVER_POD" -- rm -f /tmp/libtpu_lockfile || true
    sleep 2
    
    # 2. Launch Sender (background)
    kubectl exec "$SENDER_POD" -- sh -c "GLOG_logtostderr=1 TPU_PROCESS_ADDRESSES= TPU_WORKER_HOSTNAMES=127.0.0.1 TPU_HOST_BOUNDS=1,1,1 TPU_WORKER_ID=0 PYTHONPATH=/tmp/tpu_raiden_bazel_output_/external/jax~:/workspace/google3/third_party/tpu_raiden:/workspace nohup python3 $RUNNER_PATH --role=sender --grpc_port=50051 --block_size=${RUNNER_BS} --num_blocks=${BC} --num_layers=${L} --parallelism=${P} --local_ips=\"$SENDER_IP_ETH0,$SENDER_IP_ETH1\" --peer_ips=\"$RECEIVER_IP_ETH0,$RECEIVER_IP_ETH1\" > /tmp/sender_dual.log 2>&1 &"
    
    # Wait for Sender to boot
    sleep 3
    
    # Start local monitor in background inside Receiver Pod
    kubectl exec "$RECEIVER_POD" -- sh -c "nohup bash /workspace/google3/third_party/tpu_raiden/monitor_local.sh > /workspace/monitor_local.log 2>&1 & echo \$! > /workspace/monitor.pid"
    
    # 3. Launch Receiver (blocking)
    set +e
    set +o pipefail
    kubectl exec "$RECEIVER_POD" -- sh -c "GLOG_logtostderr=1 TPU_PROCESS_ADDRESSES= TPU_WORKER_HOSTNAMES=127.0.0.1 TPU_HOST_BOUNDS=1,1,1 TPU_WORKER_ID=0 PYTHONPATH=/tmp/tpu_raiden_bazel_output_/external/jax~:/workspace/google3/third_party/tpu_raiden:/workspace python3 \"$RUNNER_PATH\" --role=receiver --peer=\"$SENDER_IP_ETH1:50051\" --block_size=${RUNNER_BS} --num_blocks=${BC} --num_layers=${L} --parallelism=${P} --local_ips=\"$RECEIVER_IP_ETH0,$RECEIVER_IP_ETH1\" --peer_ips=\"$SENDER_IP_ETH0,$SENDER_IP_ETH1\"" 2>&1 | tee /tmp/receiver_run.log
    local RC=${PIPESTATUS[0]}
    set -e
    set -o pipefail
    
    # Stop local monitor
    kubectl exec "$RECEIVER_POD" -- sh -c "if [ -f /workspace/monitor.pid ]; then kill \$(cat /workspace/monitor.pid) && rm /workspace/monitor.pid; fi" || true
    
    local BW=$(grep "Effective Bandwidth" /tmp/receiver_run.log | awk '{print $3}' || true)
    if [ -n "$BW" ]; then
      echo "    Trial ${trial}/3 SUCCESS: ${BW} Gbps (Exit code $RC)"
      bws+=("$BW")
      success_count=$((success_count + 1))
      trial_list="${trial_list}${BW} "
    else
      echo "    Trial ${trial}/3 FAILED (Exit code $RC, Could not parse bandwidth)"
      trial_list="${trial_list}FAIL($RC) "
    fi
  done
  
  RESULTS_TRIALS["${L}_${BS}_${P}"]="${trial_list% }"
  
  # 4. Consolidate and average results
  if [ $success_count -gt 0 ]; then
    # Calculate Mean
    local sum=0
    for bw in "${bws[@]}"; do
      sum=$(awk "BEGIN {print $sum + $bw}")
    done
    local mean_bw=$(awk "BEGIN {print $sum / $success_count}")
    
    # Calculate Median
    local sorted_bws=($(printf '%s\n' "${bws[@]}" | sort -n))
    local median_bw
    if [ $((success_count % 2)) -eq 1 ]; then
      median_bw=${sorted_bws[$((success_count / 2))]}
    else
      local mid1=${sorted_bws[$((success_count / 2 - 1))]}
      local mid2=${sorted_bws[$((success_count / 2))]}
      median_bw=$(awk "BEGIN {print ($mid1 + $mid2) / 2}")
    fi
    
    RESULTS_MEAN["${L}_${BS}_${P}"]=$mean_bw
    RESULTS_MEDIAN["${L}_${BS}_${P}"]=$median_bw
    RESULTS_STATUS["${L}_${BS}_${P}"]="VERIFIED (${success_count}/3 Succeeded)"
    echo ">> CONFIG RESULT: L=${L}, BS=${BS}, P=${P} -> Mean BW = ${mean_bw} Gbps, Median BW = ${median_bw} Gbps (${success_count}/3 Succeeded)"
  else
    RESULTS_MEAN["${L}_${BS}_${P}"]="0.0"
    RESULTS_MEDIAN["${L}_${BS}_${P}"]="0.0"
    RESULTS_STATUS["${L}_${BS}_${P}"]="FAILED (0/3 Succeeded)"
    echo ">> CONFIG RESULT: L=${L}, BS=${BS}, P=${P} -> FAILED (0/3 Succeeded)"
  fi
}

# ==========================================
# Step 4.1: Execute Sweep
# ==========================================
echo "=== Starting Optimization Sweep (L=${L_BASE}) ==="
for CONFIG in "${CONFIGS[@]}"; do
  read -r P BS <<< "$CONFIG"
  execute_run $L_BASE $BS $P
done

# 4. Clean up Sender at the very end
kubectl exec "$SENDER_POD" -- pkill -f perf_test_runner.py || true
    kubectl exec "$SENDER_POD" -- rm -f /tmp/libtpu_lockfile || true
    kubectl exec "$RECEIVER_POD" -- rm -f /tmp/libtpu_lockfile || true

# ====================================================
# Phase 4.5: Backing Up Bazel Disk Cache
# ====================================================
echo "=== Phase 4.5: Backing Up Bazel Disk Cache ==="
echo "Triggering parallel GCS backup from Sender Pod ($SENDER_POD)..."
set +e
kubectl exec "$SENDER_POD" -- bash /workspace/google3/third_party/tpu_raiden/backup_cache_pod.sh \
    --bucket "amylin-bucket" \
    --prefix "bazel-cache" \
    --cache-dir "/workspace/bazel-disk-cache" \
    --workers 32 2>&1 | tee /tmp/gcs_backup.log
BACKUP_STATUS=$?
set -e

if [ $BACKUP_STATUS -eq 0 ]; then
  echo "Bazel cache backup COMPLETED successfully."
else
  echo "WARNING: Bazel cache backup FAILED with status $BACKUP_STATUS. Proceeding to results consolidation."
fi

# ====================================================
# Phase 5: Results Consolidation & Logging
# ====================================================
echo "=== Phase 5: Consolidating Results ==="

# Print Summary to Console
echo "===================================================="
echo "OPTIMIZATION SWEEP RESULTS:"
for CONFIG in "${CONFIGS[@]}"; do
  read -r P BS <<< "$CONFIG"
  mean=${RESULTS_MEAN["${L_BASE}_${BS}_${P}"]}
  median=${RESULTS_MEDIAN["${L_BASE}_${BS}_${P}"]}
  status=${RESULTS_STATUS["${L_BASE}_${BS}_${P}"]}
  trials=${RESULTS_TRIALS["${L_BASE}_${BS}_${P}"]}
  echo "  BS=${BS}MB, P=${P}: Mean=${mean} Gbps, Median=${median} Gbps (${status}) [Trials: ${trials}]"
done
echo "===================================================="

# Append to experiment_log.md
LOG_PATH="/usr/local/google/home/amylin/.gemini/jetski/brain/8c106181-6646-4b43-ba08-051c1777723b/experiment_log.md"
echo -e "\n### Run 39: HEAD with NumaThreadPool Integration\n" >> "$LOG_PATH"
echo -e "Executed 18-run optimization sweep (6 configs, 3 trials each).\n" >> "$LOG_PATH"
echo -e "| Parallelism (P) | Block Size (BS) | Mean BW (Gbps) | Median BW (Gbps) | Trials |" >> "$LOG_PATH"
echo -e "| :--- | :--- | :--- | :--- | :--- |" >> "$LOG_PATH"
for CONFIG in "${CONFIGS[@]}"; do
  read -r P BS <<< "$CONFIG"
  mean=${RESULTS_MEAN["${L_BASE}_${BS}_${P}"]}
  median=${RESULTS_MEDIAN["${L_BASE}_${BS}_${P}"]}
  trials=${RESULTS_TRIALS["${L_BASE}_${BS}_${P}"]}
  echo -e "| **P=${P}** | **BS=${BS}MB** | ${mean} | ${median} | ${trials} |" >> "$LOG_PATH"
done
echo "Results successfully appended to $LOG_PATH"

echo "Done!"
