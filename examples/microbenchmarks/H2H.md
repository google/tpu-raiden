# Multi-Node GKE Multi-NIC H2H Benchmark

This guide describes how to provision resources, compile, and execute the Multi-Node Host-to-Host (H2H) microbenchmark for **TPU Raiden** on Google Kubernetes Engine (GKE).

This benchmark evaluates cross-node memory-to-memory write throughput over high-speed networks, utilizing GKE's Multi-NIC networking and Dynamic Resource Allocation (DRA) network drivers to achieve maximum network bandwidth.

---

## GKE Cluster & Node Pool Provisioning

Setting up a dual-NIC, TPU-enabled cluster requires creating a workload placement policy, provisioning the GKE control plane with multi-networking enabled, and attaching a TPU node pool configured with GKE Net DRA.

Execute the following `gcloud` commands to provision the cluster:

### 0. Multi-NIC & DRANET Architectural Concepts

To achieve maximum throughput, the H2H benchmark separates the network planes:

- **Control Plane (`eth0`)**: The default GKE pod network interface, used exclusively for initial handshakes, TCP coordination, and barrier synchronization.
- **Data Plane (`eth1`..`ethN`)**: High-speed secondary physical network interfaces attached directly to the Pod namespace.

- **GKE Net DRA (Dynamic Resource Allocation)**: Attaches physical Mellanox NICs directly into the Pod namespace, bypassing standard overlay routing to achieve near-line-rate bare-metal TCP performance. Refer to the official guide: [Allocate network resources using DRA](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/allocate-network-resources-dra).

    - **DRANET (Automated Multi-NIC)**: Automatically provisions multiple high-speed secondary interfaces connected to isolated subnets. Leaving `--data_interface=""` (empty) triggers the runner's auto-discovery of all secondary interfaces, while `--control_interface=eth0` isolates control coordination on the primary network.

### 1. Create a Workload Placement Policy
This policy guarantees that the TPU nodes are physically co-located within the same topology slice for low-latency, high-throughput communication.

```bash
gcloud compute resource-policies create workload-policy "tpu-placement-policy" \
    --region="us-central1" \
    --project="YOUR_PROJECT_ID" \
    --accelerator-topology="2x2x1" \
    --type="HIGH_THROUGHPUT" \
    --accelerator-topology-mode="AUTO_CONNECT"
```

### 2. Create the GKE Cluster Control Plane
Enable multi-networking (`--enable-multi-networking`) and Dataplane V2 (`--enable-dataplane-v2`) to support high-speed secondary interfaces.
```bash
gcloud container clusters create "raiden-h2h-cluster" \
    --region="us-central1" \
    --project="YOUR_PROJECT_ID" \
    --network="default" \
    --subnetwork="default" \
    --cluster-version="1.35.3-gke.1389000" \
    --num-nodes=1 \
    --node-locations="us-central1-c" \
    --machine-type=e2-standard-8 \
    --enable-ip-alias \
    --enable-multi-networking \
    --enable-dataplane-v2
```

### 3. Create the TPU Node Pool with Net DRA
Provision the TPU VMs with physical high-speed NICs (`tpu7x-standard-4t`) and enable the GKE networking DRA driver label.
```bash
gcloud container node-pools create "tpu-multinic-pool" \
    --cluster="raiden-h2h-cluster" \
    --region="us-central1" \
    --project="YOUR_PROJECT_ID" \
    --node-locations="us-central1-c" \
    --num-nodes=2 \
    --machine-type="tpu7x-standard-4t" \
    --node-version="1.35.3-gke.1389000" \
    --placement-policy="tpu-placement-policy" \
    --accelerator-network-profile="auto" \
    --node-labels="cloud.google.com/gke-networking-dra-driver=true" \
    # Optional: Specify a capacity reservation if using pre-allocated slots
    # --reservation-affinity="specific" \
    # --reservation="YOUR_RESERVATION_NAME"
```

---

## Deployment & Execution Walkthrough

### Step 1: Deploy the Benchmark Pods

We provide a simple, clean Kubernetes manifest `raiden_benchmark_jobset.yaml` under `examples/microbenchmarks/` that spawns the necessary TPU and Net DRA enabled benchmark containers.

Deploy the manifest to your cluster:
```bash
kubectl apply -f examples/microbenchmarks/raiden_benchmark_jobset.yaml
```

Wait for the pods to be deployed and in the `Running` state:
```bash
kubectl get pods -w
```
This manifest will spawn two pods named `raiden-perf-test-workers-0-0-<suffix>` (the Sender) and `raiden-perf-test-workers-0-1-<suffix>` (the Receiver).

### Automated Startup Compilation & Pre-Optimized JAX Image
The pods are configured to boot from Google Cloud's official public JAX TPU container image:
`us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:latest`

This image provides the pre-optimized JAX and TPU library runtime environment (including Python, PyTorch/JAX libraries, and TPU device bindings) out-of-the-box. 

At startup, an entrypoint script automatically installs minimal compilation prerequisites (git, compiler toolchains), clones the public repository to `/workspace/tpu-raiden`, and compiles the TPU Raiden C++ extensions.

Once the pods are in the `Running` state, compilation is already complete, and the environment is fully prepared to execute the benchmark immediately.

### Step 2: Running the Benchmark

The benchmark is executed by entering the running pods using `kubectl exec` and running the compiled runner.

### Step 2: Launch the Receiver (Server)

Open a terminal, use `kubectl exec` to enter the receiver pod (index 1), and navigate to the benchmark folder:

```bash
# Enter the receiver container
kubectl exec -it raiden-perf-test-workers-0-1-<suffix> -- bash

# Navigate to the compiled workspace
cd /workspace/tpu-raiden

# Run the receiver control server
bazel run -c opt //examples/microbenchmarks:h2h_benchmark_runner -- \
    --role=receiver \
    --control_interface=eth0 \
    --data_interface="" \
    --peer_control_port=9999 \
    --block_size=134217728 \ # 128 MB (128 * 1024 * 1024 bytes)
    --parallelism=8 \
    --num_blocks=64 \
    > /tmp/receiver_perf.log 2>&1 &
```

### Step 3: Launch the Sender (Client)

Open a second terminal, dynamically resolve the receiver pod's IP address using `kubectl`, enter the sender pod (index 0), navigate to the benchmark folder, and execute the parallel transfer:

```bash
# Dynamically resolve the Receiver Pod's IP address from your local machine
RECEIVER_POD_IP=$(kubectl get pod -l jobset.x-k8s.io/replicatedjob-name=workers,jobset.x-k8s.io/job-index=1 -o jsonpath='{.items[0].status.podIP}')

# Enter the sender container
kubectl exec -it raiden-perf-test-workers-0-0-<suffix> -- bash

# Navigate to the compiled workspace
cd /workspace/tpu-raiden

# Run the sender client (pass the resolved RECEIVER_POD_IP)
bazel run -c opt //examples/microbenchmarks:h2h_benchmark_runner -- \
    --role=sender \
    --control_interface=eth0 \
    --peer_control_ip="$RECEIVER_POD_IP" \
    --peer_control_port=9999 \
    --data_interface="" \
    --block_size=134217728 \ # 128 MB (128 * 1024 * 1024 bytes)
    --parallelism=8 \
    --num_blocks=64
```

### Reference Result

Tested on v7x two-node cluster, block size 2MB.

| Parallelism (P) | Single-NIC | Multi-NIC | Speedup |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 31.72 | 58.62 | **1.85x** |
| **2** | 59.92 | 113.58 | **1.90x** |
| **4** | 101.98 | 201.29 | **1.97x** |
| **8** | 159.46 | 323.90 | **2.03x** |
| **16** | 170.50 | 314.83 | **1.85x** |

---

## Key Flags Reference

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--role` | *Required* | Set to `sender` or `receiver`. |
| `--control_interface` | `""` | The interface used for control plane TCP coordination. Must be explicitly set to `eth0` in GKE environments. |
| `--peer_control_ip` | `127.0.0.1` | (Sender only) The IP address of the receiver control plane. |
| `--peer_control_port` | `9999` | The port used for control plane communication. |
| `--data_interface` | `""` | Comma-separated list of high-speed data interfaces. Leave empty to auto-discover all active secondary interfaces. |
| `--block_size` | `1048576 (1 MB)` | Size of each transfer block in bytes. Calculate custom sizes using: `size_in_mb * 1024 * 1024` (e.g., 128MB is `134217728`). |
| `--parallelism` | `1` | Number of parallel connection streams per NIC. |
| `--num_blocks` | `64` | Total number of blocks to transfer during the test. |
| `--numa_node` | `-1` | Hard pin to a specific NUMA node (overrides automatic mapping if >= 0). |

---

### End-to-End (E2E) KV Cache Transfer Bandwidth (Device-to-Device over NIC)**

<!-- disableFinding(LINE_OVER_80) -->
We measured the end-to-end performance of transfers originating and terminating on the accelerator devices (e.g., using JAX). This includes the overhead of copying data between the device and the host, as well as the network transport between sender and receiver, which is effectively the `start_send` until the blocks are received. We evaluated the impact of NUMA awareness by comparing the default behavior (single NUMA node/NIC) against enabling multiple NUMA nodes (`ENABLE_MULTI_NUMA=1`).

#### System & Host Network Tuning Prerequisites

Achieving peak throughput relies on host-level networking optimizations:

- Enabling multi-queue packet steering (e.g. configuring 16 queues on high-speed data interfaces) to distribute RX/TX interrupt handling across CPU cores.
- Tuning host TCP socket memory buffers (increasing send/receive buffer maximums and window sizes).
- Ensuring MTU and Reverse Path Filtering (`rp_filter`) are appropriately configured for multi-interface striping.

#### Performance Results (Gbps)

The test used a **16 MB block size** (aggregate over 32 layers, 1024 blocks total) on 2-node Ironwood GKE cluster.

| Parallelism (P) | Single-NIC (NUMA Restricted) | Multi-NIC (NUMA Unrestricted) |
| :---: | :---: | :---: |
| **1** | 22.24 | 46.52 |
| **2** | 43.19 | 92.18 |
| **4** | 84.56 | 163.30 |
| **8** | 148.45 | 300.11 |
| **16** | 186.87 | 369.19 |
