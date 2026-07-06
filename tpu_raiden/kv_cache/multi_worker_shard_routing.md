# Multi-Worker Listener & Shard Routing Architecture in Raiden

This document explains how the `RaidenControllerEmbedded` manages multiple worker (listener) processes and maps/routes multiple logical shards across them during remote fetch operations.

---

## 🏗️ 1. Shard-to-Worker Mapping (Local Discovery)

Logical cache shards are distributed **sequentially** across local worker (listener) processes.

At startup, the controller discovers its local workers and their shard counts:
1.  The controller iterates from `0` to `num_listeners_ - 1` and queries each local worker listener over its control port (starting at `local_worker_port_`) using the `COMMAND_GET_ENDPOINTS` RPC.
2.  Each worker returns a list of its local data endpoints (IP:port sockets). The size of this list determines how many shards that worker manages locally:
    $$\text{num\_local\_shards}_i = \text{listener\_shard\_counts\_}[i]$$
3.  The controller maps global shard indices sequentially across these workers:
    *   **Worker 0** manages global shards: $[0, \ \text{listener\_shard\_counts\_}[0] - 1]$
    *   **Worker 1** manages global shards: $[\text{listener\_shard\_counts\_}[0], \ \text{listener\_shard\_counts\_}[0] + \text{listener\_shard\_counts\_}[1] - 1]$
    *   ... and so on.
4.  The controller consolidates all local worker endpoints sequentially in `local_worker_data_addresses_`. The index in this list matches the **global shard index**.

---

## 🤝 2. Fetch Negotiation & Routing Configuration

When a destination store initiates `FetchRemote`, the mapping is established:

1.  **Request Packaging**: The destination controller copies its entire `local_worker_data_addresses_` list (ordered by global shard index) into the `FetchNegotiationRequest` as `dst_worker_data_addresses`.
2.  **Source Scheduling**: The source controller receives the negotiation request. For each global shard `sh`:
    *   It extracts the destination worker's data address from `neg_req.dst_worker_data_addresses(sh)`.
    *   It schedules a transfer entry on source shard `sh` targeting this destination address.
    *   **Local Index Mapping**: The source controller maps the global destination shard index to its corresponding local shard index on the target destination worker. This is done by tracking how many shards have been assigned to that worker address:
        ```cpp
        int local_dst_sh = peer_to_local_shard[peer_addr]++;
        entry->set_dst_shard_idx(local_dst_sh);
        ```

---

## 🚀 3. Data Transfer Execution (PUSH Mechanism)

Raiden uses a **PUSH** model where data transfer is initiated by the source workers:

1.  **Destination Prep**: The destination controller broadcasts `COMMAND_START_TRANSFER` (with `is_sender = false`) to all its local worker listeners. Each worker registers its active plan and listens for incoming connections.
2.  **Execution Trigger**: The destination controller sends `COMMAND_EXECUTE_FETCH` to the source controller.
3.  **Source Slicing**: The source controller slices the negotiated schedule for each of its local workers:
    *   For each local worker `i`, it extracts entries where the global source shard `global_sh` matches one of the worker's local shards (`local_sh`).
    *   It broadcasts `COMMAND_START_TRANSFER` (with `is_sender = true`) containing the sliced schedules to each local source worker.
4.  **Direct PUSH**: Each source worker reads its local schedule and directly opens a TCP connection to the destination worker endpoint (`entry.dst_peer()`, which matches `neg_req.dst_worker_data_addresses(global_sh)`) to stream the blocks.

### 🔍 Direct 1-to-1 Mapping Scenario
If the source and destination configurations match exactly (e.g. 2 listeners on both sides managing 1 shard each, with no resharding):
*   `dst_worker_data_addresses` is `[ "dest_worker_0_addr", "dest_worker_1_addr" ]`.
*   **Source Worker 0** (managing global shard 0) directly connects to and PUSHes data to **Destination Worker 0** (`dest_worker_0_addr`).
*   **Source Worker 1** (managing global shard 1) directly connects to and PUSHes data to **Destination Worker 1** (`dest_worker_1_addr`).
