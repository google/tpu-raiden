# Global Prefix Caching with KVCacheStore

This document describes the Global Prefix Caching mechanism in `tpu-raiden`, which allows sharing Key-Value (KV) cache blocks across different serving nodes/jobs using a centralized `GlobalRegistry`.

## Overview

In a distributed serving environment, a request might be routed to a node that does not have the required KV cache prefix locally. Global Prefix Caching enables this node to:
1.  Query a global registry to find if another node has the cached prefix.
2.  If found, retrieve the cache from the remote node via Host-to-Host (H2H) transfer.

`KVCacheStore` serves as the logical directory. It has been extended to support falling back to the `GlobalRegistry` when a local cache miss occurs.

## Architecture & Workflow

```mermaid
sequenceDiagram
    participant Client as Connector/Scheduler
    participant Store as KVCacheStore (Local)
    participant Registry as GlobalRegistry
    participant Peer as Remote Node

    Client->>Store: Lookup(hashes, enable_global=true)
    ActiveLine Store
    Store->>Store: Look up locally (LRU Cache)
    Note over Store: Local Hits: H1, H2. Miss: H3.
    
    rect rgb(240, 240, 240)
        Note over Store: Fallback to Global Registry
        Store->>Registry: Lookup(remaining_hashes: [H3, H4])
        Registry-->>Store: Returns [Metadata(Peer, BlockID)] for H3
    end
    
    Store-->>Client: Returns [(H1, LocalRaidenId), (H2, LocalRaidenId), (H3, RemoteRaidenId)]
    DeactiveLine Store

    Note over Client: Client decides to fetch H3 from Peer
    Client->>Peer: H2H Read (RemoteRaidenId.job_name, RemoteRaidenId.data_replica_idx)
    Peer-->>Client: Send KV Cache Data
```

### Detailed Steps

1.  **Local Lookup**: `KVCacheStore::Lookup` first checks its local `LRUCache`. It processes the requested block hashes sequentially.
2.  **Stop on First Miss (Local)**: The local lookup stops at the first hash that is not present in the local cache.
3.  **Global Fallback**: If `enable_global` is set to `true` and a `global_registry_address` was provided to the `KVCacheStore` constructor, the store will query the `GlobalRegistry` for the remaining (missing) hashes.
4.  **Stop on First Miss (Global)**: The `GlobalRegistry` also performs a sequential lookup and stops at the first miss. It returns metadata (host address and block ID) for the sequential hits it found.
5.  **Result Aggregation**: `KVCacheStore` converts the remote metadata into `RaidenId` objects, where:
    *   `job_name` is set to the remote `host_address` (e.g., `IP:port`).
    *   `data_replica_idx` is set to the remote `block_id`.
    *   `job_replica_id` is set to `"0"`.
    *   `data_name` is set to `"kv_cache"`.
    これらの remote `RaidenId`s are appended to the local hits and returned to the caller.

## API Usage

### C++ API

#### Initialization
Pass the `global_registry_address` to the `KVCacheStore` constructor:

```cpp
#include "third_party/tpu_raiden/tpu_raiden/kv_cache/kv_cache_store.h"

// Initialize with capacity 100 and a global registry address
tpu_raiden::kv_cache::KVCacheStore store(100, "global-registry-service:50051");
```

#### Lookup
Call `Lookup` with `enable_global = true`:

```cpp
std::vector<std::string> hashes = {"hash_1", "hash_2", "hash_3"};
auto lookup_res_or = store.Lookup(hashes, /*enable_global=*/true);

if (lookup_res_or.ok()) {
  const auto& results = lookup_res_or.value();
  for (const auto& [hash, raiden_ids] : results) {
    for (const auto& raiden_id : raiden_ids) {
      if (raiden_id.job_name != "local_job_name") { // Identify remote
        std::string remote_host = raiden_id.job_name;
        int remote_block_id = raiden_id.data_replica_idx;
        // Trigger H2H transfer from remote_host
      }
    }
  }
}
```

### Python API (JAX/Torch)

The JAX and Torch bindings expose this functionality.

```python
from google3.third_party.tpu_raiden.tpu_raiden.api.jax import kv_cache_store

store = kv_cache_store.KVCacheStore(capacity=100, global_registry_address="global-registry-service:50051")

# Lookup with global fallback
results = store.lookup(block_hashes=["h1", "h2"], enable_global=True)
```
