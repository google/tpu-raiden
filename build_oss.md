## Update: commit `4768c29` (Save/Load/Evict lifecycle) — 2026-07-06

Commit `4768c29` ("before debug") added the **Save / Load / Evict** lifecycle to
`KVCacheStore` (on top of `FetchRemote`): `BlockStatus::HOST_AND_HBM`,
`RaidenBlockID.hbm_block_id`, per-op `{Save,Load,Evict}{Work,Completion}Queue` +
`{…}Future` + `Poll{…}Status`, proto `COMMAND_EXECUTE_{LOAD,SAVE,EVICT}` /
`COMMAND_{SAVE,EVICT}_COMPLETED`, `GlobalRegistry::Unregister`, and a Torch API.
Rebuilding it needed two *new* fixes plus re-applying the shims the commit reverted.

### New OSS breaks (this commit)
1. **Proto import path (google3 → OSS).** `global_registry.proto` now imports
   `raiden_service.proto` (the registry stores `RaidenIdProto`) via the google3 path,
   which protoc can't resolve:
   ```
   import "third_party/tpu_raiden/tpu_raiden/rpc/raiden_service.proto";  // -> File not found
   ```
   Fix — strip the `third_party/tpu_raiden/` prefix in **both** `global_registry.proto`
   and `rpc/controller_service.proto`:
   ```
   sed -i 's#import "third_party/tpu_raiden/#import "#g' \
     tpu_raiden/kv_cache/global_registry/global_registry.proto \
     tpu_raiden/rpc/controller_service.proto
   ```
   (The `global_registry_proto` proto_library already deps
   `//tpu_raiden/rpc:raiden_service_proto`; only the `import` string was wrong.)
2. **Missing include.** `raiden_orchestrator.cc` uses `absl::StrContains` without the
   header → `error: 'StrContains' is not a member of 'absl'`. Add
   `#include "absl/strings/match.h"` (the `absl/strings` BUILD dep is already present).

### Other issues
The commit re-imported clean-google3 versions of the test files + BUILD, dropping several
OSS shims from 2026-07-05. Re-applied verbatim:
- `grpc_only = True` on `global_registry_cc_grpc` (§1).
- `kv_cache_store_test.cc`: `EXPECT_OK`/`ASSERT_OK` → `absl_testing::IsOk()` shim +
  `absl/status:status_matchers` dep + `signal(SIGPIPE, SIG_IGN)` (§3).
- `api/jax/kv_cache_store_test.py`: `google3.pyglib.resources` → `try/except` bazel-bin
  fallback + OSS-only launch flags.
- Both service `_main.cc`: `signal(SIGPIPE, SIG_IGN)`.