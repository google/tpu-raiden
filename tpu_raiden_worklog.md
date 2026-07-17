# TPU Raiden Development Worklog

## Activity Log

| Date | Activity Type | Subagent Role/ID | Problem Addressed | Status | Notes |
|--- |--- |--- |--- |--- |--- |
| 2026-07-15 | Invoke Subagent | tpu-raiden-coder | Resolve conflicts & implement kv_cache_store_test_gf JAX Python TPU test | COMPLETED | Fixed conflicts, registered GL/GF tests. Fixed IPv6 gRPC worker registration bug. Fixed JAX host-caching verification bug. Updated to sequential distinct data for strict byte-exact validation. |
| 2026-07-15 | Invoke Subagent | tpu-raiden-coder | Implement asynchronous Save and Load with status polling APIs | COMPLETED | Implemented C++ internal state tracking and Poll APIs, JAX/nanobind bindings, and updated tests. Verified GL/GF tests pass. |
| 2026-07-15 | Invoke Subagent | tpu-raiden-coder | Update Python Save/Load APIs to return boolean and document return values | COMPLETED | Modified nanobind lambda to return bool, updated python wrapper/docstrings and type hints, updated mock tests. Verified all tests pass. |
| 2026-07-15 | Run Tests | Main Agent | Run comprehensive C++ and JAX tests in kv_cache and controllers | COMPLETED | Verified all 12 tests passed successfully. |
| 2026-07-15 | Invoke Subagent | tpu-raiden-coder | Restore original Allocate/Deallocate APIs and add new AllocateBlockIds/DeallocateBlockIds APIs | COMPLETED | Restored BufferProto-based Allocate/Deallocate APIs, added block-ID-based ones, updated tests and KVCacheStore. Verified all tests pass. |
| 2026-07-15 | Invoke Subagent | todo-adder-coder | Add TODO comments in TransferBuffers methods regarding peer argument | COMPLETED | Added TODO comments to both TransferBuffers overloads in raiden_controller.cc and formatted. Verified build succeeds. |
| 2026-07-15 | Invoke Subagent | tpu-raiden-poller-coder | Implement background completion poller thread in KVCacheStore | COMPLETED | Implemented PollerLoop background thread in KVCacheStore, updated PollSaveStatus/PollLoadStatus to consume status lists and snapshot pending active futures. Verified C++ and JAX tests pass. |
| 2026-07-15 | Invoke Subagent | tpu-raiden-coder (5b4cdf19-f806-4ffa-a49e-8decc5f72056) | Integrate and run E2E JAX tests (Save/Load) with registry write-through | COMPLETED | Overwrote api/jax/BUILD, updated python wrapper/tests with dynamic port allocation and __eq__ operator. Verified all tests (CPU, GL, GF) pass. |
| 2026-07-15 | Update Comments | Main Agent | Align notes in global_registry and kv_cache with new code | COMPLETED | Updated comments in global_registry_client.h and global_registry.proto to refer to workers/raiden_id instead of hosts. |
| 2026-07-15 | Invoke Subagent | tpu-raiden-oss-builder | Port OSS build fixes from cl/943119780, run Copybara export and Kokoro build | COMPLETED | Ports net/util:ports dependency removal, SIGPIPE handling, and Copybara rules; ran Copybara export and local Bazel presubmit build. All JAX build verification tests passed. |
| 2026-07-15 | Update Plan | Main Agent | Revise remote fetch design and rename FetchRemote -> ReadRemote | COMPLETED | Created remote_read_plan_v5.md detailing the asynchronous chaining of H2H and H2D to avoid race conditions. |
| 2026-07-15 | Run Tests | Main Agent | Run C++ controller, C++ kv_cache, and JAX api tests | COMPLETED | Ran 7 controller tests (passed), 7 kv_cache tests (passed), and 8 JAX api tests (passed with nocheck_visibility). |
| 2026-07-15 | Amend CL | Main Agent | Amend and upload CL 948523042 | COMPLETED | Folded build fixes, updated description with comprehensive test results and BEGIN_PUBLIC/END_PUBLIC block, excluded worklog and plan file, and uploaded to Critique. |
| 2026-07-15 | Invoke Subagent | tpu-raiden-coder (13c8495e-c582-408c-9efe-bd4cb9fb9718) | Add C++ unit test for Save write-through in kv_cache_store_test.cc | COMPLETED | Added `SaveWriteThrough` test to `kv_cache_store_test.cc` verifying async write-through to mock global registry. Verified all `kv_cache` tests pass. |
| 2026-07-15 | Amend CL | Main Agent | Amend and upload CL 948523042 | COMPLETED | Updated description to note replacement of cl/943004367 and cl/943068951, updated test links, and uploaded to Critique. |
| 2026-07-15 | Invoke Subagent | tpu-raiden-coder (4ac3a794-0d8a-49ae-8a54-229cded69368) | Replace portpicker with socket in kv_cache_store_test.py | COMPLETED | Removed `portpicker` dependency, implemented custom `_pick_unused_port()` using `socket`, and updated test/BUILD files. Verified `copybara_analysis_test` and JAX tests pass. |
| 2026-07-15 | Amend CL | Main Agent | Amend and upload CL 948523042 | COMPLETED | Updated description to note portpicker fix, and uploaded to Critique. |
| 2026-07-15 | Invoke Subagent | tpu-raiden-coder (bbd1d68d-7420-494b-8976-e5624892bb81) | Fix lint errors in CL | COMPLETED | Addressed hg lint violations in kv_cache_store.h (NOLINT thread), kv_cache_store_test.py (import ordering, assertLen, etc.), and kv_cache_store_e2e_test.py. |
| 2026-07-15 | Amend CL | Main Agent | Amend and upload CL 948523042 | COMPLETED | Updated description to note lint fixes, and uploaded to Critique. |
| 2026-07-15 | Invoke Subagent | tpu-raiden-coder (c1c25836-a51f-4f4d-bb9e-9ff408dfd1cd) | Fix AI review comments in kv_cache_store.cc | COMPLETED | Changed constructor parameters to absl::string_view and added write_through_regs.reserve() in kv_cache_store.cc. Verified C++ and JAX tests pass. |
| 2026-07-15 | Run Tests | Main Agent | Run JAX Ghostlite/GF tests after fixes | COMPLETED | Ran 7 JAX tests (GL/GF versions) and they all passed. Sponge Link: http://sponge2/a1e9ff6f-3628-4f7e-a333-ddc44602428b |
| 2026-07-15 | Amend CL | Main Agent | Amend and upload CL 948523042 | COMPLETED | Updated description with new test links and AI comments fix note, and uploaded to Critique. |
| 2026-07-15 | Update Reviewers | Main Agent | Add raiden-eng+reviews to CL | COMPLETED | Added `raiden-eng+reviews` group as a reviewer to CL 948523042 as per METADATA review_notify policy. |
| 2026-07-15 | Invoke Subagent | tpu-raiden-coder (546c2c01-8604-44c6-92dc-32a2a3462123) | Implement Evict in KVCacheStore | COMPLETED | Implemented Evict overloads and helper, added unit tests, verified pass (Sponge: http://sponge2/e326cf87-492b-44fd-a570-e09a9fcc3865), and created stacked CL 948665971. |



| 2026-07-15 | Invoke Subagent | self (095928b1-2377-4e6f-aafb-82ca1636073e) | CL Monitor Agent | CANCELLED | Cancelled by user request. (Previously monitored CL 948523042 and was waiting for approval). |

| 2026-07-16 | Invoke Subagent | tpu-raiden-coder (1b958b94-d08a-4d67-99d1-02b83c637fc8) | Revert unnecessary :utils dependency from kv_cache_manager_core_lib | COMPLETED | Reverted change in frameworks/jax/BUILD. Verified kv_cache_manager_core_lib builds successfully without python. |
| 2026-07-16 | Revert & Upload | Main Agent | Revert frameworks/jax/BUILD to HEAD and upload | COMPLETED | Reverted all changes in frameworks/jax/BUILD to parent revision to keep it clean. Uploaded changeset 67569e261b46. |
| 2026-07-16 | Plan Feature | Main Agent | Proactive Eviction on Allocation | COMPLETED | Created plan `implement_proactive_eviction_plan.md`. User requested to amend into CL 948665971. Decided to reuse the subagent's workspace to perform the amend. |
| 2026-07-16 | Invoke Subagent | tpu-raiden-coder (606704d9-9094-4d15-95bd-b86fdac356c6) | Implement Proactive Eviction (amend) | COMPLETED | Implemented proactive eviction and deallocate wrapper, amended into CL 948665971 in subagent-coder workspace. Autoreview passed with 0 findings. |
| 2026-07-16 | Rebase CL | Main Agent | Rebase CL 948523042 to HEAD | COMPLETED | Reverted local changes (CL 2), rebased CL 948523042 to HEAD (0992fabb), uploaded to Critique, and re-imported CL 948665971 changes locally. |
| 2026-07-16 | VCS Operation | Main Agent | Discard CL 948665971 and commit new CL | COMPLETED | Discarded CL 948665971 in Critique using `hg cls-drop` in coder workspace. Committed local changes in main workspace to create new CL 948689811 stacked on 948523042. |
| 2026-07-16 | Plan Feature | Main Agent | Revise LRUCache with Candidate List | COMPLETED | Created plan `revise_lru_candidate_list_plan.md`. |
| 2026-07-16 | Invoke Subagent | tpu-raiden-coder (a1dc25f9-e77f-47fa-a4cc-46209a804ca7) | Implement LRU Candidate List | COMPLETED | Spawned coder subagent to implement the LRU revision with candidate list. CL 948704974 created, then folded into CL 948689811. CL 948704974 dropped. Synced main workspace by pulling from subagent path and evolving. |
| 2026-07-16 | VCS Operation | Main Agent | Sync stack to HEAD | COMPLETED | Ran `hg sync` in main workspace to rebase CL 948523042 and CL 948689811 onto latest HEAD (93cd07bc). Uploaded changes to Critique. |
| 2026-07-16 | Plan Feature | Main Agent | Update InsertAndLock and ReleaseAndDelete | COMPLETED | Created plan `update_insert_release_lru_plan.md`. |
| 2026-07-16 | Invoke Subagent | tpu-raiden-coder (2471227e-fd22-4118-a66a-87f562a4fdbe) | Implement Insert/Release Updates | COMPLETED | Implemented InsertAndLock/ReleaseAndDelete updates, updated JAX python docstrings, updated test assertions for reversed Release. Verified all tests pass. |
| 2026-07-16 | Invoke Subagent | tpu-raiden-coder (45f1f9a5-b8d7-43a1-9a32-cd673ff01dad) | Resolve comments on parent CL 948523042 (ThreadPool & Logs) | COMPLETED | Implemented ThreadPool for write-through registrations, removed [Jetski] log prefixes. Verified parent C++ tests. Evolved child CL and verified all C++/Python JAX tests pass. Uploaded stack and replied to Critique comments. |
| 2026-07-16 | Invoke Subagent | tpu-raiden-coder (b7ce4d19-7f3b-46f9-8093-de55e2339384) | Fix TAP Presubmit (Copybara) & AI Review suggestions in CL 948689811 | COMPLETED | Hided ThreadPool from OSS using copybara:strip in kv_cache_store.cc/h. Added reserve() to Evict. Verified C++ tests, JAX tests, and copybara_analysis_test pass. Amended and uploaded. |
| 2026-07-16 | Invoke Subagent | tpu-raiden-coder (e20c8dba-ea7f-46fb-b75c-8c130d0ec855) | Implement AI Review fixes (Rollback Rescue & Evict Race) | COMPLETED | Implemented sorted hashes helper, pending_eviction_counts_ map to track evictions, and check pin_count in Evict. Added RollbackRescue and EvictRaceCondition tests. Verified all tests pass. Amended. |
| 2026-07-16 | Invoke Subagent | tpu-raiden-coder (d7a3ce82-d66e-44bd-940d-82bb490408b4) | Fix Copybara BUILD dependency (//thread) | COMPLETED | Stripped //thread dependency from kv_cache_store target in BUILD using copybara:strip. Verified copybara_analysis_test passes. Amended. |
| 2026-07-16 | Invoke Subagent | tpu-raiden-coder (6ff0d0fe-8dcf-431d-93f6-4f6041715ab9) | Dummy edit to retry presubmit | KILLED | Subagent killed due to conflict with user's manual edits. |
| 2026-07-16 | Invoke Subagent | tpu-raiden-coder (e17cfac9-ddb2-4080-a4af-00476e094d01) | Fix ClangTidy and ClangInliner warnings in CL 948689811 | COMPLETED | Fixed deprecated MutexLock calls, added missing includes in C++ store/tests, updated include paths for gRPC. Verified C++ tests. Amended and uploaded. |
| 2026-07-16 | Invoke Subagent | tpu-raiden-coder (8ba63dac-6a55-4746-9d36-54bcca0f21a8) | Add comments to RaidenBlockID::host_block_id | COMPLETED | Added comment explaining host_block_id status mapping in kv_cache_store.h. Amended and uploaded. |
| 2026-07-16 | Invoke Subagent | tpu-raiden-coder (93b803ea-e8a7-4ad6-9e23-2a81e761d4b2) | Fix Tricorder findings (VectorCheck) and lints | COMPLETED | Fixed VectorCheck in kv_cache_store.cc by bounding loop size with std::min. Cleaned up unused imports/docstring in api/torch/kv_cache_store.py. Verified C++ and Python tests. Amended and uploaded. |
| 2026-07-16 | Invoke Subagent | tpu-raiden-coder (df455312-de6f-44d2-9140-c1d52829d68f) | Refactor pending_eviction_counts_ to flat_hash_map | COMPLETED | Refactored pending_eviction_counts_ in kv_cache_store.h to absl::flat_hash_map, removed <map> include. Verified tests and amended. |

| 2026-07-16 | Invoke Subagent | tpu-raiden-coder (af64f390-0c85-4c20-a995-015099d9eb47) | Cap candidate restoration in ReleaseAndDelete | COMPLETED | Capped candidate restoration count by std::min(deleted_blocks, restoration_count) in ReleaseAndDelete. Verified C++ tests and amended. |
| 2026-07-16 | Invoke Subagent | tpu-raiden-design-reviewer (00f88bf3-5beb-41cd-a231-5b9529464862) | Review Remote Read Plan v5 | COMPLETED | Reviewed remote read plan and reported critical issues (memory leak, LRU cleanup concurrency, duplicate read protection, missing API). |
| 2026-07-16 | Invoke Subagent | tpu-raiden-coder (32412b18-fb53-4920-a9b8-7faa74c4358f) | Implement ReadRemote in RaidenController (Step 3 & 3.5) | COMPLETED | Implemented ReadRemote gRPC, updated TransferBuffers with peers, verified using existing controller tests. |
| 2026-07-16 | Invoke Subagent | tpu-raiden-coder (e15f1c63-1873-4fc2-aa1a-e06b2469919b) | Add unit test for ReadRemote in raiden_controller_test.cc | COMPLETED | Implemented ReadRemoteSuccess in raiden_controller_test.cc. Verified sorting of workers and address resolution. Uploaded to CL 948805474. |

| 2026-07-16 | Verify Codebase | self | Verify Worker H2H support & unit tests (Step 4) | COMPLETED | Verified that WorkerServiceImpl::TransferBuffers already supports H2H and is covered by TransferBuffersH2hSuccess in worker_service_test.cc. |
| 2026-07-16 | Invoke Subagent | tpu-raiden-coder (d460b066-342f-4d2e-b753-6cb3490604e4) | Implement ReadRemote core logic in KVCacheStore (Step 5) | COMPLETED | Implemented ReadRemote and PollRemoteReadStatus, background poller support. Added 5 C++ tests. Verified locally. Committed as 9d970e7e22ea. |
| 2026-07-16 | Invoke Subagent | tpu-raiden-coder (b1e056c9-a101-436c-b7a8-6e8b213c15c7) | Expose ReadRemote to Python and add E2E tests (Step 6) | IN_PROGRESS | |




























