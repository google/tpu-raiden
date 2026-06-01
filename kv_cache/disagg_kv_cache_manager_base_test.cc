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

#include "kv_cache/disagg_kv_cache_manager_base.h"

#include <chrono>  // NOLINT
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/notification.h"
#include "core/raw_transfer_core.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

class MockDisaggKVCacheManager : public DisaggKVCacheManagerBase {
 public:
  using DisaggKVCacheManagerBase::DisaggKVCacheManagerBase;

  absl::StatusOr<raiden::PjRtCopyFuture> D2h(
      const std::vector<int64_t>& src_offsets,
      const std::vector<int64_t>& dst_offsets,
      const std::vector<int64_t>& copy_sizes) override {
    d2h_called_ = true;
    return raiden::PjRtCopyFuture({});
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2d(
      const std::vector<int64_t>& src_offsets,
      const std::vector<int64_t>& dst_offsets,
      const std::vector<int64_t>& copy_sizes) override {
    h2d_called_ = true;
    return raiden::PjRtCopyFuture({});
  }

  uint8_t* GetHostPointer(size_t layer_idx, size_t shard_idx) override {
    return dummy_host_buffer_;
  }
  size_t GetHostSize(size_t layer_idx, size_t shard_idx) override {
    return kDummyHostBufferSize;
  }

  bool d2h_called() const { return d2h_called_; }
  bool h2d_called() const { return h2d_called_; }

 private:
  // 16 blocks at block_size=1 * slice_byte_size=1024 = 1024 bytes/block, so
  // non-contiguous staging block ids (e.g. {0, 2}) stay in bounds.
  static constexpr size_t kDummyHostBufferSize = 16384;
  bool d2h_called_ = false;
  bool h2d_called_ = false;
  uint8_t dummy_host_buffer_[kDummyHostBufferSize] = {0};
};

TEST(DisaggKVCacheManagerBaseTest, E2EPushWorkflowMocked) {
  MockDisaggKVCacheManager prefill(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/1024,
      /*block_size=*/1, /*local_port=*/0, /*host_blocks_to_allocate=*/4,
      /*parallelism=*/1);

  MockDisaggKVCacheManager decode(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/1024,
      /*block_size=*/1, /*local_port=*/0, /*host_blocks_to_allocate=*/4,
      /*parallelism=*/1);

  ASSERT_TRUE(prefill.Start().ok());
  ASSERT_TRUE(decode.Start().ok());

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  int decode_zmq_port = decode.zmq_control_port();
  int decode_trans_port = decode.local_port().value();

  // Bootstrap peer registration
  prefill.RegisterPeer("decode", "127.0.0.1", decode_zmq_port,
                       decode_trans_port);

  absl::Notification prefill_done;
  absl::Notification decode_done;

  // 1. Submit Decode H2D Receive Request
  DisaggTransferRequest decode_req;
  decode_req.uuid = 2001;
  decode_req.type = DisaggTransferRequest::Type::kDecodeH2D;
  decode_req.dst_offsets = {0, 1};  // block_size=1: each chunk is one block
  decode_req.sizes = {1, 1};
  decode_req.callback = [&](std::optional<std::string> s) {
    EXPECT_FALSE(s.has_value());
    decode_done.Notify();
  };
  ASSERT_TRUE(decode.SubmitRequest(decode_req).ok());

  // 2. Submit Prefill D2H Send Request
  DisaggTransferRequest prefill_req;
  prefill_req.uuid = 2001;
  prefill_req.type = DisaggTransferRequest::Type::kPrefillD2H;
  prefill_req.src_offsets = {4, 5};
  prefill_req.dst_offsets = {0, 1};
  prefill_req.sizes = {1, 1};
  prefill_req.peer = "decode";
  prefill_req.callback = [&](std::optional<std::string> s) {
    EXPECT_FALSE(s.has_value());
    prefill_done.Notify();
  };
  ASSERT_TRUE(prefill.SubmitRequest(prefill_req).ok());

  // Wait for both to complete successfully E2E!
  EXPECT_TRUE(prefill_done.WaitForNotificationWithTimeout(absl::Seconds(5)));
  EXPECT_TRUE(decode_done.WaitForNotificationWithTimeout(absl::Seconds(5)));

  prefill.Stop();
  decode.Stop();

  EXPECT_TRUE(prefill.d2h_called());
  EXPECT_TRUE(decode.h2d_called());
}

TEST(DisaggKVCacheManagerBaseTest, E2EPullWorkflowMocked) {
  // PULL mode: prefill stages (D2H) and advertises readiness; the decode pulls
  // the blocks (real H2H Read over loopback) then loads them (H2D). D2h/H2d are
  // mocked; the H2H Read + the NOTIFY_READY/PULL_COMPLETE zmq handshake are real.
  MockDisaggKVCacheManager prefill(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/1024,
      /*block_size=*/1, /*local_port=*/0, /*host_blocks_to_allocate=*/4,
      /*parallelism=*/1);
  MockDisaggKVCacheManager decode(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/1024,
      /*block_size=*/1, /*local_port=*/0, /*host_blocks_to_allocate=*/4,
      /*parallelism=*/1);

  ASSERT_TRUE(prefill.Start().ok());
  ASSERT_TRUE(decode.Start().ok());
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Pull needs BOTH directions registered: prefill -> decode for NOTIFY_READY,
  // decode -> prefill for the H2H Read pull and the PULL_COMPLETE ack.
  prefill.RegisterPeer("decode", "127.0.0.1", decode.zmq_control_port(),
                       decode.local_port().value());
  decode.RegisterPeer("prefill", "127.0.0.1", prefill.zmq_control_port(),
                      prefill.local_port().value());

  absl::Notification prefill_done;
  absl::Notification decode_done;

  // 1. Decode submits a PULL receive (peer = the prefill to pull from).
  DisaggTransferRequest decode_req;
  decode_req.uuid = 3001;
  decode_req.type = DisaggTransferRequest::Type::kDecodeH2D;
  decode_req.pull_mode = true;
  decode_req.dst_offsets = {0, 1};  // block_size=1: each chunk is one block
  decode_req.sizes = {1, 1};
  decode_req.peer = "prefill";
  decode_req.callback = [&](std::optional<std::string> s) {
    EXPECT_FALSE(s.has_value());
    decode_done.Notify();
  };
  ASSERT_TRUE(decode.SubmitRequest(decode_req).ok());

  // 2. Prefill submits a PULL stage (peer = the decode to notify).
  DisaggTransferRequest prefill_req;
  prefill_req.uuid = 3001;
  prefill_req.type = DisaggTransferRequest::Type::kPrefillD2H;
  prefill_req.pull_mode = true;
  prefill_req.src_offsets = {4, 5};
  prefill_req.dst_offsets = {0, 1};
  prefill_req.sizes = {1, 1};
  prefill_req.peer = "decode";
  prefill_req.callback = [&](std::optional<std::string> s) {
    EXPECT_FALSE(s.has_value());
    prefill_done.Notify();
  };
  ASSERT_TRUE(prefill.SubmitRequest(prefill_req).ok());

  EXPECT_TRUE(prefill_done.WaitForNotificationWithTimeout(absl::Seconds(5)));
  EXPECT_TRUE(decode_done.WaitForNotificationWithTimeout(absl::Seconds(5)));

  prefill.Stop();
  decode.Stop();

  EXPECT_TRUE(prefill.d2h_called());  // prefill staged device -> host
  EXPECT_TRUE(decode.h2d_called());   // decode loaded host -> device after pull
}

TEST(DisaggKVCacheManagerBaseTest, E2EPullWorkflowNonContiguousMocked) {
  // PULL mode with NON-CONTIGUOUS remote staging blocks: the prefill stages at
  // dst_offsets {0, 2} (block_size=1 -> staging block ids {0, 2}, skipping
  // block 1). In pull mode those ids are advertised in NOTIFY_READY and become
  // the remote ids the decode pulls, so the real H2H Read must coalesce the
  // request into two separate single-block reads (remote 0 and remote 2). D2h/
  // H2d are mocked; this verifies the manager drives the non-contiguous pull to
  // completion without erroring or hanging.
  MockDisaggKVCacheManager prefill(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/1024,
      /*block_size=*/1, /*local_port=*/0, /*host_blocks_to_allocate=*/8,
      /*parallelism=*/1);
  MockDisaggKVCacheManager decode(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/1024,
      /*block_size=*/1, /*local_port=*/0, /*host_blocks_to_allocate=*/8,
      /*parallelism=*/1);

  ASSERT_TRUE(prefill.Start().ok());
  ASSERT_TRUE(decode.Start().ok());
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  prefill.RegisterPeer("decode", "127.0.0.1", decode.zmq_control_port(),
                       decode.local_port().value());
  decode.RegisterPeer("prefill", "127.0.0.1", prefill.zmq_control_port(),
                      prefill.local_port().value());

  absl::Notification prefill_done;
  absl::Notification decode_done;

  DisaggTransferRequest decode_req;
  decode_req.uuid = 3002;
  decode_req.type = DisaggTransferRequest::Type::kDecodeH2D;
  decode_req.pull_mode = true;
  decode_req.dst_offsets = {0, 2};  // staging blocks {0, 2}: non-contiguous
  decode_req.sizes = {1, 1};
  decode_req.peer = "prefill";
  decode_req.callback = [&](std::optional<std::string> s) {
    EXPECT_FALSE(s.has_value());
    decode_done.Notify();
  };
  ASSERT_TRUE(decode.SubmitRequest(decode_req).ok());

  DisaggTransferRequest prefill_req;
  prefill_req.uuid = 3002;
  prefill_req.type = DisaggTransferRequest::Type::kPrefillD2H;
  prefill_req.pull_mode = true;
  prefill_req.src_offsets = {4, 6};
  prefill_req.dst_offsets = {0, 2};
  prefill_req.sizes = {1, 1};
  prefill_req.peer = "decode";
  prefill_req.callback = [&](std::optional<std::string> s) {
    EXPECT_FALSE(s.has_value());
    prefill_done.Notify();
  };
  ASSERT_TRUE(prefill.SubmitRequest(prefill_req).ok());

  EXPECT_TRUE(prefill_done.WaitForNotificationWithTimeout(absl::Seconds(5)));
  EXPECT_TRUE(decode_done.WaitForNotificationWithTimeout(absl::Seconds(5)));

  prefill.Stop();
  decode.Stop();

  EXPECT_TRUE(prefill.d2h_called());
  EXPECT_TRUE(decode.h2d_called());
}

// -----------------------------------------------------------------------------
// CPU-only unit tests that exercise individual base-class surfaces without
// needing the full E2E push flow (and without PJRT). They use the same
// MockDisaggKVCacheManager subclass declared above to bypass real H2D/D2H.
// -----------------------------------------------------------------------------

namespace {
std::unique_ptr<MockDisaggKVCacheManager> MakeMock() {
  return std::make_unique<MockDisaggKVCacheManager>(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/1024,
      /*block_size=*/1, /*local_port=*/0, /*host_blocks_to_allocate=*/4,
      /*parallelism=*/1);
}
}  // namespace

TEST(DisaggKVCacheManagerBaseTest, StartAllocatesEphemeralZmqPort) {
  auto m = MakeMock();
  ASSERT_TRUE(m->Start().ok());
  EXPECT_GT(m->zmq_control_port(), 0);
  m->Stop();
}

TEST(DisaggKVCacheManagerBaseTest, StartAllocatesLocalDataPort) {
  auto m = MakeMock();
  ASSERT_TRUE(m->Start().ok());
  ASSERT_TRUE(m->local_port().has_value());
  EXPECT_GT(*m->local_port(), 0);
  m->Stop();
}

TEST(DisaggKVCacheManagerBaseTest, DoubleStartIsIdempotent) {
  auto m = MakeMock();
  ASSERT_TRUE(m->Start().ok());
  int port = m->zmq_control_port();
  ASSERT_TRUE(m->Start().ok());  // second call: running_ already true
  EXPECT_EQ(m->zmq_control_port(), port);
  m->Stop();
}

TEST(DisaggKVCacheManagerBaseTest, DoubleStopIsIdempotent) {
  auto m = MakeMock();
  ASSERT_TRUE(m->Start().ok());
  m->Stop();
  m->Stop();  // second Stop must be a no-op, not crash
}

TEST(DisaggKVCacheManagerBaseTest, TwoInstancesGetDistinctPorts) {
  auto m1 = MakeMock();
  auto m2 = MakeMock();
  ASSERT_TRUE(m1->Start().ok());
  ASSERT_TRUE(m2->Start().ok());
  EXPECT_NE(m1->zmq_control_port(), m2->zmq_control_port());
  ASSERT_TRUE(m1->local_port().has_value());
  ASSERT_TRUE(m2->local_port().has_value());
  EXPECT_NE(*m1->local_port(), *m2->local_port());
  m1->Stop();
  m2->Stop();
}

TEST(DisaggKVCacheManagerBaseTest, RepeatedStartStopCyclesAreSafe) {
  auto m = MakeMock();
  for (int i = 0; i < 3; ++i) {
    ASSERT_TRUE(m->Start().ok());
    EXPECT_GT(m->zmq_control_port(), 0);
    m->Stop();
  }
}

TEST(DisaggKVCacheManagerBaseTest, SubmitBeforeStartReturnsFailedPrecondition) {
  auto m = MakeMock();
  DisaggTransferRequest req;
  req.uuid = 1;
  req.type = DisaggTransferRequest::Type::kDecodeH2D;
  req.dst_offsets = {0};
  req.sizes = {1};
  absl::Status s = m->SubmitRequest(req);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), absl::StatusCode::kFailedPrecondition);
}

TEST(DisaggKVCacheManagerBaseTest, SubmitAfterStopReturnsFailedPrecondition) {
  auto m = MakeMock();
  ASSERT_TRUE(m->Start().ok());
  m->Stop();
  DisaggTransferRequest req;
  req.uuid = 2;
  req.type = DisaggTransferRequest::Type::kDecodeH2D;
  req.dst_offsets = {0};
  req.sizes = {1};
  absl::Status s = m->SubmitRequest(req);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), absl::StatusCode::kFailedPrecondition);
}

TEST(DisaggKVCacheManagerBaseTest, RegisterPeerAcceptsMultiplePeers) {
  auto m = MakeMock();
  ASSERT_TRUE(m->Start().ok());
  m->RegisterPeer("peer_a", "127.0.0.1", 9100, 9101);
  m->RegisterPeer("peer_b", "127.0.0.1", 9200, 9201);
  m->RegisterPeer("peer_c", "127.0.0.1", 9300, 9301);
  m->Stop();
}

TEST(DisaggKVCacheManagerBaseTest, RegisterPeerOverwritesExistingName) {
  auto m = MakeMock();
  ASSERT_TRUE(m->Start().ok());
  m->RegisterPeer("peer_x", "127.0.0.1", 1, 2);
  m->RegisterPeer("peer_x", "127.0.0.1", 3, 4);  // overwrite
  m->Stop();
}

TEST(DisaggKVCacheManagerBaseTest, DestructorStopsRunningManager) {
  // ~DisaggKVCacheManagerBase() invokes Stop(); going out of scope must be
  // safe even if the user forgot to call Stop() explicitly.
  int port = 0;
  {
    auto m = MakeMock();
    ASSERT_TRUE(m->Start().ok());
    port = m->zmq_control_port();
    EXPECT_GT(port, 0);
    // m goes out of scope here; destructor must Stop() without hang/segv.
  }
}

TEST(DisaggKVCacheManagerBaseTest, DecodeRequestPendsUntilPeerNotification) {
  // Per the orchestrator contract, a DECODE_H2D request is queued and held
  // pending the peer's NOTIFY_COMPLETE; it must NOT fire H2d on its own.
  // (See OrchestrationLoop: "Waiting for peer notification to trigger H2D".)
  auto m = MakeMock();
  ASSERT_TRUE(m->Start().ok());
  absl::Notification done;
  DisaggTransferRequest req;
  req.uuid = 4242;
  req.type = DisaggTransferRequest::Type::kDecodeH2D;
  req.dst_offsets = {0};
  req.sizes = {1};
  req.callback = [&](std::optional<std::string> s) { done.Notify(); };
  ASSERT_TRUE(m->SubmitRequest(req).ok());
  // Without a peer notification, the request stays pending.
  EXPECT_FALSE(done.WaitForNotificationWithTimeout(absl::Milliseconds(200)));
  EXPECT_FALSE(m->h2d_called());
  // Stop() must still unblock cleanly while a request is queued.
  m->Stop();
}

// -----------------------------------------------------------------------------
// ThreadSafeQueue (declared in the same public header).
// -----------------------------------------------------------------------------
TEST(ThreadSafeQueueTest, PushPopOrderingIsFifo) {
  ThreadSafeQueue<int> q;
  q.Push(1);
  q.Push(2);
  q.Push(3);
  int v = 0;
  ASSERT_TRUE(q.Pop(v));
  EXPECT_EQ(v, 1);
  ASSERT_TRUE(q.Pop(v));
  EXPECT_EQ(v, 2);
  ASSERT_TRUE(q.Pop(v));
  EXPECT_EQ(v, 3);
  q.Shutdown();
  EXPECT_FALSE(q.Pop(v));  // empty + shutdown -> false
}

TEST(ThreadSafeQueueTest, ShutdownUnblocksPendingPop) {
  ThreadSafeQueue<int> q;
  absl::Notification popped;
  std::thread t([&] {
    int v = 0;
    bool ok = q.Pop(v);
    EXPECT_FALSE(ok);  // returns false because queue is empty + shutdown
    popped.Notify();
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  q.Shutdown();
  EXPECT_TRUE(popped.WaitForNotificationWithTimeout(absl::Seconds(2)));
  t.join();
}

TEST(ThreadSafeQueueTest, PostShutdownPushIsDrainedBeforeFalse) {
  ThreadSafeQueue<int> q;
  q.Push(7);
  q.Push(8);
  q.Shutdown();
  int v = 0;
  ASSERT_TRUE(q.Pop(v));  // drains existing items first
  EXPECT_EQ(v, 7);
  ASSERT_TRUE(q.Pop(v));
  EXPECT_EQ(v, 8);
  EXPECT_FALSE(q.Pop(v));  // now empty + shutdown
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
