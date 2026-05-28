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
    return 4096;
  }

  bool d2h_called() const { return d2h_called_; }
  bool h2d_called() const { return h2d_called_; }

 private:
  bool d2h_called_ = false;
  bool h2d_called_ = false;
  uint8_t dummy_host_buffer_[4096] = {0};
};

TEST(DisaggKVCacheManagerBaseTest, E2EPushWorkflowMocked) {
  MockDisaggKVCacheManager prefill(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/1024,
      /*block_size=*/2, /*local_port=*/0, /*host_blocks_to_allocate=*/4,
      /*parallelism=*/1);

  MockDisaggKVCacheManager decode(
      /*num_layers=*/1, /*num_shards=*/1, /*slice_byte_size=*/1024,
      /*block_size=*/2, /*local_port=*/0, /*host_blocks_to_allocate=*/4,
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
  decode_req.request_id = 2001;
  decode_req.type = DisaggTransferRequest::Type::kDecodeH2D;
  decode_req.dst_offsets = {0, 2};
  decode_req.sizes = {2, 2};
  decode_req.callback = [&](absl::Status s) {
    EXPECT_TRUE(s.ok());
    decode_done.Notify();
  };
  ASSERT_TRUE(decode.SubmitRequest(decode_req).ok());

  // 2. Submit Prefill D2H Send Request
  DisaggTransferRequest prefill_req;
  prefill_req.request_id = 2001;
  prefill_req.type = DisaggTransferRequest::Type::kPrefillD2H;
  prefill_req.src_offsets = {4, 6};
  prefill_req.dst_offsets = {0, 2};
  prefill_req.sizes = {2, 2};
  prefill_req.peer = "decode";
  prefill_req.callback = [&](absl::Status s) {
    EXPECT_TRUE(s.ok());
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

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
