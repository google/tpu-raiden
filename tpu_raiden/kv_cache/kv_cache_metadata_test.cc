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

#include "tpu_raiden/kv_cache/kv_cache_metadata.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

using ::testing::ElementsAre;
using ::testing::FieldsAre;
using ::testing::IsEmpty;

// 64-byte aligned backing buffer standing in for the shared memory region.
class Region {
 public:
  explicit Region(int num_blocks)
      : buffer_(KVCacheMetadata::RequiredSizeBytes(num_blocks) + 63) {}

  absl::Span<uint8_t> span() {
    auto addr = reinterpret_cast<uintptr_t>(buffer_.data());
    size_t offset = (64 - addr % 64) % 64;
    return absl::MakeSpan(buffer_.data() + offset, buffer_.size() - offset);
  }

 private:
  std::vector<uint8_t> buffer_;
};

TEST(KVCacheMetadataTest, FormatCreatesEmptyTable) {
  Region region(4);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 4);
  ASSERT_TRUE(metadata_or.ok());
  EXPECT_EQ(metadata_or->num_blocks(), 4);
  EXPECT_THAT(metadata_or->ValidEntries(), IsEmpty());
}

TEST(KVCacheMetadataTest, FormatRejectsInvalidRegions) {
  Region region(4);
  EXPECT_EQ(KVCacheMetadata::Format(region.span(), 0).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(KVCacheMetadata::Format(absl::Span<uint8_t>(), 4).status().code(),
            absl::StatusCode::kInvalidArgument);
  // Too small for 8 blocks.
  EXPECT_EQ(
      KVCacheMetadata::Format(region.span().subspan(0, 128), 8).status().code(),
      absl::StatusCode::kInvalidArgument);
  // Misaligned.
  EXPECT_EQ(
      KVCacheMetadata::Format(region.span().subspan(1), 2).status().code(),
      absl::StatusCode::kInvalidArgument);
}

TEST(KVCacheMetadataTest, SetAndClearRoundTrip) {
  Region region(4);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 4);
  ASSERT_TRUE(metadata_or.ok());

  ASSERT_TRUE(metadata_or->Set(1, "hash_b", /*seq=*/2).ok());
  ASSERT_TRUE(metadata_or->Set(3, "hash_d", /*seq=*/1).ok());
  EXPECT_THAT(
      metadata_or->ValidEntries(),
      ElementsAre(FieldsAre(1, "hash_b", 2), FieldsAre(3, "hash_d", 1)));

  ASSERT_TRUE(metadata_or->Clear(1).ok());
  EXPECT_THAT(metadata_or->ValidEntries(),
              ElementsAre(FieldsAre(3, "hash_d", 1)));
}

TEST(KVCacheMetadataTest, SetOverwritesPreviousBinding) {
  Region region(2);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 2);
  ASSERT_TRUE(metadata_or.ok());

  ASSERT_TRUE(metadata_or->Set(0, "old_hash", /*seq=*/1).ok());
  ASSERT_TRUE(metadata_or->Set(0, "new", /*seq=*/2).ok());
  EXPECT_THAT(metadata_or->ValidEntries(), ElementsAre(FieldsAre(0, "new", 2)));
}

TEST(KVCacheMetadataTest, SetValidatesArguments) {
  Region region(2);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 2);
  ASSERT_TRUE(metadata_or.ok());

  EXPECT_EQ(metadata_or->Set(-1, "h", 0).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(metadata_or->Set(2, "h", 0).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(metadata_or->Set(0, "", 0).code(),
            absl::StatusCode::kInvalidArgument);
  std::string too_long(KVCacheMetadata::kMaxHashLength + 1, 'x');
  EXPECT_EQ(metadata_or->Set(0, too_long, 0).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(metadata_or->Clear(2).code(), absl::StatusCode::kInvalidArgument);

  std::string max_length(KVCacheMetadata::kMaxHashLength, 'y');
  EXPECT_TRUE(metadata_or->Set(0, max_length, 0).ok());
  EXPECT_THAT(metadata_or->ValidEntries(),
              ElementsAre(FieldsAre(0, max_length, 0)));
}

TEST(KVCacheMetadataTest, HashesAreOpaqueBytes) {
  Region region(1);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 1);
  ASSERT_TRUE(metadata_or.ok());

  std::string binary_hash("\x00\xff\x00raiden\x01", 10);
  ASSERT_TRUE(metadata_or->Set(0, binary_hash, /*seq=*/7).ok());
  EXPECT_THAT(metadata_or->ValidEntries(),
              ElementsAre(FieldsAre(0, binary_hash, 7)));
}

TEST(KVCacheMetadataTest, AttachRecoversEntriesFromSurvivingRegion) {
  Region region(4);
  {
    auto metadata_or = KVCacheMetadata::Format(region.span(), 4);
    ASSERT_TRUE(metadata_or.ok());
    ASSERT_TRUE(metadata_or->Set(0, "hash_a", /*seq=*/3).ok());
    ASSERT_TRUE(metadata_or->Set(2, "hash_c", /*seq=*/4).ok());
    // The view is dropped here; the region survives, as shared memory would
    // across an engine crash.
  }

  auto recovered_or = KVCacheMetadata::Attach(region.span(), 4);
  ASSERT_TRUE(recovered_or.ok());
  EXPECT_THAT(
      recovered_or->ValidEntries(),
      ElementsAre(FieldsAre(0, "hash_a", 3), FieldsAre(2, "hash_c", 4)));
}

TEST(KVCacheMetadataTest, AttachRejectsUnformattedOrMismatchedRegions) {
  // Sized for 8 blocks so the mismatch below fails on the header contents,
  // not on the region size.
  Region region(8);

  // Never formatted.
  std::memset(region.span().data(), 0, KVCacheMetadata::RequiredSizeBytes(4));
  EXPECT_EQ(KVCacheMetadata::Attach(region.span(), 4).status().code(),
            absl::StatusCode::kFailedPrecondition);

  ASSERT_TRUE(KVCacheMetadata::Format(region.span(), 4).ok());
  // Formatted for 4 blocks, attached expecting 8 (block pool resized across
  // the restart): the table no longer matches the pool, treat as cold start.
  EXPECT_EQ(KVCacheMetadata::Attach(region.span(), 8).status().code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_TRUE(KVCacheMetadata::Attach(region.span(), 4).ok());
}

TEST(KVCacheMetadataTest, FormatWipesSurvivingEntries) {
  Region region(2);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 2);
  ASSERT_TRUE(metadata_or.ok());
  ASSERT_TRUE(metadata_or->Set(0, "stale", /*seq=*/1).ok());

  auto reformatted_or = KVCacheMetadata::Format(region.span(), 2);
  ASSERT_TRUE(reformatted_or.ok());
  EXPECT_THAT(reformatted_or->ValidEntries(), IsEmpty());
}

TEST(KVCacheMetadataTest, UncommittedEntryIsInvisible) {
  Region region(2);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 2);
  ASSERT_TRUE(metadata_or.ok());

  // Simulate a crash after the hash bytes landed but before the entry was
  // committed: write the fields directly and leave `valid` unset.
  auto* entry = reinterpret_cast<KVCacheMetadataEntry*>(
      region.span().data() + sizeof(KVCacheMetadataHeader));
  entry->seq = 9;
  entry->hash_len = 4;
  std::memcpy(entry->hash, "torn", 4);

  auto recovered_or = KVCacheMetadata::Attach(region.span(), 2);
  ASSERT_TRUE(recovered_or.ok());
  EXPECT_THAT(recovered_or->ValidEntries(), IsEmpty());
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
