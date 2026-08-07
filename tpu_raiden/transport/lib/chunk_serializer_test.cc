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

#include "tpu_raiden/transport/lib/chunk_serializer.h"

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "tpu_raiden/transport/lib/chunk.h"

namespace tpu_raiden::transport::lib {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

ChunkHeader MakeSampleHeaderV1() {
  return ChunkHeader{
      .version = 1,
      .op = 0xAB,
      .flags = 0xCD,
      .buffer_id = 0x1234,
      .reserved = 0x5678,
      .remote_id = 0x12345678,
      .local_id = 0x9ABCDEF0,
      .count_or_size = 0x11223344,
      .uuid = 0x0123456789ABCDEFULL,
  };
}

ChunkMetadata MakeSampleMetadataV1() {
  return ChunkMetadata{
      .layer_idx = 0x12345678,
      .dst_shard_idx = 0x9ABCDEF0,
      .dst_offset_bytes = 0x11223344,
      .size_bytes = 0x55667788,
  };
}

TEST(ChunkHeaderSerializerTest, SerializeAndDeserialize) {
  const ChunkHeader original = MakeSampleHeaderV1();
  const std::string bytes = SerializeChunkHeader(original);

  EXPECT_THAT(DeserializeChunkHeader(bytes), IsOkAndHolds(original));
}

TEST(ChunkHeaderSerializerTest, SerializeToLittleEndian) {
  const std::string wire = SerializeChunkHeader(MakeSampleHeaderV1());
  ASSERT_EQ(wire.size(), kChunkHeaderSize);

  alignas(8) const uint8_t expected_wire[64] = {
      0x52, 0x44, 0x01, 0x00, 0xAB, 0xCD, 0x34, 0x12, 0x78, 0x56, 0x00,
      0x00, 0x78, 0x56, 0x34, 0x12, 0xF0, 0xDE, 0xBC, 0x9A, 0x44, 0x33,
      0x22, 0x11, 0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01,
  };

  EXPECT_EQ(wire,
            absl::string_view(reinterpret_cast<const char*>(expected_wire),
                              sizeof(expected_wire)));
}

TEST(ChunkHeaderSerializerTest, VerifyMagicBytes) {
  const std::string s = SerializeChunkHeader(MakeSampleHeaderV1());
  ASSERT_GE(s.size(), 2);
  ASSERT_EQ(s[0], 'R');
  ASSERT_EQ(s[1], 'D');
}

TEST(ChunkHeaderSerializerTest, DeserializeLittleEndian) {
  alignas(8) const uint8_t raw_wire[64] = {
      0x52, 0x44, 0x01, 0x00, 0xAB, 0xCD, 0x34, 0x12, 0x78, 0x56, 0x00,
      0x00, 0x78, 0x56, 0x34, 0x12, 0xF0, 0xDE, 0xBC, 0x9A, 0x44, 0x33,
      0x22, 0x11, 0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01,
  };

  const std::string wire(reinterpret_cast<const char*>(raw_wire),
                         sizeof(raw_wire));

  EXPECT_THAT(DeserializeChunkHeader(wire), IsOkAndHolds(MakeSampleHeaderV1()));
}

TEST(ChunkHeaderSerializerTest, DeserializeRejectsInvalidMagic) {
  std::string bytes = SerializeChunkHeader(MakeSampleHeaderV1());
  bytes[0] ^= 0xFF;  // Corrupt the magic field.

  EXPECT_THAT(DeserializeChunkHeader(bytes),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ChunkHeaderSerializerTest, DeserializeRejectsInvalidVersion) {
  std::string bytes = SerializeChunkHeader(MakeSampleHeaderV1());
  // The `ver` field is a little-endian uint16.
  bytes[2] = 0x04;
  bytes[3] = 0x00;

  EXPECT_THAT(DeserializeChunkHeader(bytes),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(ChunkMetadataSerializerTest, SerializeAndDeserialize) {
  const ChunkMetadata original = MakeSampleMetadataV1();
  const std::string bytes = SerializeChunkMetadata(original);

  EXPECT_THAT(DeserializeChunkMetadata(bytes, /*ver=*/1),
              IsOkAndHolds(original));
}

TEST(ChunkMetadataSerializerTest, SerializeToLittleEndian) {
  const std::string wire = SerializeChunkMetadata(MakeSampleMetadataV1());
  ASSERT_EQ(wire.size(), GetChunkMetadataSize(1));

  alignas(4) const uint8_t expected_wire[16] = {
      0x78, 0x56, 0x34, 0x12, 0xF0, 0xDE, 0xBC, 0x9A,
      0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55,
  };

  EXPECT_EQ(wire,
            absl::string_view(reinterpret_cast<const char*>(expected_wire),
                              sizeof(expected_wire)));
}

TEST(ChunkMetadataSerializerTest, DeserializeLittleEndian) {
  alignas(4) const uint8_t raw_wire[16] = {
      0x78, 0x56, 0x34, 0x12, 0xF0, 0xDE, 0xBC, 0x9A,
      0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55,
  };

  const std::string wire(reinterpret_cast<const char*>(raw_wire),
                         sizeof(raw_wire));

  EXPECT_THAT(DeserializeChunkMetadata(wire, /*ver=*/1),
              IsOkAndHolds(MakeSampleMetadataV1()));
}

}  // namespace
}  // namespace tpu_raiden::transport::lib
