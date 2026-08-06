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

#ifndef THIRD_PARTY_TPU_RAIDEN_TRANSPORT_LIB_CHUNK_SERIALIZER_H_
#define THIRD_PARTY_TPU_RAIDEN_TRANSPORT_LIB_CHUNK_SERIALIZER_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tpu_raiden/transport/lib/chunk.h"
#include "tpu_raiden/transport/lib/chunk_generated.h"

namespace tpu_raiden::transport::lib {

inline constexpr size_t kChunkHeaderSize = 64;

static_assert(sizeof(flatbuf::ChunkHeader) == kChunkHeaderSize);

// Returns the size of a chunk metadata for the given version.
constexpr size_t GetChunkMetadataSize(uint16_t ver) {
  switch (ver) {
    case 1:
      return 16;
    default:
      return 0;
  }
}

// Serializes the chunk header to a binary string.
std::string SerializeChunkHeader(const ChunkHeader& header);

// Parses the chunk header from its serialized binary string.
absl::StatusOr<ChunkHeader> DeserializeChunkHeader(absl::string_view s);

// Serializes the chunk metadata to a binary string.
std::string SerializeChunkMetadata(const ChunkMetadata& meta);

// Parses the chunk metadata from its serialized binary string.
absl::StatusOr<ChunkMetadata> DeserializeChunkMetadata(absl::string_view s,
                                                       uint16_t ver);

}  // namespace tpu_raiden::transport::lib

#endif  // THIRD_PARTY_TPU_RAIDEN_TRANSPORT_LIB_CHUNK_SERIALIZER_H_
