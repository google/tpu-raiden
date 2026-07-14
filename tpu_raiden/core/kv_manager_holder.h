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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_KV_MANAGER_HOLDER_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_KV_MANAGER_HOLDER_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/core/status_macros.h"

namespace tpu_raiden {

// Type-erased wrapper for any KV Cache Manager or Transfer Manager
// implementation that provides asynchronous D2H and H2D transfers.
class KVManagerHolder {
 public:
  class Concept {
   public:
    virtual ~Concept() = default;
    virtual absl::StatusOr<raiden::PjRtCopyFuture> D2h(
        const std::vector<int64_t>& src_offsets,
        const std::vector<int64_t>& dst_offsets,
        const std::vector<int64_t>& copy_sizes) = 0;
    virtual absl::StatusOr<raiden::PjRtCopyFuture> H2d(
        const std::vector<int64_t>& src_offsets,
        const std::vector<int64_t>& dst_offsets,
        const std::vector<int64_t>& copy_sizes) = 0;
    virtual absl::StatusOr<raiden::PjRtCopyFuture> H2hWrite(
        absl::string_view peer, const std::vector<int64_t>& src_offsets,
        const std::vector<int64_t>& dst_offsets) = 0;
  };

  template <typename T>
  class Model final : public Concept {
   public:
    explicit Model(T* impl) : impl_(impl) {}
    absl::StatusOr<raiden::PjRtCopyFuture> D2h(
        const std::vector<int64_t>& src_offsets,
        const std::vector<int64_t>& dst_offsets,
        const std::vector<int64_t>& copy_sizes) override {
      return impl_->D2h(src_offsets, dst_offsets, copy_sizes);
    }
    absl::StatusOr<raiden::PjRtCopyFuture> H2d(
        const std::vector<int64_t>& src_offsets,
        const std::vector<int64_t>& dst_offsets,
        const std::vector<int64_t>& copy_sizes) override {
      return impl_->H2d(src_offsets, dst_offsets, copy_sizes);
    }
    absl::StatusOr<raiden::PjRtCopyFuture> H2hWrite(
        absl::string_view peer, const std::vector<int64_t>& src_offsets,
        const std::vector<int64_t>& dst_offsets) override {
      ASSIGN_OR_RETURN(std::vector<int> src_ids, SafeCastOffsets(src_offsets));
      ASSIGN_OR_RETURN(std::vector<int> dst_ids, SafeCastOffsets(dst_offsets));
      ASSIGN_OR_RETURN(auto res,
                       impl_->H2hWrite(std::string(peer), src_ids, dst_ids));
      return res.second;
    }

   private:
    absl::StatusOr<std::vector<int>> SafeCastOffsets(
        const std::vector<int64_t>& offsets) {
      std::vector<int> ids;
      ids.reserve(offsets.size());
      for (int64_t offset : offsets) {
        if (offset < std::numeric_limits<int>::min() ||
            offset > std::numeric_limits<int>::max()) {
          return absl::InvalidArgumentError(
              absl::StrCat("Offset ", offset, " overflows int"));
        }
        ids.push_back(static_cast<int>(offset));
      }
      return ids;
    }
    T* impl_;
  };

  KVManagerHolder() : self_(nullptr) {}

  KVManagerHolder(std::nullptr_t) : self_(nullptr) {}  // NOLINT

  template <typename T>
  KVManagerHolder(T* impl)  // NOLINT
      : self_(impl ? std::make_unique<Model<T>>(impl) : nullptr) {}

  absl::StatusOr<raiden::PjRtCopyFuture> D2h(
      const std::vector<int64_t>& src_offsets,
      const std::vector<int64_t>& dst_offsets,
      const std::vector<int64_t>& copy_sizes) const {
    if (!self_) {
      return absl::InternalError("KVManagerHolder is null");
    }
    return self_->D2h(src_offsets, dst_offsets, copy_sizes);
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2d(
      const std::vector<int64_t>& src_offsets,
      const std::vector<int64_t>& dst_offsets,
      const std::vector<int64_t>& copy_sizes) const {
    if (!self_) {
      return absl::InternalError("KVManagerHolder is null");
    }
    return self_->H2d(src_offsets, dst_offsets, copy_sizes);
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2hWrite(
      absl::string_view peer, const std::vector<int64_t>& src_offsets,
      const std::vector<int64_t>& dst_offsets) const {
    if (!self_) {
      return absl::InternalError("KVManagerHolder is null");
    }
    return self_->H2hWrite(peer, src_offsets, dst_offsets);
  }

  explicit operator bool() const { return self_ != nullptr; }
  bool operator==(std::nullptr_t) const { return self_ == nullptr; }
  bool operator!=(std::nullptr_t) const { return self_ != nullptr; }

 private:
  std::unique_ptr<Concept> self_;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_KV_MANAGER_HOLDER_H_
