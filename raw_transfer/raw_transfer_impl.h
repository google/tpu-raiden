/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef JAXLIB_PY_RAW_TRANSFER_IMPL_H_
#define JAXLIB_PY_RAW_TRANSFER_IMPL_H_

#include <Python.h>

#include <iostream>
#include <optional>
#include <typeinfo>
#include <vector>

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "jaxlib/py_array.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_external.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/shape_util.h"
#include "tsl/platform/logging.h"

namespace nb = nanobind;

struct RawBufferHolder {
  const PJRT_Api* c_api;
  const PJRT_RawBuffer_Extension* extension;
  PJRT_RawBuffer* buffer;

  RawBufferHolder(const PJRT_Api* api, const PJRT_RawBuffer_Extension* ext,
                  PJRT_RawBuffer* buf)
      : c_api(api), extension(ext), buffer(buf) {}

  ~RawBufferHolder() {
    if (buffer) {
      pjrt::PjRtCApiRawBuffer_Destroy(c_api, extension, buffer);
    }
  }
};

namespace jax {

using xla::CommonPjRtBuffer;
using xla::CommonPjRtRawBuffer;
using xla::PjRtBuffer;
using xla::PjRtCApiBuffer;
using xla::PjRtCApiClient;

struct PyArrayObject {
  PyObject_HEAD;
#if PY_VERSION_HEX < 0x030C0000
  PyObject* weakrefs;
  PyObject* dict;
#endif  // PY_VERSION_HEX < 0x030C0000
  bool initialized;
  alignas(PyArray::Storage) char array_storage[sizeof(PyArray::Storage)];
};

inline PyArray::Storage* GetPyArrayStorageFromObject(
    PyArrayObject* py_array_object) {
  return std::launder(
      reinterpret_cast<PyArray::Storage*>(py_array_object->array_storage));
}

inline xla::PjRtBuffer* GetPjrtBufferFromPyObject(PyObject* obj) {
  auto* py_array_obj = reinterpret_cast<PyArrayObject*>(obj);
  if (!py_array_obj->initialized) {
    throw std::runtime_error("PyArrayObject not initialized");
  }
  auto* storage = GetPyArrayStorageFromObject(py_array_obj);
  xla::ifrt::Array* ifrt_array = storage->ifrt_array.get();

  auto* arr =
      dynamic_cast<xla::ifrt::PjRtCompatibleArray*>(ifrt_array);
  if (arr == nullptr) {
    std::string type_name = ifrt_array ? typeid(*ifrt_array).name() : "nullptr";
    throw std::runtime_error(std::string("Not a PjRt compatible array, type: ") + type_name);
  }
  return arr->pjrt_buffers().front().get();
}

inline xla::ifrt::Array* GetIfrtArrayFromPyObject(PyObject* obj) {
  auto* py_array_obj = reinterpret_cast<PyArrayObject*>(obj);
  if (!py_array_obj->initialized) {
    throw std::runtime_error("PyArrayObject not initialized");
  }
  auto* storage = GetPyArrayStorageFromObject(py_array_obj);
  return storage->ifrt_array.get();
}

class PjRtCopyFuture {
 public:
  // Default constructor for nanobind binding
  PjRtCopyFuture() : futures_(), c_api_holds_(), holds_(), ext_holds_() {}

  PjRtCopyFuture(
      std::vector<xla::Future<>> futures,
      std::vector<std::shared_ptr<RawBufferHolder>> c_api_holds = {},
      std::vector<std::shared_ptr<CommonPjRtBuffer::ScopedHold>> holds = {})
      : futures_(std::move(futures)),
        c_api_holds_(std::move(c_api_holds)),
        holds_(std::move(holds)) {}

  PjRtCopyFuture(
      std::vector<xla::Future<>> futures,
      std::shared_ptr<RawBufferHolder> c_api_hold,
      std::shared_ptr<CommonPjRtBuffer::ScopedHold> hold = nullptr)
      : futures_(std::move(futures)) {
    if (c_api_hold) c_api_holds_.push_back(std::move(c_api_hold));
    if (hold) holds_.push_back(std::move(hold));
  }

  void Await() {
    nb::gil_scoped_release release;
    for (auto& f : futures_) {
      absl::Status status = f.Await();
      if (!status.ok()) {
        throw std::runtime_error(std::string("Async copy failed: ") +
                                 std::string(status.message()));
      }
    }
    futures_.clear();
    c_api_holds_.clear();
    holds_.clear();
    ext_holds_.clear();
  }

  void Append(PjRtCopyFuture other) {
    futures_.insert(futures_.end(),
                    std::make_move_iterator(other.futures_.begin()),
                    std::make_move_iterator(other.futures_.end()));
    c_api_holds_.insert(c_api_holds_.end(),
                        std::make_move_iterator(other.c_api_holds_.begin()),
                        std::make_move_iterator(other.c_api_holds_.end()));
    holds_.insert(holds_.end(), std::make_move_iterator(other.holds_.begin()),
                  std::make_move_iterator(other.holds_.end()));
    ext_holds_.insert(ext_holds_.end(),
                      std::make_move_iterator(other.ext_holds_.begin()),
                      std::make_move_iterator(other.ext_holds_.end()));
  }

  void Append(
      std::vector<xla::Future<>> other_futures,
      std::shared_ptr<RawBufferHolder> other_c_api_hold = nullptr,
      std::shared_ptr<CommonPjRtBuffer::ScopedHold> other_hold = nullptr) {
    futures_.insert(futures_.end(),
                    std::make_move_iterator(other_futures.begin()),
                    std::make_move_iterator(other_futures.end()));
    if (other_c_api_hold) c_api_holds_.push_back(std::move(other_c_api_hold));
    if (other_hold) holds_.push_back(std::move(other_hold));
  }

  void Append(
      std::vector<xla::Future<>> other_futures,
      std::vector<std::shared_ptr<RawBufferHolder>> other_c_api_holds,
      std::vector<std::shared_ptr<CommonPjRtBuffer::ScopedHold>> other_holds) {
    futures_.insert(futures_.end(),
                    std::make_move_iterator(other_futures.begin()),
                    std::make_move_iterator(other_futures.end()));
    c_api_holds_.insert(c_api_holds_.end(),
                        std::make_move_iterator(other_c_api_holds.begin()),
                        std::make_move_iterator(other_c_api_holds.end()));
    holds_.insert(holds_.end(), std::make_move_iterator(other_holds.begin()),
                  std::make_move_iterator(other_holds.end()));
  }

  void AddExtHold(std::unique_ptr<xla::PjRtBuffer::ExternalReference> hold) {
    ext_holds_.push_back(std::move(hold));
  }

 private:
  std::vector<xla::Future<>> futures_;
  std::vector<std::shared_ptr<RawBufferHolder>> c_api_holds_;
  std::vector<std::shared_ptr<CommonPjRtBuffer::ScopedHold>> holds_;
  std::vector<std::shared_ptr<xla::PjRtBuffer::ExternalReference>> ext_holds_;
};

inline void transfer_d2h_internal(const nb::object& src_arr,
                                  const nb::object& dst_arr,
                                  const nb::list& src_offsets_major_dim,
                                  const nb::list& dst_offsets_major_dim,
                                  const nb::list& copy_sizes_major_dim,
                                  PjRtCopyFuture& acc) {
  nb::object addressable_shards = src_arr.attr("addressable_shards");
  size_t num_shards = nb::len(addressable_shards);

  if (num_shards == 0) {
    return;
  }

  if (src_offsets_major_dim.size() != dst_offsets_major_dim.size() ||
      src_offsets_major_dim.size() != copy_sizes_major_dim.size()) {
    throw std::runtime_error("Lengths of offset and size lists must match");
  }

  nb::object first_shard_data = addressable_shards[0].attr("data");
  PjRtBuffer* first_buffer =
      jax::GetPjrtBufferFromPyObject(first_shard_data.ptr());
  const xla::Shape& shape = first_buffer->on_device_shape();

  bool is_partial = false;
  int64_t full_major_dim_size = shape.dimensions(0);
  for (size_t i = 0; i < src_offsets_major_dim.size(); ++i) {
    if (nb::cast<int64_t>(src_offsets_major_dim[i]) != 0 ||
        nb::cast<int64_t>(dst_offsets_major_dim[i]) != 0 ||
        nb::cast<int64_t>(copy_sizes_major_dim[i]) != full_major_dim_size) {
      is_partial = true;
      break;
    }
  }

  if (is_partial) {
    if (shape.dimensions_size() < 3) {
      throw std::runtime_error(
          "Only support arrays with rank >= 3 for partial copies");
    }
    nb::object sharding = src_arr.attr("sharding");
    nb::object NamedSharding =
        nb::module_::import_("jax.sharding").attr("NamedSharding");
    if (nb::isinstance(sharding, NamedSharding)) {
      nb::object spec = sharding.attr("spec");
      if (nb::len(spec) > 0) {
        nb::object first_axis = spec[0];
        if (!first_axis.is_none()) {
          throw nb::value_error(
              "Partial copy not supported for arrays sharded on major "
              "dimension");
        }
      }
    }
  }

  nb::object dst_addressable_shards = dst_arr.attr("addressable_shards");
  size_t num_dst_shards = nb::len(dst_addressable_shards);
  if (num_shards != num_dst_shards) {
    throw std::runtime_error(
        "Number of shards in source and destination must match");
  }

  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  int64_t logical_elements = 1;
  for (int i = 0; i < shape.dimensions_size(); ++i) {
    logical_elements *= shape.dimensions(i);
  }

  int64_t stride = 1;
  if (shape.dimensions_size() > 1) {
    for (int i = 1; i < shape.dimensions_size(); ++i) {
      stride *= shape.dimensions(i);
    }
  }

  int64_t non_major_product = stride;

  if (is_partial && (non_major_product * itemsize) % 4096 != 0) {
    throw std::runtime_error(
        "Unsupported shape: product of non-major dimensions must be a multiple "
        "of tile size (4KB) on device for partial copies");
  }

  PjRtBuffer* first_src_buffer =
      jax::GetPjrtBufferFromPyObject(first_shard_data.ptr());

  auto status_or_src_size = first_src_buffer->GetOnDeviceSizeInBytes();
  if (!status_or_src_size.ok()) {
    throw std::runtime_error("Failed to get source buffer size");
  }
  int64_t physical_size = status_or_src_size.value();

  CommonPjRtBuffer* first_common_buffer =
      dynamic_cast<CommonPjRtBuffer*>(first_src_buffer);
  PjRtCApiBuffer* first_capi_buffer =
      dynamic_cast<PjRtCApiBuffer*>(first_src_buffer);

  const PJRT_Api* c_api = nullptr;
  const PJRT_RawBuffer_Extension* extension = nullptr;

  if (first_capi_buffer) {
    c_api = first_capi_buffer->pjrt_c_api();
    PjRtCApiClient* capi_client =
        dynamic_cast<PjRtCApiClient*>(first_capi_buffer->client());
    extension = capi_client->FindExtension<PJRT_RawBuffer_Extension>(
        PJRT_Extension_Type::PJRT_Extension_Type_RawBuffer);
    if (!extension) {
      throw std::runtime_error(
          "RawBuffer extension not found in PjRtCApiClient");
    }
  }

  auto pjrt_layout = first_src_buffer->layout();
  const xla::Layout* xla_layout = nullptr;
  if (pjrt_layout) {
    xla_layout = &pjrt_layout->xla_layout();
  }

  int64_t size_per_major_dim = 0;
  if (is_partial) {
    if (xla_layout && !xla_layout->tiles().empty()) {
      const xla::Tile& tile = xla_layout->tiles()[0];
      auto tile_dims = tile.dimensions();
      if (tile_dims.size() != 2) {
        throw std::runtime_error("Only 2D tiling supported for now");
      }
      int64_t tH = tile_dims[0];
      int64_t tW = tile_dims[1];

      int64_t rank = shape.dimensions_size();
      int64_t H = shape.dimensions(rank - 2);
      int64_t W = shape.dimensions(rank - 1);

      int64_t num_tiles_H = (H + tH - 1) / tH;
      int64_t num_tiles_W = (W + tW - 1) / tW;

      size_per_major_dim = num_tiles_H * num_tiles_W * tH * tW * itemsize;
      for (int i = 1; i < rank - 2; ++i) {
        size_per_major_dim *= shape.dimensions(i);
      }
    }
  }

  for (size_t i = 0; i < num_shards; ++i) {
    nb::object shard = addressable_shards[i];
    nb::object shard_data = shard.attr("data");
    nb::object dst_shard = dst_addressable_shards[i];
    nb::object dst_shard_data = dst_shard.attr("data");
    size_t dst_ptr_val =
        nb::cast<size_t>(dst_shard_data.attr("unsafe_buffer_pointer")());
    uint8_t* dst_data = reinterpret_cast<uint8_t*>(dst_ptr_val);
    size_t dst_size =
        nb::cast<size_t>(dst_shard_data.attr("on_device_size_in_bytes")());

    PjRtBuffer* src_buffer = jax::GetPjrtBufferFromPyObject(shard_data.ptr());
    std::vector<xla::Future<>> shard_futures;

    CommonPjRtBuffer* common_buffer =
        dynamic_cast<CommonPjRtBuffer*>(src_buffer);
    PjRtCApiBuffer* capi_buffer = dynamic_cast<PjRtCApiBuffer*>(src_buffer);

    std::optional<CommonPjRtBuffer::ScopedHold> hold;
    tsl::RCReference<CommonPjRtRawBuffer> raw_buffer;
    PJRT_RawBuffer* c_raw_buffer = nullptr;
    std::shared_ptr<RawBufferHolder> c_api_hold;

    if (common_buffer) {
      hold.emplace(common_buffer->GetBufferWithHold(
          CommonPjRtBuffer::ScopedHold::kUsage));
      if (!hold->ok()) {
        throw std::runtime_error("Failed to acquire hold on source buffer");
      }
      raw_buffer = hold->buffer()->raw_buffer();
    } else if (capi_buffer) {
      auto status_or_ext = src_buffer->AcquireExternalReference();
      if (!status_or_ext.ok()) {
        throw std::runtime_error("Failed to acquire external reference");
      }
      acc.AddExtHold(std::move(status_or_ext.value()));
    } else {
      throw std::runtime_error(std::string("Unsupported buffer type! Type: ") +
                               typeid(*src_buffer).name());
    }

    if (!is_partial) {
      // Full copy.
      int64_t physical_size = status_or_src_size.value();
      if (dst_size < physical_size) {
        throw std::runtime_error(
            "Destination buffer too small for raw tiled copy");
      }
      xla::Future<> future;
      {
        nb::gil_scoped_release release;
        future = src_buffer->CopyRawToHost(dst_data, 0, physical_size);
      }
      shard_futures.push_back(std::move(future));
    } else {
      for (size_t j = 0; j < src_offsets_major_dim.size(); ++j) {
        int64_t src_major_dim_offset =
            nb::cast<int64_t>(src_offsets_major_dim[j]);
        int64_t dst_major_dim_offset =
            nb::cast<int64_t>(dst_offsets_major_dim[j]);
        int64_t major_dim_size = nb::cast<int64_t>(copy_sizes_major_dim[j]);

        // Partial copy.
        if (xla_layout && !xla_layout->tiles().empty()) {
          // Tiled layout!
          int64_t physical_offset = src_major_dim_offset * size_per_major_dim;
          int64_t size_to_copy = major_dim_size * size_per_major_dim;
          int64_t dst_offset = dst_major_dim_offset * size_per_major_dim;

          if (physical_offset + size_to_copy > status_or_src_size.value()) {
            throw std::runtime_error("Copy range exceeds source buffer size");
          }

          if (dst_offset + size_to_copy > dst_size) {
            throw std::runtime_error(
                "Copy range exceeds destination buffer size");
          }

          uint8_t* dst_ptr = dst_data + dst_offset;

          xla::Future<> future;
          {
            nb::gil_scoped_release release;
            future = src_buffer->CopyRawToHost(dst_ptr, physical_offset,
                                               size_to_copy);
          }
          shard_futures.push_back(std::move(future));
        } else {
          // Non-tiled or simple layout.
          int64_t src_offset = src_major_dim_offset * stride * itemsize;
          int64_t dst_offset = dst_major_dim_offset * stride * itemsize;
          int64_t size = major_dim_size * stride * itemsize;

          if (src_offset + size > status_or_src_size.value()) {
            throw std::runtime_error("Copy range exceeds source buffer size");
          }

          if (dst_offset + size > dst_size) {
            throw std::runtime_error(
                "Copy range exceeds destination buffer size");
          }

          uint8_t* dst_ptr = dst_data + dst_offset;

          xla::Future<> future;
          {
            nb::gil_scoped_release release;
            future = src_buffer->CopyRawToHost(dst_ptr, src_offset, size);
          }
          shard_futures.push_back(std::move(future));
        }
      }
    }
    acc.Append(std::move(shard_futures), c_api_hold);
  }
}

inline PjRtCopyFuture transfer_d2h_async(
    const nb::object& src_arr, const nb::object& dst_arr,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list()) {
  PjRtCopyFuture acc;
  transfer_d2h_internal(src_arr, dst_arr, src_offsets_major_dim,
                        dst_offsets_major_dim, copy_sizes_major_dim, acc);
  return acc;
}

inline PjRtCopyFuture transfer_d2h_batch_async_naive(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list()) {
  if (nb::len(src_arrs) != nb::len(dst_arrs)) {
    throw std::runtime_error("Lengths of src_arrs and dst_arrs must match");
  }
  size_t n = nb::len(src_arrs);
  PjRtCopyFuture acc;
  for (size_t i = 0; i < n; ++i) {
    transfer_d2h_internal(src_arrs[i], dst_arrs[i], src_offsets_major_dim,
                          dst_offsets_major_dim, copy_sizes_major_dim, acc);
  }
  return acc;
}

inline PjRtCopyFuture transfer_d2h_batch_async_impl(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list()) {
  if (nb::len(src_arrs) != nb::len(dst_arrs)) {
    throw std::runtime_error("Lengths of src_arrs and dst_arrs must match");
  }
  size_t n = nb::len(src_arrs);
  PjRtCopyFuture acc;
  if (n == 0) return acc;

  nb::object first_src_arr = src_arrs[0];
  nb::object first_dst_arr = dst_arrs[0];
  nb::object addressable_shards = first_src_arr.attr("addressable_shards");
  size_t num_shards = nb::len(addressable_shards);

  if (num_shards == 0) return acc;

  nb::object first_shard_data = addressable_shards[0].attr("data");
  PjRtBuffer* first_buffer =
      jax::GetPjrtBufferFromPyObject(first_shard_data.ptr());
  const xla::Shape& shape = first_buffer->on_device_shape();

  bool is_partial = false;
  if (src_offsets_major_dim.size() > 0) {
    int64_t full_major_dim_size = shape.dimensions(0);
    for (size_t i = 0; i < src_offsets_major_dim.size(); ++i) {
      if (nb::cast<int64_t>(src_offsets_major_dim[i]) != 0 ||
          nb::cast<int64_t>(dst_offsets_major_dim[i]) != 0 ||
          nb::cast<int64_t>(copy_sizes_major_dim[i]) != full_major_dim_size) {
        is_partial = true;
        break;
      }
    }
  }

  auto status_or_src_size = first_buffer->GetOnDeviceSizeInBytes();
  if (!status_or_src_size.ok()) {
    throw std::runtime_error("Failed to get source buffer size");
  }
  int64_t physical_size = status_or_src_size.value();

  nb::object first_dst_shard_data =
      first_dst_arr.attr("addressable_shards")[0].attr("data");
  PjRtBuffer* first_dst_buffer =
      jax::GetPjrtBufferFromPyObject(first_dst_shard_data.ptr());
  auto status_or_dst_size = first_dst_buffer->GetOnDeviceSizeInBytes();
  if (!status_or_dst_size.ok()) {
    throw std::runtime_error("Failed to get destination buffer size");
  }
  size_t dst_size = status_or_dst_size.value();

  const PJRT_Api* c_api = nullptr;
  const PJRT_RawBuffer_Extension* extension = nullptr;
  PjRtCApiBuffer* first_capi_buffer =
      dynamic_cast<PjRtCApiBuffer*>(first_buffer);

  if (first_capi_buffer) {
    c_api = first_capi_buffer->pjrt_c_api();
    PjRtCApiClient* capi_client =
        dynamic_cast<PjRtCApiClient*>(first_capi_buffer->client());
    extension = capi_client->FindExtension<PJRT_RawBuffer_Extension>(
        PJRT_Extension_Type::PJRT_Extension_Type_RawBuffer);
    if (!extension) {
      throw std::runtime_error(
          "RawBuffer extension not found in PjRtCApiClient");
    }
  }

  bool is_common_buffer =
      (dynamic_cast<CommonPjRtBuffer*>(first_buffer) != nullptr);

  // Fast Path for Full Array Copy!
  if (!is_partial) {
    std::vector<xla::Future<>> batch_futures;
    batch_futures.reserve(n * num_shards);
    std::vector<std::shared_ptr<RawBufferHolder>> batch_c_api_holds;
    std::vector<std::shared_ptr<CommonPjRtBuffer::ScopedHold>> batch_holds;

    if (is_common_buffer) {
      batch_holds.reserve(n * num_shards);
      for (size_t layer_idx = 0; layer_idx < n; ++layer_idx) {
        nb::object src = src_arrs[layer_idx];
        nb::object dst = dst_arrs[layer_idx];

        xla::ifrt::Array* src_ifrt_array =
            jax::GetIfrtArrayFromPyObject(src.ptr());
        xla::ifrt::Array* dst_ifrt_array =
            jax::GetIfrtArrayFromPyObject(dst.ptr());

        auto* src_compat_arr =
            dynamic_cast<xla::ifrt::PjRtCompatibleArray*>(
                src_ifrt_array);
        auto* dst_compat_arr =
            dynamic_cast<xla::ifrt::PjRtCompatibleArray*>(
                dst_ifrt_array);

        if (src_compat_arr == nullptr || dst_compat_arr == nullptr) {
          throw std::runtime_error("Not a PjRt compatible array");
        }

        auto src_buffers = src_compat_arr->pjrt_buffers();
        auto dst_buffers = dst_compat_arr->pjrt_buffers();

        if (src_buffers.size() != num_shards ||
            dst_buffers.size() != num_shards) {
          throw std::runtime_error("Number of shards mismatch");
        }

        for (size_t i = 0; i < num_shards; ++i) {
          PjRtBuffer* src_buffer = src_buffers[i].get();
          PjRtBuffer* dst_buffer = dst_buffers[i].get();

          auto status_or_ptr =
              dst_buffer->client()->UnsafeBufferPointer(dst_buffer);
          if (!status_or_ptr.ok()) {
            throw std::runtime_error(
                "Failed to get unsafe buffer pointer for dst");
          }
          uint8_t* dst_data = reinterpret_cast<uint8_t*>(status_or_ptr.value());

          if (dst_size < physical_size) {
            throw std::runtime_error(
                "Destination buffer too small for raw tiled copy");
          }

          CommonPjRtBuffer* common_buffer =
              static_cast<CommonPjRtBuffer*>(src_buffer);

          auto hold = common_buffer->GetBufferWithHold(
              CommonPjRtBuffer::ScopedHold::kUsage);
          if (!hold.ok()) {
            throw std::runtime_error("Failed to acquire hold on source buffer");
          }

          xla::Future<> future;
          {
            nb::gil_scoped_release release;
            future = src_buffer->CopyRawToHost(dst_data, 0, physical_size);
          }
          batch_futures.push_back(std::move(future));
          batch_holds.push_back(
              std::make_shared<CommonPjRtBuffer::ScopedHold>(std::move(hold)));
        }
      }
      acc.Append(std::move(batch_futures), {}, std::move(batch_holds));
    } else {
      batch_c_api_holds.reserve(n * num_shards);
      for (size_t layer_idx = 0; layer_idx < n; ++layer_idx) {
        nb::object src = src_arrs[layer_idx];
        nb::object dst = dst_arrs[layer_idx];

        xla::ifrt::Array* src_ifrt_array =
            jax::GetIfrtArrayFromPyObject(src.ptr());
        xla::ifrt::Array* dst_ifrt_array =
            jax::GetIfrtArrayFromPyObject(dst.ptr());

        auto* src_compat_arr =
            dynamic_cast<xla::ifrt::PjRtCompatibleArray*>(
                src_ifrt_array);
        auto* dst_compat_arr =
            dynamic_cast<xla::ifrt::PjRtCompatibleArray*>(
                dst_ifrt_array);

        if (src_compat_arr == nullptr || dst_compat_arr == nullptr) {
          throw std::runtime_error("Not a PjRt compatible array");
        }

        auto src_buffers = src_compat_arr->pjrt_buffers();
        auto dst_buffers = dst_compat_arr->pjrt_buffers();

        if (src_buffers.size() != num_shards ||
            dst_buffers.size() != num_shards) {
          throw std::runtime_error("Number of shards mismatch");
        }

        for (size_t i = 0; i < num_shards; ++i) {
          PjRtBuffer* src_buffer = src_buffers[i].get();
          PjRtBuffer* dst_buffer = dst_buffers[i].get();

          auto status_or_ptr =
              dst_buffer->client()->UnsafeBufferPointer(dst_buffer);
          if (!status_or_ptr.ok()) {
            throw std::runtime_error(
                "Failed to get unsafe buffer pointer for dst");
          }
          uint8_t* dst_data = reinterpret_cast<uint8_t*>(status_or_ptr.value());

          if (dst_size < physical_size) {
            throw std::runtime_error(
                "Destination buffer too small for raw tiled copy");
          }
          PjRtCApiBuffer* capi_buffer =
              static_cast<PjRtCApiBuffer*>(src_buffer);

          auto status_or_raw = pjrt::PjRtCApiBuffer_CreateRawAliasOfBuffer(
              c_api, extension, capi_buffer->c_buffer());
          if (!status_or_raw.ok()) {
            throw std::runtime_error("Failed to create raw alias of buffer");
          }
          PJRT_RawBuffer* c_raw_buffer = status_or_raw.value();
          batch_c_api_holds.push_back(std::make_shared<RawBufferHolder>(
              c_api, extension, c_raw_buffer));

          xla::Future<> future = pjrt::PjRtCApiRawBuffer_CopyRawDeviceToHost(
              c_api, extension, c_raw_buffer, dst_data, 0, physical_size);
          batch_futures.push_back(std::move(future));
        }
      }
      acc.Append(std::move(batch_futures), std::move(batch_c_api_holds), {});
    }
    return acc;
  }

  // Partial Copy Branch
  if (shape.dimensions_size() < 3) {
    throw std::runtime_error(
        "Only support arrays with rank >= 3 for partial copies");
  }
  nb::object sharding = first_src_arr.attr("sharding");
  nb::object NamedSharding =
      nb::module_::import_("jax.sharding").attr("NamedSharding");
  if (nb::isinstance(sharding, NamedSharding)) {
    nb::object spec = sharding.attr("spec");
    if (nb::len(spec) > 0) {
      nb::object first_axis = spec[0];
      if (!first_axis.is_none()) {
        throw nb::value_error(
            "Partial copy not supported for arrays sharded on major dimension");
      }
    }
  }

  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  int64_t logical_elements = 1;
  for (int i = 0; i < shape.dimensions_size(); ++i) {
    logical_elements *= shape.dimensions(i);
  }

  int64_t stride = 1;
  if (shape.dimensions_size() > 1) {
    for (int i = 1; i < shape.dimensions_size(); ++i) {
      stride *= shape.dimensions(i);
    }
  }

  int64_t non_major_product = stride;
  if ((non_major_product * itemsize) % 4096 != 0) {
    throw std::runtime_error(
        "Unsupported shape: product of non-major dimensions must be a multiple "
        "of tile size (4KB)");
  }

  auto pjrt_layout = first_buffer->layout();
  const xla::Layout* xla_layout = nullptr;
  if (pjrt_layout) {
    xla_layout = &pjrt_layout->xla_layout();
  }

  int64_t size_per_major_dim = 0;
  if (xla_layout && !xla_layout->tiles().empty()) {
    const xla::Tile& tile = xla_layout->tiles()[0];
    auto tile_dims = tile.dimensions();
    if (tile_dims.size() != 2) {
      throw std::runtime_error("Only 2D tiling supported for now");
    }
    int64_t tH = tile_dims[0];
    int64_t tW = tile_dims[1];
    int64_t rank = shape.dimensions_size();
    int64_t H = shape.dimensions(rank - 2);
    int64_t W = shape.dimensions(rank - 1);
    int64_t num_tiles_H = (H + tH - 1) / tH;
    int64_t num_tiles_W = (W + tW - 1) / tW;
    size_per_major_dim = num_tiles_H * num_tiles_W * tH * tW * itemsize;
    for (int i = 1; i < rank - 2; ++i) {
      size_per_major_dim *= shape.dimensions(i);
    }
  }

  for (size_t layer_idx = 0; layer_idx < n; ++layer_idx) {
    nb::object src = src_arrs[layer_idx];
    nb::object dst = dst_arrs[layer_idx];

    xla::ifrt::Array* src_ifrt_array = jax::GetIfrtArrayFromPyObject(src.ptr());
    xla::ifrt::Array* dst_ifrt_array = jax::GetIfrtArrayFromPyObject(dst.ptr());

    auto* src_compat_arr =
        dynamic_cast<xla::ifrt::PjRtCompatibleArray*>(src_ifrt_array);
    auto* dst_compat_arr =
        dynamic_cast<xla::ifrt::PjRtCompatibleArray*>(dst_ifrt_array);

    if (src_compat_arr == nullptr || dst_compat_arr == nullptr) {
      throw std::runtime_error("Not a PjRt compatible array");
    }

    auto src_buffers = src_compat_arr->pjrt_buffers();
    auto dst_buffers = dst_compat_arr->pjrt_buffers();

    if (src_buffers.size() != num_shards || dst_buffers.size() != num_shards) {
      throw std::runtime_error("Number of shards mismatch");
    }

    for (size_t i = 0; i < num_shards; ++i) {
      PjRtBuffer* src_buffer = src_buffers[i].get();
      PjRtBuffer* dst_buffer = dst_buffers[i].get();

      auto status_or_ptr =
          dst_buffer->client()->UnsafeBufferPointer(dst_buffer);
      if (!status_or_ptr.ok()) {
        throw std::runtime_error("Failed to get unsafe buffer pointer for dst");
      }
      uint8_t* dst_data = reinterpret_cast<uint8_t*>(status_or_ptr.value());

      std::vector<xla::Future<>> shard_futures;

      CommonPjRtBuffer* common_buffer =
          dynamic_cast<CommonPjRtBuffer*>(src_buffer);
      PjRtCApiBuffer* capi_buffer = dynamic_cast<PjRtCApiBuffer*>(src_buffer);

      std::optional<CommonPjRtBuffer::ScopedHold> hold;
      PJRT_RawBuffer* c_raw_buffer = nullptr;
      std::shared_ptr<RawBufferHolder> c_api_hold;

      if (common_buffer) {
        hold.emplace(common_buffer->GetBufferWithHold(
            CommonPjRtBuffer::ScopedHold::kUsage));
        if (!hold->ok()) {
          throw std::runtime_error("Failed to acquire hold on source buffer");
        }
      } else if (capi_buffer) {
        auto status_or_ext = src_buffer->AcquireExternalReference();
        if (!status_or_ext.ok()) {
          throw std::runtime_error("Failed to acquire external reference");
        }
        acc.AddExtHold(std::move(status_or_ext.value()));
      }

      for (size_t j = 0; j < src_offsets_major_dim.size(); ++j) {
        int64_t src_major_dim_offset =
            nb::cast<int64_t>(src_offsets_major_dim[j]);
        int64_t dst_major_dim_offset =
            nb::cast<int64_t>(dst_offsets_major_dim[j]);
        int64_t major_dim_size = nb::cast<int64_t>(copy_sizes_major_dim[j]);

        if (xla_layout && !xla_layout->tiles().empty()) {
          int64_t physical_offset = src_major_dim_offset * size_per_major_dim;
          int64_t size_to_copy = major_dim_size * size_per_major_dim;
          int64_t dst_offset = dst_major_dim_offset * size_per_major_dim;

          if (physical_offset + size_to_copy > physical_size) {
            throw std::runtime_error("Copy range exceeds source buffer size");
          }
          if (dst_offset + size_to_copy > dst_size) {
            throw std::runtime_error(
                "Copy range exceeds destination buffer size");
          }
          uint8_t* dst_ptr = dst_data + dst_offset;

          xla::Future<> future;
          {
            nb::gil_scoped_release release;
            future = src_buffer->CopyRawToHost(dst_ptr, physical_offset,
                                               size_to_copy);
          }
          shard_futures.push_back(std::move(future));
        } else {
          int64_t src_offset = src_major_dim_offset * stride * itemsize;
          int64_t dst_offset = dst_major_dim_offset * stride * itemsize;
          int64_t size = major_dim_size * stride * itemsize;

          if (src_offset + size > physical_size) {
            throw std::runtime_error("Copy range exceeds source buffer size");
          }
          if (dst_offset + size > dst_size) {
            throw std::runtime_error(
                "Copy range exceeds destination buffer size");
          }
          uint8_t* dst_ptr = dst_data + dst_offset;

          xla::Future<> future;
          {
            nb::gil_scoped_release release;
            future = src_buffer->CopyRawToHost(dst_ptr, src_offset, size);
          }
          shard_futures.push_back(std::move(future));
        }
      }
      acc.Append(std::move(shard_futures), c_api_hold);
    }
  }
  return acc;
}

inline void transfer_h2d_internal(const nb::object& src_arr,
                                  const nb::object& dst_arr,
                                  const nb::list& src_offsets_major_dim,
                                  const nb::list& dst_offsets_major_dim,
                                  const nb::list& copy_sizes_major_dim,
                                  PjRtCopyFuture& acc) {
  nb::object addressable_shards = dst_arr.attr("addressable_shards");
  size_t num_shards = nb::len(addressable_shards);

  if (num_shards == 0) {
    return;
  }

  nb::object first_shard_data = addressable_shards[0].attr("data");
  PjRtBuffer* first_buffer =
      jax::GetPjrtBufferFromPyObject(first_shard_data.ptr());
  const xla::Shape& shape = first_buffer->on_device_shape();

  bool is_partial = false;
  int64_t full_major_dim_size = shape.dimensions(0);
  for (size_t i = 0; i < src_offsets_major_dim.size(); ++i) {
    if (nb::cast<int64_t>(src_offsets_major_dim[i]) != 0 ||
        nb::cast<int64_t>(dst_offsets_major_dim[i]) != 0 ||
        nb::cast<int64_t>(copy_sizes_major_dim[i]) != full_major_dim_size) {
      is_partial = true;
      break;
    }
  }

  if (is_partial) {
    if (shape.dimensions_size() < 3) {
      throw std::runtime_error(
          "Only support arrays with rank >= 3 for partial copies");
    }
    nb::object sharding = dst_arr.attr("sharding");
    nb::object NamedSharding =
        nb::module_::import_("jax.sharding").attr("NamedSharding");
    if (nb::isinstance(sharding, NamedSharding)) {
      nb::object spec = sharding.attr("spec");
      if (nb::len(spec) > 0) {
        nb::object first_axis = spec[0];
        if (!first_axis.is_none()) {
          throw nb::value_error(
              "Partial copy not supported for arrays sharded on major "
              "dimension");
        }
      }
    }
  }

  nb::object src_addressable_shards = src_arr.attr("addressable_shards");
  size_t num_src_shards = nb::len(src_addressable_shards);
  if (num_shards != num_src_shards) {
    throw std::runtime_error(
        "Number of shards in source and destination must match");
  }

  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  int64_t stride = 1;
  if (shape.dimensions_size() == 1) {
    stride = shape.dimensions(0);
  } else {
    for (int i = 1; i < shape.dimensions_size(); ++i) {
      stride *= shape.dimensions(i);
    }
  }
  if (is_partial && (stride * itemsize) % 4096 != 0) {
    throw std::runtime_error(
        "Unsupported shape: product of non-major dimensions must be a multiple "
        "of tile size (4KB) on device for partial copies");
  }

  PjRtBuffer* first_dst_buffer =
      jax::GetPjrtBufferFromPyObject(first_shard_data.ptr());

  auto status_or_dst_size = first_dst_buffer->GetOnDeviceSizeInBytes();
  if (!status_or_dst_size.ok()) {
    throw std::runtime_error("Failed to get destination buffer size");
  }
  int64_t physical_size = status_or_dst_size.value();

  CommonPjRtBuffer* first_common_buffer =
      dynamic_cast<CommonPjRtBuffer*>(first_dst_buffer);
  PjRtCApiBuffer* first_capi_buffer =
      dynamic_cast<PjRtCApiBuffer*>(first_dst_buffer);

  const PJRT_Api* c_api = nullptr;
  const PJRT_RawBuffer_Extension* extension = nullptr;

  if (first_capi_buffer) {
    c_api = first_capi_buffer->pjrt_c_api();
    PjRtCApiClient* capi_client =
        dynamic_cast<PjRtCApiClient*>(first_capi_buffer->client());
    extension = capi_client->FindExtension<PJRT_RawBuffer_Extension>(
        PJRT_Extension_Type::PJRT_Extension_Type_RawBuffer);
    if (!extension) {
      throw std::runtime_error(
          "RawBuffer extension not found in PjRtCApiClient");
    }
  }

  auto pjrt_layout = first_dst_buffer->layout();
  const xla::Layout* xla_layout = nullptr;
  if (pjrt_layout) {
    xla_layout = &pjrt_layout->xla_layout();
  }

  int64_t size_per_major_dim = 0;
  if (is_partial) {
    if (xla_layout && !xla_layout->tiles().empty()) {
      const xla::Tile& tile = xla_layout->tiles()[0];
      auto tile_dims = tile.dimensions();
      if (tile_dims.size() != 2) {
        throw std::runtime_error("Only 2D tiling supported for now");
      }
      int64_t tH = tile_dims[0];
      int64_t tW = tile_dims[1];

      int64_t rank = shape.dimensions_size();
      int64_t H = shape.dimensions(rank - 2);
      int64_t W = shape.dimensions(rank - 1);

      int64_t num_tiles_H = (H + tH - 1) / tH;
      int64_t num_tiles_W = (W + tW - 1) / tW;

      size_per_major_dim = num_tiles_H * num_tiles_W * tH * tW * itemsize;
      for (int i = 1; i < rank - 2; ++i) {
        size_per_major_dim *= shape.dimensions(i);
      }
    }
  }

  for (size_t i = 0; i < num_shards; ++i) {
    nb::object shard = addressable_shards[i];
    nb::object shard_data = shard.attr("data");
    nb::object src_shard = src_addressable_shards[i];
    nb::object src_shard_data = src_shard.attr("data");
    size_t src_ptr_val =
        nb::cast<size_t>(src_shard_data.attr("unsafe_buffer_pointer")());
    const uint8_t* src_data = reinterpret_cast<const uint8_t*>(src_ptr_val);
    size_t src_size =
        nb::cast<size_t>(src_shard_data.attr("on_device_size_in_bytes")());

    PjRtBuffer* dst_buffer = jax::GetPjrtBufferFromPyObject(shard_data.ptr());

    CommonPjRtBuffer* common_buffer =
        dynamic_cast<CommonPjRtBuffer*>(dst_buffer);
    PjRtCApiBuffer* capi_buffer = dynamic_cast<PjRtCApiBuffer*>(dst_buffer);

    std::optional<CommonPjRtBuffer::ScopedHold> hold;
    tsl::RCReference<CommonPjRtRawBuffer> raw_buffer;
    PJRT_RawBuffer* c_raw_buffer = nullptr;

    std::shared_ptr<RawBufferHolder> c_api_hold;

    if (common_buffer) {
      hold.emplace(common_buffer->GetBufferWithHold(
          CommonPjRtBuffer::ScopedHold::kUsage));
      if (!hold->ok()) {
        throw std::runtime_error(
            "Failed to acquire hold on destination buffer");
      }
      raw_buffer = hold->buffer()->raw_buffer();
    } else if (capi_buffer) {
      auto status_or_raw = pjrt::PjRtCApiBuffer_CreateRawAliasOfBuffer(
          c_api, extension, capi_buffer->c_buffer());
      if (!status_or_raw.ok()) {
        throw std::runtime_error("Failed to create raw alias of buffer");
      }
      c_raw_buffer = status_or_raw.value();
      c_api_hold =
          std::make_shared<RawBufferHolder>(c_api, extension, c_raw_buffer);
    } else {
      throw std::runtime_error(std::string("Unsupported buffer type! Type: ") +
                               typeid(*dst_buffer).name());
    }

    std::shared_ptr<CommonPjRtBuffer::ScopedHold> shared_hold;
    if (common_buffer) {
      shared_hold =
          std::make_shared<CommonPjRtBuffer::ScopedHold>(std::move(*hold));
    }

    if (!is_partial) {
      // Full copy.
      int64_t physical_size = status_or_dst_size.value();
      if (src_size < physical_size) {
        throw std::runtime_error("Source buffer too small for raw tiled copy");
      }
      xla::Future<> future;
      if (common_buffer) {
        {
          nb::gil_scoped_release release;
          future = raw_buffer->CopyRawHostToDevice(src_data, 0, physical_size);
        }
        acc.Append({std::move(future)}, c_api_hold, shared_hold);
      } else if (capi_buffer) {
        {
          nb::gil_scoped_release release;
          future = pjrt::PjRtCApiRawBuffer_CopyRawHostToDevice(
              c_api, extension, c_raw_buffer, src_data, 0, physical_size);
        }
        acc.Append({std::move(future)}, c_api_hold, nullptr);
      }
    } else {
      for (size_t j = 0; j < src_offsets_major_dim.size(); ++j) {
        int64_t src_major_dim_offset =
            nb::cast<int64_t>(src_offsets_major_dim[j]);
        int64_t dst_major_dim_offset =
            nb::cast<int64_t>(dst_offsets_major_dim[j]);
        int64_t major_dim_size = nb::cast<int64_t>(copy_sizes_major_dim[j]);

        // Partial copy.
        if (xla_layout && !xla_layout->tiles().empty() &&
            shape.dimensions_size() >= 1) {
          int64_t physical_offset = dst_major_dim_offset * size_per_major_dim;
          int64_t size_to_copy = major_dim_size * size_per_major_dim;
          int64_t src_offset = src_major_dim_offset * size_per_major_dim;

          if (src_offset + size_to_copy > src_size) {
            throw std::runtime_error("Copy range exceeds source buffer size");
          }
          if (physical_offset + size_to_copy > physical_size) {
            throw std::runtime_error(
                "Copy range exceeds destination buffer size");
          }
          const uint8_t* src_ptr = src_data + src_offset;

          xla::Future<> future;
          if (common_buffer) {
            {
              nb::gil_scoped_release release;
              future = raw_buffer->CopyRawHostToDevice(src_ptr, physical_offset,
                                                       size_to_copy);
            }
            acc.Append({std::move(future)}, c_api_hold, shared_hold);
          } else if (capi_buffer) {
            {
              nb::gil_scoped_release release;
              future = pjrt::PjRtCApiRawBuffer_CopyRawHostToDevice(
                  c_api, extension, c_raw_buffer, src_ptr, physical_offset,
                  size_to_copy);
            }
            acc.Append({std::move(future)}, c_api_hold, nullptr);
          }
        } else {
          int64_t src_offset = src_major_dim_offset * stride * itemsize;
          int64_t dst_offset = dst_major_dim_offset * stride * itemsize;
          int64_t size = major_dim_size * stride * itemsize;

          if (src_offset + size > src_size) {
            throw std::runtime_error("Copy range exceeds source buffer size");
          }
          if (dst_offset + size > physical_size) {
            throw std::runtime_error(
                "Copy range exceeds destination buffer size");
          }
          const uint8_t* src_ptr = src_data + src_offset;

          xla::Future<> future;
          if (common_buffer) {
            {
              nb::gil_scoped_release release;
              future =
                  raw_buffer->CopyRawHostToDevice(src_ptr, dst_offset, size);
            }
            acc.Append({std::move(future)}, c_api_hold, shared_hold);
          } else if (capi_buffer) {
            {
              nb::gil_scoped_release release;
              future = pjrt::PjRtCApiRawBuffer_CopyRawHostToDevice(
                  c_api, extension, c_raw_buffer, src_ptr, dst_offset, size);
            }
            acc.Append({std::move(future)}, c_api_hold, nullptr);
          }
        }
      }
    }
  }
}

inline PjRtCopyFuture transfer_h2d_async(
    const nb::object& src_arr, const nb::object& dst_arr,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list()) {
  PjRtCopyFuture acc;
  transfer_h2d_internal(src_arr, dst_arr, src_offsets_major_dim,
                        dst_offsets_major_dim, copy_sizes_major_dim, acc);
  return acc;
}

inline PjRtCopyFuture transfer_h2d_batch_async_naive(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list()) {
  if (nb::len(src_arrs) != nb::len(dst_arrs)) {
    throw std::runtime_error("Lengths of src_arrs and dst_arrs must match");
  }
  size_t n = nb::len(src_arrs);
  PjRtCopyFuture acc;
  for (size_t i = 0; i < n; ++i) {
    transfer_h2d_internal(src_arrs[i], dst_arrs[i], src_offsets_major_dim,
                          dst_offsets_major_dim, copy_sizes_major_dim, acc);
  }
  return acc;
}

inline PjRtCopyFuture transfer_h2d_batch_async_impl(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list()) {
  if (nb::len(src_arrs) != nb::len(dst_arrs)) {
    throw std::runtime_error("Lengths of src_arrs and dst_arrs must match");
  }
  size_t n = nb::len(src_arrs);
  PjRtCopyFuture acc;
  if (n == 0) return acc;

  nb::object first_src_arr = src_arrs[0];
  nb::object first_dst_arr = dst_arrs[0];
  nb::object addressable_shards = first_dst_arr.attr("addressable_shards");
  size_t num_shards = nb::len(addressable_shards);

  if (num_shards == 0) return acc;

  nb::object first_shard_data = addressable_shards[0].attr("data");
  PjRtBuffer* first_buffer =
      jax::GetPjrtBufferFromPyObject(first_shard_data.ptr());
  const xla::Shape& shape = first_buffer->on_device_shape();

  bool is_partial = false;
  if (src_offsets_major_dim.size() > 0) {
    int64_t full_major_dim_size = shape.dimensions(0);
    for (size_t i = 0; i < src_offsets_major_dim.size(); ++i) {
      if (nb::cast<int64_t>(src_offsets_major_dim[i]) != 0 ||
          nb::cast<int64_t>(dst_offsets_major_dim[i]) != 0 ||
          nb::cast<int64_t>(copy_sizes_major_dim[i]) != full_major_dim_size) {
        is_partial = true;
        break;
      }
    }
  }

  auto status_or_dst_size = first_buffer->GetOnDeviceSizeInBytes();
  if (!status_or_dst_size.ok()) {
    throw std::runtime_error("Failed to get destination buffer size");
  }
  int64_t physical_size = status_or_dst_size.value();

  const PJRT_Api* c_api = nullptr;
  const PJRT_RawBuffer_Extension* extension = nullptr;
  PjRtCApiBuffer* first_capi_buffer =
      dynamic_cast<PjRtCApiBuffer*>(first_buffer);

  if (first_capi_buffer) {
    c_api = first_capi_buffer->pjrt_c_api();
    PjRtCApiClient* capi_client =
        dynamic_cast<PjRtCApiClient*>(first_capi_buffer->client());
    extension = capi_client->FindExtension<PJRT_RawBuffer_Extension>(
        PJRT_Extension_Type::PJRT_Extension_Type_RawBuffer);
    if (!extension) {
      throw std::runtime_error(
          "RawBuffer extension not found in PjRtCApiClient");
    }
  }

  bool is_common_buffer =
      (dynamic_cast<CommonPjRtBuffer*>(first_buffer) != nullptr);

  // Fast path for full array copies
  if (!is_partial) {
    std::vector<xla::Future<>> batch_futures;
    batch_futures.reserve(n * num_shards);
    std::vector<std::shared_ptr<RawBufferHolder>> batch_c_api_holds;
    std::vector<std::shared_ptr<CommonPjRtBuffer::ScopedHold>> batch_holds;

    if (is_common_buffer) {
      batch_holds.reserve(n * num_shards);
      for (size_t layer_idx = 0; layer_idx < n; ++layer_idx) {
        nb::object src = src_arrs[layer_idx];
        nb::object dst = dst_arrs[layer_idx];

        xla::ifrt::Array* src_ifrt_array =
            jax::GetIfrtArrayFromPyObject(src.ptr());
        xla::ifrt::Array* dst_ifrt_array =
            jax::GetIfrtArrayFromPyObject(dst.ptr());

        auto* src_compat_arr =
            dynamic_cast<xla::ifrt::PjRtCompatibleArray*>(
                src_ifrt_array);
        auto* dst_compat_arr =
            dynamic_cast<xla::ifrt::PjRtCompatibleArray*>(
                dst_ifrt_array);

        if (src_compat_arr == nullptr || dst_compat_arr == nullptr) {
          throw std::runtime_error("Not a PjRt compatible array");
        }

        auto src_buffers = src_compat_arr->pjrt_buffers();
        auto dst_buffers = dst_compat_arr->pjrt_buffers();

        if (src_buffers.size() != num_shards ||
            dst_buffers.size() != num_shards) {
          throw std::runtime_error("Number of shards mismatch");
        }

        for (size_t i = 0; i < num_shards; ++i) {
          PjRtBuffer* src_buffer = src_buffers[i].get();
          PjRtBuffer* dst_buffer = dst_buffers[i].get();

          auto status_or_ptr =
              src_buffer->client()->UnsafeBufferPointer(src_buffer);
          if (!status_or_ptr.ok()) {
            throw std::runtime_error(
                "Failed to get unsafe buffer pointer for src");
          }
          const uint8_t* src_data =
              reinterpret_cast<const uint8_t*>(status_or_ptr.value());

          size_t src_size = src_buffer->GetOnDeviceSizeInBytes().value();

          if (src_size < physical_size) {
            throw std::runtime_error(
                "Source buffer too small for raw tiled copy");
          }

          CommonPjRtBuffer* common_buffer =
              static_cast<CommonPjRtBuffer*>(dst_buffer);

          auto hold = common_buffer->GetBufferWithHold(
              CommonPjRtBuffer::ScopedHold::kUsage);
          if (!hold.ok()) {
            throw std::runtime_error(
                "Failed to acquire hold on destination buffer");
          }

          auto raw_buffer = hold.buffer()->raw_buffer();
          batch_holds.push_back(
              std::make_shared<CommonPjRtBuffer::ScopedHold>(std::move(hold)));

          xla::Future<> future;
          {
            nb::gil_scoped_release release;
            future =
                raw_buffer->CopyRawHostToDevice(src_data, 0, physical_size);
          }
          batch_futures.push_back(std::move(future));
        }
      }
      acc.Append(std::move(batch_futures), {}, std::move(batch_holds));
    } else {
      batch_c_api_holds.reserve(n * num_shards);
      for (size_t layer_idx = 0; layer_idx < n; ++layer_idx) {
        nb::object src = src_arrs[layer_idx];
        nb::object dst = dst_arrs[layer_idx];

        xla::ifrt::Array* src_ifrt_array =
            jax::GetIfrtArrayFromPyObject(src.ptr());
        xla::ifrt::Array* dst_ifrt_array =
            jax::GetIfrtArrayFromPyObject(dst.ptr());

        auto* src_compat_arr =
            dynamic_cast<xla::ifrt::PjRtCompatibleArray*>(
                src_ifrt_array);
        auto* dst_compat_arr =
            dynamic_cast<xla::ifrt::PjRtCompatibleArray*>(
                dst_ifrt_array);

        if (src_compat_arr == nullptr || dst_compat_arr == nullptr) {
          throw std::runtime_error("Not a PjRt compatible array");
        }

        auto src_buffers = src_compat_arr->pjrt_buffers();
        auto dst_buffers = dst_compat_arr->pjrt_buffers();

        if (src_buffers.size() != num_shards ||
            dst_buffers.size() != num_shards) {
          throw std::runtime_error("Number of shards mismatch");
        }

        for (size_t i = 0; i < num_shards; ++i) {
          PjRtBuffer* src_buffer = src_buffers[i].get();
          PjRtBuffer* dst_buffer = dst_buffers[i].get();

          auto status_or_ptr =
              src_buffer->client()->UnsafeBufferPointer(src_buffer);
          if (!status_or_ptr.ok()) {
            throw std::runtime_error(
                "Failed to get unsafe buffer pointer for src");
          }
          const uint8_t* src_data =
              reinterpret_cast<const uint8_t*>(status_or_ptr.value());

          size_t src_size = src_buffer->GetOnDeviceSizeInBytes().value();

          if (src_size < physical_size) {
            throw std::runtime_error(
                "Source buffer too small for raw tiled copy");
          }
          PjRtCApiBuffer* capi_buffer =
              static_cast<PjRtCApiBuffer*>(dst_buffer);

          auto status_or_raw = pjrt::PjRtCApiBuffer_CreateRawAliasOfBuffer(
              c_api, extension, capi_buffer->c_buffer());
          if (!status_or_raw.ok()) {
            throw std::runtime_error("Failed to create raw alias of buffer");
          }
          PJRT_RawBuffer* c_raw_buffer = status_or_raw.value();
          batch_c_api_holds.push_back(std::make_shared<RawBufferHolder>(
              c_api, extension, c_raw_buffer));

          xla::Future<> future = pjrt::PjRtCApiRawBuffer_CopyRawHostToDevice(
              c_api, extension, c_raw_buffer, src_data, 0, physical_size);
          batch_futures.push_back(std::move(future));
        }
      }
      acc.Append(std::move(batch_futures), std::move(batch_c_api_holds), {});
    }
    return acc;
  }

  // Partial Copy Path
  if (shape.dimensions_size() < 3) {
    throw std::runtime_error(
        "Only support arrays with rank >= 3 for partial copies");
  }
  nb::object sharding = first_dst_arr.attr("sharding");
  nb::object NamedSharding =
      nb::module_::import_("jax.sharding").attr("NamedSharding");
  if (nb::isinstance(sharding, NamedSharding)) {
    nb::object spec = sharding.attr("spec");
    if (nb::len(spec) > 0) {
      nb::object first_axis = spec[0];
      if (!first_axis.is_none()) {
        throw nb::value_error(
            "Partial copy not supported for arrays sharded on major dimension");
      }
    }
  }

  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  int64_t stride = 1;
  if (shape.dimensions_size() == 1) {
    stride = shape.dimensions(0);
  } else {
    for (int i = 1; i < shape.dimensions_size(); ++i) {
      stride *= shape.dimensions(i);
    }
  }

  if ((stride * itemsize) % 4096 != 0) {
    throw std::runtime_error(
        "Unsupported shape: product of non-major dimensions must be a multiple "
        "of tile size (4KB)");
  }

  auto pjrt_layout = first_buffer->layout();
  const xla::Layout* xla_layout = nullptr;
  if (pjrt_layout) {
    xla_layout = &pjrt_layout->xla_layout();
  }

  int64_t size_per_major_dim = 0;
  if (xla_layout && !xla_layout->tiles().empty()) {
    const xla::Tile& tile = xla_layout->tiles()[0];
    auto tile_dims = tile.dimensions();
    if (tile_dims.size() != 2) {
      throw std::runtime_error("Only 2D tiling supported for now");
    }
    int64_t tH = tile_dims[0];
    int64_t tW = tile_dims[1];
    int64_t rank = shape.dimensions_size();
    int64_t H = shape.dimensions(rank - 2);
    int64_t W = shape.dimensions(rank - 1);
    int64_t num_tiles_H = (H + tH - 1) / tH;
    int64_t num_tiles_W = (W + tW - 1) / tW;
    size_per_major_dim = num_tiles_H * num_tiles_W * tH * tW * itemsize;
    for (int i = 1; i < rank - 2; ++i) {
      size_per_major_dim *= shape.dimensions(i);
    }
  }

  for (size_t layer_idx = 0; layer_idx < n; ++layer_idx) {
    nb::object src = src_arrs[layer_idx];
    nb::object dst = dst_arrs[layer_idx];

    xla::ifrt::Array* src_ifrt_array = jax::GetIfrtArrayFromPyObject(src.ptr());
    xla::ifrt::Array* dst_ifrt_array = jax::GetIfrtArrayFromPyObject(dst.ptr());

    auto status_or_src_shards =
        src_ifrt_array->DisassembleIntoSingleDeviceArrays(
            xla::ifrt::ArrayCopySemantics::kReuseInput,
            xla::ifrt::SingleDeviceShardSemantics::kAddressableShards);
    auto status_or_dst_shards =
        dst_ifrt_array->DisassembleIntoSingleDeviceArrays(
            xla::ifrt::ArrayCopySemantics::kReuseInput,
            xla::ifrt::SingleDeviceShardSemantics::kAddressableShards);

    if (!status_or_src_shards.ok() || !status_or_dst_shards.ok()) {
      throw std::runtime_error("Failed to disassemble arrays");
    }

    const std::vector<xla::ifrt::ArrayRef>& src_shards =
        status_or_src_shards.value();
    const std::vector<xla::ifrt::ArrayRef>& dst_shards =
        status_or_dst_shards.value();

    if (src_shards.size() != num_shards || dst_shards.size() != num_shards) {
      throw std::runtime_error("Number of shards mismatch");
    }

    for (size_t i = 0; i < num_shards; ++i) {
      xla::ifrt::Array* src_shard = src_shards[i].get();
      xla::ifrt::Array* dst_shard = dst_shards[i].get();

      auto* src_compat_arr =
          dynamic_cast<xla::ifrt::PjRtCompatibleArray*>(src_shard);
      auto* dst_compat_arr =
          dynamic_cast<xla::ifrt::PjRtCompatibleArray*>(dst_shard);

      if (src_compat_arr == nullptr || dst_compat_arr == nullptr) {
        throw std::runtime_error("Not a PjRt compatible array shard");
      }

      PjRtBuffer* src_buffer = src_compat_arr->pjrt_buffers().front().get();
      PjRtBuffer* dst_buffer = dst_compat_arr->pjrt_buffers().front().get();

      auto status_or_ext = src_buffer->AcquireExternalReference();
      if (!status_or_ext.ok()) {
        throw std::runtime_error(
            "Failed to acquire external reference for src");
      }
      const void* src_ptr =
          status_or_ext.value()->OpaqueDeviceMemoryDataPointer();
      const uint8_t* src_data = reinterpret_cast<const uint8_t*>(src_ptr);

      size_t src_size = src_buffer->GetOnDeviceSizeInBytes().value();

      CommonPjRtBuffer* common_buffer =
          dynamic_cast<CommonPjRtBuffer*>(dst_buffer);
      PjRtCApiBuffer* capi_buffer = dynamic_cast<PjRtCApiBuffer*>(dst_buffer);

      std::optional<CommonPjRtBuffer::ScopedHold> hold;
      tsl::RCReference<CommonPjRtRawBuffer> raw_buffer;
      PJRT_RawBuffer* c_raw_buffer = nullptr;
      std::shared_ptr<RawBufferHolder> c_api_hold;

      if (common_buffer) {
        hold.emplace(common_buffer->GetBufferWithHold(
            CommonPjRtBuffer::ScopedHold::kUsage));
        if (!hold->ok()) {
          throw std::runtime_error(
              "Failed to acquire hold on destination buffer");
        }
        raw_buffer = hold->buffer()->raw_buffer();
      } else if (capi_buffer) {
        auto status_or_raw = pjrt::PjRtCApiBuffer_CreateRawAliasOfBuffer(
            c_api, extension, capi_buffer->c_buffer());
        if (!status_or_raw.ok()) {
          throw std::runtime_error("Failed to create raw alias of buffer");
        }
        c_raw_buffer = status_or_raw.value();
        c_api_hold =
            std::make_shared<RawBufferHolder>(c_api, extension, c_raw_buffer);
      }

      std::shared_ptr<CommonPjRtBuffer::ScopedHold> shared_hold;
      if (common_buffer) {
        shared_hold =
            std::make_shared<CommonPjRtBuffer::ScopedHold>(std::move(*hold));
      }

      for (size_t j = 0; j < src_offsets_major_dim.size(); ++j) {
        int64_t src_major_dim_offset =
            nb::cast<int64_t>(src_offsets_major_dim[j]);
        int64_t dst_major_dim_offset =
            nb::cast<int64_t>(dst_offsets_major_dim[j]);
        int64_t major_dim_size = nb::cast<int64_t>(copy_sizes_major_dim[j]);

        if (xla_layout && !xla_layout->tiles().empty() &&
            shape.dimensions_size() >= 1) {
          int64_t physical_offset = dst_major_dim_offset * size_per_major_dim;
          int64_t size_to_copy = major_dim_size * size_per_major_dim;
          int64_t src_offset = src_major_dim_offset * size_per_major_dim;

          if (src_offset + size_to_copy > src_size) {
            throw std::runtime_error("Copy range exceeds source buffer size");
          }
          if (physical_offset + size_to_copy > physical_size) {
            throw std::runtime_error(
                "Copy range exceeds destination buffer size");
          }
          const uint8_t* src_ptr = src_data + src_offset;

          xla::Future<> future;
          if (common_buffer) {
            future = raw_buffer->CopyRawHostToDevice(src_ptr, physical_offset,
                                                     size_to_copy);
            acc.Append({std::move(future)}, c_api_hold, shared_hold);
          } else if (capi_buffer) {
            future = pjrt::PjRtCApiRawBuffer_CopyRawHostToDevice(
                c_api, extension, c_raw_buffer, src_ptr, physical_offset,
                size_to_copy);
            acc.Append({std::move(future)}, c_api_hold, nullptr);
          }
        } else {
          int64_t src_offset = src_major_dim_offset * stride * itemsize;
          int64_t dst_offset = dst_major_dim_offset * stride * itemsize;
          int64_t size = major_dim_size * stride * itemsize;

          if (src_offset + size > src_size) {
            throw std::runtime_error("Copy range exceeds source buffer size");
          }
          if (dst_offset + size > physical_size) {
            throw std::runtime_error(
                "Copy range exceeds destination buffer size");
          }
          const uint8_t* src_ptr = src_data + src_offset;

          xla::Future<> future;
          if (common_buffer) {
            future = raw_buffer->CopyRawHostToDevice(src_ptr, dst_offset, size);
            acc.Append({std::move(future)}, c_api_hold, shared_hold);
          } else if (capi_buffer) {
            future = pjrt::PjRtCApiRawBuffer_CopyRawHostToDevice(
                c_api, extension, c_raw_buffer, src_ptr, dst_offset, size);
            acc.Append({std::move(future)}, c_api_hold, nullptr);
          }
        }
      }
    }
  }
  return acc;
}

inline void transfer_d2h(const nb::object& src_arr, const nb::object& dst_arr,
                         const nb::list& src_offsets_major_dim = nb::list(),
                         const nb::list& dst_offsets_major_dim = nb::list(),
                         const nb::list& copy_sizes_major_dim = nb::list()) {
  transfer_d2h_async(src_arr, dst_arr, src_offsets_major_dim,
                     dst_offsets_major_dim, copy_sizes_major_dim)
      .Await();
}

inline void transfer_h2d(const nb::object& src_arr, const nb::object& dst_arr,
                         const nb::list& src_offsets_major_dim = nb::list(),
                         const nb::list& dst_offsets_major_dim = nb::list(),
                         const nb::list& copy_sizes_major_dim = nb::list()) {
  transfer_h2d_async(src_arr, dst_arr, src_offsets_major_dim,
                     dst_offsets_major_dim, copy_sizes_major_dim)
      .Await();
}

inline void await_all(const nb::object& future_obj) {
  if (nb::isinstance<PjRtCopyFuture>(future_obj)) {
    nb::cast<PjRtCopyFuture&>(future_obj).Await();
  } else if (nb::isinstance<nb::list>(future_obj)) {
    nb::list futures = nb::cast<nb::list>(future_obj);
    for (size_t i = 0; i < futures.size(); ++i) {
      nb::cast<PjRtCopyFuture&>(futures[i]).Await();
    }
  }
}

void BuildRawTransferAPI(nanobind::module_& m);

}  // namespace jax

#endif  // JAXLIB_PY_RAW_TRANSFER_IMPL_H_
