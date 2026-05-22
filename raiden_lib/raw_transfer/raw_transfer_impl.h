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

#ifndef THIRD_PARTY_TPU_RAIDEN_RAIDEN_LIB_RAW_TRANSFER_RAW_TRANSFER_IMPL_H_
#define THIRD_PARTY_TPU_RAIDEN_RAIDEN_LIB_RAW_TRANSFER_RAW_TRANSFER_IMPL_H_

#include <Python.h>

#include <memory>
#include <optional>
#include <typeinfo>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "jaxlib/py_array.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_external.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "raiden_lib/raw_transfer/raw_transfer_core.h"

namespace nb = nanobind;

namespace jax {

inline xla::ifrt::PjRtCompatibleArray* CastToPjRtCompatibleArray(
    xla::ifrt::Array* ifrt_array) {
  return llvm::dyn_cast_or_null<xla::ifrt::PjRtCompatibleArray>(ifrt_array);
}

inline absl::Span<const std::shared_ptr<xla::PjRtBuffer>>
GetPjrtBuffersFromIfrtArray(xla::ifrt::Array* ifrt_array) {
  auto* arr = CastToPjRtCompatibleArray(ifrt_array);
  if (arr != nullptr) {
    return arr->pjrt_buffers();
  }
  if (ifrt_array->client()->runtime_type() == "pjrt_ifrt") {
    auto* pjrt_array = static_cast<xla::ifrt::PjRtArray*>(ifrt_array);
    return pjrt_array->xla::ifrt::PjRtArray::pjrt_buffers();
  }
  throw std::runtime_error("Not a PjRt compatible array");
}

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

  return GetPjrtBuffersFromIfrtArray(ifrt_array).front().get();
}

inline xla::ifrt::Array* GetIfrtArrayFromPyObject(PyObject* obj) {
  auto* py_array_obj = reinterpret_cast<PyArrayObject*>(obj);
  if (!py_array_obj->initialized) {
    throw std::runtime_error("PyArrayObject not initialized");
  }
  auto* storage = GetPyArrayStorageFromObject(py_array_obj);
  return storage->ifrt_array.get();
}

}  // namespace jax

namespace raiden {

using namespace xla;



inline absl::StatusOr<PjRtCopyFuture> transfer_d2h_internal(
    const nb::object& src_arr, const nb::object& dst_arr,
    const nb::list& src_offsets_major_dim,
    const nb::list& dst_offsets_major_dim,
    const nb::list& copy_sizes_major_dim) {
  PjRtCopyFuture acc({});
  nb::object addressable_shards = src_arr.attr("addressable_shards");
  size_t num_shards = nb::len(addressable_shards);

  if (num_shards == 0) {
    return acc;
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

  PjRtBuffer* first_src_buffer =
      jax::GetPjrtBufferFromPyObject(first_shard_data.ptr());

  TF_ASSIGN_OR_RETURN(int64_t physical_size,
                      first_src_buffer->GetOnDeviceSizeInBytes());

  const PJRT_Api* c_api = nullptr;
  const PJRT_RawBuffer_Extension* extension =
      GetRawBufferExtension(first_src_buffer, &c_api);
  PjRtCApiBuffer* first_capi_buffer =
      AsPjRtCApiBuffer(first_src_buffer);

  if (first_capi_buffer && !extension) {
    throw std::runtime_error("RawBuffer extension not found in PjRtCApiClient");
  }

  int64_t slice_byte_size = GetMajorSliceByteSize(first_src_buffer);

  if (is_partial && slice_byte_size % 4096 != 0) {
    throw std::runtime_error(
        "Unsupported shape: slice byte size must be a multiple "
        "of tile size (4KB) on device for partial copies");
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
        AsCommonPjRtBuffer(src_buffer);
    PjRtCApiBuffer* capi_buffer = AsPjRtCApiBuffer(src_buffer);

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
      TF_ASSIGN_OR_RETURN(auto ext_ref, src_buffer->AcquireExternalReference());
      acc.AddExtHold(std::move(ext_ref));
    } else {
      throw std::runtime_error(std::string("Unsupported buffer type! Type: ") +
                               typeid(*src_buffer).name());
    }

    if (!is_partial) {
      // Full copy.
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

        int64_t physical_offset = src_major_dim_offset * slice_byte_size;
        int64_t size_to_copy = major_dim_size * slice_byte_size;
        int64_t dst_offset = dst_major_dim_offset * slice_byte_size;

        if (physical_offset + size_to_copy > physical_size) {
          throw std::runtime_error("Copy range exceeds source buffer size. "
                                   "physical_offset: " + std::to_string(physical_offset) +
                                   ", size_to_copy: " + std::to_string(size_to_copy) +
                                   ", physical_size: " + std::to_string(physical_size) +
                                   ", slice_byte_size: " + std::to_string(slice_byte_size) +
                                   ", major_dim_size: " + std::to_string(major_dim_size) +
                                   ", src_major_dim_offset: " + std::to_string(src_major_dim_offset));
        }

        if (dst_offset + size_to_copy > dst_size) {
          throw std::runtime_error(
              "Copy range exceeds destination buffer size. "
              "dst_offset: " + std::to_string(dst_offset) +
              ", size_to_copy: " + std::to_string(size_to_copy) +
              ", dst_size: " + std::to_string(dst_size) +
              ", slice_byte_size: " + std::to_string(slice_byte_size) +
              ", major_dim_size: " + std::to_string(major_dim_size) +
              ", dst_major_dim_offset: " + std::to_string(dst_major_dim_offset));
        }

        uint8_t* dst_ptr = dst_data + dst_offset;

        xla::Future<> future;
        {
          nb::gil_scoped_release release;
          future =
              src_buffer->CopyRawToHost(dst_ptr, physical_offset, size_to_copy);
        }
        shard_futures.push_back(std::move(future));
      }
    }
    acc.Append(std::move(shard_futures), c_api_hold);
  }
  return acc;
}

inline absl::StatusOr<PjRtCopyFuture> transfer_d2h_async(
    const nb::object& src_arr, const nb::object& dst_arr,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list()) {
  return transfer_d2h_internal(src_arr, dst_arr, src_offsets_major_dim,
                               dst_offsets_major_dim, copy_sizes_major_dim);
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
  PjRtCopyFuture acc({});
  for (size_t i = 0; i < n; ++i) {
    acc.Append(ValueOrThrow(
        transfer_d2h_internal(src_arrs[i], dst_arrs[i], src_offsets_major_dim,
                              dst_offsets_major_dim, copy_sizes_major_dim)));
  }
  return acc;
}

inline absl::StatusOr<PjRtCopyFuture> transfer_d2h_batch_async_impl(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list()) {
  if (nb::len(src_arrs) != nb::len(dst_arrs)) {
    throw std::runtime_error("Lengths of src_arrs and dst_arrs must match");
  }
  size_t n = nb::len(src_arrs);
  PjRtCopyFuture acc({});
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

  TF_ASSIGN_OR_RETURN(int64_t physical_size,
                      first_buffer->GetOnDeviceSizeInBytes());

  nb::object first_dst_shard_data =
      first_dst_arr.attr("addressable_shards")[0].attr("data");
  PjRtBuffer* first_dst_buffer =
      jax::GetPjrtBufferFromPyObject(first_dst_shard_data.ptr());
  TF_ASSIGN_OR_RETURN(size_t dst_size,
                      first_dst_buffer->GetOnDeviceSizeInBytes());

  const PJRT_Api* c_api = nullptr;
  const PJRT_RawBuffer_Extension* extension =
      GetRawBufferExtension(first_buffer, &c_api);
  PjRtCApiBuffer* first_capi_buffer =
      AsPjRtCApiBuffer(first_buffer);

  if (first_capi_buffer && !extension) {
    throw std::runtime_error("RawBuffer extension not found in PjRtCApiClient");
  }

  bool is_common_buffer =
      (AsCommonPjRtBuffer(first_buffer) != nullptr);

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

        auto src_buffers = jax::GetPjrtBuffersFromIfrtArray(src_ifrt_array);

        auto dst_buffers = jax::GetPjrtBuffersFromIfrtArray(dst_ifrt_array);

        if (src_buffers.size() != num_shards ||
            dst_buffers.size() != num_shards) {
          throw std::runtime_error("Number of shards mismatch");
        }

        for (size_t i = 0; i < num_shards; ++i) {
          PjRtBuffer* src_buffer = src_buffers[i].get();
          PjRtBuffer* dst_buffer = dst_buffers[i].get();

          TF_ASSIGN_OR_RETURN(
              uintptr_t u_ptr,
              dst_buffer->client()->UnsafeBufferPointer(dst_buffer));
          uint8_t* dst_data = reinterpret_cast<uint8_t*>(u_ptr);

          if (dst_size < physical_size) {
            throw std::runtime_error(
                "Destination buffer too small for raw tiled copy");
          }

          CommonPjRtBuffer* common_buffer =
              AsCommonPjRtBuffer(src_buffer);

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

        auto src_buffers = jax::GetPjrtBuffersFromIfrtArray(src_ifrt_array);

        auto dst_buffers = jax::GetPjrtBuffersFromIfrtArray(dst_ifrt_array);

        if (src_buffers.size() != num_shards ||
            dst_buffers.size() != num_shards) {
          throw std::runtime_error("Number of shards mismatch");
        }

        for (size_t i = 0; i < num_shards; ++i) {
          PjRtBuffer* src_buffer = src_buffers[i].get();
          PjRtBuffer* dst_buffer = dst_buffers[i].get();

          TF_ASSIGN_OR_RETURN(
              uintptr_t u_ptr,
              dst_buffer->client()->UnsafeBufferPointer(dst_buffer));
          uint8_t* dst_data = reinterpret_cast<uint8_t*>(u_ptr);

          if (dst_size < physical_size) {
            throw std::runtime_error(
                "Destination buffer too small for raw tiled copy");
          }
          PjRtCApiBuffer* capi_buffer =
              AsPjRtCApiBuffer(src_buffer);

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

  int64_t slice_byte_size = GetMajorSliceByteSize(first_buffer);

  if (slice_byte_size % 4096 != 0) {
    throw std::runtime_error(
        "Unsupported shape: slice byte size must be a multiple "
        "of tile size (4KB)");
  }

  for (size_t layer_idx = 0; layer_idx < n; ++layer_idx) {
    nb::object src = src_arrs[layer_idx];
    nb::object dst = dst_arrs[layer_idx];

    xla::ifrt::Array* src_ifrt_array = jax::GetIfrtArrayFromPyObject(src.ptr());
    xla::ifrt::Array* dst_ifrt_array = jax::GetIfrtArrayFromPyObject(dst.ptr());

    auto src_buffers = jax::GetPjrtBuffersFromIfrtArray(src_ifrt_array);

    auto dst_buffers = jax::GetPjrtBuffersFromIfrtArray(dst_ifrt_array);

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
          AsCommonPjRtBuffer(src_buffer);
      PjRtCApiBuffer* capi_buffer = AsPjRtCApiBuffer(src_buffer);

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

        TF_ASSIGN_OR_RETURN(c_raw_buffer,
                            pjrt::PjRtCApiBuffer_CreateRawAliasOfBuffer(
                                c_api, extension, capi_buffer->c_buffer()));
        c_api_hold =
            std::make_shared<RawBufferHolder>(c_api, extension, c_raw_buffer);
      }

      for (size_t j = 0; j < src_offsets_major_dim.size(); ++j) {
        int64_t src_major_dim_offset =
            nb::cast<int64_t>(src_offsets_major_dim[j]);
        int64_t dst_major_dim_offset =
            nb::cast<int64_t>(dst_offsets_major_dim[j]);
        int64_t major_dim_size = nb::cast<int64_t>(copy_sizes_major_dim[j]);

        int64_t physical_offset = src_major_dim_offset * slice_byte_size;
        int64_t size_to_copy = major_dim_size * slice_byte_size;
        int64_t dst_offset = dst_major_dim_offset * slice_byte_size;

        if (physical_offset + size_to_copy > physical_size) {
          throw std::runtime_error("Copy range exceeds source buffer size. "
                                   "physical_offset: " + std::to_string(physical_offset) +
                                   ", size_to_copy: " + std::to_string(size_to_copy) +
                                   ", physical_size: " + std::to_string(physical_size) +
                                   ", slice_byte_size: " + std::to_string(slice_byte_size) +
                                   ", major_dim_size: " + std::to_string(major_dim_size) +
                                   ", src_major_dim_offset: " + std::to_string(src_major_dim_offset));
        }
        if (dst_offset + size_to_copy > dst_size) {
          throw std::runtime_error(
              "Copy range exceeds destination buffer size. "
              "dst_offset: " + std::to_string(dst_offset) +
              ", size_to_copy: " + std::to_string(size_to_copy) +
              ", dst_size: " + std::to_string(dst_size) +
              ", slice_byte_size: " + std::to_string(slice_byte_size) +
              ", major_dim_size: " + std::to_string(major_dim_size) +
              ", dst_major_dim_offset: " + std::to_string(dst_major_dim_offset));
        }
        uint8_t* dst_ptr = dst_data + dst_offset;

        xla::Future<> future;
        if (common_buffer) {
          future =
              src_buffer->CopyRawToHost(dst_ptr, physical_offset, size_to_copy);
        } else if (capi_buffer) {
          future = pjrt::PjRtCApiRawBuffer_CopyRawDeviceToHost(
              c_api, extension, c_raw_buffer, dst_ptr, physical_offset,
              size_to_copy);
        }
        shard_futures.push_back(std::move(future));
      }
      acc.Append(std::move(shard_futures), c_api_hold);
    }
  }
  return acc;
}

inline absl::StatusOr<PjRtCopyFuture> transfer_h2d_internal(
    const nb::object& src_arr, const nb::object& dst_arr,
    const nb::list& src_offsets_major_dim,
    const nb::list& dst_offsets_major_dim,
    const nb::list& copy_sizes_major_dim) {
  PjRtCopyFuture acc({});
  nb::object addressable_shards = dst_arr.attr("addressable_shards");
  size_t num_shards = nb::len(addressable_shards);

  if (num_shards == 0) {
    return acc;
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

  PjRtBuffer* first_dst_buffer =
      jax::GetPjrtBufferFromPyObject(first_shard_data.ptr());

  auto status_or_dst_size = first_dst_buffer->GetOnDeviceSizeInBytes();
  if (!status_or_dst_size.ok()) {
    throw std::runtime_error("Failed to get destination buffer size");
  }
  int64_t physical_size = status_or_dst_size.value();

  const PJRT_Api* c_api = nullptr;
  const PJRT_RawBuffer_Extension* extension =
      GetRawBufferExtension(first_dst_buffer, &c_api);
  PjRtCApiBuffer* first_capi_buffer =
      AsPjRtCApiBuffer(first_dst_buffer);

  if (first_capi_buffer && !extension) {
    throw std::runtime_error("RawBuffer extension not found in PjRtCApiClient");
  }

  int64_t slice_byte_size = GetMajorSliceByteSize(first_dst_buffer);

  if (is_partial && slice_byte_size % 4096 != 0) {
    throw std::runtime_error(
        "Unsupported shape: slice byte size must be a multiple "
        "of tile size (4KB) on device for partial copies");
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
        AsCommonPjRtBuffer(dst_buffer);
    PjRtCApiBuffer* capi_buffer = AsPjRtCApiBuffer(dst_buffer);

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

        int64_t physical_offset = dst_major_dim_offset * slice_byte_size;
        int64_t size_to_copy = major_dim_size * slice_byte_size;
        int64_t src_offset = src_major_dim_offset * slice_byte_size;

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
      }
    }
  }
  return acc;
}

inline absl::StatusOr<PjRtCopyFuture> transfer_h2d_async(
    const nb::object& src_arr, const nb::object& dst_arr,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list()) {
  return transfer_h2d_internal(src_arr, dst_arr, src_offsets_major_dim,
                               dst_offsets_major_dim, copy_sizes_major_dim);
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
  PjRtCopyFuture acc({});
  for (size_t i = 0; i < n; ++i) {
    acc.Append(ValueOrThrow(
        transfer_h2d_internal(src_arrs[i], dst_arrs[i], src_offsets_major_dim,
                              dst_offsets_major_dim, copy_sizes_major_dim)));
  }
  return acc;
}

inline absl::StatusOr<PjRtCopyFuture> transfer_h2d_batch_async_impl(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim = nb::list(),
    const nb::list& dst_offsets_major_dim = nb::list(),
    const nb::list& copy_sizes_major_dim = nb::list()) {
  if (nb::len(src_arrs) != nb::len(dst_arrs)) {
    throw std::runtime_error("Lengths of src_arrs and dst_arrs must match");
  }
  size_t n = nb::len(src_arrs);
  PjRtCopyFuture acc({});
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
  const PJRT_RawBuffer_Extension* extension =
      GetRawBufferExtension(first_buffer, &c_api);
  PjRtCApiBuffer* first_capi_buffer =
      AsPjRtCApiBuffer(first_buffer);

  if (first_capi_buffer) {
    if (!extension) {
      throw std::runtime_error(
          "RawBuffer extension not found in PjRtCApiClient");
    }
  }

  bool is_common_buffer =
      (AsCommonPjRtBuffer(first_buffer) != nullptr);

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

        auto src_buffers = jax::GetPjrtBuffersFromIfrtArray(src_ifrt_array);

        auto dst_buffers = jax::GetPjrtBuffersFromIfrtArray(dst_ifrt_array);

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
              AsCommonPjRtBuffer(dst_buffer);

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

        auto src_buffers = jax::GetPjrtBuffersFromIfrtArray(src_ifrt_array);

        auto dst_buffers = jax::GetPjrtBuffersFromIfrtArray(dst_ifrt_array);

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
              AsPjRtCApiBuffer(dst_buffer);

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

  int64_t slice_byte_size = GetMajorSliceByteSize(first_buffer);

  if (slice_byte_size % 4096 != 0) {
    throw std::runtime_error(
        "Unsupported shape: slice byte size must be a multiple "
        "of tile size (4KB)");
  }

  for (size_t layer_idx = 0; layer_idx < n; ++layer_idx) {
    nb::object src = src_arrs[layer_idx];
    nb::object dst = dst_arrs[layer_idx];

    xla::ifrt::Array* src_ifrt_array = jax::GetIfrtArrayFromPyObject(src.ptr());
    xla::ifrt::Array* dst_ifrt_array = jax::GetIfrtArrayFromPyObject(dst.ptr());

    auto src_buffers = jax::GetPjrtBuffersFromIfrtArray(src_ifrt_array);

    auto dst_buffers = jax::GetPjrtBuffersFromIfrtArray(dst_ifrt_array);

    if (src_buffers.size() != num_shards || dst_buffers.size() != num_shards) {
      throw std::runtime_error("Number of shards mismatch");
    }

    for (size_t i = 0; i < num_shards; ++i) {
      PjRtBuffer* src_buffer = src_buffers[i].get();
      PjRtBuffer* dst_buffer = dst_buffers[i].get();

      auto status_or_ptr =
          src_buffer->client()->UnsafeBufferPointer(src_buffer);
      if (!status_or_ptr.ok()) {
        throw std::runtime_error("Failed to get unsafe buffer pointer for src");
      }
      const uint8_t* src_data =
          reinterpret_cast<const uint8_t*>(status_or_ptr.value());

      size_t src_size = src_buffer->GetOnDeviceSizeInBytes().value();

      CommonPjRtBuffer* common_buffer =
          AsCommonPjRtBuffer(dst_buffer);
      PjRtCApiBuffer* capi_buffer = AsPjRtCApiBuffer(dst_buffer);

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

        int64_t physical_offset = dst_major_dim_offset * slice_byte_size;
        int64_t size_to_copy = major_dim_size * slice_byte_size;
        int64_t src_offset = src_major_dim_offset * slice_byte_size;

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
      }
    }
  }
  return acc;
}

inline void transfer_d2h(const nb::object& src_arr, const nb::object& dst_arr,
                         const nb::list& src_offsets_major_dim = nb::list(),
                         const nb::list& dst_offsets_major_dim = nb::list(),
                         const nb::list& copy_sizes_major_dim = nb::list()) {
  auto future = ValueOrThrow(
      transfer_d2h_async(src_arr, dst_arr, src_offsets_major_dim,
                         dst_offsets_major_dim, copy_sizes_major_dim));
  nb::gil_scoped_release release;
  future.Await();
}

inline void transfer_h2d(const nb::object& src_arr, const nb::object& dst_arr,
                         const nb::list& src_offsets_major_dim = nb::list(),
                         const nb::list& dst_offsets_major_dim = nb::list(),
                         const nb::list& copy_sizes_major_dim = nb::list()) {
  auto future = ValueOrThrow(
      transfer_h2d_async(src_arr, dst_arr, src_offsets_major_dim,
                         dst_offsets_major_dim, copy_sizes_major_dim));
  nb::gil_scoped_release release;
  future.Await();
}

inline void await_all(const nb::object& future_obj) {
  if (nb::isinstance<PjRtCopyFuture>(future_obj)) {
    PjRtCopyFuture& future = nb::cast<PjRtCopyFuture&>(future_obj);
    nb::gil_scoped_release release;
    future.Await();
  } else if (nb::isinstance<nb::list>(future_obj)) {
    nb::list futures = nb::cast<nb::list>(future_obj);
    for (size_t i = 0; i < futures.size(); ++i) {
      PjRtCopyFuture& future = nb::cast<PjRtCopyFuture&>(futures[i]);
      nb::gil_scoped_release release;
      future.Await();
    }
  }
}

inline bool is_ready(const nb::object& future_obj) {
  if (nb::isinstance<PjRtCopyFuture>(future_obj)) {
    return nb::cast<const PjRtCopyFuture&>(future_obj).IsReady();
  } else if (nb::isinstance<nb::list>(future_obj)) {
    nb::list futures = nb::cast<nb::list>(future_obj);
    for (size_t i = 0; i < futures.size(); ++i) {
      if (!nb::cast<const PjRtCopyFuture&>(futures[i]).IsReady()) {
        return false;
      }
    }
    return true;
  }
  return true;
}

}  // namespace raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_RAIDEN_LIB_RAW_TRANSFER_RAW_TRANSFER_IMPL_H_
