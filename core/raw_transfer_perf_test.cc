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

// This file contains performance benchmarks for raw transfers (H2D and D2H).
//
// By default, the test suite runs a lightweight configuration (4 data types,
// 64 layers, 16 blocks) to keep presubmit runs fast (~1 minute).
//
// To manually reproduce the deeper benchmark scenarios previously defined in
// this file (e.g., to test PCIe roofline saturation), you can run the test with
// the following absl flags:
//
// 1. Base (Default):
//    blaze test ... --test_arg=--num_layers=64 --test_arg=--num_blocks=16
//
// 2. Medium:
//    blaze test ... --test_arg=--num_layers=128 --test_arg=--num_blocks=32
//
// 3. Large:
//    blaze test ... --test_arg=--num_layers=256 --test_arg=--num_blocks=32
//    (Note: INT32_Large historically used --num_blocks=16)
//
// 4. Extreme:
//    blaze test ... --test_arg=--num_layers=1024 --test_arg=--num_blocks=8
//
// You can also control the number of TPUs to use with --test_arg=--num_tpus=N.
// For example, to run the Extreme benchmark on all 8 TPUs of a Ghostfish host:
//   blaze test --nocheck_visibility -c opt \
//     --test_arg=--num_tpus=8 \
//     --test_arg=--num_layers=1024 \
//     --test_arg=--num_blocks=8 \
//     //core:raw_transfer_perf_test

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "core/host_memory_allocator.h"
#include "core/raw_transfer_impl.h"
#include "core/tpu_pjrt_manager.h"

ABSL_FLAG(int, num_tpus, 1, "Number of TPUs to use");
ABSL_FLAG(int, num_layers, 64, "Number of layers to use for the benchmark.");
ABSL_FLAG(int64_t, num_blocks, 16,
          "Number of blocks to use for the benchmark.");

namespace raiden {
namespace {

// Static initializer to load and initialize libtpu.so in OSS environment.
bool InitializeLibtpuOnce() {
  const char* libtpu_path = std::getenv("TPU_LIBRARY_PATH");
  if (!libtpu_path || std::string(libtpu_path).empty()) {
    libtpu_path = "libtpu.so";
  }
  std::cout << "[INFO] Dynamically loading libtpu from: " << libtpu_path
            << std::endl;
  void* library = dlopen(libtpu_path, RTLD_NOW | RTLD_GLOBAL);
  if (!library) {
    std::cerr << "[WARNING] Failed to dlopen libtpu.so: " << dlerror()
              << ". This might be expected if not running on TPU hardware."
              << std::endl;
    return false;
  }
  auto initialize_fn = reinterpret_cast<void (*)(bool, int, const char**)>(
      dlsym(library, "TfTpu_Initialize"));
  if (!initialize_fn) {
    std::cerr << "[WARNING] TfTpu_Initialize symbol not found in libtpu.so"
              << std::endl;
    dlclose(library);
    return false;
  }
  std::cout << "[INFO] Calling TfTpu_Initialize(true, 0, nullptr)."
            << std::endl;
  initialize_fn(/*init_library=*/true, 0, nullptr);
  std::cout << "[INFO] libtpu.so initialized successfully." << std::endl;
  return true;
}

static bool libtpu_dummy = InitializeLibtpuOnce();

}  // namespace
}  // namespace raiden

#ifndef ASSERT_OK
#define ASSERT_OK(status) ASSERT_TRUE((status).ok())
#endif

namespace raiden {
namespace {


// Helper to compute and print the distribution of bandwidth numbers
void PrintDistribution(const std::string& label,
                       std::vector<double> bandwidths) {
  if (bandwidths.empty()) return;

  std::sort(bandwidths.begin(), bandwidths.end());
  double min = bandwidths.front();
  double max = bandwidths.back();
  double sum = 0.0;
  for (double bw : bandwidths) sum += bw;
  double mean = sum / bandwidths.size();

  double median = 0.0;
  if (bandwidths.size() % 2 == 0) {
    median = (bandwidths[bandwidths.size() / 2 - 1] +
              bandwidths[bandwidths.size() / 2]) /
             2.0;
  } else {
    median = bandwidths[bandwidths.size() / 2];
  }

  double variance = 0.0;
  for (double bw : bandwidths) {
    variance += (bw - mean) * (bw - mean);
  }
  double stddev = std::sqrt(variance / bandwidths.size());

  std::cout << "[DISTRIBUTION] " << label << " (over " << bandwidths.size()
            << " runs):" << std::endl;
  std::cout << "  Min:    " << min << " GB/s" << std::endl;
  std::cout << "  Max:    " << max << " GB/s" << std::endl;
  std::cout << "  Mean:   " << mean << " GB/s" << std::endl;
  std::cout << "  Median: " << median << " GB/s" << std::endl;
  std::cout << "  StdDev: " << stddev << " GB/s" << std::endl;
}

// Templated benchmark executor for Scenario A (Fragmented Batch)
template <typename T>
void RunBenchmarkScenarioA(tpu_raiden::TpuPjrtManager* manager,
                           const std::vector<xla::PjRtDevice*>& devices,
                           xla::PrimitiveType primitive_type,
                           const std::string& type_label, int num_layers,
                           int64_t num_blocks, double min_d2h_bandwidth_gb_s,
                           double min_h2d_bandwidth_gb_s,
                           bool should_gate_performance) {
  const int kNumLayers = num_layers;
  const int64_t kNumBlocks = num_blocks;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kNumHeads = 8;
  constexpr int64_t kHeadDim = 128;

  std::vector<int64_t> layer_shape = {kNumBlocks, kBlockSize, kNumHeads, 2,
                                      kHeadDim};
  int64_t elements_per_layer =
      kNumBlocks * kBlockSize * kNumHeads * 2 * kHeadDim;
  size_t bytes_per_layer = elements_per_layer * sizeof(T);
  size_t total_bytes = bytes_per_layer * kNumLayers;

  // We only copy ODD blocks (1, 3, 5, 7, 9, 11, 13, 15) -> exactly 50% of data
  std::vector<int64_t> src_offsets;
  std::vector<int64_t> dst_offsets;
  std::vector<int64_t> copy_sizes;
  for (int64_t b = 1; b < kNumBlocks; b += 2) {
    src_offsets.push_back(b);
    dst_offsets.push_back(b);
    copy_sizes.push_back(1);
  }
  size_t bytes_transferred = (bytes_per_layer / 2) * kNumLayers;

  std::cout << "[INFO] Scenario A: Fragmented Batch (ODD BLOCKS ONLY) ["
            << type_label << "]. Total layers: " << kNumLayers
            << ", Layer size: " << bytes_per_layer / (1024.0 * 1024.0) << " MB"
            << ", Total size: " << total_bytes / (1024.0 * 1024.0) << " MB"
            << ", Transferred size: " << bytes_transferred / (1024.0 * 1024.0)
            << " MB" << std::endl;

  // 1. Allocate Host Staging Buffers (Page Aligned, Zero-initialized)
  auto allocator_or =
      tpu_raiden::HostMemoryAllocator::Create(manager->client());
  ASSERT_OK(allocator_or.status());
  auto host_allocator = std::move(allocator_or).value();

  std::vector<tpu_raiden::HostBufferAllocation> host_src_buffers;
  std::vector<tpu_raiden::HostBufferAllocation> host_dst_buffers;
  std::vector<const uint8_t*> host_src_ptrs;
  std::vector<uint8_t*> host_dst_ptrs;
  std::vector<size_t> host_sizes;

  host_src_buffers.reserve(kNumLayers);
  host_dst_buffers.reserve(kNumLayers);
  host_src_ptrs.reserve(kNumLayers);
  host_dst_ptrs.reserve(kNumLayers);
  host_sizes.reserve(kNumLayers);

  for (int i = 0; i < kNumLayers; ++i) {
    auto src_alloc_or = host_allocator->AllocateDmaMapped(bytes_per_layer);
    ASSERT_OK(src_alloc_or.status());
    host_src_buffers.push_back(std::move(src_alloc_or).value());
    std::memset(host_src_buffers.back().ptr, 0, bytes_per_layer);

    auto dst_alloc_or = host_allocator->AllocateDmaMapped(bytes_per_layer);
    ASSERT_OK(dst_alloc_or.status());
    host_dst_buffers.push_back(std::move(dst_alloc_or).value());
    std::memset(host_dst_buffers.back().ptr, 0, bytes_per_layer);

    // Initialize source host memory with dummy values
    T* src_ptr = reinterpret_cast<T*>(host_src_buffers.back().ptr);
    for (int64_t e = 0; e < elements_per_layer; ++e) {
      if constexpr (std::is_same_v<T, float>) {
        src_ptr[e] = static_cast<float>((i + e) % 65536) / 1000.0f;
      } else {
        // Fallback for BF16 raw bit representation
        src_ptr[e] = static_cast<T>((i + e) % 65536);
      }
    }

    host_src_ptrs.push_back(host_src_buffers.back().ptr);
    host_dst_ptrs.push_back(host_dst_buffers.back().ptr);
    host_sizes.push_back(bytes_per_layer);
  }

  // 2. Allocate TPU Device Buffers (fully populated from host_src)
  std::vector<std::unique_ptr<xla::PjRtBuffer>> device_buffers;
  std::vector<xla::PjRtBuffer*> device_buffer_ptrs;
  device_buffers.reserve(kNumLayers);
  device_buffer_ptrs.reserve(kNumLayers);

  for (int i = 0; i < kNumLayers; ++i) {
    xla::PjRtDevice* device = devices[i % devices.size()];
    auto buf_or = manager->BufferFromHost(host_src_ptrs[i], primitive_type,
                                          layer_shape, device);
    ASSERT_OK(buf_or.status());
    device_buffers.push_back(std::move(buf_or).value());
    device_buffer_ptrs.push_back(device_buffers.back().get());
  }

  // Await TPU buffer allocation/copies to complete
  for (auto* buf : device_buffer_ptrs) {
    ASSERT_OK(buf->GetReadyFuture().Await());
  }

  // 3. Warmup Phase
  constexpr int kWarmupIterations = 3;
  for (int iter = 0; iter < kWarmupIterations; ++iter) {
    auto d2h_future_or =
        transfer_d2h_core(device_buffer_ptrs, host_dst_ptrs, host_sizes,
                          src_offsets, dst_offsets, copy_sizes);
    ASSERT_OK(d2h_future_or.status());
    ASSERT_OK(d2h_future_or.value().Await().status());

    auto h2d_future_or =
        transfer_h2d_core(device_buffer_ptrs, host_src_ptrs, host_sizes,
                          src_offsets, dst_offsets, copy_sizes);
    ASSERT_OK(h2d_future_or.status());
    ASSERT_OK(h2d_future_or.value().Await().status());
  }

  // 4. Timed Benchmark Loop (D2H) - Measure each iteration
  constexpr int kBenchmarkIterations = 50;
  std::vector<double> d2h_bandwidths;
  d2h_bandwidths.reserve(kBenchmarkIterations);

  for (int iter = 0; iter < kBenchmarkIterations; ++iter) {
    absl::Time start = absl::Now();
    auto d2h_future_or =
        transfer_d2h_core(device_buffer_ptrs, host_dst_ptrs, host_sizes,
                          src_offsets, dst_offsets, copy_sizes);
    ASSERT_OK(d2h_future_or.status());
    ASSERT_OK(d2h_future_or.value().Await().status());
    absl::Duration duration = absl::Now() - start;
    double seconds = absl::ToDoubleSeconds(duration);
    double bandwidth =
        (bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / seconds;
    d2h_bandwidths.push_back(bandwidth);
  }
  PrintDistribution(
      "Scenario A D2H Bandwidth (Odd Blocks) [" + type_label + "]",
      d2h_bandwidths);

  // 5. Timed Benchmark Loop (H2D) - Measure each iteration
  std::vector<double> h2d_bandwidths;
  h2d_bandwidths.reserve(kBenchmarkIterations);

  for (int iter = 0; iter < kBenchmarkIterations; ++iter) {
    absl::Time start = absl::Now();
    auto h2d_future_or =
        transfer_h2d_core(device_buffer_ptrs, host_src_ptrs, host_sizes,
                          src_offsets, dst_offsets, copy_sizes);
    ASSERT_OK(h2d_future_or.status());
    ASSERT_OK(h2d_future_or.value().Await().status());
    absl::Duration duration = absl::Now() - start;
    double seconds = absl::ToDoubleSeconds(duration);
    double bandwidth =
        (bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / seconds;
    h2d_bandwidths.push_back(bandwidth);
  }
  PrintDistribution(
      "Scenario A H2D Bandwidth (Odd Blocks) [" + type_label + "]",
      h2d_bandwidths);

  // 6. Verify Correctness (Odd blocks must match, Even blocks must remain 0)
  size_t bytes_per_block = bytes_per_layer / kNumBlocks;
  for (int i = 0; i < kNumLayers; ++i) {
    uint8_t* src_ptr = const_cast<uint8_t*>(host_src_ptrs[i]);
    uint8_t* dst_ptr = host_dst_ptrs[i];
    for (int64_t b = 0; b < kNumBlocks; ++b) {
      if (b % 2 == 1) {
        int cmp = std::memcmp(src_ptr + b * bytes_per_block,
                              dst_ptr + b * bytes_per_block, bytes_per_block);
        EXPECT_EQ(cmp, 0) << "Data mismatch in layer " << i << ", odd block "
                          << b;
      } else {
        for (size_t k = 0; k < bytes_per_block; ++k) {
          EXPECT_EQ(dst_ptr[b * bytes_per_block + k], 0)
              << "Even block " << b << " in layer " << i
              << " was touched at byte " << k;
        }
      }
    }
  }

  // 7. Gate Performance against median (Only on supported GLP/GF platforms)
  if (should_gate_performance) {
    std::sort(d2h_bandwidths.begin(), d2h_bandwidths.end());
    std::sort(h2d_bandwidths.begin(), h2d_bandwidths.end());
    double median_d2h = d2h_bandwidths[kBenchmarkIterations / 2];
    double median_h2d = h2d_bandwidths[kBenchmarkIterations / 2];

    EXPECT_GE(median_d2h, min_d2h_bandwidth_gb_s)
        << "Median D2H Bandwidth in Scenario A [" << type_label
        << "] dropped below threshold of " << min_d2h_bandwidth_gb_s << " GB/s";
    EXPECT_GE(median_h2d, min_h2d_bandwidth_gb_s)
        << "Median H2D Bandwidth in Scenario A [" << type_label
        << "] dropped below threshold of " << min_h2d_bandwidth_gb_s << " GB/s";
  }
}

// Templated benchmark executor for Scenario B (Baked-in Tensor)
template <typename T>
void RunBenchmarkScenarioB(tpu_raiden::TpuPjrtManager* manager,
                           const std::vector<xla::PjRtDevice*>& devices,
                           xla::PrimitiveType primitive_type,
                           const std::string& type_label, int num_layers,
                           int64_t num_blocks, double min_d2h_bandwidth_gb_s,
                           double min_h2d_bandwidth_gb_s,
                           bool should_gate_performance) {
  int num_devices = devices.size();
  const int kNumLayers = std::max(1, num_layers / num_devices);
  const int64_t kNumBlocks = num_blocks;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kNumHeads = 8;
  constexpr int64_t kHeadDim = 128;

  std::vector<int64_t> baked_shape = {kNumBlocks, kNumLayers, kBlockSize,
                                      kNumHeads,  2,          kHeadDim};
  int64_t total_elements =
      kNumBlocks * kNumLayers * kBlockSize * kNumHeads * 2 * kHeadDim;
  size_t total_bytes = total_elements * sizeof(T);

  // We only copy ODD blocks (1, 3, 5, 7, 9, 11, 13, 15) -> exactly 50% of data
  std::vector<int64_t> src_offsets;
  std::vector<int64_t> dst_offsets;
  std::vector<int64_t> copy_sizes;
  for (int64_t b = 1; b < kNumBlocks; b += 2) {
    src_offsets.push_back(b);
    dst_offsets.push_back(b);
    copy_sizes.push_back(1);
  }

  size_t bytes_transferred = (total_bytes / 2) * num_devices;

  std::cout << "[INFO] Scenario B: Baked-in Tensor (ODD BLOCKS ONLY) ["
            << type_label << "]. Shape: [" << kNumBlocks << ", " << kNumLayers
            << ", " << kBlockSize << ", " << kNumHeads << ", 2, " << kHeadDim
            << "]"
            << ", Total size: "
            << (total_bytes * num_devices) / (1024.0 * 1024.0) << " MB"
            << ", Transferred size: " << bytes_transferred / (1024.0 * 1024.0)
            << " MB" << ", Devices: " << num_devices << std::endl;

  // 1. Allocate Host Staging Buffers (Page Aligned, Zero-initialized)
  auto allocator_or =
      tpu_raiden::HostMemoryAllocator::Create(manager->client());
  ASSERT_OK(allocator_or.status());
  auto host_allocator = std::move(allocator_or).value();

  // Optimize: Allocate only ONE host src buffer and reuse it for all devices
  auto src_alloc_or = host_allocator->AllocateDmaMapped(total_bytes);
  ASSERT_OK(src_alloc_or.status());
  auto host_src = std::move(src_alloc_or).value();
  std::memset(host_src.ptr, 0, total_bytes);

  // Initialize source host memory with dummy values
  T* src_ptr = reinterpret_cast<T*>(host_src.ptr);
  for (int64_t e = 0; e < total_elements; ++e) {
    if constexpr (std::is_same_v<T, float>) {
      src_ptr[e] = static_cast<float>(e % 65536) / 1000.0f;
    } else {
      src_ptr[e] = static_cast<T>(e % 65536);
    }
  }

  std::vector<tpu_raiden::HostBufferAllocation> host_dst_buffers;
  std::vector<const uint8_t*> host_src_ptrs;
  std::vector<uint8_t*> host_dst_ptrs;
  std::vector<size_t> host_sizes;

  host_dst_buffers.reserve(num_devices);
  host_src_ptrs.reserve(num_devices);
  host_dst_ptrs.reserve(num_devices);
  host_sizes.reserve(num_devices);

  for (int i = 0; i < num_devices; ++i) {
    auto dst_alloc_or = host_allocator->AllocateDmaMapped(total_bytes);
    ASSERT_OK(dst_alloc_or.status());
    host_dst_buffers.push_back(std::move(dst_alloc_or).value());
    std::memset(host_dst_buffers.back().ptr, 0, total_bytes);

    host_src_ptrs.push_back(host_src.ptr);  // Reuse the same src pointer
    host_dst_ptrs.push_back(host_dst_buffers.back().ptr);
    host_sizes.push_back(total_bytes);
  }

  // 2. Allocate TPU Device Buffers (fully populated from host_src)
  std::vector<std::unique_ptr<xla::PjRtBuffer>> device_buffers;
  std::vector<xla::PjRtBuffer*> device_buffer_ptrs;
  device_buffers.reserve(num_devices);
  device_buffer_ptrs.reserve(num_devices);

  for (int i = 0; i < num_devices; ++i) {
    auto device_buffer_or = manager->BufferFromHost(
        host_src_ptrs[i], primitive_type, baked_shape, devices[i]);
    ASSERT_OK(device_buffer_or.status());
    device_buffers.push_back(std::move(device_buffer_or).value());
    device_buffer_ptrs.push_back(device_buffers.back().get());
  }

  for (auto* buf : device_buffer_ptrs) {
    ASSERT_OK(buf->GetReadyFuture().Await());
  }

  // 3. Warmup Phase
  constexpr int kWarmupIterations = 3;
  for (int iter = 0; iter < kWarmupIterations; ++iter) {
    auto d2h_future_or =
        transfer_d2h_core(device_buffer_ptrs, host_dst_ptrs, host_sizes,
                          src_offsets, dst_offsets, copy_sizes);
    ASSERT_OK(d2h_future_or.status());
    ASSERT_OK(d2h_future_or.value().Await().status());

    auto h2d_future_or =
        transfer_h2d_core(device_buffer_ptrs, host_src_ptrs, host_sizes,
                          src_offsets, dst_offsets, copy_sizes);
    ASSERT_OK(h2d_future_or.status());
    ASSERT_OK(h2d_future_or.value().Await().status());
  }

  // 4. Timed Benchmark Loop (D2H) - Measure each iteration
  constexpr int kBenchmarkIterations = 50;
  std::vector<double> d2h_bandwidths;
  d2h_bandwidths.reserve(kBenchmarkIterations);

  for (int iter = 0; iter < kBenchmarkIterations; ++iter) {
    absl::Time start = absl::Now();
    auto d2h_future_or =
        transfer_d2h_core(device_buffer_ptrs, host_dst_ptrs, host_sizes,
                          src_offsets, dst_offsets, copy_sizes);
    ASSERT_OK(d2h_future_or.status());
    ASSERT_OK(d2h_future_or.value().Await().status());
    absl::Duration duration = absl::Now() - start;
    double seconds = absl::ToDoubleSeconds(duration);
    double bandwidth =
        (bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / seconds;
    d2h_bandwidths.push_back(bandwidth);
  }
  PrintDistribution(
      "Scenario B D2H Bandwidth (Odd Blocks) [" + type_label + "]",
      d2h_bandwidths);

  // 5. Timed Benchmark Loop (H2D) - Measure each iteration
  std::vector<double> h2d_bandwidths;
  h2d_bandwidths.reserve(kBenchmarkIterations);

  for (int iter = 0; iter < kBenchmarkIterations; ++iter) {
    absl::Time start = absl::Now();
    auto h2d_future_or =
        transfer_h2d_core(device_buffer_ptrs, host_src_ptrs, host_sizes,
                          src_offsets, dst_offsets, copy_sizes);
    ASSERT_OK(h2d_future_or.status());
    ASSERT_OK(h2d_future_or.value().Await().status());
    absl::Duration duration = absl::Now() - start;
    double seconds = absl::ToDoubleSeconds(duration);
    double bandwidth =
        (bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / seconds;
    h2d_bandwidths.push_back(bandwidth);
  }
  PrintDistribution(
      "Scenario B H2D Bandwidth (Odd Blocks) [" + type_label + "]",
      h2d_bandwidths);

  // 6. Verify Correctness
  size_t bytes_per_block = total_bytes / kNumBlocks;
  uint8_t* src_bytes = host_src.ptr;  // Use the single src buffer
  for (int i = 0; i < num_devices; ++i) {
    uint8_t* dst_bytes = host_dst_ptrs[i];
    for (int64_t b = 0; b < kNumBlocks; ++b) {
      if (b % 2 == 1) {
        int cmp = std::memcmp(src_bytes + b * bytes_per_block,
                              dst_bytes + b * bytes_per_block, bytes_per_block);
        EXPECT_EQ(cmp, 0) << "Data mismatch in Scenario B, device " << i
                          << ", odd block " << b;
      } else {
        for (size_t k = 0; k < bytes_per_block; ++k) {
          EXPECT_EQ(dst_bytes[b * bytes_per_block + k], 0)
              << "Even block " << b << " in Scenario B, device " << i
              << " was touched at byte " << k;
        }
      }
    }
  }

  // 7. Gate Performance against median (Only on supported GLP/GF platforms)
  if (should_gate_performance) {
    std::sort(d2h_bandwidths.begin(), d2h_bandwidths.end());
    std::sort(h2d_bandwidths.begin(), h2d_bandwidths.end());
    double median_d2h = d2h_bandwidths[kBenchmarkIterations / 2];
    double median_h2d = h2d_bandwidths[kBenchmarkIterations / 2];

    EXPECT_GE(median_d2h, min_d2h_bandwidth_gb_s)
        << "Median D2H Bandwidth in Scenario B [" << type_label
        << "] dropped below threshold of " << min_d2h_bandwidth_gb_s << " GB/s";
    EXPECT_GE(median_h2d, min_h2d_bandwidth_gb_s)
        << "Median H2D Bandwidth in Scenario B [" << type_label
        << "] dropped below threshold of " << min_h2d_bandwidth_gb_s << " GB/s";
  }
}

class RawTransferPerfTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(manager_, tpu_raiden::TpuPjrtManager::GetDefault());

    int num_tpus = absl::GetFlag(FLAGS_num_tpus);
    auto all_devices = manager_->client()->addressable_devices();
    int available_tpus = all_devices.size();
    std::cout << "[INFO] Available TPUs: " << available_tpus << std::endl;

    if (num_tpus > available_tpus) {
      std::cout << "[WARNING] Requested " << num_tpus << " TPUs, but only "
                << available_tpus << " are available. Using " << available_tpus
                << " TPUs." << std::endl;
      num_tpus = available_tpus;
    }

    for (int i = 0; i < num_tpus; ++i) {
      devices_.push_back(all_devices[i]);
    }

    std::cout << "[INFO] Using " << devices_.size() << " TPUs:" << std::endl;
    for (auto* dev : devices_) {
      std::cout << "  Device " << dev->id() << ": " << dev->device_kind()
                << std::endl;
    }

    // Use the first device for platform detection and default device pointer
    device_ = devices_[0];

    // Detect platform (GhostFish vs GhostFishLite)
    std::string device_kind(device_->device_kind());
    std::cout << "[INFO] Detected TPU Device Kind: " << device_kind
              << std::endl;

    min_d2h_bandwidth_gb_s_ = 24.0;
    min_h2d_bandwidth_gb_s_ = 24.0;
    should_gate_performance_ = true;

    if (should_gate_performance_) {
      std::cout << "[INFO] Platform identified as: " << device_kind
                << ", Gating D2H: " << min_d2h_bandwidth_gb_s_ << " GB/s"
                << ", Gating H2D: " << min_h2d_bandwidth_gb_s_ << " GB/s"
                << std::endl;
    }
  }

  tpu_raiden::TpuPjrtManager* manager_ = nullptr;
  std::vector<xla::PjRtDevice*> devices_;
  xla::PjRtDevice* device_ = nullptr;
  double min_d2h_bandwidth_gb_s_ = 12.0;
  double min_h2d_bandwidth_gb_s_ = 12.0;
  bool should_gate_performance_ = false;
};

struct BenchmarkParams {
  xla::PrimitiveType primitive_type;
  std::string type_label;
  bool should_gate_performance;
};

class ParameterizedRawTransferPerfTest
    : public RawTransferPerfTest,
      public ::testing::WithParamInterface<BenchmarkParams> {};

// Scenario A: Fragmented Batch (Independent Layer Buffers)
TEST_P(ParameterizedRawTransferPerfTest, ScenarioA_FragmentedBatch) {
  const BenchmarkParams& params = GetParam();
  int num_layers = absl::GetFlag(FLAGS_num_layers);
  int64_t num_blocks = absl::GetFlag(FLAGS_num_blocks);
  bool should_gate =
      params.should_gate_performance ? should_gate_performance_ : false;
  switch (params.primitive_type) {
    case xla::BF16:
      RunBenchmarkScenarioA<uint16_t>(manager_, devices_, params.primitive_type,
                                      params.type_label, num_layers, num_blocks,
                                      min_d2h_bandwidth_gb_s_,
                                      min_h2d_bandwidth_gb_s_, should_gate);
      break;
    case xla::F32:
      RunBenchmarkScenarioA<float>(manager_, devices_, params.primitive_type,
                                   params.type_label, num_layers, num_blocks,
                                   min_d2h_bandwidth_gb_s_,
                                   min_h2d_bandwidth_gb_s_, should_gate);
      break;
    case xla::S32:
      RunBenchmarkScenarioA<int32_t>(manager_, devices_, params.primitive_type,
                                     params.type_label, num_layers, num_blocks,
                                     min_d2h_bandwidth_gb_s_,
                                     min_h2d_bandwidth_gb_s_, should_gate);
      break;
    case xla::F8E4M3FN:
      RunBenchmarkScenarioA<uint8_t>(manager_, devices_, params.primitive_type,
                                     params.type_label, num_layers, num_blocks,
                                     min_d2h_bandwidth_gb_s_,
                                     min_h2d_bandwidth_gb_s_, should_gate);
      break;
    default:
      FAIL() << "Unsupported primitive type: " << params.primitive_type;
  }
}

// Scenario B: Baked-in Layer Dimension (Single Massive Buffer)
TEST_P(ParameterizedRawTransferPerfTest, ScenarioB_BakedInTensor) {
  const BenchmarkParams& params = GetParam();
  int num_layers = absl::GetFlag(FLAGS_num_layers);
  int64_t num_blocks = absl::GetFlag(FLAGS_num_blocks);
  bool should_gate =
      params.should_gate_performance ? should_gate_performance_ : false;
  switch (params.primitive_type) {
    case xla::BF16:
      RunBenchmarkScenarioB<uint16_t>(manager_, devices_, params.primitive_type,
                                      params.type_label, num_layers, num_blocks,
                                      min_d2h_bandwidth_gb_s_,
                                      min_h2d_bandwidth_gb_s_, should_gate);
      break;
    case xla::F32:
      RunBenchmarkScenarioB<float>(manager_, devices_, params.primitive_type,
                                   params.type_label, num_layers, num_blocks,
                                   min_d2h_bandwidth_gb_s_,
                                   min_h2d_bandwidth_gb_s_, should_gate);
      break;
    case xla::S32:
      RunBenchmarkScenarioB<int32_t>(manager_, devices_, params.primitive_type,
                                     params.type_label, num_layers, num_blocks,
                                     min_d2h_bandwidth_gb_s_,
                                     min_h2d_bandwidth_gb_s_, should_gate);
      break;
    case xla::F8E4M3FN:
      RunBenchmarkScenarioB<uint8_t>(manager_, devices_, params.primitive_type,
                                     params.type_label, num_layers, num_blocks,
                                     min_d2h_bandwidth_gb_s_,
                                     min_h2d_bandwidth_gb_s_, should_gate);
      break;
    default:
      FAIL() << "Unsupported primitive type: " << params.primitive_type;
  }
}

INSTANTIATE_TEST_SUITE_P(
    RawTransferPerfTestInstantiation, ParameterizedRawTransferPerfTest,
    ::testing::Values(BenchmarkParams{xla::BF16, "BF16", true},
                      BenchmarkParams{xla::F32, "F32", true},
                      BenchmarkParams{xla::S32, "INT32", false},
                      BenchmarkParams{xla::F8E4M3FN, "FP8", false}),
    [](const ::testing::TestParamInfo<BenchmarkParams>& info) {
      return info.param.type_label;
    });

}  // namespace
}  // namespace raiden
