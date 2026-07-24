#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kMaxBlocks = 65535;
constexpr int kMaxWorldSize = 32;
constexpr int kLayerPlanes = 3;

__host__ __device__ __forceinline__ int64_t record_byte_offset(
    int64_t record_index,
    int64_t record_bytes) {
  return record_index * record_bytes;
}

__global__ void barrier_all_peers_kernel(
    const int64_t* __restrict__ publish_flag_ptrs,
    const int64_t* __restrict__ wait_flag_ptrs,
    int32_t* __restrict__ send_counters,
    int32_t* __restrict__ wait_counters,
    int world_size,
    uint64_t timeout_cycles,
    int phase) {
  const int peer = static_cast<int>(threadIdx.x);
  if (peer >= world_size) {
    return;
  }

  const int64_t counter_offset =
      static_cast<int64_t>(phase) * world_size + peer;
  const int32_t publish_value = send_counters[counter_offset] + 1;
  send_counters[counter_offset] = publish_value;
  __threadfence_system();
  auto* publish_flag = reinterpret_cast<int32_t*>(static_cast<uintptr_t>(
      publish_flag_ptrs[counter_offset]));
  asm volatile(
      "st.relaxed.sys.global.u32 [%1], %0;"
      :
      : "r"(publish_value), "l"(publish_flag));

  const int32_t expected = wait_counters[counter_offset] + 1;
  wait_counters[counter_offset] = expected;
  const auto* wait_flag = reinterpret_cast<const int32_t*>(
      static_cast<uintptr_t>(wait_flag_ptrs[counter_offset]));
  int32_t observed;
  const uint64_t started = clock64();
  uint32_t spins = 0;
  do {
    asm volatile(
        "ld.acquire.sys.global.u32 %0, [%1];"
        : "=r"(observed)
        : "l"(wait_flag));
    if (((++spins & 0x3ffU) == 0U) &&
        (clock64() - started > timeout_cycles)) {
      printf(
          "B12X copy-engine selected-record barrier timed out: phase=%d "
          "peer=%d expected=%d observed=%d\n",
          phase,
          peer,
          expected,
          observed);
      asm volatile("trap;");
      return;
    }
  } while (static_cast<int32_t>(observed - expected) < 0);
}

__global__ void publish_all_peers_kernel(
    const int64_t* __restrict__ publish_flag_ptrs,
    int32_t* __restrict__ send_counters,
    int world_size,
    int phase) {
  const int peer = static_cast<int>(threadIdx.x);
  if (peer >= world_size) {
    return;
  }

  const int64_t counter_offset =
      static_cast<int64_t>(phase) * world_size + peer;
  const int32_t publish_value = send_counters[counter_offset] + 1;
  send_counters[counter_offset] = publish_value;
  __threadfence_system();
  auto* publish_flag = reinterpret_cast<int32_t*>(static_cast<uintptr_t>(
      publish_flag_ptrs[counter_offset]));
  asm volatile(
      "st.relaxed.sys.global.u32 [%1], %0;"
      :
      : "r"(publish_value), "l"(publish_flag));
}

__global__ void wait_all_peers_kernel(
    const int64_t* __restrict__ wait_flag_ptrs,
    int32_t* __restrict__ wait_counters,
    int world_size,
    uint64_t timeout_cycles,
    int phase) {
  const int peer = static_cast<int>(threadIdx.x);
  if (peer >= world_size) {
    return;
  }

  const int64_t counter_offset =
      static_cast<int64_t>(phase) * world_size + peer;
  const int32_t expected = wait_counters[counter_offset] + 1;
  wait_counters[counter_offset] = expected;
  const auto* wait_flag = reinterpret_cast<const int32_t*>(
      static_cast<uintptr_t>(wait_flag_ptrs[counter_offset]));
  int32_t observed;
  const uint64_t started = clock64();
  uint32_t spins = 0;
  do {
    asm volatile(
        "ld.acquire.sys.global.u32 %0, [%1];"
        : "=r"(observed)
        : "l"(wait_flag));
    if (((++spins & 0x3ffU) == 0U) &&
        (clock64() - started > timeout_cycles)) {
      printf(
          "B12X copy-engine selected-record release wait timed out: phase=%d "
          "peer=%d expected=%d observed=%d\n",
          phase,
          peer,
          expected,
          observed);
      asm volatile("trap;");
      return;
    }
  } while (static_cast<int32_t>(observed - expected) < 0);
}

__device__ __forceinline__ uint64_t broadcast_lane_zero(uint64_t value) {
  uint32_t low = static_cast<uint32_t>(value);
  uint32_t high = static_cast<uint32_t>(value >> 32);
  low = __shfl_sync(0xffffffff, low, 0);
  high = __shfl_sync(0xffffffff, high, 0);
  return (static_cast<uint64_t>(high) << 32) | low;
}

template <typename index_t, typename copy_t>
__global__ void pack_compact_records_kernel(
    const uint8_t* __restrict__ records,
    const index_t* __restrict__ local_indices,
    uint8_t* __restrict__ primary,
    uint8_t* __restrict__ overflow,
    int64_t selected_records,
    int64_t record_bytes,
    int64_t primary_capacity,
    int64_t primary_positions_offset,
    int64_t primary_records_offset,
    int64_t overflow_positions_offset,
    int64_t overflow_records_offset) {
  const int lane = threadIdx.x % kWarpSize;
  const int64_t first_warp =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) /
      kWarpSize;
  const int64_t warp_stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x / kWarpSize;

  for (int64_t selected = first_warp; selected < selected_records;
       selected += warp_stride) {
    const int64_t local_record = static_cast<int64_t>(local_indices[selected]);
    if (local_record < 0) {
      continue;
    }

    uint64_t ordinal = 0;
    if (lane == 0) {
      ordinal = atomicAdd(
          reinterpret_cast<unsigned long long*>(primary),
          static_cast<unsigned long long>(1));
    }
    ordinal = broadcast_lane_zero(ordinal);
    const bool use_primary = ordinal < static_cast<uint64_t>(primary_capacity);
    const int64_t compact_index = use_primary
        ? static_cast<int64_t>(ordinal)
        : static_cast<int64_t>(ordinal) - primary_capacity;
    uint8_t* payload = use_primary ? primary : overflow;
    const int64_t positions_offset =
        use_primary ? primary_positions_offset : overflow_positions_offset;
    const int64_t records_offset =
        use_primary ? primary_records_offset : overflow_records_offset;

    if (lane == 0) {
      reinterpret_cast<int64_t*>(payload + positions_offset)[compact_index] =
          selected;
    }

    const int64_t units_per_record = record_bytes / sizeof(copy_t);
    const auto* source = reinterpret_cast<const copy_t*>(
        records + record_byte_offset(local_record, record_bytes));
    auto* destination = reinterpret_cast<copy_t*>(
        payload + records_offset +
        record_byte_offset(compact_index, record_bytes));
    for (int64_t unit = lane; unit < units_per_record; unit += kWarpSize) {
      destination[unit] = source[unit];
    }
  }
}

template <typename copy_t>
__global__ void unpack_compact_records_kernel(
    const uint8_t* __restrict__ primary_base,
    int64_t primary_stride,
    const int64_t* __restrict__ peer_overflow_ptrs,
    uint8_t* __restrict__ output,
    int64_t selected_records,
    int64_t record_bytes,
    int64_t primary_capacity,
    int64_t primary_positions_offset,
    int64_t primary_records_offset,
    int64_t overflow_positions_offset,
    int64_t overflow_records_offset,
    int world_size) {
  const int lane = threadIdx.x % kWarpSize;
  const int64_t first_warp =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) /
      kWarpSize;
  const int64_t warp_stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x / kWarpSize;
  const int64_t total_compact_records =
      static_cast<int64_t>(world_size) * selected_records;

  for (int64_t compact_record = first_warp;
       compact_record < total_compact_records;
       compact_record += warp_stride) {
    const int source = static_cast<int>(compact_record / selected_records);
    const int64_t ordinal = compact_record % selected_records;
    const uint8_t* primary = primary_base + source * primary_stride;
    const uint64_t total_records =
        *reinterpret_cast<const uint64_t*>(primary);
    if (ordinal >= static_cast<int64_t>(total_records)) {
      continue;
    }

    const bool use_primary = ordinal < primary_capacity;
    const int64_t compact_index =
        use_primary ? ordinal : ordinal - primary_capacity;
    const uint8_t* payload = use_primary
        ? primary
        : reinterpret_cast<const uint8_t*>(
              static_cast<uintptr_t>(peer_overflow_ptrs[source]));
    const int64_t positions_offset =
        use_primary ? primary_positions_offset : overflow_positions_offset;
    const int64_t records_offset =
        use_primary ? primary_records_offset : overflow_records_offset;
    const int64_t selected =
        reinterpret_cast<const int64_t*>(payload + positions_offset)[compact_index];
    if (selected < 0 || selected >= selected_records) {
      continue;
    }

    const int64_t units_per_record = record_bytes / sizeof(copy_t);
    const auto* source_record = reinterpret_cast<const copy_t*>(
        payload + records_offset +
        record_byte_offset(compact_index, record_bytes));
    auto* destination = reinterpret_cast<copy_t*>(
        output + record_byte_offset(selected, record_bytes));
    for (int64_t unit = lane; unit < units_per_record; unit += kWarpSize) {
      destination[unit] = source_record[unit];
    }
  }
}

template <typename index_t, typename copy_t>
__global__ void pack_compact_record_layers_kernel(
    const uint8_t* __restrict__ records0,
    const uint8_t* __restrict__ records1,
    const uint8_t* __restrict__ records2,
    const index_t* __restrict__ local_indices,
    uint8_t* __restrict__ primary,
    uint8_t* __restrict__ overflow,
    int64_t selected_records,
    int64_t record_bytes,
    int64_t primary_capacity,
    int64_t primary_positions_offset,
    int64_t primary_records_offset,
    int64_t overflow_positions_offset,
    int64_t overflow_records_offset) {
  const int lane = threadIdx.x % kWarpSize;
  const int64_t first_warp =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) /
      kWarpSize;
  const int64_t warp_stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x / kWarpSize;
  const int64_t packet_record_bytes = record_bytes * kLayerPlanes;

  for (int64_t selected = first_warp; selected < selected_records;
       selected += warp_stride) {
    const int64_t local_record = static_cast<int64_t>(local_indices[selected]);
    if (local_record < 0) {
      continue;
    }

    uint64_t ordinal = 0;
    if (lane == 0) {
      ordinal = atomicAdd(
          reinterpret_cast<unsigned long long*>(primary),
          static_cast<unsigned long long>(1));
    }
    ordinal = broadcast_lane_zero(ordinal);
    const bool use_primary = ordinal < static_cast<uint64_t>(primary_capacity);
    const int64_t compact_index = use_primary
        ? static_cast<int64_t>(ordinal)
        : static_cast<int64_t>(ordinal) - primary_capacity;
    uint8_t* payload = use_primary ? primary : overflow;
    const int64_t positions_offset =
        use_primary ? primary_positions_offset : overflow_positions_offset;
    const int64_t records_offset =
        use_primary ? primary_records_offset : overflow_records_offset;

    if (lane == 0) {
      reinterpret_cast<int64_t*>(payload + positions_offset)[compact_index] =
          selected;
    }

    const int64_t units_per_record = record_bytes / sizeof(copy_t);
    const uint8_t* source_bytes[kLayerPlanes] = {records0, records1, records2};
    uint8_t* packet = payload + records_offset +
        record_byte_offset(compact_index, packet_record_bytes);
#pragma unroll
    for (int layer = 0; layer < kLayerPlanes; ++layer) {
      const auto* source = reinterpret_cast<const copy_t*>(
          source_bytes[layer] + record_byte_offset(local_record, record_bytes));
      auto* destination = reinterpret_cast<copy_t*>(
          packet + static_cast<int64_t>(layer) * record_bytes);
      for (int64_t unit = lane; unit < units_per_record; unit += kWarpSize) {
        destination[unit] = source[unit];
      }
    }
  }
}

template <typename copy_t>
__global__ void unpack_compact_record_layers_kernel(
    const uint8_t* __restrict__ primary_base,
    int64_t primary_stride,
    const int64_t* __restrict__ peer_overflow_ptrs,
    uint8_t* __restrict__ output0,
    uint8_t* __restrict__ output1,
    uint8_t* __restrict__ output2,
    int64_t selected_records,
    int64_t record_bytes,
    int64_t primary_capacity,
    int64_t primary_positions_offset,
    int64_t primary_records_offset,
    int64_t overflow_positions_offset,
    int64_t overflow_records_offset,
    int world_size) {
  const int lane = threadIdx.x % kWarpSize;
  const int64_t first_warp =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) /
      kWarpSize;
  const int64_t warp_stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x / kWarpSize;
  const int64_t total_compact_records =
      static_cast<int64_t>(world_size) * selected_records;
  const int64_t packet_record_bytes = record_bytes * kLayerPlanes;

  for (int64_t compact_record = first_warp;
       compact_record < total_compact_records;
       compact_record += warp_stride) {
    const int source_rank =
        static_cast<int>(compact_record / selected_records);
    const int64_t ordinal = compact_record % selected_records;
    const uint8_t* primary = primary_base + source_rank * primary_stride;
    const uint64_t total_records =
        *reinterpret_cast<const uint64_t*>(primary);
    if (ordinal >= static_cast<int64_t>(total_records)) {
      continue;
    }

    const bool use_primary = ordinal < primary_capacity;
    const int64_t compact_index =
        use_primary ? ordinal : ordinal - primary_capacity;
    const uint8_t* payload = use_primary
        ? primary
        : reinterpret_cast<const uint8_t*>(
              static_cast<uintptr_t>(peer_overflow_ptrs[source_rank]));
    const int64_t positions_offset =
        use_primary ? primary_positions_offset : overflow_positions_offset;
    const int64_t records_offset =
        use_primary ? primary_records_offset : overflow_records_offset;
    const int64_t selected =
        reinterpret_cast<const int64_t*>(payload + positions_offset)[compact_index];
    if (selected < 0 || selected >= selected_records) {
      continue;
    }

    const int64_t units_per_record = record_bytes / sizeof(copy_t);
    const uint8_t* packet = payload + records_offset +
        record_byte_offset(compact_index, packet_record_bytes);
    uint8_t* output_bytes[kLayerPlanes] = {output0, output1, output2};
#pragma unroll
    for (int layer = 0; layer < kLayerPlanes; ++layer) {
      const auto* source = reinterpret_cast<const copy_t*>(
          packet + static_cast<int64_t>(layer) * record_bytes);
      auto* destination = reinterpret_cast<copy_t*>(
          output_bytes[layer] + record_byte_offset(selected, record_bytes));
      for (int64_t unit = lane; unit < units_per_record; unit += kWarpSize) {
        destination[unit] = source[unit];
      }
    }
  }
}

void validate_tensor(
    const torch::Tensor& tensor,
    const char* name,
    torch::ScalarType scalar_type) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == scalar_type, name, " has wrong dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

int launch_blocks_for_warps(int64_t warps) {
  const int64_t requested =
      (warps * kWarpSize + kThreads - 1) / kThreads;
  return static_cast<int>(
      std::max<int64_t>(1, std::min<int64_t>(requested, kMaxBlocks)));
}

template <typename index_t, typename copy_t>
void launch_pack(
    const torch::Tensor& records,
    const torch::Tensor& local_indices,
    int64_t primary_ptr,
    int64_t overflow_ptr,
    int64_t record_bytes,
    int64_t primary_capacity,
    int64_t primary_positions_offset,
    int64_t primary_records_offset,
    int64_t overflow_positions_offset,
    int64_t overflow_records_offset,
    cudaStream_t stream) {
  const int64_t selected_records = local_indices.numel();
  const int blocks = launch_blocks_for_warps(selected_records);
  pack_compact_records_kernel<index_t, copy_t><<<blocks, kThreads, 0, stream>>>(
      records.data_ptr<uint8_t>(),
      local_indices.data_ptr<index_t>(),
      reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(primary_ptr)),
      reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(overflow_ptr)),
      selected_records,
      record_bytes,
      primary_capacity,
      primary_positions_offset,
      primary_records_offset,
      overflow_positions_offset,
      overflow_records_offset);
}

void pack_compact_records(
    torch::Tensor records,
    torch::Tensor local_indices,
    int64_t primary_ptr,
    int64_t overflow_ptr,
    int64_t record_bytes,
    int64_t primary_capacity,
    int64_t primary_positions_offset,
    int64_t primary_records_offset,
    int64_t overflow_positions_offset,
    int64_t overflow_records_offset) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(records));
  validate_tensor(records, "records", torch::kUInt8);
  TORCH_CHECK(local_indices.is_cuda(), "local indices must be CUDA");
  TORCH_CHECK(
      local_indices.scalar_type() == torch::kInt32 ||
          local_indices.scalar_type() == torch::kInt64,
      "local indices must be int32 or int64");
  TORCH_CHECK(local_indices.is_contiguous(), "local indices must be contiguous");
  TORCH_CHECK(records.dim() >= 2, "records must have rank at least 2");
  TORCH_CHECK(record_bytes > 0, "record_bytes must be positive");
  TORCH_CHECK(
      records.size(-1) == record_bytes,
      "records must end in record_bytes");
  TORCH_CHECK(primary_ptr != 0, "primary pointer must be nonzero");
  TORCH_CHECK(overflow_ptr != 0, "overflow pointer must be nonzero");
  TORCH_CHECK(primary_capacity > 0, "primary capacity must be positive");
  TORCH_CHECK(
      primary_capacity <= local_indices.numel(),
      "primary capacity cannot exceed selected records");

  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  AT_CUDA_CHECK(cudaMemsetAsync(
      reinterpret_cast<void*>(static_cast<uintptr_t>(primary_ptr)),
      0,
      sizeof(uint64_t),
      stream));
  if (local_indices.numel() == 0) {
    return;
  }

  const bool vectorized =
      record_bytes % static_cast<int64_t>(sizeof(uint4)) == 0 &&
      reinterpret_cast<uintptr_t>(records.data_ptr<uint8_t>()) % alignof(uint4) == 0 &&
      (static_cast<uint64_t>(primary_ptr) + primary_records_offset) %
              alignof(uint4) ==
          0 &&
      (static_cast<uint64_t>(overflow_ptr) + overflow_records_offset) %
              alignof(uint4) ==
          0;
  if (local_indices.scalar_type() == torch::kInt32) {
    if (vectorized) {
      launch_pack<int32_t, uint4>(
          records,
          local_indices,
          primary_ptr,
          overflow_ptr,
          record_bytes,
          primary_capacity,
          primary_positions_offset,
          primary_records_offset,
          overflow_positions_offset,
          overflow_records_offset,
          stream);
    } else {
      launch_pack<int32_t, uint8_t>(
          records,
          local_indices,
          primary_ptr,
          overflow_ptr,
          record_bytes,
          primary_capacity,
          primary_positions_offset,
          primary_records_offset,
          overflow_positions_offset,
          overflow_records_offset,
          stream);
    }
  } else if (vectorized) {
    launch_pack<int64_t, uint4>(
        records,
        local_indices,
        primary_ptr,
        overflow_ptr,
        record_bytes,
        primary_capacity,
        primary_positions_offset,
        primary_records_offset,
        overflow_positions_offset,
        overflow_records_offset,
        stream);
  } else {
    launch_pack<int64_t, uint8_t>(
        records,
        local_indices,
        primary_ptr,
        overflow_ptr,
        record_bytes,
        primary_capacity,
        primary_positions_offset,
        primary_records_offset,
        overflow_positions_offset,
        overflow_records_offset,
        stream);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

template <typename index_t, typename copy_t>
void launch_pack_layers(
    const torch::Tensor& records0,
    const torch::Tensor& records1,
    const torch::Tensor& records2,
    const torch::Tensor& local_indices,
    int64_t primary_ptr,
    int64_t overflow_ptr,
    int64_t record_bytes,
    int64_t primary_capacity,
    int64_t primary_positions_offset,
    int64_t primary_records_offset,
    int64_t overflow_positions_offset,
    int64_t overflow_records_offset,
    cudaStream_t stream) {
  const int64_t selected_records = local_indices.numel();
  const int blocks = launch_blocks_for_warps(selected_records);
  pack_compact_record_layers_kernel<index_t, copy_t>
      <<<blocks, kThreads, 0, stream>>>(
          records0.data_ptr<uint8_t>(),
          records1.data_ptr<uint8_t>(),
          records2.data_ptr<uint8_t>(),
          local_indices.data_ptr<index_t>(),
          reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(primary_ptr)),
          reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(overflow_ptr)),
          selected_records,
          record_bytes,
          primary_capacity,
          primary_positions_offset,
          primary_records_offset,
          overflow_positions_offset,
          overflow_records_offset);
}

void pack_compact_record_layers(
    torch::Tensor records0,
    torch::Tensor records1,
    torch::Tensor records2,
    torch::Tensor local_indices,
    int64_t primary_ptr,
    int64_t overflow_ptr,
    int64_t record_bytes,
    int64_t primary_capacity,
    int64_t primary_positions_offset,
    int64_t primary_records_offset,
    int64_t overflow_positions_offset,
    int64_t overflow_records_offset) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(records0));
  validate_tensor(records0, "layer 0 records", torch::kUInt8);
  validate_tensor(records1, "layer 1 records", torch::kUInt8);
  validate_tensor(records2, "layer 2 records", torch::kUInt8);
  TORCH_CHECK(
      records1.device() == records0.device() &&
          records2.device() == records0.device(),
      "all layer records must use the same CUDA device");
  TORCH_CHECK(
      records0.sizes() == records1.sizes() &&
          records0.sizes() == records2.sizes(),
      "all layer record tensors must have the same shape");
  TORCH_CHECK(local_indices.is_cuda(), "local indices must be CUDA");
  TORCH_CHECK(
      local_indices.device() == records0.device(),
      "local indices and records must use the same CUDA device");
  TORCH_CHECK(
      local_indices.scalar_type() == torch::kInt32 ||
          local_indices.scalar_type() == torch::kInt64,
      "local indices must be int32 or int64");
  TORCH_CHECK(local_indices.is_contiguous(), "local indices must be contiguous");
  TORCH_CHECK(records0.dim() >= 2, "records must have rank at least 2");
  TORCH_CHECK(record_bytes > 0, "record_bytes must be positive");
  TORCH_CHECK(
      record_bytes <= std::numeric_limits<int64_t>::max() / kLayerPlanes,
      "layered record width exceeds int64 capacity");
  TORCH_CHECK(
      records0.size(-1) == record_bytes,
      "layer records must end in record_bytes");
  TORCH_CHECK(primary_ptr != 0, "primary pointer must be nonzero");
  TORCH_CHECK(overflow_ptr != 0, "overflow pointer must be nonzero");
  TORCH_CHECK(primary_capacity > 0, "primary capacity must be positive");
  TORCH_CHECK(
      primary_capacity <= local_indices.numel(),
      "primary capacity cannot exceed selected records");

  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  AT_CUDA_CHECK(cudaMemsetAsync(
      reinterpret_cast<void*>(static_cast<uintptr_t>(primary_ptr)),
      0,
      sizeof(uint64_t),
      stream));
  if (local_indices.numel() == 0) {
    return;
  }

  const bool vectorized =
      record_bytes % static_cast<int64_t>(sizeof(uint4)) == 0 &&
      reinterpret_cast<uintptr_t>(records0.data_ptr<uint8_t>()) %
              alignof(uint4) ==
          0 &&
      reinterpret_cast<uintptr_t>(records1.data_ptr<uint8_t>()) %
              alignof(uint4) ==
          0 &&
      reinterpret_cast<uintptr_t>(records2.data_ptr<uint8_t>()) %
              alignof(uint4) ==
          0 &&
      (static_cast<uint64_t>(primary_ptr) + primary_records_offset) %
              alignof(uint4) ==
          0 &&
      (static_cast<uint64_t>(overflow_ptr) + overflow_records_offset) %
              alignof(uint4) ==
          0;
  if (local_indices.scalar_type() == torch::kInt32) {
    if (vectorized) {
      launch_pack_layers<int32_t, uint4>(
          records0,
          records1,
          records2,
          local_indices,
          primary_ptr,
          overflow_ptr,
          record_bytes,
          primary_capacity,
          primary_positions_offset,
          primary_records_offset,
          overflow_positions_offset,
          overflow_records_offset,
          stream);
    } else {
      launch_pack_layers<int32_t, uint8_t>(
          records0,
          records1,
          records2,
          local_indices,
          primary_ptr,
          overflow_ptr,
          record_bytes,
          primary_capacity,
          primary_positions_offset,
          primary_records_offset,
          overflow_positions_offset,
          overflow_records_offset,
          stream);
    }
  } else if (vectorized) {
    launch_pack_layers<int64_t, uint4>(
        records0,
        records1,
        records2,
        local_indices,
        primary_ptr,
        overflow_ptr,
        record_bytes,
        primary_capacity,
        primary_positions_offset,
        primary_records_offset,
        overflow_positions_offset,
        overflow_records_offset,
        stream);
  } else {
    launch_pack_layers<int64_t, uint8_t>(
        records0,
        records1,
        records2,
        local_indices,
        primary_ptr,
        overflow_ptr,
        record_bytes,
        primary_capacity,
        primary_positions_offset,
        primary_records_offset,
        overflow_positions_offset,
        overflow_records_offset,
        stream);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

template <typename copy_t>
void launch_unpack(
    int64_t primary_base_ptr,
    int64_t primary_stride,
    const torch::Tensor& peer_overflow_ptrs,
    torch::Tensor& output,
    int64_t selected_records,
    int64_t record_bytes,
    int64_t primary_capacity,
    int64_t primary_positions_offset,
    int64_t primary_records_offset,
    int64_t overflow_positions_offset,
    int64_t overflow_records_offset,
    cudaStream_t stream) {
  const int world_size = static_cast<int>(peer_overflow_ptrs.numel());
  const int64_t warps = static_cast<int64_t>(world_size) * selected_records;
  const int blocks = launch_blocks_for_warps(warps);
  unpack_compact_records_kernel<copy_t><<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(
          static_cast<uintptr_t>(primary_base_ptr)),
      primary_stride,
      peer_overflow_ptrs.data_ptr<int64_t>(),
      output.data_ptr<uint8_t>(),
      selected_records,
      record_bytes,
      primary_capacity,
      primary_positions_offset,
      primary_records_offset,
      overflow_positions_offset,
      overflow_records_offset,
      world_size);
}

void unpack_compact_records(
    int64_t primary_base_ptr,
    int64_t primary_stride,
    torch::Tensor peer_overflow_ptrs,
    torch::Tensor output,
    int64_t selected_records,
    int64_t record_bytes,
    int64_t primary_capacity,
    int64_t primary_positions_offset,
    int64_t primary_records_offset,
    int64_t overflow_positions_offset,
    int64_t overflow_records_offset) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  validate_tensor(peer_overflow_ptrs, "peer overflow pointers", torch::kInt64);
  validate_tensor(output, "output", torch::kUInt8);
  TORCH_CHECK(primary_base_ptr != 0, "primary base pointer must be nonzero");
  TORCH_CHECK(primary_stride > 0, "primary stride must be positive");
  TORCH_CHECK(record_bytes > 0, "record_bytes must be positive");
  TORCH_CHECK(selected_records > 0, "selected_records must be positive");
  TORCH_CHECK(primary_capacity > 0, "primary capacity must be positive");
  const int64_t world_size = peer_overflow_ptrs.numel();
  TORCH_CHECK(
      world_size >= 2 && world_size <= kMaxWorldSize,
      "world size must be in [2, 32]");
  TORCH_CHECK(
      selected_records <= std::numeric_limits<int64_t>::max() / record_bytes,
      "selected payload exceeds int64 capacity");
  TORCH_CHECK(
      output.numel() == selected_records * record_bytes,
      "output size must exactly match the selected payload");

  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  const bool vectorized =
      record_bytes % static_cast<int64_t>(sizeof(uint4)) == 0 &&
      reinterpret_cast<uintptr_t>(output.data_ptr<uint8_t>()) % alignof(uint4) == 0 &&
      (static_cast<uint64_t>(primary_base_ptr) + primary_records_offset) %
              alignof(uint4) ==
          0;
  if (vectorized) {
    launch_unpack<uint4>(
        primary_base_ptr,
        primary_stride,
        peer_overflow_ptrs,
        output,
        selected_records,
        record_bytes,
        primary_capacity,
        primary_positions_offset,
        primary_records_offset,
        overflow_positions_offset,
        overflow_records_offset,
        stream);
  } else {
    launch_unpack<uint8_t>(
        primary_base_ptr,
        primary_stride,
        peer_overflow_ptrs,
        output,
        selected_records,
        record_bytes,
        primary_capacity,
        primary_positions_offset,
        primary_records_offset,
        overflow_positions_offset,
        overflow_records_offset,
        stream);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

template <typename copy_t>
void launch_unpack_layers(
    int64_t primary_base_ptr,
    int64_t primary_stride,
    const torch::Tensor& peer_overflow_ptrs,
    torch::Tensor& output0,
    torch::Tensor& output1,
    torch::Tensor& output2,
    int64_t selected_records,
    int64_t record_bytes,
    int64_t primary_capacity,
    int64_t primary_positions_offset,
    int64_t primary_records_offset,
    int64_t overflow_positions_offset,
    int64_t overflow_records_offset,
    cudaStream_t stream) {
  const int world_size = static_cast<int>(peer_overflow_ptrs.numel());
  const int64_t warps = static_cast<int64_t>(world_size) * selected_records;
  const int blocks = launch_blocks_for_warps(warps);
  unpack_compact_record_layers_kernel<copy_t><<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(
          static_cast<uintptr_t>(primary_base_ptr)),
      primary_stride,
      peer_overflow_ptrs.data_ptr<int64_t>(),
      output0.data_ptr<uint8_t>(),
      output1.data_ptr<uint8_t>(),
      output2.data_ptr<uint8_t>(),
      selected_records,
      record_bytes,
      primary_capacity,
      primary_positions_offset,
      primary_records_offset,
      overflow_positions_offset,
      overflow_records_offset,
      world_size);
}

void unpack_compact_record_layers(
    int64_t primary_base_ptr,
    int64_t primary_stride,
    torch::Tensor peer_overflow_ptrs,
    torch::Tensor output0,
    torch::Tensor output1,
    torch::Tensor output2,
    int64_t selected_records,
    int64_t record_bytes,
    int64_t primary_capacity,
    int64_t primary_positions_offset,
    int64_t primary_records_offset,
    int64_t overflow_positions_offset,
    int64_t overflow_records_offset) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output0));
  validate_tensor(peer_overflow_ptrs, "peer overflow pointers", torch::kInt64);
  validate_tensor(output0, "layer 0 output", torch::kUInt8);
  validate_tensor(output1, "layer 1 output", torch::kUInt8);
  validate_tensor(output2, "layer 2 output", torch::kUInt8);
  TORCH_CHECK(
      output1.device() == output0.device() &&
          output2.device() == output0.device() &&
          peer_overflow_ptrs.device() == output0.device(),
      "layer outputs and peer pointers must use the same CUDA device");
  TORCH_CHECK(
      output0.sizes() == output1.sizes() &&
          output0.sizes() == output2.sizes(),
      "all layer output tensors must have the same shape");
  TORCH_CHECK(primary_base_ptr != 0, "primary base pointer must be nonzero");
  TORCH_CHECK(primary_stride > 0, "primary stride must be positive");
  TORCH_CHECK(record_bytes > 0, "record_bytes must be positive");
  TORCH_CHECK(
      record_bytes <= std::numeric_limits<int64_t>::max() / kLayerPlanes,
      "layered record width exceeds int64 capacity");
  TORCH_CHECK(selected_records > 0, "selected_records must be positive");
  TORCH_CHECK(primary_capacity > 0, "primary capacity must be positive");
  const int64_t world_size = peer_overflow_ptrs.numel();
  TORCH_CHECK(
      world_size >= 2 && world_size <= kMaxWorldSize,
      "world size must be in [2, 32]");
  TORCH_CHECK(
      selected_records <= std::numeric_limits<int64_t>::max() / record_bytes,
      "selected payload exceeds int64 capacity");
  TORCH_CHECK(
      output0.numel() == selected_records * record_bytes,
      "each output size must exactly match the selected payload");

  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  const bool vectorized =
      record_bytes % static_cast<int64_t>(sizeof(uint4)) == 0 &&
      reinterpret_cast<uintptr_t>(output0.data_ptr<uint8_t>()) %
              alignof(uint4) ==
          0 &&
      reinterpret_cast<uintptr_t>(output1.data_ptr<uint8_t>()) %
              alignof(uint4) ==
          0 &&
      reinterpret_cast<uintptr_t>(output2.data_ptr<uint8_t>()) %
              alignof(uint4) ==
          0 &&
      (static_cast<uint64_t>(primary_base_ptr) + primary_records_offset) %
              alignof(uint4) ==
          0;
  if (vectorized) {
    launch_unpack_layers<uint4>(
        primary_base_ptr,
        primary_stride,
        peer_overflow_ptrs,
        output0,
        output1,
        output2,
        selected_records,
        record_bytes,
        primary_capacity,
        primary_positions_offset,
        primary_records_offset,
        overflow_positions_offset,
        overflow_records_offset,
        stream);
  } else {
    launch_unpack_layers<uint8_t>(
        primary_base_ptr,
        primary_stride,
        peer_overflow_ptrs,
        output0,
        output1,
        output2,
        selected_records,
        record_bytes,
        primary_capacity,
        primary_positions_offset,
        primary_records_offset,
        overflow_positions_offset,
        overflow_records_offset,
        stream);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

void barrier_all_peers(
    torch::Tensor publish_flag_ptrs,
    torch::Tensor wait_flag_ptrs,
    torch::Tensor send_counters,
    torch::Tensor wait_counters,
    int64_t phase,
    int64_t timeout_cycles) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(send_counters));
  validate_tensor(publish_flag_ptrs, "publish flag pointers", torch::kInt64);
  validate_tensor(wait_flag_ptrs, "wait flag pointers", torch::kInt64);
  validate_tensor(send_counters, "send counters", torch::kInt32);
  validate_tensor(wait_counters, "wait counters", torch::kInt32);
  TORCH_CHECK(phase == 0 || phase == 1, "phase must be zero or one");
  TORCH_CHECK(timeout_cycles > 0, "timeout_cycles must be positive");
  TORCH_CHECK(
      publish_flag_ptrs.numel() == wait_flag_ptrs.numel(),
      "publish and wait pointer counts must match");
  TORCH_CHECK(
      send_counters.numel() == wait_counters.numel(),
      "send and wait counter counts must match");
  TORCH_CHECK(
      publish_flag_ptrs.numel() == send_counters.numel(),
      "barrier pointer and counter counts must match");
  TORCH_CHECK(
      publish_flag_ptrs.dim() == 2 && publish_flag_ptrs.size(0) == 2,
      "barrier pointers must have shape [2, world_size]");
  const int world_size = static_cast<int>(publish_flag_ptrs.size(1));
  TORCH_CHECK(
      world_size >= 2 && world_size <= kMaxWorldSize,
      "world size must be in [2, 32]");

  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  barrier_all_peers_kernel<<<1, kMaxWorldSize, 0, stream>>>(
      publish_flag_ptrs.data_ptr<int64_t>(),
      wait_flag_ptrs.data_ptr<int64_t>(),
      send_counters.data_ptr<int32_t>(),
      wait_counters.data_ptr<int32_t>(),
      world_size,
      static_cast<uint64_t>(timeout_cycles),
      static_cast<int>(phase));
  AT_CUDA_CHECK(cudaGetLastError());
}

void publish_all_peers(
    torch::Tensor publish_flag_ptrs,
    torch::Tensor send_counters,
    int64_t phase) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(send_counters));
  validate_tensor(publish_flag_ptrs, "publish flag pointers", torch::kInt64);
  validate_tensor(send_counters, "send counters", torch::kInt32);
  TORCH_CHECK(phase == 0 || phase == 1, "phase must be zero or one");
  TORCH_CHECK(
      publish_flag_ptrs.numel() == send_counters.numel(),
      "publish pointer and counter counts must match");
  TORCH_CHECK(
      publish_flag_ptrs.dim() == 2 && publish_flag_ptrs.size(0) == 2,
      "publish pointers must have shape [2, world_size]");
  const int world_size = static_cast<int>(publish_flag_ptrs.size(1));
  TORCH_CHECK(
      world_size >= 2 && world_size <= kMaxWorldSize,
      "world size must be in [2, 32]");

  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  publish_all_peers_kernel<<<1, kMaxWorldSize, 0, stream>>>(
      publish_flag_ptrs.data_ptr<int64_t>(),
      send_counters.data_ptr<int32_t>(),
      world_size,
      static_cast<int>(phase));
  AT_CUDA_CHECK(cudaGetLastError());
}

void wait_all_peers(
    torch::Tensor wait_flag_ptrs,
    torch::Tensor wait_counters,
    int64_t phase,
    int64_t timeout_cycles) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(wait_counters));
  validate_tensor(wait_flag_ptrs, "wait flag pointers", torch::kInt64);
  validate_tensor(wait_counters, "wait counters", torch::kInt32);
  TORCH_CHECK(phase == 0 || phase == 1, "phase must be zero or one");
  TORCH_CHECK(timeout_cycles > 0, "timeout_cycles must be positive");
  TORCH_CHECK(
      wait_flag_ptrs.numel() == wait_counters.numel(),
      "wait pointer and counter counts must match");
  TORCH_CHECK(
      wait_flag_ptrs.dim() == 2 && wait_flag_ptrs.size(0) == 2,
      "wait pointers must have shape [2, world_size]");
  const int world_size = static_cast<int>(wait_flag_ptrs.size(1));
  TORCH_CHECK(
      world_size >= 2 && world_size <= kMaxWorldSize,
      "world size must be in [2, 32]");

  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  wait_all_peers_kernel<<<1, kMaxWorldSize, 0, stream>>>(
      wait_flag_ptrs.data_ptr<int64_t>(),
      wait_counters.data_ptr<int32_t>(),
      world_size,
      static_cast<uint64_t>(timeout_cycles),
      static_cast<int>(phase));
  AT_CUDA_CHECK(cudaGetLastError());
}

int64_t record_byte_offset_for_test(
    int64_t record_index,
    int64_t record_bytes) {
  TORCH_CHECK(record_index >= 0, "record_index must be non-negative");
  TORCH_CHECK(record_bytes > 0, "record_bytes must be positive");
  TORCH_CHECK(
      record_index <= std::numeric_limits<int64_t>::max() / record_bytes,
      "record byte offset exceeds int64 capacity");
  return record_byte_offset(record_index, record_bytes);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "pack_compact_records",
      &pack_compact_records,
      "Pack locally owned selected records into copy-engine staging packets");
  module.def(
      "pack_compact_record_layers",
      &pack_compact_record_layers,
      "Pack three record planes into one copy-engine staging packet");
  module.def(
      "unpack_compact_records",
      &unpack_compact_records,
      "Unpack source packets into destination-selected record order");
  module.def(
      "unpack_compact_record_layers",
      &unpack_compact_record_layers,
      "Unpack one packet directly into three destination record planes");
  module.def(
      "barrier_all_peers",
      &barrier_all_peers,
      "Publish and wait for every selected-record peer");
  module.def(
      "publish_all_peers",
      &publish_all_peers,
      "Publish one selected-record generation to every peer");
  module.def(
      "wait_all_peers",
      &wait_all_peers,
      "Wait for one selected-record generation from every peer");
  module.def(
      "record_byte_offset_for_test",
      &record_byte_offset_for_test,
      "Compute the Int64 byte offset used by selected-record copy exchange");
}
