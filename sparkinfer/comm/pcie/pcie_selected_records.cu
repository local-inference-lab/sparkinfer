#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 65535;
constexpr int kMaxWorldSize = 32;

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
          "B12X selected-record barrier timed out: phase=%d peer=%d "
          "expected=%d observed=%d\n",
          phase,
          peer,
          expected,
          observed);
      asm volatile("trap;");
      return;
    }
  } while (static_cast<int32_t>(observed - expected) < 0);
}

__host__ __device__ __forceinline__ int64_t record_byte_offset(
    int64_t record_index,
    int64_t record_bytes) {
  return record_index * record_bytes;
}

template <typename index_t, typename copy_t>
__global__ void scatter_records_kernel(
    const uint8_t* __restrict__ records,
    const index_t* __restrict__ local_indices_by_destination,
    const int64_t* __restrict__ peer_payload_ptrs,
    int64_t selected_records,
    int64_t units_per_record,
    int64_t record_bytes,
    int world_size) {
  const int destination = static_cast<int>(blockIdx.y);
  if (destination >= world_size) {
    return;
  }

  const int64_t payload_units = selected_records * units_per_record;
  const int64_t grid_stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t payload_unit =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       payload_unit < payload_units;
       payload_unit += grid_stride) {
    const int64_t selected = payload_unit / units_per_record;
    const int64_t unit_in_record = payload_unit - selected * units_per_record;
    const int64_t byte_in_record = unit_in_record * sizeof(copy_t);
    const int64_t map_offset =
        static_cast<int64_t>(destination) * selected_records + selected;
    const int64_t local_record = static_cast<int64_t>(
        local_indices_by_destination[map_offset]);
    if (local_record < 0) {
      continue;
    }

    // Keep both pool- and output-scaled products in Int64. A valid record
    // index can cross the 2 GiB byte boundary even when the index is Int32.
    const int64_t source_offset =
        record_byte_offset(local_record, record_bytes) + byte_in_record;
    const int64_t destination_offset =
        record_byte_offset(selected, record_bytes) + byte_in_record;
    const auto* source = reinterpret_cast<const copy_t*>(
        records + source_offset);
    auto* destination_ptr = reinterpret_cast<copy_t*>(
        static_cast<uintptr_t>(peer_payload_ptrs[destination]) +
        destination_offset);
    *destination_ptr = *source;
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

int64_t record_byte_offset_for_test(
    int64_t record_index,
    int64_t record_bytes) {
  TORCH_CHECK(record_index >= 0, "record_index must be non-negative");
  TORCH_CHECK(record_bytes > 0, "record_bytes must be positive");
  TORCH_CHECK(
      record_index <= INT64_MAX / record_bytes,
      "record byte offset exceeds int64 capacity");
  return record_byte_offset(record_index, record_bytes);
}

void exchange(
    torch::Tensor records,
    torch::Tensor local_indices_by_destination,
    torch::Tensor peer_payload_ptrs,
    int64_t local_payload_ptr,
    torch::Tensor barrier_publish_ptrs,
    torch::Tensor barrier_wait_ptrs,
    torch::Tensor send_counters,
    torch::Tensor wait_counters,
    torch::Tensor output,
    int64_t record_bytes,
    int64_t timeout_cycles) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(records));
  validate_tensor(records, "records", torch::kUInt8);
  TORCH_CHECK(
      local_indices_by_destination.is_cuda(), "local indices must be CUDA");
  TORCH_CHECK(
      local_indices_by_destination.scalar_type() == torch::kInt32 ||
          local_indices_by_destination.scalar_type() == torch::kInt64,
      "local indices must be int32 or int64");
  TORCH_CHECK(
      local_indices_by_destination.is_contiguous(),
      "local indices must be contiguous");
  validate_tensor(peer_payload_ptrs, "peer payload pointers", torch::kInt64);
  validate_tensor(
      barrier_publish_ptrs, "barrier publish pointers", torch::kInt64);
  validate_tensor(barrier_wait_ptrs, "barrier wait pointers", torch::kInt64);
  validate_tensor(send_counters, "send counters", torch::kInt32);
  validate_tensor(wait_counters, "wait counters", torch::kInt32);
  validate_tensor(output, "output", torch::kUInt8);

  TORCH_CHECK(record_bytes > 0, "record_bytes must be positive");
  TORCH_CHECK(timeout_cycles > 0, "timeout_cycles must be positive");
  TORCH_CHECK(local_payload_ptr != 0, "local payload pointer must be nonzero");
  TORCH_CHECK(
      local_indices_by_destination.dim() >= 2,
      "local indices must have rank at least 2");
  const int64_t world_size = local_indices_by_destination.size(0);
  TORCH_CHECK(
      world_size >= 1 && world_size <= kMaxWorldSize,
      "world size must be in [1, 32]");
  TORCH_CHECK(
      peer_payload_ptrs.numel() == world_size,
      "peer payload pointer count must match world size");
  TORCH_CHECK(
      barrier_publish_ptrs.numel() == 2 * world_size,
      "publish pointer count must cover two barrier phases");
  TORCH_CHECK(
      barrier_wait_ptrs.numel() == 2 * world_size,
      "wait pointer count must cover two barrier phases");
  TORCH_CHECK(
      send_counters.numel() == 2 * world_size,
      "send counter count must cover two barrier phases");
  TORCH_CHECK(
      wait_counters.numel() == 2 * world_size,
      "wait counter count must cover two barrier phases");
  TORCH_CHECK(
      records.dim() >= 2 && records.size(-1) == record_bytes,
      "records must end in record_bytes");
  TORCH_CHECK(
      output.dim() >= 2 && output.size(-1) == record_bytes,
      "output must end in record_bytes");

  const int64_t selected_records =
      local_indices_by_destination.numel() / world_size;
  TORCH_CHECK(
      selected_records <= INT64_MAX / record_bytes,
      "selected payload exceeds int64 capacity");
  const int64_t payload_bytes = selected_records * record_bytes;
  TORCH_CHECK(
      output.numel() == payload_bytes,
      "output size must exactly match the selected payload");

  const auto stream = c10::cuda::getCurrentCUDAStream().stream();
  if (payload_bytes > 0) {
    const bool vectorized =
        record_bytes % static_cast<int64_t>(sizeof(uint4)) == 0 &&
        reinterpret_cast<uintptr_t>(records.data_ptr<uint8_t>()) %
                alignof(uint4) ==
            0;
    const int64_t unit_bytes = vectorized ? sizeof(uint4) : sizeof(uint8_t);
    const int64_t units_per_record = record_bytes / unit_bytes;
    const int64_t payload_units = selected_records * units_per_record;
    const int64_t requested_blocks =
        (payload_units + kThreads - 1) / kThreads;
    const int blocks = static_cast<int>(
        std::min<int64_t>(requested_blocks, kMaxBlocks));
    const dim3 grid(
        static_cast<unsigned int>(blocks),
        static_cast<unsigned int>(world_size));
    if (local_indices_by_destination.scalar_type() == torch::kInt32) {
      if (vectorized) {
        scatter_records_kernel<int32_t, uint4><<<grid, kThreads, 0, stream>>>(
            records.data_ptr<uint8_t>(),
            local_indices_by_destination.data_ptr<int32_t>(),
            peer_payload_ptrs.data_ptr<int64_t>(),
            selected_records,
            units_per_record,
            record_bytes,
            static_cast<int>(world_size));
      } else {
        scatter_records_kernel<int32_t, uint8_t><<<grid, kThreads, 0, stream>>>(
            records.data_ptr<uint8_t>(),
            local_indices_by_destination.data_ptr<int32_t>(),
            peer_payload_ptrs.data_ptr<int64_t>(),
            selected_records,
            units_per_record,
            record_bytes,
            static_cast<int>(world_size));
      }
    } else {
      if (vectorized) {
        scatter_records_kernel<int64_t, uint4><<<grid, kThreads, 0, stream>>>(
            records.data_ptr<uint8_t>(),
            local_indices_by_destination.data_ptr<int64_t>(),
            peer_payload_ptrs.data_ptr<int64_t>(),
            selected_records,
            units_per_record,
            record_bytes,
            static_cast<int>(world_size));
      } else {
        scatter_records_kernel<int64_t, uint8_t><<<grid, kThreads, 0, stream>>>(
            records.data_ptr<uint8_t>(),
            local_indices_by_destination.data_ptr<int64_t>(),
            peer_payload_ptrs.data_ptr<int64_t>(),
            selected_records,
            units_per_record,
            record_bytes,
            static_cast<int>(world_size));
      }
    }
    AT_CUDA_CHECK(cudaGetLastError());
  }

  barrier_all_peers_kernel<<<1, 32, 0, stream>>>(
      barrier_publish_ptrs.data_ptr<int64_t>(),
      barrier_wait_ptrs.data_ptr<int64_t>(),
      send_counters.data_ptr<int32_t>(),
      wait_counters.data_ptr<int32_t>(),
      static_cast<int>(world_size),
      static_cast<uint64_t>(timeout_cycles),
      0);
  AT_CUDA_CHECK(cudaGetLastError());

  if (payload_bytes > 0) {
    AT_CUDA_CHECK(cudaMemcpyAsync(
        output.data_ptr<uint8_t>(),
        reinterpret_cast<const void*>(
            static_cast<uintptr_t>(local_payload_ptr)),
        static_cast<size_t>(payload_bytes),
        cudaMemcpyDeviceToDevice,
        stream));
  }

  barrier_all_peers_kernel<<<1, 32, 0, stream>>>(
      barrier_publish_ptrs.data_ptr<int64_t>(),
      barrier_wait_ptrs.data_ptr<int64_t>(),
      send_counters.data_ptr<int32_t>(),
      wait_counters.data_ptr<int32_t>(),
      static_cast<int>(world_size),
      static_cast<uint64_t>(timeout_cycles),
      1);
  AT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "record_byte_offset_for_test",
      &record_byte_offset_for_test,
      "Compute the Int64 byte offset used by selected-record scatter");
  module.def(
      "exchange",
      &exchange,
      "Direct-scatter selected records with two device-side barriers");
}
