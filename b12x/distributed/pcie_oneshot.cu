// PCIe-only custom allreduce extension.
// System-scope barriers for cross-GPU visibility over PCIe switch fabric.

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#define CHECK_CUDA_SUCCESS(cmd)                                         \
  do {                                                                  \
    cudaError_t e = cmd;                                                \
    if (e != cudaSuccess) {                                             \
      std::stringstream _message;                                       \
      auto s = cudaGetErrorString(e);                                   \
      _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__; \
      throw std::runtime_error(_message.str());                         \
    }                                                                   \
  } while (0)

namespace pcie_allreduce {

constexpr int kMaxBlocks = 36;
constexpr int kMaxRanks = 16;
using FlagType = uint32_t;

// 128B stride per rank to avoid PCIe false sharing.
constexpr int kFlagStride = 32;

struct Signal {
  alignas(128) FlagType self_counter[kMaxBlocks][kMaxRanks];
  alignas(128) FlagType peer_counter[2][kMaxBlocks][kMaxRanks * kFlagStride];
  alignas(128) FlagType rms_completion[kMaxBlocks];
  alignas(128) float rms_partial[kMaxBlocks];
};

struct __align__(16) RankData {
  const void* __restrict__ ptrs[kMaxRanks];
};

struct __align__(16) RankSignals {
  Signal* signals[kMaxRanks];
};

template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

template <typename T>
struct packed_t {
  using P = array_t<T, 16 / sizeof(T)>;
  using A = array_t<float, 16 / sizeof(T)>;
};

#define DINLINE __device__ __forceinline__

DINLINE float upcast_s(float val) { return val; }
DINLINE float upcast_s(half val) { return __half2float(val); }

template <typename T>
DINLINE T downcast_s(float val);
template <>
DINLINE half downcast_s(float val) {
  return __float2half(val);
}

DINLINE half& assign_add(half& a, half b) {
  a = __hadd(a, b);
  return a;
}
DINLINE float& assign_add(float& a, float b) {
  return a += b;
}

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
DINLINE float upcast_s(nv_bfloat16 val) { return __bfloat162float(val); }
template <>
DINLINE nv_bfloat16 downcast_s(float val) {
  return __float2bfloat16(val);
}
DINLINE nv_bfloat16& assign_add(nv_bfloat16& a, nv_bfloat16 b) {
  a = __hadd(a, b);
  return a;
}
#endif

template <typename T, int N>
DINLINE array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; i++) assign_add(a.data[i], b.data[i]);
  return a;
}

template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> val) {
  if constexpr (std::is_same<T, float>::value) {
    return val;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; i++) out.data[i] = upcast_s(val.data[i]);
    return out;
  }
}

template <typename O>
DINLINE O downcast(array_t<float, O::size> val) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return val;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; i++) out.data[i] = downcast_s<typename O::type>(val.data[i]);
    return out;
  }
}

static DINLINE void st_flag_relaxed(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.relaxed.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static DINLINE FlagType ld_flag_relaxed(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;
}

template <int ngpus, bool is_start>
DINLINE void multi_gpu_barrier(const RankSignals& sg, Signal* self_sg, int rank) {
  if constexpr (!is_start) __syncthreads();
  if (threadIdx.x < ngpus) {
    __threadfence_system();
    auto val = self_sg->self_counter[blockIdx.x][threadIdx.x] += 1;
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->peer_counter[val % 2][blockIdx.x][rank * kFlagStride];
    auto self_counter_ptr = &self_sg->peer_counter[val % 2][blockIdx.x][threadIdx.x * kFlagStride];
    st_flag_relaxed(peer_counter_ptr, val);
    while (ld_flag_relaxed(self_counter_ptr) != val)
      ;
  }
  __syncthreads();
}

template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) packed_assign_add(tmp, upcast(ptrs[i][idx]));
  return downcast<P>(tmp);
}

template <typename T, int N>
DINLINE float packed_square_sum(array_t<T, N> value) {
  auto value_f = upcast(value);
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < N; i++) sum += value_f.data[i] * value_f.data[i];
  return sum;
}

DINLINE float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

DINLINE float block_reduce_sum(float value, float* warp_sums) {
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  value = warp_reduce_sum(value);
  if (lane == 0) warp_sums[warp] = value;
  __syncthreads();

  value = threadIdx.x < blockDim.x / warpSize ? warp_sums[lane] : 0.0f;
  if (warp == 0) value = warp_reduce_sum(value);
  return value;
}

// 1-stage allreduce with staggered peer reads and start-barrier-only. The end
// barrier is avoided by alternating between eager IPC buffers in a single
// ordered channel. Callers must not share a channel concurrently across CUDA
// streams; multi-stream use needs separate signal and staging buffers.
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) pcie_allreduce_kernel(
    RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result, int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  auto dp = *_dp;
  const P* rotated[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    rotated[i] = (const P*)dp.ptrs[(rank + i) % ngpus];
  }
  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>(rotated, idx);
  }
}

// Complete the allreduce, residual add, and RMSNorm in one launch. Reduction
// is split across several CTAs per row to preserve peer-read parallelism. The
// last CTA to publish its row partial performs the local normalization; no CTA
// waits for an unscheduled CTA, so this does not require cooperative launch.
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) pcie_allreduce_fused_add_rms_norm_kernel(
    RankData* _dp,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual,
    const T* __restrict__ weight,
    T* __restrict__ output,
    T* __restrict__ residual_output,
    int rank,
    int rows,
    int hidden_packs,
    int base_ctas_per_row,
    int extra_cta_rows,
    float epsilon) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  __shared__ float warp_sums[32];
  __shared__ float inv_rms;
  __shared__ int normalize_row;
  auto dp = *_dp;
  const P* rotated[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    rotated[i] = (const P*)dp.ptrs[(rank + i) % ngpus];
  }

  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);
  const P* residual_p = reinterpret_cast<const P*>(residual);
  const P* weight_p = reinterpret_cast<const P*>(weight);
  P* output_p = reinterpret_cast<P*>(output);
  P* residual_output_p = reinterpret_cast<P*>(residual_output);
  const int large_row_ctas = base_ctas_per_row + 1;
  const int large_row_blocks = extra_cta_rows * large_row_ctas;
  const bool in_large_row = blockIdx.x < large_row_blocks;
  const int row = in_large_row
                      ? blockIdx.x / large_row_ctas
                      : extra_cta_rows +
                            (blockIdx.x - large_row_blocks) / base_ctas_per_row;
  const int row_first_block =
      in_large_row ? row * large_row_ctas
                   : large_row_blocks +
                         (row - extra_cta_rows) * base_ctas_per_row;
  const int ctas_for_row =
      base_ctas_per_row + static_cast<int>(row < extra_cta_rows);
  const int row_cta = blockIdx.x - row_first_block;
  const int row_offset = row * hidden_packs;
  float square_sum = 0.0f;
  for (int col = row_cta * blockDim.x + threadIdx.x; col < hidden_packs;
       col += ctas_for_row * blockDim.x) {
    const int index = row_offset + col;
    P value = packed_reduce<P, ngpus, A>(rotated, index);
    packed_assign_add(value, residual_p[index]);
    residual_output_p[index] = value;
    square_sum += packed_square_sum(value);
  }
  square_sum = block_reduce_sum(square_sum, warp_sums);
  if (threadIdx.x == 0) {
    self_sg->rms_partial[blockIdx.x] = square_sum;
    __threadfence();
    const FlagType previous = atomicAdd(&self_sg->rms_completion[row], 1u);
    normalize_row = previous == static_cast<FlagType>(ctas_for_row - 1);
  }
  __syncthreads();
  if (!normalize_row) return;

  if (threadIdx.x == 0) {
    float total = 0.0f;
#pragma unroll 1
    for (int row_part = 0; row_part < ctas_for_row; row_part++) {
      total += self_sg->rms_partial[row_first_block + row_part];
    }
    inv_rms = rsqrtf(total / (hidden_packs * P::size) + epsilon);
    atomicExch(&self_sg->rms_completion[row], 0u);
  }
  __syncthreads();
  for (int col = threadIdx.x; col < hidden_packs; col += blockDim.x) {
    const int index = row_offset + col;
    A value = upcast(residual_output_p[index]);
    A scale = upcast(weight_p[col]);
#pragma unroll
    for (int i = 0; i < A::size; i++) {
      value.data[i] *= inv_rms * scale.data[i];
    }
    output_p[index] = downcast<P>(value);
  }
}

using IPC_KEY = std::array<uint8_t, sizeof(cudaIpcMemHandle_t)>;

class PCIeAllreduce {
 public:
  int rank_;
  int world_size_;

  RankSignals sg_;
  std::unordered_map<void*, RankData*> buffers_;
  Signal* self_sg_;

  RankData *d_rank_data_base_, *d_rank_data_end_;
  std::vector<void*> graph_unreg_buffers_;
  std::map<IPC_KEY, char*> ipc_handles_;

  bool dbuf_enabled_ = false;
  int dbuf_slot_ = 0;
  void* dbuf_raw_[2][kMaxRanks] = {};
  RankData* dbuf_rd_[2] = {};

  PCIeAllreduce(Signal** signals, void* rank_data, size_t rank_data_sz, int rank, int world_size)
      : rank_(rank),
        world_size_(world_size),
        self_sg_(signals[rank]),
        d_rank_data_base_(reinterpret_cast<RankData*>(rank_data)),
        d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData)) {
    for (int i = 0; i < world_size_; i++) sg_.signals[i] = signals[i];
  }

  char* open_ipc_handle(const void* ipc_handle) {
    auto [it, new_handle] = ipc_handles_.insert({*((IPC_KEY*)ipc_handle), nullptr});
    if (new_handle) {
      char* ipc_ptr;
      CHECK_CUDA_SUCCESS(cudaIpcOpenMemHandle(
          (void**)&ipc_ptr, *((const cudaIpcMemHandle_t*)ipc_handle), cudaIpcMemLazyEnablePeerAccess));
      it->second = ipc_ptr;
    }
    return it->second;
  }

  std::pair<std::string, std::vector<int64_t>> get_graph_buffer_ipc_meta() {
    auto num_buffers = graph_unreg_buffers_.size();
    auto handle_sz = sizeof(cudaIpcMemHandle_t);
    std::string handles(handle_sz * num_buffers, static_cast<char>(0));
    std::vector<int64_t> offsets(num_buffers);
    for (size_t i = 0; i < num_buffers; i++) {
      auto ptr = graph_unreg_buffers_[i];
      void* base_ptr;
      if (cuPointerGetAttribute(&base_ptr, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, (CUdeviceptr)ptr) != CUDA_SUCCESS)
        throw std::runtime_error("failed to get pointer attr");
      CHECK_CUDA_SUCCESS(cudaIpcGetMemHandle((cudaIpcMemHandle_t*)&handles[i * handle_sz], base_ptr));
      offsets[i] = ((char*)ptr) - ((char*)base_ptr);
    }
    return std::make_pair(handles, offsets);
  }

  void check_rank_data_capacity(size_t num = 1) {
    if (d_rank_data_base_ + num > d_rank_data_end_)
      throw std::runtime_error("Rank data buffer overflow");
  }

  void register_pcie_buffers(void** ptrs0, void** ptrs1) {
    check_rank_data_capacity(2);
    for (int s = 0; s < 2; s++) {
      void** ptrs = s == 0 ? ptrs0 : ptrs1;
      RankData data;
      for (int i = 0; i < world_size_; i++) {
        data.ptrs[i] = ptrs[i];
        dbuf_raw_[s][i] = ptrs[i];
      }
      dbuf_rd_[s] = d_rank_data_base_++;
      CHECK_CUDA_SUCCESS(cudaMemcpy(dbuf_rd_[s], &data, sizeof(RankData), cudaMemcpyHostToDevice));
    }
    buffers_[ptrs0[rank_]] = dbuf_rd_[0];
    buffers_[ptrs1[rank_]] = dbuf_rd_[1];
    dbuf_enabled_ = true;
    dbuf_slot_ = 0;
  }

  void register_buffer(void** ptrs) {
    check_rank_data_capacity();
    RankData data;
    for (int i = 0; i < world_size_; i++) data.ptrs[i] = ptrs[i];
    auto d_data = d_rank_data_base_++;
    CHECK_CUDA_SUCCESS(cudaMemcpy(d_data, &data, sizeof(RankData), cudaMemcpyHostToDevice));
    buffers_[ptrs[rank_]] = d_data;
  }

  void register_graph_buffers(
      const std::vector<std::string>& handles, const std::vector<std::vector<int64_t>>& offsets) {
    auto num_buffers = graph_unreg_buffers_.size();
    check_rank_data_capacity(num_buffers);
    std::vector<RankData> rank_data(num_buffers);
    for (size_t i = 0; i < num_buffers; i++) {
      auto self_ptr = graph_unreg_buffers_[i];
      auto& rd = rank_data[i];
      for (int j = 0; j < world_size_; j++) {
        if (j != rank_) {
          char* handle = open_ipc_handle(&handles[j][i * sizeof(cudaIpcMemHandle_t)]);
          handle += offsets[j][i];
          rd.ptrs[j] = handle;
        } else {
          rd.ptrs[j] = self_ptr;
        }
      }
    }
    CHECK_CUDA_SUCCESS(
        cudaMemcpy(d_rank_data_base_, rank_data.data(), sizeof(RankData) * num_buffers, cudaMemcpyHostToDevice));
    d_rank_data_base_ += num_buffers;
    graph_unreg_buffers_.clear();
  }

  template <typename T>
  void allreduce(cudaStream_t stream, T* input, T* output, int size, int threads = 512, int block_limit = 36) {
    auto d = packed_t<T>::P::size;
    if (size % d != 0)
      throw std::runtime_error("allreduce requires input length to be multiple of " + std::to_string(d));
    if (block_limit > kMaxBlocks)
      throw std::runtime_error("max supported block limit is " + std::to_string(kMaxBlocks));

    RankData* ptrs;
    cudaStreamCaptureStatus status;
    CHECK_CUDA_SUCCESS(cudaStreamIsCapturing(stream, &status));

    if (dbuf_enabled_) {
      int slot = dbuf_slot_ % 2;
      dbuf_slot_++;
      auto input_size = size * sizeof(T);
      AT_CUDA_CHECK(cudaMemcpyAsync(dbuf_raw_[slot][rank_], input, input_size, cudaMemcpyDeviceToDevice, stream));
      ptrs = dbuf_rd_[slot];
    } else if (status == cudaStreamCaptureStatusActive) {
      throw std::runtime_error(
          "PCIe oneshot graph capture requires eager IPC buffers; construct the runtime with eager buffers or use "
          "PCIeOneshotAllReducePool");
    } else {
      auto it = buffers_.find(input);
      if (it == buffers_.end())
        throw std::runtime_error(
            "buffer address " + std::to_string(reinterpret_cast<uint64_t>(input)) + " is not registered!");
      ptrs = it->second;
    }

    size /= d;
    int blocks = std::min(block_limit, (size + threads - 1) / threads);
    blocks = std::max(blocks, 1);

#define KL(ngpus) pcie_allreduce_kernel<T, ngpus><<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, rank_, size);
    switch (world_size_) {
      case 2:
        KL(2);
        break;
      case 4:
        KL(4);
        break;
      case 6:
        KL(6);
        break;
      case 8:
        KL(8);
        break;
      case 10:
        KL(10);
        break;
      default:
        throw std::runtime_error("only supports (2,4,6,8,10) gpus, got " + std::to_string(world_size_));
    }
#undef KL
  }

  template <typename T>
  void allreduce_fused_add_rms_norm(
      cudaStream_t stream,
      T* input,
      const T* residual,
      const T* weight,
      T* output,
      T* residual_output,
      int size,
      int hidden_size,
      float epsilon) {
    using P = typename packed_t<T>::P;
    const int pack_size = P::size;
    if (hidden_size <= 0 || size <= 0 || size % hidden_size != 0)
      throw std::runtime_error("fused allreduce RMSNorm requires complete non-empty rows");
    if (hidden_size % pack_size != 0)
      throw std::runtime_error(
          "fused allreduce RMSNorm requires hidden size to be a multiple of " + std::to_string(pack_size));

    RankData* ptrs;
    cudaStreamCaptureStatus status;
    CHECK_CUDA_SUCCESS(cudaStreamIsCapturing(stream, &status));
    if (dbuf_enabled_) {
      int slot = dbuf_slot_ % 2;
      dbuf_slot_++;
      AT_CUDA_CHECK(cudaMemcpyAsync(
          dbuf_raw_[slot][rank_], input, size * sizeof(T), cudaMemcpyDeviceToDevice, stream));
      ptrs = dbuf_rd_[slot];
    } else if (status == cudaStreamCaptureStatusActive) {
      throw std::runtime_error(
          "PCIe oneshot graph capture requires eager IPC buffers; construct the runtime with eager buffers or use "
          "PCIeOneshotAllReducePool");
    } else {
      auto it = buffers_.find(input);
      if (it == buffers_.end())
        throw std::runtime_error(
            "buffer address " + std::to_string(reinterpret_cast<uint64_t>(input)) + " is not registered!");
      ptrs = it->second;
    }

    int rows = size / hidden_size;
    int hidden_packs = hidden_size / pack_size;
    if (rows > kMaxBlocks)
      throw std::runtime_error(
          "fused allreduce RMSNorm supports at most " + std::to_string(kMaxBlocks) + " rows");
    const int threads = 512;
    const int size_packs = size / pack_size;
    const int blocks =
        std::min(kMaxBlocks, std::max(rows, (size_packs + threads - 1) / threads));
    const int base_ctas_per_row = blocks / rows;
    const int extra_cta_rows = blocks % rows;

#define KL(ngpus)                                                                                                    \
  pcie_allreduce_fused_add_rms_norm_kernel<T, ngpus><<<blocks, threads, 0, stream>>>(                                 \
      ptrs, sg_, self_sg_, residual, weight, output, residual_output, rank_, rows, hidden_packs, base_ctas_per_row,  \
      extra_cta_rows, epsilon);
      switch (world_size_) {
        case 2:
          KL(2);
          break;
        case 4:
          KL(4);
          break;
        case 6:
          KL(6);
          break;
        case 8:
          KL(8);
          break;
        case 10:
          KL(10);
          break;
        default:
          throw std::runtime_error("only supports (2,4,6,8,10) gpus, got " + std::to_string(world_size_));
      }
#undef KL
  }

  ~PCIeAllreduce() {
    for (auto [_, ptr] : ipc_handles_) CHECK_CUDA_SUCCESS(cudaIpcCloseMemHandle(ptr));
  }
};

}  // namespace pcie_allreduce

using fptr_t = int64_t;

static fptr_t init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs, torch::Tensor& rank_data, int64_t rank) {
  int world_size = fake_ipc_ptrs.size();
  if (world_size > pcie_allreduce::kMaxRanks)
    throw std::invalid_argument("world size > " + std::to_string(pcie_allreduce::kMaxRanks) + " is not supported");
  if (world_size % 2 != 0) throw std::invalid_argument("Odd num gpus is not supported");
  if (rank < 0 || rank >= world_size) throw std::invalid_argument("invalid rank");

  pcie_allreduce::Signal* ipc_ptrs[pcie_allreduce::kMaxRanks];
  for (int i = 0; i < world_size; i++) ipc_ptrs[i] = reinterpret_cast<pcie_allreduce::Signal*>(fake_ipc_ptrs[i]);
  return (fptr_t) new pcie_allreduce::PCIeAllreduce(ipc_ptrs, rank_data.data_ptr(), rank_data.numel(), rank, world_size);
}

static bool _is_weak_contiguous(torch::Tensor& t) {
  return t.is_contiguous() ||
         (t.storage().nbytes() - t.storage_offset() * t.element_size() == t.numel() * t.element_size());
}

static void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out, fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(_is_weak_contiguous(inp));
  auto input_size = inp.numel() * inp.element_size();

  // When double-buffer is active, allreduce handles the memcpy and slot
  // alternation internally, including under CUDA graph capture.
  void* reg_buffer;
  if (fa->dbuf_enabled_) {
    reg_buffer = inp.data_ptr();
  } else if (_reg_buffer) {
    reg_buffer = reinterpret_cast<void*>(_reg_buffer);
    TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
    AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size, cudaMemcpyDeviceToDevice, stream));
  } else {
    reg_buffer = inp.data_ptr();
  }

  switch (out.scalar_type()) {
    case at::ScalarType::Float:
      fa->allreduce<float>(stream, (float*)reg_buffer, (float*)out.data_ptr(), out.numel());
      break;
    case at::ScalarType::Half:
      fa->allreduce<half>(stream, (half*)reg_buffer, (half*)out.data_ptr(), out.numel());
      break;
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16:
      fa->allreduce<nv_bfloat16>(stream, (nv_bfloat16*)reg_buffer, (nv_bfloat16*)out.data_ptr(), out.numel());
      break;
#endif
    default:
      throw std::runtime_error("only supports float32, float16 and bfloat16");
  }
}

static void all_reduce_fused_add_rms_norm(
    fptr_t _fa,
    torch::Tensor& inp,
    torch::Tensor& residual,
    torch::Tensor& weight,
    torch::Tensor& out,
    torch::Tensor& residual_out,
    double epsilon,
    fptr_t _reg_buffer,
    int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_GE(inp.dim(), 1);
  TORCH_CHECK_EQ(inp.scalar_type(), residual.scalar_type());
  TORCH_CHECK_EQ(inp.scalar_type(), weight.scalar_type());
  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.scalar_type(), residual_out.scalar_type());
  TORCH_CHECK_EQ(inp.device(), residual.device());
  TORCH_CHECK_EQ(inp.device(), weight.device());
  TORCH_CHECK_EQ(inp.device(), out.device());
  TORCH_CHECK_EQ(inp.device(), residual_out.device());
  TORCH_CHECK_EQ(inp.numel(), residual.numel());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK_EQ(inp.numel(), residual_out.numel());
  TORCH_CHECK_EQ(weight.dim(), 1);
  TORCH_CHECK_EQ(weight.numel(), inp.size(-1));
  TORCH_CHECK_GE(epsilon, 0.0);
  TORCH_CHECK(_is_weak_contiguous(inp));
  TORCH_CHECK(_is_weak_contiguous(residual));
  TORCH_CHECK(weight.is_contiguous());
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(_is_weak_contiguous(residual_out));
  TORCH_CHECK_NE(out.data_ptr(), residual_out.data_ptr());

  auto input_size = inp.numel() * inp.element_size();
  void* reg_buffer;
  if (fa->dbuf_enabled_) {
    reg_buffer = inp.data_ptr();
  } else if (_reg_buffer) {
    reg_buffer = reinterpret_cast<void*>(_reg_buffer);
    TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
    AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size, cudaMemcpyDeviceToDevice, stream));
  } else {
    reg_buffer = inp.data_ptr();
  }

#define CALL_FUSED(T)                                                                                              \
  fa->allreduce_fused_add_rms_norm<T>(                                                                             \
      stream,                                                                                                      \
      reinterpret_cast<T*>(reg_buffer),                                                                           \
      reinterpret_cast<const T*>(residual.data_ptr()),                                                             \
      reinterpret_cast<const T*>(weight.data_ptr()),                                                               \
      reinterpret_cast<T*>(out.data_ptr()),                                                                        \
      reinterpret_cast<T*>(residual_out.data_ptr()),                                                               \
      inp.numel(),                                                                                                  \
      inp.size(-1),                                                                                                 \
      static_cast<float>(epsilon));
  switch (inp.scalar_type()) {
    case at::ScalarType::Float:
      CALL_FUSED(float);
      break;
    case at::ScalarType::Half:
      CALL_FUSED(half);
      break;
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16:
      CALL_FUSED(nv_bfloat16);
      break;
#endif
    default:
      throw std::runtime_error("only supports float32, float16 and bfloat16");
  }
#undef CALL_FUSED
}

static void dispose(fptr_t _fa) {
  delete reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
}

static int64_t meta_size() {
  return sizeof(pcie_allreduce::Signal);
}

static void register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  TORCH_CHECK(fake_ipc_ptrs.size() == (size_t)fa->world_size_);
  void* ipc_ptrs[pcie_allreduce::kMaxRanks];
  for (size_t i = 0; i < fake_ipc_ptrs.size(); i++) ipc_ptrs[i] = reinterpret_cast<void*>(fake_ipc_ptrs[i]);
  fa->register_buffer(ipc_ptrs);
}

static void register_pcie_buffers(fptr_t _fa, const std::vector<fptr_t>& ptrs0, const std::vector<fptr_t>& ptrs1) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  TORCH_CHECK(ptrs0.size() == (size_t)fa->world_size_);
  TORCH_CHECK(ptrs1.size() == (size_t)fa->world_size_);
  void* p0[pcie_allreduce::kMaxRanks];
  void* p1[pcie_allreduce::kMaxRanks];
  for (size_t i = 0; i < ptrs0.size(); i++) {
    p0[i] = reinterpret_cast<void*>(ptrs0[i]);
    p1[i] = reinterpret_cast<void*>(ptrs1[i]);
  }
  fa->register_pcie_buffers(p0, p1);
}

static std::tuple<std::vector<int64_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(fptr_t _fa) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  auto [handle, offsets] = fa->get_graph_buffer_ipc_meta();
  std::vector<int64_t> bytes(handle.begin(), handle.end());
  return std::make_tuple(bytes, offsets);
}

static void register_graph_buffers(
    fptr_t _fa, const std::vector<std::vector<int64_t>>& handles, const std::vector<std::vector<int64_t>>& offsets) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  std::vector<std::string> bytes;
  bytes.reserve(handles.size());
  for (size_t i = 0; i < handles.size(); i++) bytes.emplace_back(handles[i].begin(), handles[i].end());
  fa->register_graph_buffers(bytes, offsets);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_custom_ar", &init_custom_ar, "init PCIe allreduce");
  m.def("all_reduce", &all_reduce, "PCIe allreduce");
  m.def(
      "all_reduce_fused_add_rms_norm",
      &all_reduce_fused_add_rms_norm,
      "PCIe allreduce fused with residual add and RMSNorm");
  m.def("dispose", &dispose, "dispose PCIe allreduce");
  m.def("meta_size", &meta_size, "signal metadata size");
  m.def("register_buffer", &register_buffer, "register IPC buffer");
  m.def("register_pcie_buffers", &register_pcie_buffers, "register double-buffered IPC buffers");
  m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta, "get graph buffer IPC meta");
  m.def("register_graph_buffers", &register_graph_buffers, "register graph buffers");
}
