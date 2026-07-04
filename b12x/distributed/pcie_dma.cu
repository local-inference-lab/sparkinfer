// PCIe ring allreduce transport primitives.
//
// Prefill-size allreduce is bandwidth-bound and the per-GPU x16 link is the
// invariant bottleneck (2*(N-1)/N * S egress per rank for any algorithm).
// On this fabric CE peer copies sustain ~56 GB/s while SM peer reads run at
// ~27 GB/s and SM peer writes at ~3 GB/s, so the data plane is CE
// (cudaMemcpyAsync peer copies) and the SM only synchronizes and reduces:
//
//   copy chunk -> peer scratch (CE, stream-ordered)
//   set_flag   -> peer flag    (tiny SM kernel, monotonic device counter)
//   wait_flag  -> local flag   (spin kernel, graph-replay safe)
//   add        -> accumulate received chunk into the working buffer
//
// Flag values come from device-resident monotonic counters so captured
// graphs replay without host-side value patching, mirroring the oneshot
// barrier's self_counter scheme.

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/extension.h>

#include <sstream>
#include <stdexcept>

#define CHECK_CUDA_SUCCESS(cmd)                                         \
  do {                                                                  \
    cudaError_t e = cmd;                                                \
    if (e != cudaSuccess) {                                             \
      std::stringstream _message;                                       \
      _message << cudaGetErrorString(e) << "\n"                         \
               << __FILE__ << ':' << __LINE__;                          \
      throw std::runtime_error(_message.str());                        \
    }                                                                   \
  } while (0)

namespace pcie_dma {

using FlagType = unsigned int;

__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T from_float(float v);
template <>
__device__ __forceinline__ float from_float<float>(float v) { return v; }
template <>
__device__ __forceinline__ half from_float<half>(float v) { return __float2half(v); }
template <>
__device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

__global__ void set_flag_kernel(FlagType* peer_flag, FlagType* local_counter) {
  const FlagType value = *local_counter + 1;
  *local_counter = value;
  // The CE copy this flag publishes completed in stream order before this
  // kernel launched; the system fence orders any outstanding writes.
  __threadfence_system();
  asm volatile("st.relaxed.sys.global.u32 [%1], %0;" ::"r"(value), "l"(peer_flag));
}

__global__ void wait_flag_kernel(FlagType* flag, FlagType* expected_counter) {
  const FlagType expected = *expected_counter + 1;
  *expected_counter = expected;
  FlagType observed;
  do {
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(observed) : "l"(flag));
  } while (static_cast<int>(observed - expected) < 0);
}

template <typename T>
__global__ void __launch_bounds__(256, 1) add_kernel(T* __restrict__ dst,
                                                     const T* __restrict__ src,
                                                     long long packs) {
  using Pack = uint4;
  Pack* dst_p = reinterpret_cast<Pack*>(dst);
  const Pack* src_p = reinterpret_cast<const Pack*>(src);
  constexpr int kElems = sizeof(Pack) / sizeof(T);
  for (long long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < packs;
       idx += gridDim.x * blockDim.x) {
    Pack a = dst_p[idx];
    Pack b = src_p[idx];
    T* av = reinterpret_cast<T*>(&a);
    const T* bv = reinterpret_cast<const T*>(&b);
#pragma unroll
    for (int e = 0; e < kElems; ++e) {
      av[e] = from_float<T>(to_float(av[e]) + to_float(bv[e]));
    }
    dst_p[idx] = a;
  }
}

}  // namespace pcie_dma

static void dma_copy(int64_t dst_ptr, int64_t src_ptr, int64_t bytes) {
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  CHECK_CUDA_SUCCESS(cudaMemcpyAsync(reinterpret_cast<void*>(dst_ptr),
                                     reinterpret_cast<const void*>(src_ptr),
                                     static_cast<size_t>(bytes),
                                     cudaMemcpyDeviceToDevice, stream));
}

static void dma_set_flag(int64_t peer_flag_ptr, int64_t counter_ptr) {
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  pcie_dma::set_flag_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<pcie_dma::FlagType*>(peer_flag_ptr),
      reinterpret_cast<pcie_dma::FlagType*>(counter_ptr));
}

static void dma_wait_flag(int64_t flag_ptr, int64_t counter_ptr) {
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  pcie_dma::wait_flag_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<pcie_dma::FlagType*>(flag_ptr),
      reinterpret_cast<pcie_dma::FlagType*>(counter_ptr));
}

static void dma_add(int64_t dst_ptr, int64_t src_ptr, int64_t elems,
                     int64_t dtype_code) {
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  const int threads = 256;
  const long long packs16 = elems * (dtype_code == 2 ? 4 : 2) / 16;
  const int blocks = static_cast<int>(
      std::max<long long>(1, std::min<long long>(64, (packs16 + threads - 1) / threads)));
  if (dtype_code == 0) {
    const long long packs = elems / 8;
    pcie_dma::add_kernel<nv_bfloat16><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<nv_bfloat16*>(dst_ptr),
        reinterpret_cast<const nv_bfloat16*>(src_ptr), packs);
  } else if (dtype_code == 1) {
    const long long packs = elems / 8;
    pcie_dma::add_kernel<half><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<half*>(dst_ptr), reinterpret_cast<const half*>(src_ptr),
        packs);
  } else {
    const long long packs = elems / 4;
    pcie_dma::add_kernel<float><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<float*>(dst_ptr), reinterpret_cast<const float*>(src_ptr),
        packs);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dma_copy", &dma_copy, "CE peer copy on the current stream");
  m.def("dma_set_flag", &dma_set_flag, "publish a monotonic flag to a peer");
  m.def("dma_wait_flag", &dma_wait_flag, "wait for a monotonic peer flag");
  m.def("dma_add", &dma_add, "elementwise add src into dst");
}
