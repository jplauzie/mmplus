#include "cudalaunch.hpp"
#include "rigidbodymodes_common.hpp"

// T is deduced from the sdata array's element type
template <int NFIELDS, typename T>
__device__ inline void blockReduceSum(T sdata, int tid) {
  for (unsigned int s = BLOCKDIM / 2; s > 0; s >>= 1) {
    if (tid < s) {
      #pragma unroll
      for (int f = 0; f < NFIELDS; f++)
        sdata[f][tid] += sdata[f][tid + s];
    }
    __syncthreads();
  }
}

// Warp-shuffle, no risk of divergence
//warps are 32 threads on all current CUDA GPUs
constexpr int warp_size = 32;
template <typename T, int WIDTH = 32>
__device__ __forceinline__ T warpReduceSum(T val) {
  static_assert(WIDTH > 0 && (WIDTH & (WIDTH - 1)) == 0, "WIDTH must be a power of two");
  #pragma unroll
  for (int offset = WIDTH / 2; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffffu, val, offset, WIDTH);
  return val;
}
