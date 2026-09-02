#pragma once

#include "datatypes.hpp"
#include "cudalaunch.hpp"
#include "system.hpp"

// f64 cell position — for accumulation kernels where cancellation matters.
__device__ inline double3 cellPositionDeviceD(const CuSystem& system, int idx) {
  int3 coord = system.grid.index2coord(idx);
  return double3{double(coord.x) * double(system.cellsize.x),
                 double(coord.y) * double(system.cellsize.y),
                 double(coord.z) * double(system.cellsize.z)};
}

// f32 cell position
__device__ inline real3 cellPositionDevice(const CuSystem& system, int idx) {
  int3 coord = system.grid.index2coord(idx);
  return real3{real(coord.x) * system.cellsize.x,
               real(coord.y) * system.cellsize.y,
               real(coord.z) * system.cellsize.z};
}

__host__ __device__ inline double3 addD3(double3 a, double3 b) {
  return double3{a.x + b.x, a.y + b.y, a.z + b.z};
}
__host__ __device__ inline double3 subD3(double3 a, double3 b) {
  return double3{a.x - b.x, a.y - b.y, a.z - b.z};
}
__host__ __device__ inline double3 crossD3(double3 a, double3 b) {
  return double3{a.y * b.z - a.z * b.y,
                 a.z * b.x - a.x * b.z,
                 a.x * b.y - a.y * b.x};
}

__host__ __device__ inline real3 toReal3(double3 v) {
  return real3{real(v.x), real(v.y), real(v.z)};
}
__host__ __device__ inline double3 toDouble3(real3 v) {
  return double3{double(v.x), double(v.y), double(v.z)};
}


// T is deduced from the sdata array's element type
template <int N_accums, typename T>
__device__ inline void blockReduceSum(T sdata[N_accums][BLOCKDIM], int tid) {
  for (unsigned int s = BLOCKDIM / 2; s > 0; s >>= 1) {
    if (tid < s) {
      #pragma unroll
      for (int f = 0; f < N_accums; f++)
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
