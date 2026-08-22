#pragma once

#include "datatypes.hpp"
#include "cudalaunch.hpp"  
#include "system.hpp"      

// Small math/reduction helpers shared by rigidbodymodes.cu and rigidbodymodes4.cu. 

// f64 cell position. cellsize is f32 (CuSystem::cellsize is real3), avoids compounding additional rounding in
// everything downstream (com subtraction, cross products, sums). f64 maybe unnecessary?
__device__ inline double3 cellPositionDeviceD(const CuSystem& system, int idx) {
  int3 coord = system.grid.index2coord(idx);
  return double3{double(coord.x) * double(system.cellsize.x),
                 double(coord.y) * double(system.cellsize.y),
                 double(coord.z) * double(system.cellsize.z)};
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

template <int NFIELDS>
__device__ inline void blockReduceSum(double sdata[NFIELDS][BLOCKDIM], int tid) {
  for (unsigned int s = BLOCKDIM / 2; s > 0; s >>= 1) {
    if (tid < s) {
      #pragma unroll
      for (int f = 0; f < NFIELDS; f++)
        sdata[f][tid] += sdata[f][tid + s];
    }
    __syncthreads();
  }
}