#include "rigidbodymodes_diag.hpp"

#include <stdexcept>

#include "cudaerror.hpp"
#include "cudalaunch.hpp"
#include "cudastream.hpp"
#include "gpubuffer.hpp"

namespace {

// Double-precision atomicAdd is native on compute capability >= 6.0.
// CAS-loop fallback kept here defensively; this file has no other
// dependency on rigidbodymodes.cu so it should build standalone either way.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAddDiag(double* address, double val) {
  unsigned long long int* addr = (unsigned long long int*)address;
  unsigned long long int old = *addr, assumed;
  do {
    assumed = old;
    old = atomicCAS(addr, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#else
__device__ inline double atomicAddDiag(double* address, double val) {
  return atomicAdd(address, val);
}
#endif

// Straightforward, unweighted grid-stride-free sum of u plus a cell count,
// over cells in the geometry. Uses atomics rather than a tree reduction on
// purpose -- this is a verification path, so simplicity/independence from
// rigidbodymodes.cu matters more than performance.
__global__ void k_sumUnweighted(double* sums, int* count, CuField u) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!u.cellInGrid(idx) || !u.cellInGeometry(idx))
    return;

  real3 val = u.vectorAt(idx);
  atomicAddDiag(&sums[0], double(val.x));
  atomicAddDiag(&sums[1], double(val.y));
  atomicAddDiag(&sums[2], double(val.z));
  atomicAdd(count, 1);
}

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

void invert3x3Diag(const double I[3][3], double out[3][3]) {
  double det =
      I[0][0]*(I[1][1]*I[2][2] - I[1][2]*I[2][1]) -
      I[0][1]*(I[1][0]*I[2][2] - I[1][2]*I[2][0]) +
      I[0][2]*(I[1][0]*I[2][1] - I[1][1]*I[2][0]);

  if (det == 0.0)
    throw std::runtime_error("Rigid-body inertia tensor is singular.");

  double invDet = 1.0 / det;
  out[0][0] = ( I[1][1]*I[2][2] - I[1][2]*I[2][1]) * invDet;
  out[0][1] = (-I[0][1]*I[2][2] + I[0][2]*I[2][1]) * invDet;
  out[0][2] = ( I[0][1]*I[1][2] - I[0][2]*I[1][1]) * invDet;
  out[1][0] = (-I[1][0]*I[2][2] + I[1][2]*I[2][0]) * invDet;
  out[1][1] = ( I[0][0]*I[2][2] - I[0][2]*I[2][0]) * invDet;
  out[1][2] = (-I[0][0]*I[1][2] + I[0][2]*I[1][0]) * invDet;
  out[2][0] = ( I[1][0]*I[2][1] - I[1][1]*I[2][0]) * invDet;
  out[2][1] = (-I[0][0]*I[2][1] + I[0][1]*I[2][0]) * invDet;
  out[2][2] = ( I[0][0]*I[1][1] - I[0][1]*I[1][0]) * invDet;
}

__host__ __device__ inline double3 crossD3(double3 a, double3 b) {
  return double3{a.y * b.z - a.z * b.y,
                 a.z * b.x - a.x * b.z,
                 a.x * b.y - a.y * b.x};
}

}  // namespace

real3 computeNetTranslationUnweighted(const Field& u) {
  int ncells = u.system()->grid().ncells();

  GpuBuffer<double> d_sums(3);
  GpuBuffer<int> d_count(1);
  checkCudaError(cudaMemsetAsync(d_sums.get(), 0, 3 * sizeof(double), getCudaStream()));
  checkCudaError(cudaMemsetAsync(d_count.get(), 0, sizeof(int), getCudaStream()));

  cudaLaunch(ncells, k_sumUnweighted, d_sums.get(), d_count.get(), u.cu());

  double sums[3];
  int count;
  checkCudaError(cudaMemcpyAsync(sums, d_sums.get(), 3 * sizeof(double),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaMemcpyAsync(&count, d_count.get(), sizeof(int),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));

  if (count <= 0)
    throw std::runtime_error("computeNetTranslationUnweighted: zero cells in geometry.");

  return real3{real(sums[0] / count), real(sums[1] / count), real(sums[2] / count)};
}

__global__ void k_addTranslation(CuField u, real3 T) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!u.cellInGrid(idx) || !u.cellInGeometry(idx))
    return;

  real3 val = u.vectorAt(idx);
  u.setVectorInCell(idx, val + T);
}

void addTranslationUnweighted(Field& u, real3 T) {
  int ncells = u.system()->grid().ncells();
  cudaLaunch(ncells, k_addTranslation, u.cu(), T);
}

// rigidbodymodes_diag.cu
__global__ void k_addRotation(CuField u, double3 com, double3 omega) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!u.cellInGrid(idx) || !u.cellInGeometry(idx))
    return;

  double3 r = subD3(cellPositionDeviceD(u.system, idx), com);  // needs the same helpers as rigidbodymodes.cu, or a local copy per your "no shared code" rule
  double3 rot = {omega.y * r.z - omega.z * r.y,
                omega.z * r.x - omega.x * r.z,
                omega.x * r.y - omega.y * r.x};

  real3 val = u.vectorAt(idx);
  u.setVectorInCell(idx, val + real3{real(rot.x), real(rot.y), real(rot.z)});
}

__global__ void k_sumRotation(double* sums, int* count, CuField u, double3 com) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!u.cellInGrid(idx) || !u.cellInGeometry(idx))
    return;

  double3 r = subD3(cellPositionDeviceD(u.system, idx), com);
  real3 val = u.vectorAt(idx);
  double3 uD = {double(val.x), double(val.y), double(val.z)};
  double3 rxu = crossD3(r, uD);

  // L = sum(r x u), unweighted
  atomicAddDiag(&sums[0], rxu.x);
  atomicAddDiag(&sums[1], rxu.y);
  atomicAddDiag(&sums[2], rxu.z);

  // geometric (unweighted) inertia tensor about com, upper triangle
  double x = r.x, y = r.y, z = r.z;
  atomicAddDiag(&sums[3], y*y + z*z);   // Ixx
  atomicAddDiag(&sums[4], x*x + z*z);   // Iyy
  atomicAddDiag(&sums[5], x*x + y*y);   // Izz
  atomicAddDiag(&sums[6], -x*y);        // Ixy
  atomicAddDiag(&sums[7], -x*z);        // Ixz
  atomicAddDiag(&sums[8], -y*z);        // Iyz

  atomicAdd(count, 1);
}

real3 computeNetRotationUnweighted(const Field& u, real3 com) {
  int ncells = u.system()->grid().ncells();
  double3 comD = {double(com.x), double(com.y), double(com.z)};

  GpuBuffer<double> d_sums(9);
  GpuBuffer<int> d_count(1);
  checkCudaError(cudaMemsetAsync(d_sums.get(), 0, 9 * sizeof(double), getCudaStream()));
  checkCudaError(cudaMemsetAsync(d_count.get(), 0, sizeof(int), getCudaStream()));

  cudaLaunch(ncells, k_sumRotation, d_sums.get(), d_count.get(), u.cu(), comD);

  double s[9];
  int count;
  checkCudaError(cudaMemcpyAsync(s, d_sums.get(), 9 * sizeof(double), cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaMemcpyAsync(&count, d_count.get(), sizeof(int), cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));

  if (count <= 0)
    throw std::runtime_error("computeNetRotationUnweighted: zero cells in geometry.");

  double I[3][3] = {{s[3], s[6], s[7]},
                    {s[6], s[4], s[8]},
                    {s[7], s[8], s[5]}};
  double Iinv[3][3];
  invert3x3Diag(I, Iinv);  // local copy, don't reuse rigidbodymodes.cu's invert3x3

  double3 L = {s[0], s[1], s[2]};
  double3 omega = {
    Iinv[0][0]*L.x + Iinv[0][1]*L.y + Iinv[0][2]*L.z,
    Iinv[1][0]*L.x + Iinv[1][1]*L.y + Iinv[1][2]*L.z,
    Iinv[2][0]*L.x + Iinv[2][1]*L.y + Iinv[2][2]*L.z
  };
  return real3{real(omega.x), real(omega.y), real(omega.z)};
}

void addRotationUnweighted(Field& u, real3 com, real3 omega) {
  int ncells = u.system()->grid().ncells();
  double3 comD = {double(com.x), double(com.y), double(com.z)};
  double3 omegaD = {double(omega.x), double(omega.y), double(omega.z)};
  cudaLaunch(ncells, k_addRotation, u.cu(), comD, omegaD);
}