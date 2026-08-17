#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "cudaerror.hpp"
#include "cudalaunch.hpp"
#include "cudastream.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "gpubuffer.hpp"
#include "magnet.hpp"
#include "parameter.hpp"
#include "rigidbodymodes2.hpp"
#include "system.hpp"

namespace {

// ---- mass-weighted center of mass (double precision, single block) -------
// Deliberately duplicated from rigidbodymodes.cu rather than shared: this
// file is meant to be a self-contained, independently comparable
// implementation. Anonymous-namespace internal linkage means the
// duplicate symbol names don't collide across translation units.

__device__ inline double3 cellPositionDeviceD2(const CuSystem& system, int idx) {
  int3 coord = system.grid.index2coord(idx);
  return double3{double(coord.x) * double(system.cellsize.x),
                 double(coord.y) * double(system.cellsize.y),
                 double(coord.z) * double(system.cellsize.z)};
}

__global__ void k_com2(double4* result, CuSystem system, CuParameter rho) {
  __shared__ double sx[BLOCKDIM], sy[BLOCKDIM], sz[BLOCKDIM], sw[BLOCKDIM];
  int ncells = system.grid.ncells();
  int tid = threadIdx.x;

  double lx = 0, ly = 0, lz = 0, lw = 0;
  for (int i = tid; i < ncells; i += BLOCKDIM) {
    if (!system.inGeometry(i))
      continue;
    double w = rho.valueAt(i);
    double3 p = cellPositionDeviceD2(system, i);
    lx += w * p.x; ly += w * p.y; lz += w * p.z; lw += w;
  }
  sx[tid] = lx; sy[tid] = ly; sz[tid] = lz; sw[tid] = lw;
  __syncthreads();

  for (unsigned int s = BLOCKDIM / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sx[tid] += sx[tid + s]; sy[tid] += sy[tid + s];
      sz[tid] += sz[tid + s]; sw[tid] += sw[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0)
    *result = {sx[0], sy[0], sz[0], sw[0]};
}

double3 computeCom2(const Magnet* magnet) {
  std::shared_ptr<const System> system = magnet->system();
  CuSystem cusys = system->cu();
  CuParameter rho = magnet->rho.cu();

  GpuBuffer<double4> d_result(1);
  cudaLaunchReductionKernel(k_com2, d_result.get(), cusys, rho);

  double4 sum;
  checkCudaError(cudaMemcpyAsync(&sum, d_result.get(), sizeof(double4),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));

  if (sum.w <= 0.0)
    throw std::runtime_error(
        "computeRigidBodyModes2: total mass (sum of rho) in geometry is zero or negative.");

  return double3{sum.x / sum.w, sum.y / sum.w, sum.z / sum.w};
}

// ---- generic inner product: rho-weighted (kinematic) or 1/rho-weighted
//      (force) -- see rigidbodymodes2.hpp for which is which. ------------

__global__ void k_genericWeightedDot(double* result, CuField a, CuField b,
                                     CuParameter rho, bool inverseWeight) {
  __shared__ double sdata[BLOCKDIM];
  int ncells = a.system.grid.ncells();
  int tid = threadIdx.x;

  double threadValue = 0.0;
  for (int i = tid; i < ncells; i += BLOCKDIM) {
    if (!a.cellInGeometry(i))
      continue;
    double w = double(rho.valueAt(i));
    if (inverseWeight) {
      if (w <= 0.0) continue;  // guard, shouldn't happen inside geometry
      w = 1.0 / w;
    }
    real3 va = a.vectorAt(i);
    real3 vb = b.vectorAt(i);
    threadValue += w * (double(va.x) * double(vb.x) +
                        double(va.y) * double(vb.y) +
                        double(va.z) * double(vb.z));
  }
  sdata[tid] = threadValue;
  __syncthreads();

  for (unsigned int s = BLOCKDIM / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0)
    *result = sdata[0];
}

double genericWeightedDot(const Field& a, const Field& b, const Magnet* magnet, bool inverseWeight) {
  CuParameter rho = magnet->rho.cu();
  GpuBuffer<double> d_result(1);
  cudaLaunchReductionKernel(k_genericWeightedDot, d_result.get(), a.cu(), b.cu(), rho, inverseWeight);

  double result;
  checkCudaError(cudaMemcpyAsync(&result, d_result.get(), sizeof(double),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));
  return result;
}

// ---- generic mode construction: bare direction (kinematic) or
//      rho(r)-scaled direction (force) -- see rigidbodymodes2.hpp --------

__global__ void k_setTranslationModeGeneric(CuField f, real3 dir, CuParameter rho, bool scaleByRho) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!f.cellInGrid(idx)) return;
  if (!f.cellInGeometry(idx)) { f.setVectorInCell(idx, real3{0, 0, 0}); return; }
  real scale = scaleByRho ? real(rho.valueAt(idx)) : real(1.0);
  f.setVectorInCell(idx, scale * dir);
}

__global__ void k_setRotationModeGeneric(CuField f, real3 axis, real3 com,
                                         CuParameter rho, bool scaleByRho) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!f.cellInGrid(idx)) return;
  if (!f.cellInGeometry(idx)) { f.setVectorInCell(idx, real3{0, 0, 0}); return; }
  int3 coord = f.system.grid.index2coord(idx);
  real3 pos = int3_to_real3(coord) * f.system.cellsize;
  real3 r = pos - com;
  real scale = scaleByRho ? real(rho.valueAt(idx)) : real(1.0);
  f.setVectorInCell(idx, scale * cross(axis, r));
}

std::array<std::array<double, 3>, 3> inertiaViaProjectionImpl(const Magnet* magnet) {
  std::shared_ptr<const System> system = magnet->system();
  double3 comD = computeCom2(magnet);
  real3 com = {real(comD.x), real(comD.y), real(comD.z)};
  CuParameter rho = magnet->rho.cu();

  real3 axes[3] = {real3{1, 0, 0}, real3{0, 1, 0}, real3{0, 0, 1}};
  Field rot[3] = {Field(system, 3), Field(system, 3), Field(system, 3)};
  for (int k = 0; k < 3; k++) {
    int ncells = rot[k].grid().ncells();
    cudaLaunch(ncells, k_setRotationModeGeneric, rot[k].cu(), axes[k], com, rho, /*scaleByRho=*/false);
  }

  std::array<std::array<double, 3>, 3> I{};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      I[i][j] = genericWeightedDot(rot[i], rot[j], magnet, /*inverseWeight=*/false);
  return I;
}

}  // namespace

std::array<std::array<double, 3>, 3> inertiaTensorViaProjection2(const Magnet* magnet) {
  return inertiaViaProjectionImpl(magnet);
}

// isForce = false: kinematic (rho-weighted).
// isForce = true: force (1/rho-weighted -- see rigidbodymodes2.hpp).
double weightedDot(const Field& a, const Field& b, const Magnet* magnet, bool isForce) {
  return genericWeightedDot(a, b, magnet, /*inverseWeight=*/isForce);
}

std::array<double, 6> rigidBodyModeCoefficients2(const Field& f, const RigidBodyModes2& modes,
                                                  const Magnet* magnet, bool isForce) {
  std::array<double, 6> c;
  for (int k = 0; k < 6; k++)
    c[k] = weightedDot(f, modes.modes[k], magnet, isForce);
  return c;
}

RigidBodyModes2 computeRigidBodyModes2(const Magnet* magnet, bool isForce) {
  std::shared_ptr<const System> system = magnet->system();

  double3 comD = computeCom2(magnet);
  real3 com = {real(comD.x), real(comD.y), real(comD.z)};
  CuParameter rho = magnet->rho.cu();

  RigidBodyModes2 result;

  // Translation modes: unit direction, rho(r)-scaled if isForce.
  real3 dirs[3] = {real3{1, 0, 0}, real3{0, 1, 0}, real3{0, 0, 1}};
  for (int k = 0; k < 3; k++) {
    result.modes[k] = Field(system, 3);
    int ncells = result.modes[k].grid().ncells();
    cudaLaunch(ncells, k_setTranslationModeGeneric, result.modes[k].cu(), dirs[k], rho, isForce);
  }

  // Rotation modes: v(r) = axis x (r - com), rho(r)-scaled if isForce,
  // zero outside geometry.
  for (int k = 0; k < 3; k++) {
    result.modes[3 + k] = Field(system, 3);
    int ncells = result.modes[3 + k].grid().ncells();
    cudaLaunch(ncells, k_setRotationModeGeneric, result.modes[3 + k].cu(), dirs[k], com, rho, isForce);
  }

  // Modified Gram-Schmidt, under the rho-weighted (kinematic) or
  // 1/rho-weighted (force) inner product. Translations are already
  // exactly orthogonal to each other and (analytically, given an exact
  // center of mass) to the rotations -- so this mainly orthogonalizes the
  // 3 rotation modes against each other. That's the direct-projection
  // equivalent of diagonalizing the inertia tensor, but without ever
  // forming or inverting a 3x3 matrix.
  const double normEpsilon = 1e-12;  // TODO: tune, or scale relative to system size
  for (int k = 0; k < 6; k++) {
    for (int j = 0; j < k; j++) {
      double c = weightedDot(result.modes[k], result.modes[j], magnet, isForce);
      addTo(result.modes[k], real(-c), result.modes[j]);
    }
    double normSq = weightedDot(result.modes[k], result.modes[k], magnet, isForce);
    double norm = std::sqrt(std::max(normSq, 0.0));
    if (norm > normEpsilon) {
      result.modes[k] = real(1.0 / norm) * result.modes[k];
    } else {
      // Degenerate / linearly-dependent mode (e.g. not enough independent
      // cells to support this rotation) -- zero it out so it never
      // contributes to the projection, instead of blowing up like a
      // near-singular inertia-tensor inverse would.
      result.modes[k].makeZero();
    }
  }

  return result;
}

void removeRigidBodyModes2(Field& f, const RigidBodyModes2& modes,
                           const Magnet* magnet, bool isForce) {
  for (int k = 0; k < 6; k++) {
    double c = weightedDot(f, modes.modes[k], magnet, isForce);
    addTo(f, real(-c), modes.modes[k]);
  }
}