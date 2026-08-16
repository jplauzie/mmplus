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

// ---- rotation mode construction: v(r) = axis x (r - com) ------------------
// Single elementwise pass (not a reduction), so real3/float is fine here,
// same precision argument as k_subtractRigidModes in rigidbodymodes.cu.

__global__ void k_setRotationMode(CuField f, real3 axis, real3 com) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!f.cellInGrid(idx))
    return;
  if (!f.cellInGeometry(idx)) {
    f.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  int3 coord = f.system.grid.index2coord(idx);
  real3 pos = int3_to_real3(coord) * f.system.cellsize;
  real3 r = pos - com;
  f.setVectorInCell(idx, cross(axis, r));
}

// ---- mass-weighted inner product (double precision, single block) --------
// Same reduction shape as reduce.cu's dotSum, but rho-weighted; not
// available there, so implemented locally.

__global__ void k_weightedDot(double* result, CuField a, CuField b, CuParameter rho) {
  __shared__ double sdata[BLOCKDIM];
  int ncells = a.system.grid.ncells();
  int tid = threadIdx.x;

  double threadValue = 0.0;
  for (int i = tid; i < ncells; i += BLOCKDIM) {
    if (!a.cellInGeometry(i))
      continue;
    double w = double(rho.valueAt(i));
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

double weightedDot(const Field& a, const Field& b, const Magnet* magnet) {
  CuParameter rho = magnet->rho.cu();
  GpuBuffer<double> d_result(1);
  cudaLaunchReductionKernel(k_weightedDot, d_result.get(), a.cu(), b.cu(), rho);

  double result;
  checkCudaError(cudaMemcpyAsync(&result, d_result.get(), sizeof(double),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));
  return result;
}

std::array<std::array<double, 3>, 3> inertiaViaProjectionImpl(const Magnet* magnet) {
  std::shared_ptr<const System> system = magnet->system();
  double3 comD = computeCom2(magnet);
  real3 com = {real(comD.x), real(comD.y), real(comD.z)};

  real3 axes[3] = {real3{1, 0, 0}, real3{0, 1, 0}, real3{0, 0, 1}};
  Field rot[3] = {Field(system, 3), Field(system, 3), Field(system, 3)};
  for (int k = 0; k < 3; k++) {
    int ncells = rot[k].grid().ncells();
    cudaLaunch(ncells, k_setRotationMode, rot[k].cu(), axes[k], com);
  }

  std::array<std::array<double, 3>, 3> I{};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      I[i][j] = weightedDot(rot[i], rot[j], magnet);
  return I;
}

}  // namespace

std::array<std::array<double, 3>, 3> inertiaTensorViaProjection2(const Magnet* magnet) {
  return inertiaViaProjectionImpl(magnet);
}

std::array<double, 6> rigidBodyModeCoefficients2(const Field& f,
                                                  const RigidBodyModes2& modes,
                                                  const Magnet* magnet) {
  std::array<double, 6> c;
  for (int k = 0; k < 6; k++)
    c[k] = weightedDot(f, modes.modes[k], magnet);
  return c;
}




RigidBodyModes2 computeRigidBodyModes2(const Magnet* magnet) {
  std::shared_ptr<const System> system = magnet->system();

  double3 comD = computeCom2(magnet);
  real3 com = {real(comD.x), real(comD.y), real(comD.z)};

  RigidBodyModes2 result;

  // Translation modes: uniform unit vectors. Field's real3-valued
  // constructor already masks outside geometry (see field.cu
  // k_setVectorValue), so no custom kernel needed.
  result.modes[0] = Field(system, 3, real3{1, 0, 0});
  result.modes[1] = Field(system, 3, real3{0, 1, 0});
  result.modes[2] = Field(system, 3, real3{0, 0, 1});

  // Rotation modes: v(r) = axis x (r - com), zero outside geometry.
  real3 axes[3] = {real3{1, 0, 0}, real3{0, 1, 0}, real3{0, 0, 1}};
  for (int k = 0; k < 3; k++) {
    result.modes[3 + k] = Field(system, 3);
    int ncells = result.modes[3 + k].grid().ncells();
    cudaLaunch(ncells, k_setRotationMode, result.modes[3 + k].cu(), axes[k], com);
  }

  // Modified Gram-Schmidt, mass-weighted inner product. Translations are
  // already exactly orthogonal to each other and (analytically, given an
  // exact center of mass: sum(rho * r) = 0 by definition) to the
  // rotations -- so this mainly orthogonalizes the 3 rotation modes
  // against each other. That's the direct-projection equivalent of
  // diagonalizing the inertia tensor, but without ever forming or
  // inverting a 3x3 matrix.
  const double normEpsilon = 1e-12;  // TODO: tune, or scale relative to system size
  for (int k = 0; k < 6; k++) {
    for (int j = 0; j < k; j++) {
      double c = weightedDot(result.modes[k], result.modes[j], magnet);
      addTo(result.modes[k], real(-c), result.modes[j]);
    }
    double normSq = weightedDot(result.modes[k], result.modes[k], magnet);
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

void removeRigidBodyModes2(Field& f, const RigidBodyModes2& modes, const Magnet* magnet) {
  for (int k = 0; k < 6; k++) {
    double c = weightedDot(f, modes.modes[k], magnet);
    addTo(f, real(-c), modes.modes[k]);
  }
}