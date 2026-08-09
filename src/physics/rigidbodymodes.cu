#include <stdexcept>
#include <vector>

#include "cudaerror.hpp"
#include "cudalaunch.hpp"
#include "cudastream.hpp"
#include "field.hpp"
#include "gpubuffer.hpp"
#include "magnet.hpp"
#include "parameter.hpp"
#include "reduce.hpp"
#include "rigidbodymodes.hpp"
#include "system.hpp"

namespace {

constexpr int MAX_REDUCTION_BLOCKS = 4096;

int numReductionBlocks(int ncells) {
  int n = (ncells + BLOCKDIM - 1) / BLOCKDIM;
  if (n < 1) n = 1;
  if (n > MAX_REDUCTION_BLOCKS) n = MAX_REDUCTION_BLOCKS;
  return n;
}

// Double-precision cell position. cellsize itself is only float32 at the
// source (CuSystem::cellsize is real3), so this can't recover precision
// already lost there -- it only avoids compounding additional rounding in
// everything downstream (com subtraction, cross products, sums).
__device__ inline double3 cellPositionDeviceD(const CuSystem& system, int idx) {
  int3 coord = system.grid.index2coord(idx);
  return double3{double(coord.x) * double(system.cellsize.x),
                 double(coord.y) * double(system.cellsize.y),
                 double(coord.z) * double(system.cellsize.z)};
}

__device__ inline double3 subD(double3 a, double3 b) {
  return double3{a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ inline double3 crossD(double3 a, double3 b) {
  return double3{a.y * b.z - a.z * b.y,
                 a.z * b.x - a.x * b.z,
                 a.x * b.y - a.y * b.x};
}

// ---- pass 1: mass-weighted center of mass, fused with total-mass sum ----

__global__ void k_comPartialSums(double4* blockSums, CuSystem system,
                                 CuParameter rho) {
  __shared__ double sx[BLOCKDIM], sy[BLOCKDIM], sz[BLOCKDIM], sw[BLOCKDIM];
  int ncells = system.grid.ncells();
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  double lx = 0, ly = 0, lz = 0, lw = 0;
  for (int i = gid; i < ncells; i += stride) {
    if (!system.inGeometry(i))
      continue;
    double w = rho.valueAt(i);
    double3 p = cellPositionDeviceD(system, i);
    lx += w * p.x;
    ly += w * p.y;
    lz += w * p.z;
    lw += w;
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
    blockSums[blockIdx.x] = {sx[0], sy[0], sz[0], sw[0]};
}

struct ComResult { double3 com; double totalRho; };

ComResult computeCom(const CuSystem& cusys, const CuParameter& rho, int ncells) {
  int numBlocks = numReductionBlocks(ncells);
  GpuBuffer<double4> d_partials(numBlocks);

  k_comPartialSums<<<numBlocks, BLOCKDIM, 0, getCudaStream()>>>(
      d_partials.get(), cusys, rho);
  checkCudaError(cudaPeekAtLastError());

  std::vector<double4> partials(numBlocks);
  checkCudaError(cudaMemcpyAsync(partials.data(), d_partials.get(),
                                 numBlocks * sizeof(double4),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));

  double firstmoment_x = 0, firstmoment_y = 0, firstmoment_z = 0, totalmass = 0;
  for (const double4& p : partials) {
    firstmoment_x += p.x; firstmoment_y += p.y; firstmoment_z += p.z; totalmass += p.w;
  }
  if (totalmass <= 0.0)
    throw std::runtime_error("computeRigidBodyGeometry: total mass (sum of rho) in geometry is zero or negative.");

  ComResult result;
  result.com = double3{firstmoment_x / totalmass, firstmoment_y / totalmass, firstmoment_z / totalmass};
  result.totalRho = totalmass;
  return result;
}

// ---- pass 2: mass-weighted inertia tensor about com -----------------------

struct InertiaPartial { double3 diag; double3 offdiag; };

__global__ void k_inertiaPartialSums(InertiaPartial* blockSums,
                                     CuSystem system, CuParameter rho,
                                     double3 com) {
  __shared__ double sxx[BLOCKDIM], syy[BLOCKDIM], szz[BLOCKDIM];
  __shared__ double sxy[BLOCKDIM], sxz[BLOCKDIM], syz[BLOCKDIM];
  int ncells = system.grid.ncells();
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  double xx = 0, yy = 0, zz = 0, xy = 0, xz = 0, yz = 0;
  for (int i = gid; i < ncells; i += stride) {
    if (!system.inGeometry(i))
      continue;
    double w = rho.valueAt(i);
    double3 r = subD(cellPositionDeviceD(system, i), com);
    double x = r.x, y = r.y, z = r.z;
    xx += w * (y * y + z * z);
    yy += w * (x * x + z * z);
    zz += w * (x * x + y * y);
    xy += w * (-x * y);
    xz += w * (-x * z);
    yz += w * (-y * z);
  }
  sxx[tid] = xx; syy[tid] = yy; szz[tid] = zz;
  sxy[tid] = xy; sxz[tid] = xz; syz[tid] = yz;
  __syncthreads();

  for (unsigned int s = BLOCKDIM / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sxx[tid] += sxx[tid + s]; syy[tid] += syy[tid + s]; szz[tid] += szz[tid + s];
      sxy[tid] += sxy[tid + s]; sxz[tid] += sxz[tid + s]; syz[tid] += syz[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    blockSums[blockIdx.x].diag = {sxx[0], syy[0], szz[0]};
    blockSums[blockIdx.x].offdiag = {sxy[0], sxz[0], syz[0]};
  }
}

void invert3x3(const double I[3][3], double out[3][3]) {
  double det =
      I[0][0]*(I[1][1]*I[2][2] - I[1][2]*I[2][1]) -
      I[0][1]*(I[1][0]*I[2][2] - I[1][2]*I[2][0]) +
      I[0][2]*(I[1][0]*I[2][1] - I[1][1]*I[2][0]);

  if (det == 0.0)
    throw std::runtime_error("Rigid-body inertia tensor is singular even "
                             "after regularization.");

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

// ---- translation + angular momentum, fused, mass-weighted -----------------

struct TransRotPartial { double3 rhoF; double3 rhoRxF; };

__global__ void k_transRotPartialSums(TransRotPartial* blockSums, CuField f,
                                      CuParameter rho, double3 com) {
  __shared__ double sfx[BLOCKDIM], sfy[BLOCKDIM], sfz[BLOCKDIM];
  __shared__ double slx[BLOCKDIM], sly[BLOCKDIM], slz[BLOCKDIM];
  int ncells = f.system.grid.ncells();
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  double fx = 0, fy = 0, fz = 0, lx = 0, ly = 0, lz = 0;
  for (int i = gid; i < ncells; i += stride) {
    if (!f.cellInGeometry(i))
      continue;
    double w = rho.valueAt(i);
    real3 vf = f.vectorAt(i);
    double3 v = double3{double(vf.x), double(vf.y), double(vf.z)};
    fx += w * v.x; fy += w * v.y; fz += w * v.z;

    double3 r = subD(cellPositionDeviceD(f.system, i), com);
    double3 c = crossD(r, v);
    lx += w * c.x; ly += w * c.y; lz += w * c.z;
  }
  sfx[tid] = fx; sfy[tid] = fy; sfz[tid] = fz;
  slx[tid] = lx; sly[tid] = ly; slz[tid] = lz;
  __syncthreads();

  for (unsigned int s = BLOCKDIM / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sfx[tid] += sfx[tid + s]; sfy[tid] += sfy[tid + s]; sfz[tid] += sfz[tid + s];
      slx[tid] += slx[tid + s]; sly[tid] += sly[tid + s]; slz[tid] += slz[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    blockSums[blockIdx.x].rhoF = {sfx[0], sfy[0], sfz[0]};
    blockSums[blockIdx.x].rhoRxF = {slx[0], sly[0], slz[0]};
  }
}

// ---- fused elementwise apply: f -= T + omega x r, geometry-masked ---------

__global__ void k_subtractRigidModes(CuField f, double3 com, double3 T,
                                     double3 omega) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!f.cellInGrid(idx))
    return;
  if (!f.cellInGeometry(idx))
    return;

  double3 r = subD(cellPositionDeviceD(f.system, idx), com);
  double3 correction = {T.x + (omega.y * r.z - omega.z * r.y),
                        T.y + (omega.z * r.x - omega.x * r.z),
                        T.z + (omega.x * r.y - omega.y * r.x)};

  real3 v = f.vectorAt(idx);
  real3 result = {real(double(v.x) - correction.x),
                  real(double(v.y) - correction.y),
                  real(double(v.z) - correction.z)};
  f.setVectorInCell(idx, result);
}

}  // namespace ends

RigidBodyGeometry computeRigidBodyGeometry(const Magnet* magnet) {
  std::shared_ptr<const System> system = magnet->system();
  int ncells = system->grid().ncells();
  CuSystem cusys = system->cu();
  CuParameter rho = magnet->rho.cu();

  ComResult comResult = computeCom(cusys, rho, ncells);

  int numBlocks = numReductionBlocks(ncells);
  GpuBuffer<InertiaPartial> d_partials(numBlocks);
  k_inertiaPartialSums<<<numBlocks, BLOCKDIM, 0, getCudaStream()>>>(
      d_partials.get(), cusys, rho, comResult.com);
  checkCudaError(cudaPeekAtLastError());

  std::vector<InertiaPartial> partials(numBlocks);
  checkCudaError(cudaMemcpyAsync(partials.data(), d_partials.get(),
                                 numBlocks * sizeof(InertiaPartial),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));

  double Ixx = 0, Iyy = 0, Izz = 0, Ixy = 0, Ixz = 0, Iyz = 0;
  for (const InertiaPartial& p : partials) {
    Ixx += p.diag.x; Iyy += p.diag.y; Izz += p.diag.z;
    Ixy += p.offdiag.x; Ixz += p.offdiag.y; Iyz += p.offdiag.z;
  }
  double I[3][3] = {{Ixx, Ixy, Ixz}, {Ixy, Iyy, Iyz}, {Ixz, Iyz, Izz}};

  double trace = Ixx + Iyy + Izz;
  double lambda = 1e-10 * trace;
  for (int d = 0; d < 3; d++) I[d][d] += lambda;

  RigidBodyGeometry geom;
  geom.com = comResult.com;
  geom.totalRho = comResult.totalRho;
  invert3x3(I, geom.Iinv);
  return geom;
}

RigidModeMoments computeRigidModeMoments(const Field& f, const RigidBodyGeometry& geom,
                                         const Magnet* magnet) {
  CuParameter rho = magnet->rho.cu();
  int ncells = f.system()->grid().ncells();
  int numBlocks = numReductionBlocks(ncells);

  GpuBuffer<TransRotPartial> d_partials(numBlocks);
  k_transRotPartialSums<<<numBlocks, BLOCKDIM, 0, getCudaStream()>>>(
      d_partials.get(), f.cu(), rho, geom.com);
  checkCudaError(cudaPeekAtLastError());

  std::vector<TransRotPartial> partials(numBlocks);
  checkCudaError(cudaMemcpyAsync(partials.data(), d_partials.get(),
                                 numBlocks * sizeof(TransRotPartial),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));

  double3 rhoF{0, 0, 0}, rhoRxF{0, 0, 0};
  for (const TransRotPartial& p : partials) {
    rhoF.x += p.rhoF.x; rhoF.y += p.rhoF.y; rhoF.z += p.rhoF.z;
    rhoRxF.x += p.rhoRxF.x; rhoRxF.y += p.rhoRxF.y; rhoRxF.z += p.rhoRxF.z;
  }

  RigidModeMoments m;
  m.T = {rhoF.x / geom.totalRho, rhoF.y / geom.totalRho, rhoF.z / geom.totalRho};
  double3 L = rhoRxF;
  m.omega.x = geom.Iinv[0][0]*L.x + geom.Iinv[0][1]*L.y + geom.Iinv[0][2]*L.z;
  m.omega.y = geom.Iinv[1][0]*L.x + geom.Iinv[1][1]*L.y + geom.Iinv[1][2]*L.z;
  m.omega.z = geom.Iinv[2][0]*L.x + geom.Iinv[2][1]*L.y + geom.Iinv[2][2]*L.z;
  return m;
}

void removeRigidBodyModes(Field& f, const RigidBodyGeometry& geom, const Magnet* magnet) {
  RigidModeMoments m = computeRigidModeMoments(f, geom, magnet);
  int ncells = f.system()->grid().ncells();
  cudaLaunch(ncells, k_subtractRigidModes, f.cu(), geom.com, m.T, m.omega);
}