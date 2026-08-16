#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "cudaerror.hpp"
#include "cudalaunch.hpp"
#include "cudastream.hpp"
#include "field.hpp"
#include "gpubuffer.hpp"
#include "magnet.hpp"
#include "parameter.hpp"
#include "rigidbodymodes3.hpp"
#include "system.hpp"

namespace {

// Deliberately duplicated from rigidbodymodes.cu / rigidbodymodes2.cu rather
// than shared: this file is meant to be a self-contained, independently
// comparable implementation. Anonymous-namespace internal linkage means the
// duplicate symbol names don't collide across translation units.

__device__ inline double3 cellPositionDeviceD3(const CuSystem& system, int idx) {
  int3 coord = system.grid.index2coord(idx);
  return double3{double(coord.x) * double(system.cellsize.x),
                 double(coord.y) * double(system.cellsize.y),
                 double(coord.z) * double(system.cellsize.z)};
}

__host__ __device__ inline double3 subD3(double3 a, double3 b) {
  return double3{a.x - b.x, a.y - b.y, a.z - b.z};
}

// ---- generic single-block reduction into shared memory --------------------
// This file's per-call reductions are small enough (one pass over the
// geometry, done rarely -- not a per-minimizer-step hot path) that the
// single-block style used by rigidbodymodes2.cu / reduce.cu is preferred
// over rigidbodymodes.cu's multi-block + host-merge pattern: simpler, and
// there's no performance reason to reach for the more complex version here.

template <int NFIELDS>
__device__ inline void blockReduceSum3(double sdata[NFIELDS][BLOCKDIM], int tid) {
  for (unsigned int s = BLOCKDIM / 2; s > 0; s >>= 1) {
    if (tid < s) {
      #pragma unroll
      for (int f = 0; f < NFIELDS; f++)
        sdata[f][tid] += sdata[f][tid + s];
    }
    __syncthreads();
  }
}

// ---- pass 1: reference mass-weighted center of mass + total mass ----------

__global__ void k_com3(double4* result, CuSystem system, CuParameter rho) {
  __shared__ double sdata[4][BLOCKDIM];
  int ncells = system.grid.ncells();
  int tid = threadIdx.x;

  double acc[4] = {0, 0, 0, 0};
  for (int i = tid; i < ncells; i += BLOCKDIM) {
    if (!system.inGeometry(i))
      continue;
    double w = rho.valueAt(i);
    double3 p = cellPositionDeviceD3(system, i);
    acc[0] += w * p.x; acc[1] += w * p.y; acc[2] += w * p.z; acc[3] += w;
  }
  #pragma unroll
  for (int f = 0; f < 4; f++)
    sdata[f][tid] = acc[f];
  __syncthreads();

  blockReduceSum3<4>(sdata, tid);

  if (tid == 0)
    *result = {sdata[0][0], sdata[1][0], sdata[2][0], sdata[3][0]};
}

// ---- pass 2: reference shape covariance S0 = Sum w*(r0-com0)(r0-com0)^T ---
// Symmetric, so only 6 unique entries are reduced; expanded to a full 3x3
// on the host afterward (same pattern as the inertia tensor in
// rigidbodymodes.cu).

__global__ void k_s0PartialSum(double* result6, CuSystem system, CuParameter rho,
                               double3 com0) {
  __shared__ double sdata[6][BLOCKDIM];  // xx yy zz xy xz yz
  int ncells = system.grid.ncells();
  int tid = threadIdx.x;

  double acc[6] = {0, 0, 0, 0, 0, 0};
  for (int i = tid; i < ncells; i += BLOCKDIM) {
    if (!system.inGeometry(i))
      continue;
    double w = rho.valueAt(i);
    double3 p = subD3(cellPositionDeviceD3(system, i), com0);
    acc[0] += w * p.x * p.x;
    acc[1] += w * p.y * p.y;
    acc[2] += w * p.z * p.z;
    acc[3] += w * p.x * p.y;
    acc[4] += w * p.x * p.z;
    acc[5] += w * p.y * p.z;
  }
  #pragma unroll
  for (int f = 0; f < 6; f++)
    sdata[f][tid] = acc[f];
  __syncthreads();

  blockReduceSum3<6>(sdata, tid);

  if (tid == 0)
    for (int f = 0; f < 6; f++)
      result6[f] = sdata[f][0];
}

// ---- per-call pass: C = Sum w*(r0-com0)*u^T  and  Sum w*u ------------------
// C is NOT symmetric (outer product of two different vectors), so all 9
// entries are needed. Fused with the u-sum (needed for comX) into one pass
// over the field, since both need the same per-cell data.
//
// Note: this does NOT need uMean subtracted out of u first. See the
// derivation in the surrounding chat / file header comment: the uMean
// cross-term in the full covariance Sum w*(r0-com0)(u-uMean)^T vanishes
// identically because Sum w*(r0-com0) = 0 by construction of com0. So
// H = S0 + C, with C computed directly from raw u -- no second pass needed.

__global__ void k_kabschPartialSum(double* result12, CuSystem system, CuField u,
                                   CuParameter rho, double3 com0) {
  __shared__ double sdata[12][BLOCKDIM];  // C[0..8] row-major, then uSum[9..11]
  int ncells = system.grid.ncells();
  int tid = threadIdx.x;

  double acc[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = tid; i < ncells; i += BLOCKDIM) {
    if (!u.cellInGeometry(i))
      continue;
    double w = rho.valueAt(i);
    double3 p = subD3(cellPositionDeviceD3(system, i), com0);
    real3 uf = u.vectorAt(i);
    double3 uv = double3{double(uf.x), double(uf.y), double(uf.z)};

    // C[a][b] += w * p[a] * uv[b], row-major flattened: C[a][b] -> acc[3a+b]
    acc[0] += w * p.x * uv.x; acc[1] += w * p.x * uv.y; acc[2] += w * p.x * uv.z;
    acc[3] += w * p.y * uv.x; acc[4] += w * p.y * uv.y; acc[5] += w * p.y * uv.z;
    acc[6] += w * p.z * uv.x; acc[7] += w * p.z * uv.y; acc[8] += w * p.z * uv.z;

    acc[9]  += w * uv.x;
    acc[10] += w * uv.y;
    acc[11] += w * uv.z;
  }
  #pragma unroll
  for (int f = 0; f < 12; f++)
    sdata[f][tid] = acc[f];
  __syncthreads();

  blockReduceSum3<12>(sdata, tid);

  if (tid == 0)
    for (int f = 0; f < 12; f++)
      result12[f] = sdata[f][0];
}

// ---- host-side 3x3 linear algebra ------------------------------------------

inline double det3(const double M[3][3]) {
  return M[0][0]*(M[1][1]*M[2][2] - M[1][2]*M[2][1])
       - M[0][1]*(M[1][0]*M[2][2] - M[1][2]*M[2][0])
       + M[0][2]*(M[1][0]*M[2][1] - M[1][1]*M[2][0]);
}

inline double sgn(double x) { return x >= 0.0 ? 1.0 : -1.0; }

// Classic cyclic Jacobi eigenvalue algorithm for a small symmetric matrix.
// Overwrites eigval with the (unsorted) eigenvalues and V with the
// corresponding eigenvectors as columns. Chosen over a closed-form 3x3
// eigensolver for simplicity/robustness -- this runs once per call on the
// host on a 3x3 matrix, so cost is negligible regardless of method.
void jacobiEigenSymmetric3x3(const double A[3][3], double V[3][3], double eigval[3]) {
  double a[3][3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      a[i][j] = A[i][j];
      V[i][j] = (i == j) ? 1.0 : 0.0;
    }

  const int pq[3][2] = {{0, 1}, {0, 2}, {1, 2}};
  const int maxSweeps = 50;
  for (int sweep = 0; sweep < maxSweeps; sweep++) {
    double off = std::abs(a[0][1]) + std::abs(a[0][2]) + std::abs(a[1][2]);
    if (off < 1e-30)
      break;

    for (int pair = 0; pair < 3; pair++) {
      int p = pq[pair][0], q = pq[pair][1];
      if (std::abs(a[p][q]) < 1e-300)
        continue;

      double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
      double t = (theta == 0.0) ? 1.0
                                : sgn(theta) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
      double c = 1.0 / std::sqrt(t * t + 1.0);
      double s = t * c;

      double app = a[p][p], aqq = a[q][q], apq = a[p][q];
      a[p][p] = app - t * apq;
      a[q][q] = aqq + t * apq;
      a[p][q] = a[q][p] = 0.0;

      for (int k = 0; k < 3; k++) {
        if (k == p || k == q) continue;
        double akp = a[k][p], akq = a[k][q];
        a[k][p] = a[p][k] = c * akp - s * akq;
        a[k][q] = a[q][k] = s * akp + c * akq;
      }
      for (int k = 0; k < 3; k++) {
        double vkp = V[k][p], vkq = V[k][q];
        V[k][p] = c * vkp - s * vkq;
        V[k][q] = s * vkp + c * vkq;
      }
    }
  }

  for (int i = 0; i < 3; i++)
    eigval[i] = a[i][i];
}

// Sorts eigval descending, permuting the corresponding columns of V to match.
void sortEigenDescending(double eigval[3], double V[3][3]) {
  int order[3] = {0, 1, 2};
  std::sort(order, order + 3, [&](int i, int j) { return eigval[i] > eigval[j]; });

  double sortedVal[3];
  double sortedV[3][3];
  for (int i = 0; i < 3; i++) {
    sortedVal[i] = eigval[order[i]];
    for (int k = 0; k < 3; k++)
      sortedV[k][i] = V[k][order[i]];
  }
  for (int i = 0; i < 3; i++) {
    eigval[i] = sortedVal[i];
    for (int k = 0; k < 3; k++)
      V[k][i] = sortedV[k][i];
  }
}

// 3x3 SVD via eigendecomposition of H^T H, followed by the Kabsch rotation
// formula. H = U * diag(sigma) * V^T (standard SVD convention); Kabsch's
// R = V * diag(1,1,d) * U^T with d = sign(det(V)*det(U)) is the closest
// proper rotation (det = +1, no reflection) aligning the reference-centered
// points onto the current-centered points.
void kabschRotationFromH(const double H[3][3], double R[3][3]) {
  double HtH[3][3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      double sum = 0.0;
      for (int k = 0; k < 3; k++)
        sum += H[k][i] * H[k][j];  // (H^T H)_ij = Sum_k H_ki H_kj
      HtH[i][j] = sum;
    }

  double V[3][3], eigval[3];
  jacobiEigenSymmetric3x3(HtH, V, eigval);
  sortEigenDescending(eigval, V);

  const double sigmaEpsilon = 1e-12;  // TODO: tune, or scale relative to system size
  double sigma[3], U[3][3];
  for (int i = 0; i < 3; i++) {
    sigma[i] = std::sqrt(std::max(eigval[i], 0.0));
    if (sigma[i] > sigmaEpsilon) {
      for (int k = 0; k < 3; k++) {
        double Hv_k = H[k][0] * V[0][i] + H[k][1] * V[1][i] + H[k][2] * V[2][i];
        U[k][i] = Hv_k / sigma[i];
      }
    } else {
      // Degenerate direction (e.g. no displacement signal along this axis --
      // could happen for a purely 2D/planar deformation, or zero
      // displacement). Fall back to identity for this column, corrected
      // below via Gram-Schmidt against whichever columns ARE well-defined,
      // rather than risking an ill-conditioned division.
      for (int k = 0; k < 3; k++)
        U[k][i] = (k == i) ? 1.0 : 0.0;
    }
  }
  // Re-orthonormalize U (Gram-Schmidt) in case any column used the fallback
  // above and isn't already orthogonal to the well-defined columns.
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < i; j++) {
      double dot = U[0][i]*U[0][j] + U[1][i]*U[1][j] + U[2][i]*U[2][j];
      for (int k = 0; k < 3; k++)
        U[k][i] -= dot * U[k][j];
    }
    double norm = std::sqrt(U[0][i]*U[0][i] + U[1][i]*U[1][i] + U[2][i]*U[2][i]);
    if (norm > sigmaEpsilon)
      for (int k = 0; k < 3; k++)
        U[k][i] /= norm;
  }

  double d = sgn(det3(V) * det3(U));
  double D[3] = {1.0, 1.0, d};

  for (int a = 0; a < 3; a++)
    for (int b = 0; b < 3; b++) {
      double sum = 0.0;
      for (int i = 0; i < 3; i++)
        sum += V[a][i] * D[i] * U[b][i];
      R[a][b] = sum;
    }
}

__host__ __device__ inline double3 matVec3(const double R[3][3], double3 v) {
  return double3{R[0][0]*v.x + R[0][1]*v.y + R[0][2]*v.z,
                 R[1][0]*v.x + R[1][1]*v.y + R[1][2]*v.z,
                 R[2][0]*v.x + R[2][1]*v.y + R[2][2]*v.z};
}

// ---- fused elementwise apply: u -= (R*r0 + T - r0), geometry-masked -------

struct Mat3 { double m[3][3]; };

__global__ void k_subtractKabschMode(CuField f, double3 com0Unused, Mat3 R, double3 T) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!f.cellInGrid(idx))
    return;
  if (!f.cellInGeometry(idx))
    return;

  double3 r0 = cellPositionDeviceD3(f.system, idx);
  double3 Rr0 = matVec3(R.m, r0);
  double3 correction = {Rr0.x + T.x - r0.x,
                        Rr0.y + T.y - r0.y,
                        Rr0.z + T.z - r0.z};

  real3 v = f.vectorAt(idx);
  real3 result = {real(double(v.x) - correction.x),
                  real(double(v.y) - correction.y),
                  real(double(v.z) - correction.z)};
  f.setVectorInCell(idx, result);
}

}  // namespace

RigidBodyGeometry3 computeRigidBodyGeometry3(const Magnet* magnet) {
  std::shared_ptr<const System> system = magnet->system();
  CuSystem cusys = system->cu();
  CuParameter rho = magnet->rho.cu();

  GpuBuffer<double4> d_com(1);
  cudaLaunchReductionKernel(k_com3, d_com.get(), cusys, rho);
  double4 comSum;
  checkCudaError(cudaMemcpyAsync(&comSum, d_com.get(), sizeof(double4),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));

  if (comSum.w <= 0.0)
    throw std::runtime_error(
        "computeRigidBodyGeometry3: total mass (sum of rho) in geometry is zero or negative.");

  RigidBodyGeometry3 geom;
  geom.com0 = double3{comSum.x / comSum.w, comSum.y / comSum.w, comSum.z / comSum.w};
  geom.totalRho = comSum.w;

  GpuBuffer<double> d_s0(6);
  cudaLaunchReductionKernel(k_s0PartialSum, d_s0.get(), cusys, rho, geom.com0);
  double s0Flat[6];
  checkCudaError(cudaMemcpyAsync(s0Flat, d_s0.get(), 6 * sizeof(double),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));

  double xx = s0Flat[0], yy = s0Flat[1], zz = s0Flat[2];
  double xy = s0Flat[3], xz = s0Flat[4], yz = s0Flat[5];
  geom.S0[0][0] = xx; geom.S0[0][1] = xy; geom.S0[0][2] = xz;
  geom.S0[1][0] = xy; geom.S0[1][1] = yy; geom.S0[1][2] = yz;
  geom.S0[2][0] = xz; geom.S0[2][1] = yz; geom.S0[2][2] = zz;

  return geom;
}

KabschResult computeKabschAlignment(const Field& u, const RigidBodyGeometry3& geom,
                                    const Magnet* magnet) {
  CuParameter rho = magnet->rho.cu();
  CuSystem cusys = magnet->system()->cu();

  GpuBuffer<double> d_result(12);
  cudaLaunchReductionKernel(k_kabschPartialSum, d_result.get(), cusys, u.cu(), rho, geom.com0);
  double r12[12];
  checkCudaError(cudaMemcpyAsync(r12, d_result.get(), 12 * sizeof(double),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));

  double C[3][3] = {{r12[0], r12[1], r12[2]},
                    {r12[3], r12[4], r12[5]},
                    {r12[6], r12[7], r12[8]}};
  double3 uSum = {r12[9], r12[10], r12[11]};

  double H[3][3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      H[i][j] = geom.S0[i][j] + C[i][j];

  KabschResult result;
  kabschRotationFromH(H, result.R);

  double3 uMean = {uSum.x / geom.totalRho, uSum.y / geom.totalRho, uSum.z / geom.totalRho};
  result.com = double3{geom.com0.x + uMean.x, geom.com0.y + uMean.y, geom.com0.z + uMean.z};

  double3 Rcom0 = matVec3(result.R, geom.com0);
  result.T = double3{result.com.x - Rcom0.x, result.com.y - Rcom0.y, result.com.z - Rcom0.z};

  return result;
}

void removeRigidBodyModesKabsch(Field& u, const RigidBodyGeometry3& geom, const Magnet* magnet) {
  KabschResult result = computeKabschAlignment(u, geom, magnet);
  int ncells = u.system()->grid().ncells();

  Mat3 R;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      R.m[i][j] = result.R[i][j];

  cudaLaunch(ncells, k_subtractKabschMode, u.cu(), geom.com0, R, result.T);
}

double kabschRotationAngle(const KabschResult& result) {
  double trace = result.R[0][0] + result.R[1][1] + result.R[2][2];
  double c = std::max(-1.0, std::min(1.0, (trace - 1.0) / 2.0));
  return std::acos(c);
}
