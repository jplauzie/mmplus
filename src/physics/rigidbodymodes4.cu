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
#include "rigidbodymodes4.hpp"
#include "rigidbodymodes_common.hpp"
#include "system.hpp"

namespace {

// ---- pass 1: reference mass-weighted center of mass + total mass ----------
// com0 is always rho-weighted (see rigidbodymodes4.hpp) -- no unweighted
// variant needed here.

__global__ void k_com4(double4* result, CuSystem system, CuParameter rho, bool unweighted) {
  __shared__ double sdata[4][BLOCKDIM];
  int ncells = system.grid.ncells();
  int tid = threadIdx.x;

  double acc[4] = {0, 0, 0, 0};
  for (int i = tid; i < ncells; i += BLOCKDIM) {
    if (!system.inGeometry(i))
      continue;
    double w = unweighted ? 1.0 : double(rho.valueAt(i));
    double3 p = cellPositionDeviceD(system, i);
    acc[0] += w * p.x; acc[1] += w * p.y; acc[2] += w * p.z; acc[3] += w;
  }
  #pragma unroll
  for (int f = 0; f < 4; f++)
    sdata[f][tid] = acc[f];
  __syncthreads();

  blockReduceSum<4>(sdata, tid);

  if (tid == 0)
    *result = {sdata[0][0], sdata[1][0], sdata[2][0], sdata[3][0]};
}

// ---- pass 2: reference shape covariance S0 = Sum w*(r0-com0)(r0-com0)^T ---
// unweighted: w=1 for every in-geometry cell instead of rho. 7th field is
// the plain in-geometry cell count, accumulated unconditionally so a single
// call (either weighting) yields ncellsInGeometry.

__global__ void k_s04PartialSum(double* result7, CuSystem system, CuParameter rho,
                                double3 com0, bool unweighted) {
  __shared__ double sdata[7][BLOCKDIM];  // xx yy zz xy xz yz count
  int ncells = system.grid.ncells();
  int tid = threadIdx.x;

  double acc[7] = {0, 0, 0, 0, 0, 0, 0};
  for (int i = tid; i < ncells; i += BLOCKDIM) {
    if (!system.inGeometry(i))
      continue;
    double w = unweighted ? 1.0 : double(rho.valueAt(i));
    double3 p = subD3(cellPositionDeviceD(system, i), com0);
    acc[0] += w * p.x * p.x;
    acc[1] += w * p.y * p.y;
    acc[2] += w * p.z * p.z;
    acc[3] += w * p.x * p.y;
    acc[4] += w * p.x * p.z;
    acc[5] += w * p.y * p.z;
    acc[6] += 1.0;
  }
  #pragma unroll
  for (int f = 0; f < 7; f++)
    sdata[f][tid] = acc[f];
  __syncthreads();

  blockReduceSum<7>(sdata, tid);

  if (tid == 0)
    for (int f = 0; f < 7; f++)
      result7[f] = sdata[f][0];
}

// ---- per-call pass: C = Sum w*(r0-com0)*u^T  and  Sum w*u ------------------
// unweighted: w=1 for every in-geometry cell instead of rho.

__global__ void k_quatPartialSum(double* result12, CuSystem system, CuField u,
                                 CuParameter rho, double3 com0, bool unweighted) {
  __shared__ double sdata[12][BLOCKDIM];  // C[0..8] row-major, then uSum[9..11]
  int ncells = system.grid.ncells();
  int tid = threadIdx.x;

  double acc[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = tid; i < ncells; i += BLOCKDIM) {
    if (!u.cellInGeometry(i))
      continue;
    double w = unweighted ? 1.0 : double(rho.valueAt(i));
    double3 p = subD3(cellPositionDeviceD(system, i), com0);
    real3 uf = u.vectorAt(i);
    double3 uv = double3{double(uf.x), double(uf.y), double(uf.z)};

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

  blockReduceSum<12>(sdata, tid);

  if (tid == 0)
    for (int f = 0; f < 12; f++)
      result12[f] = sdata[f][0];
}

// ---- host-side: Horn's N matrix + 4x4 symmetric eigensolver ---------------
// (unchanged -- sgn4, buildHornMatrix, jacobiEigenSymmetric4x4,
// quaternionToMatrix, matVec3_4 all stay exactly as before, no weighting
// dependence at this stage)

inline double sgn4(double x) { return x >= 0.0 ? 1.0 : -1.0; }

void buildHornMatrix(const double H[3][3], double N[4][4]) {
  double Sxx = H[0][0], Sxy = H[0][1], Sxz = H[0][2];
  double Syx = H[1][0], Syy = H[1][1], Syz = H[1][2];
  double Szx = H[2][0], Szy = H[2][1], Szz = H[2][2];

  N[0][0] = Sxx + Syy + Szz;
  N[0][1] = N[1][0] = Syz - Szy;
  N[0][2] = N[2][0] = Szx - Sxz;
  N[0][3] = N[3][0] = Sxy - Syx;

  N[1][1] = Sxx - Syy - Szz;
  N[1][2] = N[2][1] = Sxy + Syx;
  N[1][3] = N[3][1] = Szx + Sxz;

  N[2][2] = -Sxx + Syy - Szz;
  N[2][3] = N[3][2] = Syz + Szy;

  N[3][3] = -Sxx - Syy + Szz;
}

void jacobiEigenSymmetric4x4(const double A[4][4], double V[4][4], double eigval[4]) {
  double a[4][4];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) {
      a[i][j] = A[i][j];
      V[i][j] = (i == j) ? 1.0 : 0.0;
    }

  const int pq[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
  const int maxSweeps = 60;
  for (int sweep = 0; sweep < maxSweeps; sweep++) {
    double off = 0.0;
    for (int k = 0; k < 6; k++)
      off += std::abs(a[pq[k][0]][pq[k][1]]);
    if (off < 1e-30)
      break;

    for (int pair = 0; pair < 6; pair++) {
      int p = pq[pair][0], q = pq[pair][1];
      if (std::abs(a[p][q]) < 1e-300)
        continue;

      double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
      double t = (theta == 0.0) ? 1.0
                                : sgn4(theta) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
      double c = 1.0 / std::sqrt(t * t + 1.0);
      double s = t * c;

      double app = a[p][p], aqq = a[q][q], apq = a[p][q];
      a[p][p] = app - t * apq;
      a[q][q] = aqq + t * apq;
      a[p][q] = a[q][p] = 0.0;

      for (int k = 0; k < 4; k++) {
        if (k == p || k == q) continue;
        double akp = a[k][p], akq = a[k][q];
        a[k][p] = a[p][k] = c * akp - s * akq;
        a[k][q] = a[q][k] = s * akp + c * akq;
      }
      for (int k = 0; k < 4; k++) {
        double vkp = V[k][p], vkq = V[k][q];
        V[k][p] = c * vkp - s * vkq;
        V[k][q] = s * vkp + c * vkq;
      }
    }
  }

  for (int i = 0; i < 4; i++)
    eigval[i] = a[i][i];
}

void quaternionToMatrix(const double q[4], double R[3][3]) {
  double q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];
  R[0][0] = 1 - 2*(q2*q2 + q3*q3); R[0][1] = 2*(q1*q2 - q0*q3);     R[0][2] = 2*(q1*q3 + q0*q2);
  R[1][0] = 2*(q1*q2 + q0*q3);     R[1][1] = 1 - 2*(q1*q1 + q3*q3); R[1][2] = 2*(q2*q3 - q0*q1);
  R[2][0] = 2*(q1*q3 - q0*q2);     R[2][1] = 2*(q2*q3 + q0*q1);     R[2][2] = 1 - 2*(q1*q1 + q2*q2);
}

__host__ __device__ inline double3 matVec3_4(const double R[3][3], double3 v) {
  return double3{R[0][0]*v.x + R[0][1]*v.y + R[0][2]*v.z,
                 R[1][0]*v.x + R[1][1]*v.y + R[1][2]*v.z,
                 R[2][0]*v.x + R[2][1]*v.y + R[2][2]*v.z};
}

// ---- fused elementwise apply: u -= (R*r0 + T - r0), geometry-masked -------

struct Mat3_4 { double m[3][3]; };

__global__ void k_subtractQuatMode(CuField f, Mat3_4 R, double3 T) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!f.cellInGrid(idx))
    return;
  if (!f.cellInGeometry(idx))
    return;

  double3 r0 = cellPositionDeviceD(f.system, idx);
  double3 Rr0 = matVec3_4(R.m, r0);
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

RigidBodyGeometry4 computeRigidBodyGeometry4(const Magnet* magnet) {
  std::shared_ptr<const System> system = magnet->system();
  CuSystem cusys = system->cu();
  CuParameter rho = magnet->rho.cu();

  GpuBuffer<double4> d_com(1);
  cudaLaunchReductionKernel(k_com4, d_com.get(), cusys, rho, /*unweighted=*/false);
  double4 comSum;
  checkCudaError(cudaMemcpyAsync(&comSum, d_com.get(), sizeof(double4),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));
  if (comSum.w <= 0.0)
    throw std::runtime_error("computeRigidBodyGeometry4: total mass (sum of rho) in geometry is zero or negative.");

  GpuBuffer<double4> d_comU(1);
  cudaLaunchReductionKernel(k_com4, d_comU.get(), cusys, rho, /*unweighted=*/true);
  double4 comSumU;
  checkCudaError(cudaMemcpyAsync(&comSumU, d_comU.get(), sizeof(double4),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));
  if (comSumU.w <= 0.0)
    throw std::runtime_error("computeRigidBodyGeometry4: zero cells in geometry.");

  RigidBodyGeometry4 geom;
  geom.com0 = double3{comSum.x / comSum.w, comSum.y / comSum.w, comSum.z / comSum.w};
  geom.totalRho = comSum.w;
  geom.com0Unweighted = double3{comSumU.x / comSumU.w, comSumU.y / comSumU.w, comSumU.z / comSumU.w};
  geom.ncellsInGeometry = comSumU.w;

  GpuBuffer<double> d_s0(6);
  cudaLaunchReductionKernel(k_s04PartialSum, d_s0.get(), cusys, rho, geom.com0, /*unweighted=*/false);
  double s0Flat[6];
  checkCudaError(cudaMemcpyAsync(s0Flat, d_s0.get(), 6 * sizeof(double),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));

  GpuBuffer<double> d_s0u(6);
  cudaLaunchReductionKernel(k_s04PartialSum, d_s0u.get(), cusys, rho, geom.com0Unweighted, /*unweighted=*/true);
  double s0uFlat[6];
  checkCudaError(cudaMemcpyAsync(s0uFlat, d_s0u.get(), 6 * sizeof(double),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));

  geom.S0[0][0] = s0Flat[0]; geom.S0[0][1] = s0Flat[3]; geom.S0[0][2] = s0Flat[4];
  geom.S0[1][0] = s0Flat[3]; geom.S0[1][1] = s0Flat[1]; geom.S0[1][2] = s0Flat[5];
  geom.S0[2][0] = s0Flat[4]; geom.S0[2][1] = s0Flat[5]; geom.S0[2][2] = s0Flat[2];

  geom.S0Unweighted[0][0] = s0uFlat[0]; geom.S0Unweighted[0][1] = s0uFlat[3]; geom.S0Unweighted[0][2] = s0uFlat[4];
  geom.S0Unweighted[1][0] = s0uFlat[3]; geom.S0Unweighted[1][1] = s0uFlat[1]; geom.S0Unweighted[1][2] = s0uFlat[5];
  geom.S0Unweighted[2][0] = s0uFlat[4]; geom.S0Unweighted[2][1] = s0uFlat[5]; geom.S0Unweighted[2][2] = s0uFlat[2];

  return geom;
}

QuatAlignResult computeQuaternionAlignment(const Field& u, const RigidBodyGeometry4& geom,
                                           const Magnet* magnet, bool unweighted) {
  CuParameter rho = magnet->rho.cu();
  CuSystem cusys = magnet->system()->cu();
  double3 com0 = unweighted ? geom.com0Unweighted : geom.com0;

  GpuBuffer<double> d_result(12);
  cudaLaunchReductionKernel(k_quatPartialSum, d_result.get(), cusys, u.cu(), rho, com0, unweighted);
  double r12[12];
  checkCudaError(cudaMemcpyAsync(r12, d_result.get(), 12 * sizeof(double),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));

  double C[3][3] = {{r12[0], r12[1], r12[2]},
                    {r12[3], r12[4], r12[5]},
                    {r12[6], r12[7], r12[8]}};
  double3 uSum = {r12[9], r12[10], r12[11]};

  const double (*S0ToUse)[3] = unweighted ? geom.S0Unweighted : geom.S0;
  double denom = unweighted ? geom.ncellsInGeometry : geom.totalRho;

  double H[3][3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      H[i][j] = S0ToUse[i][j] + C[i][j];

  double N[4][4];
  buildHornMatrix(H, N);

  double V[4][4], eigval[4];
  jacobiEigenSymmetric4x4(N, V, eigval);

  // Find the largest eigenvalue; its eigenvector is the optimal quaternion.
  int best = 0;
  for (int i = 1; i < 4; i++)
    if (eigval[i] > eigval[best])
      best = i;
  int second = -1;
  for (int i = 0; i < 4; i++)
    if (i != best && (second < 0 || eigval[i] > eigval[second]))
      second = i;

  QuatAlignResult result;
  for (int k = 0; k < 4; k++)
    result.q[k] = V[k][best];
  result.eigenGap = eigval[best] - eigval[second];

  quaternionToMatrix(result.q, result.R);

  double3 uMean = {uSum.x / denom, uSum.y / denom, uSum.z / denom};
  result.com = double3{com0.x + uMean.x, com0.y + uMean.y, com0.z + uMean.z};

  double3 Rcom0 = matVec3_4(result.R, com0);
  result.T = double3{result.com.x - Rcom0.x, result.com.y - Rcom0.y, result.com.z - Rcom0.z};

  return result;
}

void removeRigidBodyModesQuaternion(Field& u, const RigidBodyGeometry4& geom,
                                    const Magnet* magnet, bool unweighted) {
  QuatAlignResult result = computeQuaternionAlignment(u, geom, magnet, unweighted);
  int ncells = u.system()->grid().ncells();

  Mat3_4 R;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      R.m[i][j] = result.R[i][j];

  cudaLaunch(ncells, k_subtractQuatMode, u.cu(), R, result.T);
}

double quaternionRotationAngle(const QuatAlignResult& result) {
  double q0clamped = std::max(-1.0, std::min(1.0, std::abs(result.q[0])));
  return 2.0 * std::acos(q0clamped);
}