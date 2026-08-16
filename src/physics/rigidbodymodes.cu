#include <numeric>
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

// Double-precision cell position. cellsize is f32 (CuSystem::cellsize is real3), avoids compounding additional rounding in
// everything downstream (com subtraction, cross products, sums). maybe unnecessary?
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

__host__ __device__ inline double4 operator+(double4 a, double4 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

// ---- shared block-reduction helper (all pass kernels use this) -----------

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

// ---- generic device-reduce-then-gather-on-host --------------------------

template <typename T, typename... Args>
T reduceOnDevice(int numBlocks, void (*kernel)(T*, Args...), Args... args) {
  GpuBuffer<T> d_partials(numBlocks);
  kernel<<<numBlocks, BLOCKDIM, 0, getCudaStream()>>>(d_partials.get(), args...);
  checkCudaError(cudaPeekAtLastError());

  std::vector<T> partials(numBlocks);
  checkCudaError(cudaMemcpyAsync(partials.data(), d_partials.get(),
                                 numBlocks * sizeof(T),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));

  T sum{};
  for (const T& p : partials)
    sum = sum + p;
  return sum;
}

// pass 1: center of mass (COM), total-mass sum 
__global__ void k_comPartialSums(double4* blockSums, CuSystem system,
                                 CuParameter rho) {
  __shared__ double sdata[4][BLOCKDIM];
  int ncells = system.grid.ncells();
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  double accumulator[4] = {0, 0, 0, 0};
  for (int i = gid; i < ncells; i += stride) {
    if (!system.inGeometry(i))
      continue;
    double w = rho.valueAt(i);
    double3 r = cellPositionDeviceD(system, i);
    accumulator[0] += w * r.x;
    accumulator[1] += w * r.y;
    accumulator[2] += w * r.z;
    accumulator[3] += w;
  }
  #pragma unroll
  for (int f = 0; f < 4; f++)
    sdata[f][tid] = accumulator[f];
  __syncthreads();

  blockReduceSum<4>(sdata, tid);

  if (tid == 0)
    blockSums[blockIdx.x] = {sdata[0][0], sdata[1][0], sdata[2][0], sdata[3][0]};
}

struct ComResult { double3 com; double totalRho; };

ComResult computeCom(const CuSystem& cusys, const CuParameter& rho, int ncells) {
  int numBlocks = numReductionBlocks(ncells);
  double4 s = reduceOnDevice<double4>(numBlocks, k_comPartialSums, cusys, rho);

  if (s.w <= 0.0)
    throw std::runtime_error("computeRigidBodyGeometry: total mass (sum of rho) in geometry is zero or negative.");

  return ComResult{double3{s.x / s.w, s.y / s.w, s.z / s.w}, s.w};
}

// pass 2: inertia tensor about COM

struct InertiaPartial { double3 diag; double3 offdiag; };
__host__ __device__ inline InertiaPartial operator+(InertiaPartial a, InertiaPartial b) {
  return {addD3(a.diag, b.diag), addD3(a.offdiag, b.offdiag)};
}

__global__ void k_inertiaPartialSums(InertiaPartial* blockSums,
                                     CuSystem system, CuParameter rho,
                                     double3 com) {
  __shared__ double sdata[6][BLOCKDIM];  // xx yy zz xy xz yz
  int ncells = system.grid.ncells();
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  double accumulator[6] = {0, 0, 0, 0, 0, 0};
  for (int i = gid; i < ncells; i += stride) {
    if (!system.inGeometry(i))
      continue;
    double w = rho.valueAt(i);
    double3 r = subD3(cellPositionDeviceD(system, i), com);
    double x = r.x, y = r.y, z = r.z;
    accumulator[0] += w * (y * y + z * z);
    accumulator[1] += w * (x * x + z * z);
    accumulator[2] += w * (x * x + y * y);
    accumulator[3] += w * (-x * y);
    accumulator[4] += w * (-x * z);
    accumulator[5] += w * (-y * z);
  }
  #pragma unroll
  for (int f = 0; f < 6; f++)
    sdata[f][tid] = accumulator[f];
  __syncthreads();

  blockReduceSum<6>(sdata, tid);

  if (tid == 0) {
    blockSums[blockIdx.x].diag = {sdata[0][0], sdata[1][0], sdata[2][0]};
    blockSums[blockIdx.x].offdiag = {sdata[3][0], sdata[4][0], sdata[5][0]};
  }
}

void invert3x3(const double I[3][3], double out[3][3]) {
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

// translation + angular rotation, fused, mass-weighted

struct TransRotPartial { double3 rhoF; double3 rhoRxF; };
__host__ __device__ inline TransRotPartial operator+(TransRotPartial a, TransRotPartial b) {
  return {addD3(a.rhoF, b.rhoF), addD3(a.rhoRxF, b.rhoRxF)};
}

__global__ void k_transRotPartialSums(TransRotPartial* blockSums, CuField f,
                                      CuParameter rho, double3 com) {
  __shared__ double sdata[6][BLOCKDIM];  // fx fy fz lx ly lz
  int ncells = f.system.grid.ncells();
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // Sum (w * v_i) and (w * r x v_i) 
  double accumulator[6] = {0, 0, 0, 0, 0, 0};
  for (int i = gid; i < ncells; i += stride) {
    if (!f.cellInGeometry(i))
      continue;
    double density = rho.valueAt(i);
    real3 f_real = f.vectorAt(i);
    double3 f_double = double3{double(f_real.x), double(f_real.y), double(f_real.z)};
    accumulator[0] += density * f_double.x; accumulator[1] += density * f_double.y; accumulator[2] += density * f_double.z;

    //position relative to COM
    double3 r = subD3(cellPositionDeviceD(f.system, i), com);
    // angular momentum contribution: r x f
    double3 rxf = crossD3(r, f_double);
    accumulator[3] += density * rxf.x; accumulator[4] += density * rxf.y; accumulator[5] += density * rxf.z;
  }
  #pragma unroll
  for (int f_ = 0; f_ < 6; f_++)
    sdata[f_][tid] = accumulator[f_];
  __syncthreads();

  blockReduceSum<6>(sdata, tid);

  if (tid == 0) {
    blockSums[blockIdx.x].rhoF = {sdata[0][0], sdata[1][0], sdata[2][0]};
    blockSums[blockIdx.x].rhoRxF = {sdata[3][0], sdata[4][0], sdata[5][0]};
  }
}

// ---- fused elementwise apply: u -= T + θ x r, geometry-masked ---------
// symbols are for displacement, but reused for force, velocity, etc
// v-= v_net_trans + ω x r , f-= f_net_trans+ α x r

__global__ void k_subtractRigidModes(CuField f, double3 com, double3 T,
                                     double3 omega) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!f.cellInGrid(idx) || !f.cellInGeometry(idx))
    return;
  

  double3 r = subD3(cellPositionDeviceD(f.system, idx), com);
  double3 correction = {T.x + (omega.y * r.z - omega.z * r.y),
                        T.y + (omega.z * r.x - omega.x * r.z),
                        T.z + (omega.x * r.y - omega.y * r.x)};

  real3 f_real = f.vectorAt(idx);
  real3 result = {real(double(f_real.x) - correction.x),
                  real(double(f_real.y) - correction.y),
                  real(double(f_real.z) - correction.z)};
  f.setVectorInCell(idx, result);
}

}  // namespace


// compute center of mass (COM) COM= (Σrho_i*r_i)/Σrho_i (factor of V canceled off) and moment of Inertia tensor (I) about COM
// I = Σrho_i*(|r_comi|^2*I - r_comi*r_comi^T). r_comi=r_i - COM . (factor of V also cancels off with L in L=I*ω or ω=I^-1*L)
// L= Σrho_i*(r_comi x f_i) . COM, I computed once and reused
RigidBodyGeometry computeRigidBodyGeometry(const Magnet* magnet) {
  std::shared_ptr<const System> system = magnet->system();
  int ncells = system->grid().ncells();
  CuSystem cusys = system->cu();
  CuParameter rho = magnet->rho.cu();

  ComResult COM_result = computeCom(cusys, rho, ncells);

  int numBlocks = numReductionBlocks(ncells);
  InertiaPartial inertiaP = reduceOnDevice<InertiaPartial>(numBlocks, k_inertiaPartialSums, cusys, rho, COM_result.com);

  double Ixx = inertiaP.diag.x, Iyy = inertiaP.diag.y, Izz = inertiaP.diag.z;
  // I is symmetric, Ixy=Iyx, Ixz=Izx, Iyz=Izy
  double Ixy = inertiaP.offdiag.x, Ixz = inertiaP.offdiag.y, Iyz = inertiaP.offdiag.z;
  double I[3][3] = {{Ixx, Ixy, Ixz}, {Ixy, Iyy, Iyz}, {Ixz, Iyz, Izz}};

  double trace = Ixx + Iyy + Izz;
  // Tikhonov regularization trick: add a small nonzero term to avoid singularity/instability
  // see Numerical Recipes, 3rd ed, sec 19.5. probably no longer needed with cuboid self-inertia, but leaving just in case
  //double lambda = 1e-10 * trace;
  //for (int d = 0; d < 3; d++) I[d][d] += lambda;

  RigidBodyGeometry geom;
  geom.com = COM_result.com;
  geom.totalRho = COM_result.totalRho;
  for (int a = 0; a < 3; a++)
  for (int b = 0; b < 3; b++)
    geom.I[a][b] = I[a][b];
  invert3x3(I, geom.Iinv);
  return geom;
}

RigidModeMoments computeRigidModeMoments(const Field& f, const RigidBodyGeometry& geom,
                                         const Magnet* magnet) {
  CuParameter rho = magnet->rho.cu();
  int ncells = f.system()->grid().ncells();
  int numBlocks = numReductionBlocks(ncells);

  TransRotPartial trp = reduceOnDevice<TransRotPartial>(numBlocks, k_transRotPartialSums, f.cu(), rho, geom.com);

  RigidModeMoments moments;
  moments.T = {trp.rhoF.x / geom.totalRho, trp.rhoF.y / geom.totalRho, trp.rhoF.z / geom.totalRho};
  double3 L = trp.rhoRxF;
  //ω= I^-1 * L
  moments.omega.x = geom.Iinv[0][0]*L.x + geom.Iinv[0][1]*L.y + geom.Iinv[0][2]*L.z;
  moments.omega.y = geom.Iinv[1][0]*L.x + geom.Iinv[1][1]*L.y + geom.Iinv[1][2]*L.z;
  moments.omega.z = geom.Iinv[2][0]*L.x + geom.Iinv[2][1]*L.y + geom.Iinv[2][2]*L.z;
  return moments;
}

void removeRigidBodyModes(Field& f, const RigidBodyGeometry& geom, const Magnet* magnet) {
  RigidModeMoments moments = computeRigidModeMoments(f, geom, magnet);
  int ncells = f.system()->grid().ncells();
  cudaLaunch(ncells, k_subtractRigidModes, f.cu(), geom.com, moments.T, moments.omega);
}