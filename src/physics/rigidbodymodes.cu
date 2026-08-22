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
#include "rigidbodymodes_common.hpp"
#include "system.hpp"

namespace {

constexpr int MAX_REDUCTION_BLOCKS = 4096;
int numReductionBlocks(int ncells) {
  int n = (ncells + BLOCKDIM - 1) / BLOCKDIM;
  if (n < 1) n = 1;
  if (n > MAX_REDUCTION_BLOCKS) n = MAX_REDUCTION_BLOCKS;
  return n;
}

__host__ __device__ inline double4 operator+(double4 a, double4 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

//device-reduce-then-gather-on-host. gather-on-host is inefficient
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

// pass 1: center of mass (COM) (if unweighted=true:  w=1 , COM is geometric centroid)
// factor of V cancels elsewhere, so actually just rho weighted if unweighted=false
__global__ void k_comPartialSums(double4* blockSums, CuSystem system,
                                 CuParameter rho, bool unweighted) {
  __shared__ double sdata[4][BLOCKDIM];
  int ncells = system.grid.ncells();
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  double accumulator[4] = {0, 0, 0, 0};
  for (int i = gid; i < ncells; i += stride) {
    if (!system.inGeometry(i))
      continue;
    double w;
    if (unweighted) {
      w = 1.0;
    } else {
      w = double(rho.valueAt(i));
    }
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

struct ComResult { double3 com; double weightSum; };

ComResult computeCom(const CuSystem& cusys, const CuParameter& rho, int ncells, bool unweighted) {
  int numBlocks = numReductionBlocks(ncells);
  double4 s = reduceOnDevice<double4>(numBlocks, k_comPartialSums, cusys, rho, unweighted);

  if (s.w <= 0.0){
    throw std::runtime_error("computeRigidBodyGeometry: total mass (sum of rho) in geometry is zero or negative.");
  }

  return ComResult{double3{s.x / s.w, s.y / s.w, s.z / s.w}, s.w};
}

// pass 2: inertia tensor about a given COM
struct InertiaPartial { double3 diag; double3 offdiag; };
__host__ __device__ inline InertiaPartial operator+(InertiaPartial a, InertiaPartial b) {
  return {addD3(a.diag, b.diag), addD3(a.offdiag, b.offdiag)};
}

__global__ void k_inertiaPartialSums(InertiaPartial* blockSums,
                                     CuSystem system, CuParameter rho,
                                     double3 com, bool unweighted) {
  __shared__ double sdata[6][BLOCKDIM];  // xx yy zz xy xz yz
  int ncells = system.grid.ncells();
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  double accumulator[6] = {0, 0, 0, 0, 0, 0};
  for (int i = gid; i < ncells; i += stride) {
    if (!system.inGeometry(i))
      continue;
    double w;
    if (unweighted) {
      w = 1.0;
    } else {
      w = double(rho.valueAt(i));
    }
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

// fused translation + angular rotation
struct TransRotPartial { double3 rhoF; double3 rhoRxF; };
__host__ __device__ inline TransRotPartial operator+(TransRotPartial a, TransRotPartial b) {
  return {addD3(a.rhoF, b.rhoF), addD3(a.rhoRxF, b.rhoRxF)};
}

__global__ void k_transRotPartialSumsGeneric(TransRotPartial* blockSums, CuField f,
                                             CuParameter rho, double3 com,
                                             bool unweighted) {
  __shared__ double sdata[6][BLOCKDIM];  // fx fy fz lx ly lz
  int ncells = f.system.grid.ncells();
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  double accumulator[6] = {0, 0, 0, 0, 0, 0};
  for (int i = gid; i < ncells; i += stride) {
    if (!f.cellInGeometry(i))
      continue;
    double w;
    if (unweighted) {
      w = 1.0;
    } else {
      w = double(rho.valueAt(i));
    }
    real3 f_real = f.vectorAt(i);
    double3 f_double = double3{double(f_real.x), double(f_real.y), double(f_real.z)};
    accumulator[0] += w * f_double.x; accumulator[1] += w * f_double.y; accumulator[2] += w * f_double.z;

    //position relative to COM
    double3 r = subD3(cellPositionDeviceD(f.system, i), com);
    // angular momentum contribution: r x f
    double3 rxf = crossD3(r, f_double);
    accumulator[3] += w * rxf.x; accumulator[4] += w * rxf.y; accumulator[5] += w * rxf.z;
  }
  #pragma unroll
  for (int f_ = 0; f_ < 6; f_++)
    sdata[f_][tid] = accumulator[f_];
  __syncthreads();

  blockReduceSum<6>(sdata, tid);

  if (tid == 0) {
    blockSums[blockIdx.x].rhoF   = {sdata[0][0], sdata[1][0], sdata[2][0]};
    blockSums[blockIdx.x].rhoRxF = {sdata[3][0], sdata[4][0], sdata[5][0]};
  }
}

// u -= T + theta x r, or f-= f_trans + τ 
__global__ void k_subtractRigidModesGeneric(CuField f, double3 com,
                                            double3 T, double3 omega) {
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


// Computes both the rho-weighted and the fully unweighted (geometric)
// versions of com, inertia tensor, and normalization

// if rho-weighted, compute center of mass (COM) COM= (Σrho_i*r_i)/Σrho_i (factor of V canceled off) and moment of Inertia tensor (I) 
//about COM I = Σrho_i*(|r_comi|^2*I - r_comi*r_comi^T). r_comi=r_i - COM . (factor of V also cancels off with L in L=I*ω or ω=I^-1*L)
// L= Σrho_i*(r_comi x f_i) . COM, I computed once and reused
RigidBodyGeometry computeRigidBodyGeometry(const Magnet* magnet) {
  std::shared_ptr<const System> system = magnet->system();
  int ncells = system->grid().ncells();
  CuSystem cusys = system->cu();
  CuParameter rho = magnet->rho.cu();

  ComResult comWeighted = computeCom(cusys, rho, ncells, false);
  ComResult comUnweighted = computeCom(cusys, rho, ncells, true);

  int numBlocks = numReductionBlocks(ncells);
  InertiaPartial inertiaP = reduceOnDevice<InertiaPartial>(
      numBlocks, k_inertiaPartialSums, cusys, rho, comWeighted.com, false);
  InertiaPartial inertiaPUnweighted = reduceOnDevice<InertiaPartial>(
      numBlocks, k_inertiaPartialSums, cusys, rho, comUnweighted.com, true);

  // I is symmetric, Ixy=Iyx, Ixz=Izx, Iyz=Izy
  double I[3][3] = {{inertiaP.diag.x, inertiaP.offdiag.x, inertiaP.offdiag.y},
                    {inertiaP.offdiag.x, inertiaP.diag.y, inertiaP.offdiag.z},
                    {inertiaP.offdiag.y, inertiaP.offdiag.z, inertiaP.diag.z}};
  double Iu[3][3] = {{inertiaPUnweighted.diag.x, inertiaPUnweighted.offdiag.x, inertiaPUnweighted.offdiag.y},
                     {inertiaPUnweighted.offdiag.x, inertiaPUnweighted.diag.y, inertiaPUnweighted.offdiag.z},
                     {inertiaPUnweighted.offdiag.y, inertiaPUnweighted.offdiag.z, inertiaPUnweighted.diag.z}};

  
  // Tikhonov regularization trick: add a small nonzero term to avoid singularity/instability
  // see Numerical Recipes, 3rd ed, sec 19.5. probably no longer needed with cuboid self-inertia, but leaving just in case
  //double trace = Ixx + Iyy + Izz;
  //double lambda = 1e-10 * trace;
  //for (int d = 0; d < 3; d++) I[d][d] += lambda;

  RigidBodyGeometry geom;
  geom.com = comWeighted.com;
  geom.comUnweighted = comUnweighted.com;
  geom.totalRho = comWeighted.weightSum;
  geom.ncellsInGeometry = comUnweighted.weightSum;  
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) {
      geom.I[a][b] = I[a][b];
      geom.IUnweighted[a][b] = Iu[a][b];
    }
  }
  invert3x3(I, geom.Iinv);
  invert3x3(Iu, geom.IinvUnweighted);
  return geom;
}

// least-squares fit about geom.com
RigidModeMoments computeRigidModeMoments(const Field& f, const RigidBodyGeometry& geom,
                                         const Magnet* magnet, bool unweighted) {
  CuParameter rho = magnet->rho.cu();
  int ncells = f.system()->grid().ncells();
  int numBlocks = numReductionBlocks(ncells);

  double3 com;
  double denom;
  const double (*IinvToUse)[3];
  if (unweighted) {
    com = geom.comUnweighted;
    denom = geom.ncellsInGeometry;
    IinvToUse = geom.IinvUnweighted;
  } else {
    com = geom.com;
    denom = geom.totalRho;
    IinvToUse = geom.Iinv;
  }

  TransRotPartial trp = reduceOnDevice<TransRotPartial>(numBlocks, k_transRotPartialSumsGeneric,
                                                        f.cu(), rho, com, unweighted);

  RigidModeMoments moments;
  moments.T = {trp.rhoF.x / denom, trp.rhoF.y / denom, trp.rhoF.z / denom};
  double3 L = trp.rhoRxF;
  moments.omega.x = IinvToUse[0][0]*L.x + IinvToUse[0][1]*L.y + IinvToUse[0][2]*L.z;
  moments.omega.y = IinvToUse[1][0]*L.x + IinvToUse[1][1]*L.y + IinvToUse[1][2]*L.z;
  moments.omega.z = IinvToUse[2][0]*L.x + IinvToUse[2][1]*L.y + IinvToUse[2][2]*L.z;
  return moments;
}

void removeRigidBodyModes(Field& f, const RigidBodyGeometry& geom,
                          const Magnet* magnet, bool unweighted) {
  RigidModeMoments moments = computeRigidModeMoments(f, geom, magnet, unweighted);
  double3 com;
  if (unweighted) {
    com = geom.comUnweighted;
  } else {
    com = geom.com;
  }
  int ncells = f.system()->grid().ncells();
  cudaLaunch(ncells, k_subtractRigidModesGeneric, f.cu(), com, moments.T, moments.omega);
}

