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

// Multiplier on the occupancy-derived block cap (numSMs * maxActiveBlocksPerSM).
constexpr int REDUCTION_OCCUPANCY_MULTIPLIER = 1;

template <typename Kernel>
int occupancyMaxBlocks(Kernel kernel, int blockDim, size_t sharedMemBytes = 0) {
  static int cached = -1;
  if (cached < 0) {
    int device;
    checkCudaError(cudaGetDevice(&device));
    int smCount;
    checkCudaError(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));

    int maxActiveBlocksPerSm = 0;
    checkCudaError(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocksPerSm, kernel, blockDim, sharedMemBytes));

    cached = smCount * maxActiveBlocksPerSm * REDUCTION_OCCUPANCY_MULTIPLIER;
  }
  return cached;
}

//sensible choice of number of blocks
template <typename Kernel>
int numReductionBlocks(int ncells, Kernel kernel) {
  int n = (ncells + BLOCKDIM - 1) / BLOCKDIM;
  if (n < 1) n = 1;
  int maxBlocks = occupancyMaxBlocks(kernel, BLOCKDIM);
  if (maxBlocks > 0 && n > maxBlocks) n = maxBlocks;
  return n;
}

// device-reduce-then-gather-on-host. Used by the COM/inertia passes. host gather is inefficient, but only runs once
template <typename T, typename... Args>
T reduceOnDevice(int ncells, void (*kernel)(T*, Args...), Args... args) {
  int numBlocks = numReductionBlocks(ncells, kernel);

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

//pass 1: center of mass (COM)
struct ComPartial { real3 rw; real w; };
__host__ __device__ inline ComPartial operator+(ComPartial a, ComPartial b) {
  return {a.rw + b.rw, a.w + b.w};
}

__global__ void k_comPartialSums(ComPartial* blockSums, CuSystem system,
                                 CuParameter rho, bool unweighted) {
  __shared__ real sdata[4][BLOCKDIM];
  int ncells = system.grid.ncells();
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  real3 accumulator_pos{0, 0, 0};
  real accumulator_w = 0;
  for (int i = gid; i < ncells; i += stride) {
    if (!system.inGeometry(i)){
      continue;
    }
    real w;
    if (unweighted) {
      w = 1.0;
    } else {
      w = rho.valueAt(i);
    }
    accumulator_pos = accumulator_pos + w * cellPositionDevice(system, i);
    accumulator_w += w;
  }
  sdata[0][tid] = accumulator_pos.x; sdata[1][tid] = accumulator_pos.y;
  sdata[2][tid] = accumulator_pos.z; sdata[3][tid] = accumulator_w;
  __syncthreads();

  blockReduceSum<4>(sdata, tid);

  if (tid == 0)
    blockSums[blockIdx.x] = {real3{sdata[0][0], sdata[1][0], sdata[2][0]}, sdata[3][0]};
}

struct ComResult { real3 com; real weightSum; };

ComResult computeCom(const CuSystem& cusys, const CuParameter& rho, int ncells, bool unweighted) {
  ComPartial s = reduceOnDevice<ComPartial>(ncells, k_comPartialSums, cusys, rho, unweighted);

  if (s.w <= real(0.0))
    throw std::runtime_error("computeRigidBodyGeometry: total mass (sum of rho) in geometry is zero or negative.");

  return ComResult{s.rw / s.w, s.w};
}


// pass 2: inertia tensor about a given COM
struct InertiaPartial { real3 diag; real3 offdiag; };
__host__ __device__ inline InertiaPartial operator+(InertiaPartial a, InertiaPartial b) {
  return {a.diag + b.diag, a.offdiag + b.offdiag};
}

__global__ void k_inertiaPartialSums(InertiaPartial* blockSums,
                                     CuSystem system, CuParameter rho,
                                     real3 com, bool unweighted) {
  __shared__ real sdata[6][BLOCKDIM];  // xx yy zz xy xz yz
  int ncells = system.grid.ncells();
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  real accumulator[6] = {0, 0, 0, 0, 0, 0};
  for (int i = gid; i < ncells; i += stride) {
    if (!system.inGeometry(i)){
      continue;
    }
    real w;
    if (unweighted) {
      w = 1.0;
    } else {
      w = rho.valueAt(i);
    }
    real3 r = cellPositionDevice(system, i) - com;
    real x = r.x, y = r.y, z = r.z;
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


// fused translation/rotation moment reduction, 2 stage reduction
// level 1: grid-stride load + two-level warp-shuffle tree reduce.
//  Each block writes its own partial to a small global array. 
// level 2: exactly one block. Combines level-1 partials using f64,
//  then the same warp-shuffle tree.

struct TransRotAccum { real fx, fy, fz, lx, ly, lz; };
struct TransRotAccumD { double fx, fy, fz, lx, ly, lz; };

// level 1
__global__ void k_transRotSumsPartial(TransRotAccum* blockPartials, CuField f,
                                      CuParameter rho, real3 com, bool unweighted) {
  int ncells = f.system.grid.ncells();
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  real accumulator_fx = 0, accumulator_fy = 0, accumulator_fz = 0;
  real accumulator_lx = 0, accumulator_ly = 0, accumulator_lz = 0;

  for (int i = gid; i < ncells; i += stride) {
    if (!f.cellInGeometry(i)) continue;
    real w = unweighted ? real(1.0) : rho.valueAt(i);
    real3 fv = f.vectorAt(i);
    accumulator_fx += w * fv.x; accumulator_fy += w * fv.y; accumulator_fz += w * fv.z;

    real3 r = cellPositionDevice(f.system, i) - com;
    real3 rxf = cross(r, fv);
    accumulator_lx += w * rxf.x; accumulator_ly += w * rxf.y; accumulator_lz += w * rxf.z;
  }


  accumulator_fx = warpReduceSum<real, warp_size>(accumulator_fx);
  accumulator_fy = warpReduceSum<real, warp_size>(accumulator_fy);
  accumulator_fz = warpReduceSum<real, warp_size>(accumulator_fz);
  accumulator_lx = warpReduceSum<real, warp_size>(accumulator_lx);
  accumulator_ly = warpReduceSum<real, warp_size>(accumulator_ly);
  accumulator_lz = warpReduceSum<real, warp_size>(accumulator_lz);

  static_assert(BLOCKDIM % warp_size == 0, "assumes full warps");
  constexpr int NUM_WARPS = BLOCKDIM / warp_size;
  __shared__ real sFx[NUM_WARPS], sFy[NUM_WARPS], sFz[NUM_WARPS];
  __shared__ real sLx[NUM_WARPS], sLy[NUM_WARPS], sLz[NUM_WARPS];

  int lane = threadIdx.x % warpSize;
  int warpId = threadIdx.x / warpSize;
  if (lane == 0) {
    sFx[warpId] = accumulator_fx; sFy[warpId] = accumulator_fy; sFz[warpId] = accumulator_fz;
    sLx[warpId] = accumulator_lx; sLy[warpId] = accumulator_ly; sLz[warpId] = accumulator_lz;
  }
  __syncthreads();

  if (warpId == 0) {
    real vFx = (lane < NUM_WARPS) ? sFx[lane] : real(0);
    real vFy = (lane < NUM_WARPS) ? sFy[lane] : real(0);
    real vFz = (lane < NUM_WARPS) ? sFz[lane] : real(0);
    real vLx = (lane < NUM_WARPS) ? sLx[lane] : real(0);
    real vLy = (lane < NUM_WARPS) ? sLy[lane] : real(0);
    real vLz = (lane < NUM_WARPS) ? sLz[lane] : real(0);

    vFx = warpReduceSum<real, NUM_WARPS>(vFx); vFy = warpReduceSum<real, NUM_WARPS>(vFy); vFz = warpReduceSum<real, NUM_WARPS>(vFz);
    vLx = warpReduceSum<real, NUM_WARPS>(vLx); vLy = warpReduceSum<real, NUM_WARPS>(vLy); vLz = warpReduceSum<real, NUM_WARPS>(vLz);

    if (lane == 0)
      blockPartials[blockIdx.x] = TransRotAccum{vFx, vFy, vFz, vLx, vLy, vLz};
  }
}

// level 2: single block, double accumulation
__global__ void k_transRotSumsCombineDouble(TransRotAccumD* out,
                                            const TransRotAccum* blockPartials,
                                            int numPartials) {
  int tid = threadIdx.x;

  double accFx = 0, accFy = 0, accFz = 0;
  double accLx = 0, accLy = 0, accLz = 0;

  for (int i = tid; i < numPartials; i += blockDim.x) {
    TransRotAccum p = blockPartials[i];
    accFx += double(p.fx); accFy += double(p.fy); accFz += double(p.fz);
    accLx += double(p.lx); accLy += double(p.ly); accLz += double(p.lz);
  }

  accFx = warpReduceSum<double, warp_size>(accFx); accFy = warpReduceSum<double, warp_size>(accFy); accFz = warpReduceSum<double, warp_size>(accFz);
  accLx = warpReduceSum<double, warp_size>(accLx); accLy = warpReduceSum<double, warp_size>(accLy); accLz = warpReduceSum<double, warp_size>(accLz);

  static_assert(BLOCKDIM % warp_size == 0, "assumes full warps");
  constexpr int NUM_WARPS = BLOCKDIM / warp_size;
  __shared__ double sFx[NUM_WARPS], sFy[NUM_WARPS], sFz[NUM_WARPS];
  __shared__ double sLx[NUM_WARPS], sLy[NUM_WARPS], sLz[NUM_WARPS];

  int lane = tid % warpSize;
  int warpId = tid / warpSize;
  if (lane == 0) {
    sFx[warpId] = accFx; sFy[warpId] = accFy; sFz[warpId] = accFz;
    sLx[warpId] = accLx; sLy[warpId] = accLy; sLz[warpId] = accLz;
  }
  __syncthreads();

  if (warpId == 0) {
    double vFx = (lane < NUM_WARPS) ? sFx[lane] : 0.0;
    double vFy = (lane < NUM_WARPS) ? sFy[lane] : 0.0;
    double vFz = (lane < NUM_WARPS) ? sFz[lane] : 0.0;
    double vLx = (lane < NUM_WARPS) ? sLx[lane] : 0.0;
    double vLy = (lane < NUM_WARPS) ? sLy[lane] : 0.0;
    double vLz = (lane < NUM_WARPS) ? sLz[lane] : 0.0;

    vFx = warpReduceSum<double, NUM_WARPS>(vFx); vFy = warpReduceSum<double, NUM_WARPS>(vFy); vFz = warpReduceSum<double, NUM_WARPS>(vFz);
    vLx = warpReduceSum<double, NUM_WARPS>(vLx); vLy = warpReduceSum<double, NUM_WARPS>(vLy); vLz = warpReduceSum<double, NUM_WARPS>(vLz);

    if (lane == 0)
      *out = TransRotAccumD{vFx, vFy, vFz, vLx, vLy, vLz};
  }
}

// Host launcher
TransRotAccumD reduceTransRotSums(CuField f, CuParameter rho, real3 com,
                                  bool unweighted, int ncells) {
  int numBlocksA = numReductionBlocks(ncells, k_transRotSumsPartial);

  GpuBuffer<TransRotAccum> d_partials(numBlocksA);
  GpuBuffer<TransRotAccumD> d_out(1);


  k_transRotSumsPartial<<<numBlocksA, BLOCKDIM, 0, getCudaStream()>>>(
      d_partials.get(), f, rho, com, unweighted);
  checkCudaError(cudaPeekAtLastError());

  k_transRotSumsCombineDouble<<<1, BLOCKDIM, 0, getCudaStream()>>>(
      d_out.get(), d_partials.get(), numBlocksA);
  checkCudaError(cudaPeekAtLastError());

  TransRotAccumD result;
  checkCudaError(cudaMemcpyAsync(&result, d_out.get(), sizeof(TransRotAccumD),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));
  return result;
}

// u -= T + theta x r, or f -= f_trans + tau. 
__global__ void k_subtractRigidModesGeneric(CuField f, real3 com,
                                            real3 T, real3 omega) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!f.cellInGrid(idx) || !f.cellInGeometry(idx))
    return;

  real3 r = cellPositionDevice(f.system, idx) - com;
  real3 correction = {T.x + (omega.y * r.z - omega.z * r.y),
                      T.y + (omega.z * r.x - omega.x * r.z),
                      T.z + (omega.x * r.y - omega.y * r.x)};

  real3 f_real = f.vectorAt(idx);
  f.setVectorInCell(idx, f_real - correction);
}

}  // namespace

// Computes both the rho-weighted and the fully unweighted (geometric)
// versions of com, inertia tensor, and normalization.
//
// if rho-weighted, compute center of mass (COM) COM= (Σrho_i*r_i)/Σrho_i (factor of V canceled off) and moment of Inertia tensor (I)
// about COM I = Σrho_i*(|r_comi|^2*I - r_comi*r_comi^T). r_comi=r_i - COM . (factor of V also cancels off with L in L=I*ω or ω=I^-1*L)
// L= Σrho_i*(r_comi x f_i) . COM, I computed once and reused
RigidBodyGeometry computeRigidBodyGeometry(const Magnet* magnet) {
  std::shared_ptr<const System> system = magnet->system();
  int ncells = system->grid().ncells();
  CuSystem cusys = system->cu();
  CuParameter rho = magnet->rho.cu();

  ComResult comWeighted = computeCom(cusys, rho, ncells, false);
  ComResult comUnweighted = computeCom(cusys, rho, ncells, true);

  InertiaPartial inertiaP = reduceOnDevice<InertiaPartial>(
      ncells, k_inertiaPartialSums, cusys, rho, comWeighted.com, false);
  InertiaPartial inertiaPUnweighted = reduceOnDevice<InertiaPartial>(
      ncells, k_inertiaPartialSums, cusys, rho, comUnweighted.com, true);

 // I is symmetric, Ixy=Iyx, Ixz=Izx, Iyz=Izy, f64 in case of conditioning for flat/thin geometries
  double I[3][3] = {{double(inertiaP.diag.x), double(inertiaP.offdiag.x), double(inertiaP.offdiag.y)},
                    {double(inertiaP.offdiag.x), double(inertiaP.diag.y), double(inertiaP.offdiag.z)},
                    {double(inertiaP.offdiag.y), double(inertiaP.offdiag.z), double(inertiaP.diag.z)}};
  double Iu[3][3] = {{double(inertiaPUnweighted.diag.x), double(inertiaPUnweighted.offdiag.x), double(inertiaPUnweighted.offdiag.y)},
                     {double(inertiaPUnweighted.offdiag.x), double(inertiaPUnweighted.diag.y), double(inertiaPUnweighted.offdiag.z)},
                     {double(inertiaPUnweighted.offdiag.y), double(inertiaPUnweighted.offdiag.z), double(inertiaPUnweighted.diag.z)}};


  //account for the self-inertia of cuboids, not point masses
  real3 cellsize = system->cellsize();
  double dx2 = double(cellsize.x) * double(cellsize.x);
  double dy2 = double(cellsize.y) * double(cellsize.y);
  double dz2 = double(cellsize.z) * double(cellsize.z);

  double selfFactor = double(comWeighted.weightSum) / 12.0;
  I[0][0] += selfFactor * (dy2 + dz2);
  I[1][1] += selfFactor * (dx2 + dz2);
  I[2][2] += selfFactor * (dx2 + dy2);

  double selfFactorU = double(comUnweighted.weightSum) / 12.0;
  Iu[0][0] += selfFactorU * (dy2 + dz2);
  Iu[1][1] += selfFactorU * (dx2 + dz2);
  Iu[2][2] += selfFactorU * (dx2 + dy2);

  // Tikhonov regularization trick: add a small nonzero term to avoid singularity/instability
  // see Numerical Recipes, 3rd ed, sec 19.5. probably no longer needed with cuboid self-inertia, but leaving just in case
  //double trace = I[0][0] + I[1][1] + I[2][2];
  //double lambda = 1e-10 * trace;
  //for (int d = 0; d < 3; d++) I[d][d] += lambda;

  RigidBodyGeometry geom;
  geom.com = comWeighted.com;
  geom.comUnweighted = comUnweighted.com;
  geom.totalRho = comWeighted.weightSum;
  geom.ncellsInGeometry = comUnweighted.weightSum;
  for (int a = 0; a < 3; a++)
    for (int b = 0; b < 3; b++) {
      geom.I[a][b] = I[a][b];
      geom.IUnweighted[a][b] = Iu[a][b];
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

  real3 com = unweighted ? geom.comUnweighted : geom.com;
  double denom = unweighted ? double(geom.ncellsInGeometry) : double(geom.totalRho);
  const double (*IinvToUse)[3] = unweighted ? geom.IinvUnweighted : geom.Iinv;

  TransRotAccumD trp = reduceTransRotSums(f.cu(), rho, com, unweighted, ncells);

  RigidModeMoments moments;
  moments.T = {trp.fx / denom, trp.fy / denom, trp.fz / denom};
  moments.omega.x = IinvToUse[0][0]*trp.lx + IinvToUse[0][1]*trp.ly + IinvToUse[0][2]*trp.lz;
  moments.omega.y = IinvToUse[1][0]*trp.lx + IinvToUse[1][1]*trp.ly + IinvToUse[1][2]*trp.lz;
  moments.omega.z = IinvToUse[2][0]*trp.lx + IinvToUse[2][1]*trp.ly + IinvToUse[2][2]*trp.lz;
  return moments;
}

void removeRigidBodyModes(Field& f, const RigidBodyGeometry& geom,
                          const Magnet* magnet, bool unweighted) {
  RigidModeMoments moments = computeRigidModeMoments(f, geom, magnet, unweighted);
  real3 com = unweighted ? geom.comUnweighted : geom.com;

  int ncells = f.system()->grid().ncells();
  cudaLaunch(ncells, k_subtractRigidModesGeneric, f.cu(), com,
             toReal3(moments.T), toReal3(moments.omega));
}
