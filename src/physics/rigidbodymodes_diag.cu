// rigidbodymodes_diag.cu
//
// Deliberately independent, deliberately naive verification of net
// force/torque. Single GPU thread, sequential loop, double precision.
// Shares NO code (reduction templates, block-sum kernels, COM/inertia
// helpers) with rigidbodymodes.cu or rigidbodymodes2.cu -- the only things
// reused are the basic per-cell accessors (vectorAt, valueAt,
// cellInGeometry, grid.index2coord, cellsize), which are simple enough
// that a bug there would need separate verification anyway.
//
// NOT for the hot path -- this is O(ncells) on a single thread, meant to
// be called a handful of times during debugging, not every minimizer step.

#include "cudaerror.hpp"
#include "cudalaunch.hpp"
#include "cudastream.hpp"
#include "field.hpp"
#include "gpubuffer.hpp"
#include "magnet.hpp"
#include "parameter.hpp"
#include "system.hpp"
#include <stdexcept>
#include "rigidbodymodes_diag.hpp"



namespace {

__global__ void k_dumbComSingleThread(double4* out, CuSystem system, CuParameter rho) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  int ncells = system.grid.ncells();
  double sx = 0, sy = 0, sz = 0, sw = 0;
  for (int i = 0; i < ncells; i++) {
    if (!system.inGeometry(i)) continue;
    int3 coord = system.grid.index2coord(i);
    double x = double(coord.x) * double(system.cellsize.x);
    double y = double(coord.y) * double(system.cellsize.y);
    double z = double(coord.z) * double(system.cellsize.z);
    double w = double(rho.valueAt(i));
    sx += w * x;
    sy += w * y;
    sz += w * z;
    sw += w;
  }
  *out = {sx, sy, sz, sw};
}

__global__ void k_dumbForceTorqueSingleThread(double3* outForce, double3* outTorque,
                                              CuField f, double3 com) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  int ncells = f.system.grid.ncells();
  double fx = 0, fy = 0, fz = 0;
  double tx = 0, ty = 0, tz = 0;
  for (int i = 0; i < ncells; i++) {
    if (!f.cellInGeometry(i)) continue;
    int3 coord = f.system.grid.index2coord(i);
    double rx = double(coord.x) * double(f.system.cellsize.x) - com.x;
    double ry = double(coord.y) * double(f.system.cellsize.y) - com.y;
    double rz = double(coord.z) * double(f.system.cellsize.z) - com.z;

    real3 fv = f.vectorAt(i);
    double fvx = double(fv.x), fvy = double(fv.y), fvz = double(fv.z);

    fx += fvx; fy += fvy; fz += fvz;
    tx += ry * fvz - rz * fvy;
    ty += rz * fvx - rx * fvz;
    tz += rx * fvy - ry * fvx;
  }
  *outForce = {fx, fy, fz};
  *outTorque = {tx, ty, tz};
}

}  // namespace

NetForceTorqueDumb computeNetForceTorqueDumbLoop(const Field& f, const Magnet* magnet) {
  CuSystem cusys = f.system()->cu();
  CuParameter rho = magnet->rho.cu();

  // independent COM
  GpuBuffer<double4> d_com(1);
  k_dumbComSingleThread<<<1, 1, 0, getCudaStream()>>>(d_com.get(), cusys, rho);
  checkCudaError(cudaPeekAtLastError());
  double4 comSum;
  checkCudaError(cudaMemcpyAsync(&comSum, d_com.get(), sizeof(double4),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));
  if (comSum.w <= 0.0)
    throw std::runtime_error("computeNetForceTorqueDumbLoop: zero/negative total mass.");
  double3 com = {comSum.x / comSum.w, comSum.y / comSum.w, comSum.z / comSum.w};

  // independent net force / torque about that COM
  GpuBuffer<double3> d_force(1), d_torque(1);
  k_dumbForceTorqueSingleThread<<<1, 1, 0, getCudaStream()>>>(d_force.get(), d_torque.get(), f.cu(), com);
  checkCudaError(cudaPeekAtLastError());

  NetForceTorqueDumb result;
  result.com = com;
  checkCudaError(cudaMemcpyAsync(&result.netForce, d_force.get(), sizeof(double3),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaMemcpyAsync(&result.netTorque, d_torque.get(), sizeof(double3),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));
  return result;
}

// rigidbodymodes_diag.cu

namespace {

__global__ void k_dumbCentroidSingleThread(double4* out, CuSystem system,
                                           CuParameter rho, bool weighted) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  int ncells = system.grid.ncells();
  double sx = 0, sy = 0, sz = 0, sw = 0;
  for (int i = 0; i < ncells; i++) {
    if (!system.inGeometry(i)) continue;
    int3 coord = system.grid.index2coord(i);
    double x = double(coord.x) * double(system.cellsize.x);
    double y = double(coord.y) * double(system.cellsize.y);
    double z = double(coord.z) * double(system.cellsize.z);
    double w = weighted ? double(rho.valueAt(i)) : 1.0;
    sx += w * x; sy += w * y; sz += w * z; sw += w;
  }
  *out = {sx, sy, sz, sw};
}

__global__ void k_dumbKinematicMomentsSingleThread(double3* outT, double* outI,
                                                    double3* outL, CuField u,
                                                    CuParameter rho, double3 com,
                                                    bool weighted) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  int ncells = u.system.grid.ncells();
  double Tx = 0, Ty = 0, Tz = 0;
  double Ixx = 0, Iyy = 0, Izz = 0, Ixy = 0, Ixz = 0, Iyz = 0;
  double Lx = 0, Ly = 0, Lz = 0;

  for (int i = 0; i < ncells; i++) {
    if (!u.cellInGeometry(i)) continue;
    int3 coord = u.system.grid.index2coord(i);
    double rx = double(coord.x) * double(u.system.cellsize.x) - com.x;
    double ry = double(coord.y) * double(u.system.cellsize.y) - com.y;
    double rz = double(coord.z) * double(u.system.cellsize.z) - com.z;
    double w = weighted ? double(rho.valueAt(i)) : 1.0;

    real3 uv = u.vectorAt(i);
    double ux = double(uv.x), uy = double(uv.y), uz = double(uv.z);

    Tx += w * ux; Ty += w * uy; Tz += w * uz;

    Ixx += w * (ry*ry + rz*rz);
    Iyy += w * (rx*rx + rz*rz);
    Izz += w * (rx*rx + ry*ry);
    Ixy += w * (-rx*ry);
    Ixz += w * (-rx*rz);
    Iyz += w * (-ry*rz);

    Lx += w * (ry*uz - rz*uy);
    Ly += w * (rz*ux - rx*uz);
    Lz += w * (rx*uy - ry*ux);
  }

  *outT = {Tx, Ty, Tz};
  outI[0]=Ixx; outI[1]=Iyy; outI[2]=Izz; outI[3]=Ixy; outI[4]=Ixz; outI[5]=Iyz;
  *outL = {Lx, Ly, Lz};
}

void invert3x3Dumb(const double I[3][3], double out[3][3]) {
  double det = I[0][0]*(I[1][1]*I[2][2]-I[1][2]*I[2][1])
             - I[0][1]*(I[1][0]*I[2][2]-I[1][2]*I[2][0])
             + I[0][2]*(I[1][0]*I[2][1]-I[1][1]*I[2][0]);
  if (det == 0.0)
    throw std::runtime_error("computeKinematicMomentDumbLoop: singular inertia tensor.");
  double invDet = 1.0 / det;
  out[0][0] = ( I[1][1]*I[2][2]-I[1][2]*I[2][1]) * invDet;
  out[0][1] = (-I[0][1]*I[2][2]+I[0][2]*I[2][1]) * invDet;
  out[0][2] = ( I[0][1]*I[1][2]-I[0][2]*I[1][1]) * invDet;
  out[1][0] = (-I[1][0]*I[2][2]+I[1][2]*I[2][0]) * invDet;
  out[1][1] = ( I[0][0]*I[2][2]-I[0][2]*I[2][0]) * invDet;
  out[1][2] = (-I[0][0]*I[1][2]+I[0][2]*I[1][0]) * invDet;
  out[2][0] = ( I[1][0]*I[2][1]-I[1][1]*I[2][0]) * invDet;
  out[2][1] = (-I[0][0]*I[2][1]+I[0][1]*I[2][0]) * invDet;
  out[2][2] = ( I[0][0]*I[1][1]-I[0][1]*I[1][0]) * invDet;
}

}  // namespace

KinematicMomentDumb computeKinematicMomentDumbLoop(const Field& u, const Magnet* magnet, bool weighted) {
  CuSystem cusys = u.system()->cu();
  CuParameter rho = magnet->rho.cu();

  GpuBuffer<double4> d_com(1);
  k_dumbCentroidSingleThread<<<1, 1, 0, getCudaStream()>>>(d_com.get(), cusys, rho, weighted);
  checkCudaError(cudaPeekAtLastError());
  double4 comSum;
  checkCudaError(cudaMemcpyAsync(&comSum, d_com.get(), sizeof(double4),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));
  if (comSum.w <= 0.0)
    throw std::runtime_error("computeKinematicMomentDumbLoop: zero/negative total weight.");
  double3 com = {comSum.x / comSum.w, comSum.y / comSum.w, comSum.z / comSum.w};

  GpuBuffer<double3> d_T(1), d_L(1);
  GpuBuffer<double> d_I(6);
  k_dumbKinematicMomentsSingleThread<<<1, 1, 0, getCudaStream()>>>(
      d_T.get(), d_I.get(), d_L.get(), u.cu(), rho, com, weighted);
  checkCudaError(cudaPeekAtLastError());

  double3 Tsum, L;
  double Iflat[6];
  checkCudaError(cudaMemcpyAsync(&Tsum, d_T.get(), sizeof(double3), cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaMemcpyAsync(&L, d_L.get(), sizeof(double3), cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaMemcpyAsync(Iflat, d_I.get(), 6*sizeof(double), cudaMemcpyDeviceToHost, getCudaStream()));
  checkCudaError(cudaStreamSynchronize(getCudaStream()));

  double I[3][3] = {{Iflat[0], Iflat[3], Iflat[4]},
                    {Iflat[3], Iflat[1], Iflat[5]},
                    {Iflat[4], Iflat[5], Iflat[2]}};
  double Iinv[3][3];
  invert3x3Dumb(I, Iinv);

  KinematicMomentDumb result;
  result.com = com;
  result.T = {Tsum.x / comSum.w, Tsum.y / comSum.w, Tsum.z / comSum.w};
  result.theta.x = Iinv[0][0]*L.x + Iinv[0][1]*L.y + Iinv[0][2]*L.z;
  result.theta.y = Iinv[1][0]*L.x + Iinv[1][1]*L.y + Iinv[1][2]*L.z;
  result.theta.z = Iinv[2][0]*L.x + Iinv[2][1]*L.y + Iinv[2][2]*L.z;
  result.thetaNorm = std::sqrt(result.theta.x*result.theta.x +
                               result.theta.y*result.theta.y +
                               result.theta.z*result.theta.z);
  return result;
}