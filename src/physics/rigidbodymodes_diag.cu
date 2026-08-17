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

struct NetForceTorqueDumb {
  double3 com;        // independently computed mass-weighted COM
  double3 netForce;   // Sum_i f(r_i)
  double3 netTorque;  // Sum_i (r_i - com) x f(r_i)
};

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