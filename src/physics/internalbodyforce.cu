#include "elastodynamics.hpp"
#include "internalbodyforce.hpp"
#include "cudalaunch.hpp"
#include "magnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "stresstensor.hpp"
#include "traction.hpp"


__device__ int tensorComp(int row, int col) {
  return (row == col) ? row : row+col+2;
}

__device__ int3 tensorRowComps(int row) {
  return int3{tensorComp(row, 0), tensorComp(row, 1), tensorComp(row, 2)};
}

/**
 * Numerical divergence of stress with central five-point stencil in bulk material.
 * Lower order accuracy (three-point stencil) central difference is used in bulk
 * 1 cell away from boundary.
 * The traction is applied at the boundary, implemented using a custom
 * second-order-accurate three-point stencil.
*/
__global__ void k_internalBodyForce(CuField fField,
                                    const CuField stressTensor,
                                    const CuBoundaryTraction traction,
                                    const real3 w,  // 1 / cellsize
                                    const Grid mastergrid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = fField.system;
  const Grid grid = system.grid;

  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      fField.setVectorInCell(idx, real3{0, 0, 0});
    }
    return;
  }

  const real ws[3] = {w.x, w.y, w.z};
  const int3 im1_arr[3] = {int3{-1,0,0}, int3{0,-1,0}, int3{0,0,-1}};
  const int3 ip1_arr[3] = {int3{ 1,0,0}, int3{0, 1,0}, int3{0,0, 1}};
  const int3 coo = grid.index2coord(idx);

  real3 f = {0, 0, 0};

  for (int i = 0; i < 3; i++) {
    int3 stressRow = tensorRowComps(i);

    int3 coo_im1 = mastergrid.wrap(coo + im1_arr[i]);
    int3 coo_ip1 = mastergrid.wrap(coo + ip1_arr[i]);
    bool im1_inGeo = system.inGeometry(coo_im1);
    bool ip1_inGeo = system.inGeometry(coo_ip1);

    real3 sigma_i = stressTensor.vectorAt(idx, stressRow);

    // Interior divergence: backward difference (only if backward neighbor exists)
    if (im1_inGeo) {
      real3 sigma_im1 = stressTensor.vectorAt(coo_im1, stressRow);
      f += ws[i] * (sigma_i - sigma_im1);
    }
    // If backward neighbor is outside: ghost σ_{-1} = σ_0, so interior
    // divergence contributes 0. Nothing to add here.

    // SAT at the -i boundary face (outward normal = -e_i)
    if (!im1_inGeo) {
      f += ws[i] * (sigma_i + traction.getSide(i, -1).vectorAt(idx));
    }

    // SAT at the +i boundary face (outward normal = +e_i)
    if (!ip1_inGeo) {
      f += ws[i] * (-sigma_i + traction.getSide(i, 1).vectorAt(idx));
    }
  }

  fField.setVectorInCell(idx, f);
}


Field evalInternalBodyForce(const Magnet* magnet) {

  Field fField(magnet->system(), 3);
  if (stressTensorAssuredZero(magnet)) {
    fField.makeZero();
    return fField;
  }

  int ncells = fField.grid().ncells();
  Field stressTensor = evalStressTensor(magnet);
  CuBoundaryTraction traction = magnet->boundaryTraction.cu();
  real3 w = 1. / magnet->cellsize();
  Grid mastergrid = magnet->world()->mastergrid();

  cudaLaunch(ncells, k_internalBodyForce, fField.cu(), stressTensor.cu(),
             traction, w, mastergrid);

  return fField;
}

M_FieldQuantity internalBodyForceQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalInternalBodyForce, 3, "internal_body_force", "N/m3");
}
