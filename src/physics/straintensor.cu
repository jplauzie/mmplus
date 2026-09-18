#include "cudalaunch.hpp"
#include "magnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "straintensor.hpp"


bool strainTensorAssuredZero(const Magnet* magnet) {
  return !magnet->enableElastodynamics();
}


__global__ void k_strainTensor(CuField strain,
                               const CuField u,
                               const real3 w,  // w = 1/cellsize
                               const Grid mastergrid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = strain.system;
  const Grid grid = system.grid;

  // Outside geometry: zero and return
  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      for (int i = 0; i < strain.ncomp; i++)
        strain.setValueInCell(idx, i, 0);
    }
    return;
  }

  const real ws[3] = {w.x, w.y, w.z};
  const int3 ip1_arr[3] = {int3{1,0,0}, int3{0,1,0}, int3{0,0,1}};
  const int3 coo = grid.index2coord(idx);

  real3 u_0 = u.vectorAt(idx);
  real der[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};  // der[i][j] = ∂_i u_j

  for (int i = 0; i < 3; i++) {
    int3 coo_ip1 = mastergrid.wrap(coo + ip1_arr[i]);
    if (system.inGeometry(coo_ip1)) {
      real3 dudi = (u.vectorAt(coo_ip1) - u_0) * ws[i];
      der[i][0] = dudi.x;
      der[i][1] = dudi.y;
      der[i][2] = dudi.z;
    }
    // else: ghost u_{i+1} = u_i, so der[i][*] stays 0 (SBP-consistent free surface)
  }

  // Assemble symmetric strain tensor (same layout as before)
  for (int i = 0; i < 3; i++) {
    for (int j = i; j < 3; j++) {
      if (i == j) {
        strain.setValueInCell(idx, i, der[i][j]);
      } else {
        strain.setValueInCell(idx, i+j+2, 0.5 * (der[i][j] + der[j][i]));
      }
    }
  }
}


Field evalStrainTensor(const Magnet* magnet) {
  Field strain(magnet->system(), 6);
  if (strainTensorAssuredZero(magnet)) {
    strain.makeZero();
    return strain;
  }

  int ncells = strain.grid().ncells();
  CuField u = magnet->elasticDisplacement()->field().cu();
  real3 w = 1 / magnet->cellsize();
  Grid mastergrid = magnet->world()->mastergrid();

  cudaLaunch(ncells, k_strainTensor, strain.cu(), u, w, mastergrid);
  return strain;
}


M_FieldQuantity strainTensorQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalStrainTensor, 6, "strain_tensor", "");
}

// --------------------
// Strain Rate

Field evalStrainRate(const Magnet* magnet) {
  Field strainRate(magnet->system(), 6);  // symmetric 3x3 tensor
  if (strainTensorAssuredZero(magnet)) {  // same condition
    strainRate.makeZero();
    return strainRate;
  }

  int ncells = strainRate.grid().ncells();
  CuField v = magnet->elasticVelocity()->field().cu();
  real3 w = 1/ magnet->cellsize();
  Grid mastergrid = magnet->world()->mastergrid();

  // The math for strain rate is exactly the same as for strain tensor,
  // but applied to velocity instead of displacement.
  cudaLaunch(ncells, k_strainTensor, strainRate.cu(), v, w, mastergrid);

  return strainRate;
}

M_FieldQuantity strainRateQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalStrainRate, 6, "strain_rate", "1/s");
}
