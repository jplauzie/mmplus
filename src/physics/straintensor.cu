#include "cudalaunch.hpp"
#include "magnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "straintensor.hpp"
#include "strainstencil.hpp"


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

  // When outside the geometry, set to zero and return early
  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      for (int i = 0; i < strain.ncomp; i++)
        strain.setValueInCell(idx, i, 0);
    }
    return;
  }

  const real ws[3] = {w.x, w.y, w.z};
  const int3 dir[3] = {int3{1,0,0}, int3{0,1,0}, int3{0,0,1}};
  const int3 coo = grid.index2coord(idx);

  real der[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};  // derivatives ∂i(uj)
  const real3 u_0 = u.vectorAt(idx);
#pragma unroll
  for (int i = 0; i < 3; i++) {  // i is a {x, y, z} direction
    int3 cn[5];
    bool g[5];
    for (int t = -2; t <= 2; t++) {
      cn[t+2] = (t == 0) ? coo
                         : mastergrid.wrap(int3{coo.x + t*dir[i].x,
                                                coo.y + t*dir[i].y,
                                                coo.z + t*dir[i].z});
      g[t+2] = (t == 0) ? true : system.inGeometry(cn[t+2]);
    }
    real s[5];
    derivativeStencil(g[0], g[1], g[3], g[4], s);

    real3 dudi = real3{0, 0, 0};
    for (int k = -2; k <= 2; k++) {
      if (s[k+2] != 0)
        dudi += s[k+2] * (k == 0 ? u_0 : u.vectorAt(cn[k+2]));
    }
    dudi *= ws[i];

    der[i][0] = dudi.x;
    der[i][1] = dudi.y;
    der[i][2] = dudi.z;
  }

  // create the strain tensor
  for (int i = 0; i < 3; i++){
    for (int j = i; j < 3; j++){
      if (i == j) {  // diagonals
        strain.setValueInCell(idx, i, der[i][j]);
      }
      else {  // off-diagonal
        strain.setValueInCell(idx, i+j+2,
                              0.5 * (der[i][j] + der[j][i]));
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
