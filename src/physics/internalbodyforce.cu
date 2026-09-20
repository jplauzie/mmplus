#include "elastodynamics.hpp"
#include "internalbodyforce.hpp"
#include "cudalaunch.hpp"
#include "magnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "stresstensor.hpp"
#include "traction.hpp"
#include "strainstencil.hpp"


__device__ int tensorComp(int row, int col) {
  return (row == col) ? row : row+col+2;
}

__device__ int3 tensorRowComps(int row) {
  return int3{tensorComp(row, 0), tensorComp(row, 1), tensorComp(row, 2)};
}

//
// Force = transpose of the strain operator applied to the stress (F = -S^T σ), so that
// the discrete elastic operator is symmetric (energy conserving) on any geometry:
//  f_b(c) = - sum_a w_a  sum_{n = c+k e_a, |k|<=2} s^a(n -> c) σ_ab(n)
// where s^a(n -> c) is the weight that the strain stencil of cell n (axis a) gives to
// cell c. Boundary traction is a face load: t/h on every exposed face.
//
__global__ void k_stressDivergence(CuField fField,
                                   const CuField stress,
                                   const CuBoundaryTraction traction,
                                   const bool applyTraction,
                                   const real3 w,  // 1 / cellsize
                                   const Grid mastergrid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = fField.system;
  const Grid grid = system.grid;

  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx))
      fField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  const real ws[3] = {w.x, w.y, w.z};
  const int3 dir[3] = {int3{1,0,0}, int3{0,1,0}, int3{0,0,1}};
  const int3 coo = grid.index2coord(idx);

  real3 f = {0, 0, 0};
  for (int a = 0; a < 3; a++) {
    const int3 stressRow = tensorRowComps(a);  // (σ_a0, σ_a1, σ_a2)

    // cells at offsets -4..+4 along a: wrapped coordinate and in-geometry flag
    int3 cn[9];
    bool g[9];
    for (int t = -4; t <= 4; t++) {
      cn[t+4] = (t == 0) ? coo
                         : mastergrid.wrap(int3{coo.x + t*dir[a].x,
                                                coo.y + t*dir[a].y,
                                                coo.z + t*dir[a].z});
      g[t+4] = (t == 0) ? true : system.inGeometry(cn[t+4]);
    }

    for (int k = -2; k <= 2; k++) {
      if (!g[k+4]) continue;  // no stress outside the geometry
      real sn[5];
      // strain stencil of neighbour n = c + k
      derivativeStencil(g[k+2], g[k+3], g[k+5], g[k+6], sn);
      const real s = sn[2 - k];  // weight neighbour n gives to c (offset -k)
      if (s != 0)
        f += (-ws[a] * s) * stress.vectorAt(cn[k+4], stressRow);
    }

    if (applyTraction) {
      // exposed faces: force per volume = traction / cellsize
      if (!g[3]) f += ws[a] * traction.getSide(a, -1).vectorAt(idx);
      if (!g[5]) f += ws[a] * traction.getSide(a,  1).vectorAt(idx);
    }
  }
  fField.setVectorInCell(idx, f);
}

Field evalStressDivergence(const Magnet* magnet, const Field& stress,
                           bool applyTraction) {
  Field fField(magnet->system(), 3);
  int ncells = fField.grid().ncells();
  CuBoundaryTraction traction = magnet->boundaryTraction.cu();
  real3 w = 1. / magnet->cellsize();
  Grid mastergrid = magnet->world()->mastergrid();
  cudaLaunch(ncells, k_stressDivergence, fField.cu(), stress.cu(), traction,
             applyTraction, w, mastergrid);
  return fField;
}

Field evalInternalBodyForce(const Magnet* magnet) {
  if (stressTensorAssuredZero(magnet)) {
    Field fField(magnet->system(), 3);
    fField.makeZero();
    return fField;
  }
  Field stressTensor = evalStressTensor(magnet);
  return evalStressDivergence(magnet, stressTensor, true);
}

M_FieldQuantity internalBodyForceQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalInternalBodyForce, 3, "internal_body_force", "N/m3");
}
