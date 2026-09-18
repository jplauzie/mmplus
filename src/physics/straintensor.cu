#include "cudalaunch.hpp"
#include "magnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "straintensor.hpp"


// ---------------------------------------------------------------------------
// Helpers
//
// insideGrid: is c a valid cell of the master grid? (No wrapping.)
// inGridAndGeom: is c a valid grid cell that is also inside the geometry?
//
// These deliberately do NOT call wrap(). If wrap() is used, a cell at the
// grid edge sees its periodic neighbour, dL/dR never reach 0, and the boundary
// stencil is never selected — which is the bug for box-filling geometries.
//
// Adapt the accessors if your Grid type exposes size/origin differently.
// ---------------------------------------------------------------------------
// Bounds check against the system's own grid (NOT the mastergrid,
// which may have z-size 0 for 2D worlds).
__device__ __forceinline__ bool insideGrid(const Grid& g, int3 c) {
  const int3 s = g.size();
  const int3 o = g.origin();
  // If the mastergrid has zero extent in any dimension, treat that
  // dimension as unbounded (use c's value).
  const int sx = (s.x > 0) ? s.x : 1;
  const int sy = (s.y > 0) ? s.y : 1;
  const int sz = (s.z > 0) ? s.z : 1;
  return c.x >= o.x && c.x < o.x + sx &&
         c.y >= o.y && c.y < o.y + sy &&
         c.z >= o.z && c.z < o.z + sz;
}

// Bounds check against the system's own grid, NOT the mastergrid.
// The mastergrid can have size 1 or 0 in some dimensions (e.g. a 2D world),
// which would reject every legitimate neighbour coordinate.
__device__ __forceinline__ bool inGridAndGeom(const Grid& sysGrid,
                                              const CuSystem& sys,
                                              int3 c) {
  const int3 s = sysGrid.size();
  const int3 o = sysGrid.origin();
  if (c.x < o.x || c.x >= o.x + s.x) return false;
  if (c.y < o.y || c.y >= o.y + s.y) return false;
  if (c.z < o.z || c.z >= o.z + s.z) return false;
  return sys.inGeometry(c);
}



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

  // Outside the geometry: zero and return.
  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      for (int i = 0; i < strain.ncomp; i++)
        strain.setValueInCell(idx, i, 0);
    }
    return;
  }

  const real ws[3]   = {w.x, w.y, w.z};
  const int3 dirs[3] = {int3{1,0,0}, int3{0,1,0}, int3{0,0,1}};
  const int3 coo     = grid.index2coord(idx);
  const real3 u_0    = u.vectorAt(idx);

  real der[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};  // der[i][j] = ∂_i u_j

#pragma unroll
  for (int i = 0; i < 3; i++) {
    const int3 di  = dirs[i];
    const real wi  = ws[i];
    const int3 cm1 = coo - di;
    const int3 cp1 = coo + di;

    const bool m1_ok = inGridAndGeom(grid, system, cm1);
    const bool p1_ok = inGridAndGeom(grid, system, cp1);

    real3 dudi;
    if (m1_ok && p1_ok) {
      // Interior of a 1D interval: SBP 2-1-2 interior row (central).
      dudi = 0.5 * (u.vectorAt(cp1) - u.vectorAt(cm1));
    } else if (p1_ok) {
      // Left end of interval: SBP 2-1-2 boundary row.
      dudi = -u_0 + u.vectorAt(cp1);
    } else if (m1_ok) {
      // Right end of interval: SBP 2-1-2 boundary row (mirror).
      dudi = -u.vectorAt(cm1) + u_0;
    } else {
      // Single-cell interval in this direction: no gradient contribution.
      dudi = real3{0, 0, 0};
    }
    dudi *= wi;

    der[i][0] = dudi.x;
    der[i][1] = dudi.y;
    der[i][2] = dudi.z;
  }

  // Assemble the strain tensor (Voigt: xx, yy, zz, xy, xz, yz).
  for (int i = 0; i < 3; i++) {
    for (int j = i; j < 3; j++) {
      if (i == j) {
        strain.setValueInCell(idx, i, der[i][j]);
      } else {
        strain.setValueInCell(idx, i + j + 2, 0.5 * (der[i][j] + der[j][i]));
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
  CuField u  = magnet->elasticDisplacement()->field().cu();
  real3 w    = 1 / magnet->cellsize();
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
  Field strainRate(magnet->system(), 6);
  if (strainTensorAssuredZero(magnet)) {
    strainRate.makeZero();
    return strainRate;
  }

  int ncells = strainRate.grid().ncells();
  CuField v  = magnet->elasticVelocity()->field().cu();
  real3 w    = 1 / magnet->cellsize();
  Grid mastergrid = magnet->world()->mastergrid();

  // Same math as for strain, applied to velocity.
  cudaLaunch(ncells, k_strainTensor, strainRate.cu(), v, w, mastergrid);

  return strainRate;
}

M_FieldQuantity strainRateQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalStrainRate, 6, "strain_rate", "1/s");
}
