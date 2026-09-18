#include "elastodynamics.hpp"
#include "internalbodyforce.hpp"
#include "cudalaunch.hpp"
#include "magnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "stresstensor.hpp"
#include "traction.hpp"

#define TEST_IDX_LX    0     // pick a cell on the -x boundary of geometry
#define TEST_IDX_RX    0     // pick a cell on the +x boundary of geometry
#define TEST_IDX_LX_Y  0     // second cell on -x boundary, same y
#define TEST_IDX_RX_Y  0     // second cell on +x boundary, same y


__device__ int tensorComp(int row, int col) {
  return (row == col) ? row : row + col + 2;
}

__device__ int3 tensorRowComps(int row) {
  return int3{tensorComp(row, 0), tensorComp(row, 1), tensorComp(row, 2)};
}


// ---------------------------------------------------------------------------
// Helpers (identical to those in straintensor.cu; move to a shared header
// eventually). No wrapping here — this is what makes the boundary detection
// work for box-filling geometries.
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



__global__ void k_internalBodyForce(CuField fField,
                                    const CuField stressTensor,
                                    const CuBoundaryTraction traction,
                                    const real3 w,  // 1 / cellsize
                                    const Grid mastergrid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = fField.system;
  const Grid grid = system.grid;

  // Outside the geometry: zero and return.
  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      fField.setVectorInCell(idx, real3{0, 0, 0});
    }
    return;
  }

  const real ws[3]   = {w.x, w.y, w.z};
  const int3 dirs[3] = {int3{1,0,0}, int3{0,1,0}, int3{0,0,1}};
  const int3 coo     = grid.index2coord(idx);

  real3 f = {0, 0, 0};

  // SAT penalty. Σ = 2 is H^{-1} for the 2-1-2 SBP norm (boundary weight h/2),
  // and gives exact energy conservation with a free surface. Try Σ = 1 or 3
  // if the BC looks too soft / too stiff.
  const real Sigma = 2.0;

#pragma unroll
  for (int i = 0; i < 3; i++) {
    const int3 di        = dirs[i];
    const real wi        = ws[i];
    const int3 stressRow = tensorRowComps(i);
    const int3 cm1       = coo - di;
    const int3 cp1       = coo + di;

    const bool m1_ok = inGridAndGeom(grid, system, cm1);
    const bool p1_ok = inGridAndGeom(grid, system, cp1);

    // --- Divergence: SBP 2-1-2, plain (no H^-1 here). ---
    if (m1_ok && p1_ok) {
        f += 0.5 * wi * (stressTensor.vectorAt(cp1, stressRow)
                      - stressTensor.vectorAt(cm1, stressRow));
    } else if (p1_ok) {
        f += wi * (-stressTensor.vectorAt(idx, stressRow)
                  + stressTensor.vectorAt(cp1, stressRow));
    } else if (m1_ok) {
        f += wi * (-stressTensor.vectorAt(cm1, stressRow)
                  + stressTensor.vectorAt(idx, stressRow));
    }
    // else: single-cell interval in this direction — no divergence here.

    const real SatSign = +1.0;
    const real Sigma   = 2.0;

    if (!m1_ok) {
        const real3 sigma_n = -1.0 * stressTensor.vectorAt(idx, stressRow);
        const real3 tau     =        traction.getSide(i, -1).vectorAt(idx);
        const real3 SAT     = SatSign * wi * Sigma * (tau - sigma_n);
        f += SAT;

        if (idx == TEST_IDX_LX || idx == TEST_IDX_LX_Y) {
            printf("MINUS face i=%d idx=%d coo=(%d,%d,%d) "
                  "sigma_row=(%+.4e,%+.4e,%+.4e) tau=(%+.4e,%+.4e,%+.4e) SAT=(%+.4e,%+.4e,%+.4e)\n",
                  i, idx, coo.x, coo.y, coo.z,
                  stressTensor.vectorAt(idx, stressRow).x,
                  stressTensor.vectorAt(idx, stressRow).y,
                  stressTensor.vectorAt(idx, stressRow).z,
                  tau.x, tau.y, tau.z,
                  SAT.x, SAT.y, SAT.z);
        }
    }
    if (!p1_ok) {
        const real3 sigma_n = +1.0 * stressTensor.vectorAt(idx, stressRow);
        const real3 tau     =        traction.getSide(i,  1).vectorAt(idx);
        const real3 SAT     = SatSign * wi * Sigma * (tau - sigma_n);
        f += SAT;

        if (idx == TEST_IDX_RX || idx == TEST_IDX_RX_Y) {
            printf("PLUS  face i=%d idx=%d coo=(%d,%d,%d) "
                  "sigma_row=(%+.4e,%+.4e,%+.4e) tau=(%+.4e,%+.4e,%+.4e) SAT=(%+.4e,%+.4e,%+.4e)\n",
                  i, idx, coo.x, coo.y, coo.z,
                  stressTensor.vectorAt(idx, stressRow).x,
                  stressTensor.vectorAt(idx, stressRow).y,
                  stressTensor.vectorAt(idx, stressRow).z,
                  tau.x, tau.y, tau.z,
                  SAT.x, SAT.y, SAT.z);
        }
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
  return M_FieldQuantity(magnet, evalInternalBodyForce, 3,
                         "internal_body_force", "N/m3");
}
