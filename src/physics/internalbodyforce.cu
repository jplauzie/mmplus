#include "derivativestencil.hpp"
#include "elastodynamics.hpp"
#include "internalbodyforce.hpp"
#include "cudalaunch.hpp"
#include "magnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "straintensor.hpp"
#include "stresstensor.hpp"
#include "traction.hpp"


__device__ int tensorComp(int row, int col) {
  return (row == col) ? row : row+col+2;
}

__device__ int3 tensorRowComps(int row) {
  return int3{tensorComp(row, 0), tensorComp(row, 1), tensorComp(row, 2)};
}

// ============================================================
// Interior elastic force: direct jump-aware second/mixed derivatives of u
// ============================================================
//
// Ported from magnum.np's llg_with_le_solver.py `dpd`, which never forms an
// intermediate stress field for the force calculation: each force component is
// built directly from magnum's _2nd_derivative/_mixed_derivative operators
// acting on the displacement `u`, so that the same jump data used to build the
// (traction-consistent) strain tensor also governs the force -- guaranteeing
// the two are mutually consistent at interior material interfaces by
// construction, rather than by two independently-written stencils that only
// happen to agree in bulk.
//
// Under mumax's cubic-symmetry stiffness (C11, C12, C44; no separate C13/C23/
// C55/C66 as magnum's fully general trigonal stiffness allows), each cubic
// force component f_p decomposes into exactly 7 terms (verified symbolically
// against the standard Navier-Cauchy form for cubic elasticity):
//   for p in {x,y,z}, with {j,k} = the other two directions:
//     diagonal:    d/dp( C11 * d(u_p)/dp )
//     C12 cross:   d/dp( C12 * d(u_j)/dj )
//                  d/dp( C12 * d(u_k)/dk )
//     C44 "self":  d/dj( C44 * d(u_p)/dj )
//                  d/dk( C44 * d(u_p)/dk )
//     C44 "cross": d/dj( C44 * d(u_j)/dp )
//                  d/dk( C44 * d(u_k)/dp )
//
// The "diagonal" and both "C44 self" terms are direct second derivatives
// (directSecondDerivative) and carry the magnetoelastic eigenstrain jump
// offset -- for a term differentiated in direction `d` of component `u_c`,
// magnum's _2nd_derivative_homogeneous_Neumann reads its B offset from
// diff_data._Bl/Br_jump_conditions[c][d], i.e. indexed by (component, direction)
// with the *direction* selecting which row of the eigenstrain traction to use
// -- NOT necessarily the same as the term's own physical direction p.
// Concretely: the diagonal term uses row p, component p; each C44-"self" term
// at differentiation direction j uses row j, component p (not row p). This was
// checked explicitly against magnum's term compiler before implementation,
// since assuming "row == p" throughout would have silently mismatched the
// eigenstrain coupling on the C44-self terms.
//
// The two "C12 cross" and two "C44 cross" terms are mixed derivatives
// (directMixedDerivative) of an already-computed (jump-corrected) companion
// gradient component, and carry NO eigenstrain offset, matching magnum's
// _mixed_derivative (which passes no B to its internal gradient_with_pbc call).
__global__ void k_elasticForceDirect(CuField fField,
                                     const CuField u,
                                     const CuField gradU,  // 9-comp, jump-corrected (from evalDisplacementGradientTensor)
                                     const CuField sigmaFaceL,  // 9-comp, eigenstrain
                                     const CuField sigmaFaceR,
                                     const CuParameter C11,
                                     const CuParameter C12,
                                     const CuParameter C44,
                                     const real3 w,
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

  const int3 coo = grid.index2coord(idx);
  const real hs[3] = {1 / w.x, 1 / w.y, 1 / w.z};
  const int3 step_arr[3] = {int3{1, 0, 0}, int3{0, 1, 0}, int3{0, 0, 1}};

  real f[3] = {0, 0, 0};

#pragma unroll
  for (int p = 0; p < 3; p++) {
    // if this cell is adjacent to the OUTER geometry boundary in direction p
    // (or in either companion direction below), the corresponding term is left
    // to internalbodyforce's existing boundary-traction handling: skip direct-
    // derivative contributions in that direction entirely, matching the
    // existing kernel's own per-direction outer-boundary branch structure.
    int j = (p + 1) % 3;
    int k = (p + 2) % 3;

    // ---- diagonal term: d/dp( C11 * d(u_p)/dp ) ----
    if (interiorInDirection(system, mastergrid, coo, p)) {
      int3 coo_pm = mastergrid.wrap(coo - step_arr[p]);
      int3 coo_pp = mastergrid.wrap(coo + step_arr[p]);
      int idx_pm = grid.coord2index(coo_pm);
      int idx_pp = grid.coord2index(coo_pp);

      real u_self = u.valueAt(idx, p);
      real u_pm = u.valueAt(idx_pm, p);
      real u_pp = u.valueAt(idx_pp, p);
      real C_self = C11.valueAt(idx);
      real C_pm = C11.valueAt(idx_pm);
      real C_pp = C11.valueAt(idx_pp);

      int3 rowP = gradRowComps(p);
      real3 BselfL3 = sigmaFaceL.vectorAt(idx, rowP);
      real3 BselfR3 = sigmaFaceR.vectorAt(idx, rowP);
      real3 BpmR3 = sigmaFaceR.vectorAt(coo_pm, rowP);
      real3 BppL3 = sigmaFaceL.vectorAt(coo_pp, rowP);
      real BselfLarr[3] = {BselfL3.x, BselfL3.y, BselfL3.z};
      real BselfRarr[3] = {BselfR3.x, BselfR3.y, BselfR3.z};
      real BpmRarr[3] = {BpmR3.x, BpmR3.y, BpmR3.z};
      real BppLarr[3] = {BppL3.x, BppL3.y, BppL3.z};

      f[p] += directSecondDerivative(u_self, u_pm, u_pp, C_self, C_pm, C_pp,
                                     BselfLarr[p], BselfRarr[p],
                                     BpmRarr[p], BppLarr[p], hs[p]);

      // ---- C12 cross terms: d/dp( C12 * d(u_j)/dj ), d/dp( C12 * d(u_k)/dk ) ----
      real g_j_self = gradU.valueAt(idx, 3 * j + j);
      real g_j_pm = gradU.valueAt(idx_pm, 3 * j + j);
      real g_j_pp = gradU.valueAt(idx_pp, 3 * j + j);
      real g_k_self = gradU.valueAt(idx, 3 * k + k);
      real g_k_pm = gradU.valueAt(idx_pm, 3 * k + k);
      real g_k_pp = gradU.valueAt(idx_pp, 3 * k + k);

      real C12_self = C12.valueAt(idx);
      real C12_pm = C12.valueAt(idx_pm);
      real C12_pp = C12.valueAt(idx_pp);

      f[p] += directMixedDerivative(g_j_self, g_j_pm, g_j_pp,
                                    C12_self, C12_pm, C12_pp, hs[p]);
      f[p] += directMixedDerivative(g_k_self, g_k_pm, g_k_pp,
                                    C12_self, C12_pm, C12_pp, hs[p]);
    }

    // ---- C44 "self" term: d/dj( C44 * d(u_p)/dj ) ----
    if (interiorInDirection(system, mastergrid, coo, j)) {
      int3 coo_jm = mastergrid.wrap(coo - step_arr[j]);
      int3 coo_jp = mastergrid.wrap(coo + step_arr[j]);
      int idx_jm = grid.coord2index(coo_jm);
      int idx_jp = grid.coord2index(coo_jp);

      real u_self = u.valueAt(idx, p);
      real u_jm = u.valueAt(idx_jm, p);
      real u_jp = u.valueAt(idx_jp, p);
      real C_self = C44.valueAt(idx);
      real C_jm = C44.valueAt(idx_jm);
      real C_jp = C44.valueAt(idx_jp);

      // eigenstrain row = j (the differentiation direction), column = p
      int3 rowJ = gradRowComps(j);
      real3 BselfL3 = sigmaFaceL.vectorAt(idx, rowJ);
      real3 BselfR3 = sigmaFaceR.vectorAt(idx, rowJ);
      real3 BjmR3 = sigmaFaceR.vectorAt(coo_jm, rowJ);
      real3 BjpL3 = sigmaFaceL.vectorAt(coo_jp, rowJ);
      real BselfLarr[3] = {BselfL3.x, BselfL3.y, BselfL3.z};
      real BselfRarr[3] = {BselfR3.x, BselfR3.y, BselfR3.z};
      real BjmRarr[3] = {BjmR3.x, BjmR3.y, BjmR3.z};
      real BjpLarr[3] = {BjpL3.x, BjpL3.y, BjpL3.z};

      f[p] += directSecondDerivative(u_self, u_jm, u_jp, C_self, C_jm, C_jp,
                                     BselfLarr[p], BselfRarr[p],
                                     BjmRarr[p], BjpLarr[p], hs[j]);

      // ---- C44 "cross" term: d/dj( C44 * d(u_j)/dp ) ----
      real g_self = gradU.valueAt(idx, 3 * p + j);   // d(u_j)/dp at self
      real g_jm = gradU.valueAt(idx_jm, 3 * p + j);
      real g_jp = gradU.valueAt(idx_jp, 3 * p + j);

      f[p] += directMixedDerivative(g_self, g_jm, g_jp, C_self, C_jm, C_jp, hs[j]);
    }

    // ---- C44 "self" term: d/dk( C44 * d(u_p)/dk ) ----
    if (interiorInDirection(system, mastergrid, coo, k)) {
      int3 coo_km = mastergrid.wrap(coo - step_arr[k]);
      int3 coo_kp = mastergrid.wrap(coo + step_arr[k]);
      int idx_km = grid.coord2index(coo_km);
      int idx_kp = grid.coord2index(coo_kp);

      real u_self = u.valueAt(idx, p);
      real u_km = u.valueAt(idx_km, p);
      real u_kp = u.valueAt(idx_kp, p);
      real C_self = C44.valueAt(idx);
      real C_km = C44.valueAt(idx_km);
      real C_kp = C44.valueAt(idx_kp);

      int3 rowK = gradRowComps(k);
      real3 BselfL3 = sigmaFaceL.vectorAt(idx, rowK);
      real3 BselfR3 = sigmaFaceR.vectorAt(idx, rowK);
      real3 BkmR3 = sigmaFaceR.vectorAt(coo_km, rowK);
      real3 BkpL3 = sigmaFaceL.vectorAt(coo_kp, rowK);
      real BselfLarr[3] = {BselfL3.x, BselfL3.y, BselfL3.z};
      real BselfRarr[3] = {BselfR3.x, BselfR3.y, BselfR3.z};
      real BkmRarr[3] = {BkmR3.x, BkmR3.y, BkmR3.z};
      real BkpLarr[3] = {BkpL3.x, BkpL3.y, BkpL3.z};

      f[p] += directSecondDerivative(u_self, u_km, u_kp, C_self, C_km, C_kp,
                                     BselfLarr[p], BselfRarr[p],
                                     BkmRarr[p], BkpLarr[p], hs[k]);

      // ---- C44 "cross" term: d/dk( C44 * d(u_k)/dp ) ----
      real g_self = gradU.valueAt(idx, 3 * p + k);   // d(u_k)/dp at self
      real g_km = gradU.valueAt(idx_km, 3 * p + k);
      real g_kp = gradU.valueAt(idx_kp, 3 * p + k);

      f[p] += directMixedDerivative(g_self, g_km, g_kp, C_self, C_km, C_kp, hs[k]);
    }
  }

  fField.setVectorInCell(idx, real3{f[0], f[1], f[2]});
}

/**
 * Numerical divergence of stress with central five-point stencil, used ONLY
 * for the viscous-damping stress (eta11/eta12/eta44), which has no magnum.np
 * analogue and is therefore left as a plain (non-jump-aware) divergence of the
 * precomputed viscous stress field, exactly as before this port.
 *
 * `includeTraction` controls whether this kernel also applies the outer-
 * boundary traction condition. It must be true only when this is the SOLE
 * contribution to the outer-boundary force (i.e. elastodynamics is disabled or
 * assured-zero), and false whenever k_elasticBoundaryTraction is also being
 * launched for the same evaluation -- otherwise the boundary traction would be
 * added twice (once here, once there) at cells adjacent to the outer geometry
 * boundary. See evalInternalBodyForce for how this is decided.
*/
__global__ void k_viscousBodyForce(CuField fField,
                                   const CuField stressTensor,
                                   const CuBoundaryTraction traction,
                                   const real3 w,
                                   const Grid mastergrid,
                                   const bool includeTraction) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = fField.system;
  const Grid grid = system.grid;

  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      fField.setVectorInCell(idx, real3{0, 0, 0});
    }
    return;
  }

  const real tf = includeTraction ? real(1) : real(0);
  const real ws[3] = {w.x, w.y, w.z};
  const int3 im2_arr[3] = {int3{-2, 0, 0}, int3{0,-2, 0}, int3{0, 0,-2}};
  const int3 im1_arr[3] = {int3{-1, 0, 0}, int3{0,-1, 0}, int3{0, 0,-1}};
  const int3 ip1_arr[3] = {int3{ 1, 0, 0}, int3{0, 1, 0}, int3{0, 0, 1}};
  const int3 ip2_arr[3] = {int3{ 2, 0, 0}, int3{0, 2, 0}, int3{0, 0, 2}};
  const int3 coo = grid.index2coord(idx);

  real3 f = {0, 0, 0};
  for (int i = 0; i < 3; i++) {
    int3 stressRow = tensorRowComps(i);

    int3 im2 = im2_arr[i], im1 = im1_arr[i];
    int3 ip1 = ip1_arr[i], ip2 = ip2_arr[i];

    int3 coo_im2 = mastergrid.wrap(coo + im2);
    int3 coo_im1 = mastergrid.wrap(coo + im1);
    int3 coo_ip1 = mastergrid.wrap(coo + ip1);
    int3 coo_ip2 = mastergrid.wrap(coo + ip2);

    bool im2_inGeo = system.inGeometry(coo_im2);
    bool im1_inGeo = system.inGeometry(coo_im1);
    bool ip1_inGeo = system.inGeometry(coo_ip1);
    bool ip2_inGeo = system.inGeometry(coo_ip2);

    if (!im1_inGeo && !ip1_inGeo) {
      f += ws[i] * tf * (traction.getSide(i, 1).vectorAt(idx)
                    + traction.getSide(i, -1).vectorAt(idx));
    } else if (!im1_inGeo) {
      f += ws[i] * (
        4./3. * tf * traction.getSide(i, -1).vectorAt(idx)
        + stressTensor.vectorAt(idx, stressRow)
        + 1./3. * stressTensor.vectorAt(coo_ip1, stressRow)
      );
    } else if (!ip1_inGeo) {
      f += ws[i] * (
        - 1./3. * stressTensor.vectorAt(coo_im1, stressRow)
        - stressTensor.vectorAt(idx, stressRow)
        + 4./3. * tf * traction.getSide(i, 1).vectorAt(idx)
      );
    } else if (!im2_inGeo || !ip2_inGeo) {
      f += 0.5*ws[i] * (stressTensor.vectorAt(coo_ip1, stressRow) -
                        stressTensor.vectorAt(coo_im1, stressRow));
    } else {
      f += ws[i] * ((4./6.) * (stressTensor.vectorAt(coo_ip1, stressRow) -
                               stressTensor.vectorAt(coo_im1, stressRow)) +
                    (1./12.)* (stressTensor.vectorAt(coo_im2, stressRow) -
                               stressTensor.vectorAt(coo_ip2, stressRow)));
    }
  }

  fField.setVectorInCell(idx, f);
}

// ============================================================
// Outer-boundary elastic traction contribution (magnum-consistent port)
// ============================================================
//
// Direct port of magnum.np's dpd() free-surface boundary block combined with
// _update_neumann_bcs's natural (zero-traction, or user-set boundaryTraction)
// Neumann-plane construction, specialized to the boundary_nodes=1 /
// non-second-order branch (see the "still open" notes below evalInternalBodyForce
// for what boundary_nodes=2,3 would additionally require -- NOT yet ported).
//
// For each of the 3 directions i, if a cell is adjacent to the outer geometry
// boundary in direction i (interiorInDirection(...)==false for that i -- the
// SAME per-direction check k_elasticForceDirect uses to decide which terms to
// skip), this cell's boundary-normal-i Neumann plane contributes THREE force
// corrections (magnum's g_i/g_j/g_k), one per force component, all built from
// the SAME one-sided formula:
//
//   f_c_contribution = sign_i * (t_val_c - sigma_ic(self)) / h_i
//
// where sign_i = +1 for the +i face (t_sign in magnum), -1 for the -i face;
// t_val_c is the c-th component of the applied traction on that face
// (CuBoundaryTraction::getSide(i, sign_i), a full real3 -- ALL three traction
// components matter, not just the direction-i one, since off-diagonal/shear
// tractions drive g_j/g_k); sigma_ic(self) is stressTensor's row-i, column-c
// Voigt component evaluated at THIS cell (there is no virtual/ghost node past
// the boundary in mumax+'s convention -- the last cell inside the geometry IS
// the boundary node, matching magnum's boundary_nodes=1). stressTensor must be
// the ELASTIC-ONLY stress (C:eps - sigma_mel); evalInternalBodyForce performs
// that subtraction before calling this kernel.
//
// tensorRowComps(i) already returns [sigma_i0, sigma_i1, sigma_i2] in Voigt
// form, i.e. its c-th entry IS sigma_ic -- so all three of g_i/g_j/g_k share
// one vectorAt call and one component-wise formula; no separate "shear"
// indexing is needed here (tensorComp is symmetric: sigma_ic == sigma_ci).
//
// CORNER / MULTI-DIRECTION AVERAGING: unlike a simple sum, magnum's ft_weight
// mechanism AVERAGES contributions when more than one Neumann plane writes to
// the same (cell, force component) -- e.g. at a staircase corner of a curved
// geometry (very common on this ring geometry) where two different directions
// i can both be non-interior at the same cell. This kernel accumulates a
// per-component sum and a per-component count, then divides at the end,
// exactly mirroring magnum's `f_ij[bc_mask] = ft[bc_mask] / ft_weigth[bc_mask]`.
// The old kernel's implicit "just sum both directions" behavior was WRONG at
// any such corner cell (double-counts rather than averages); this is fixed.
//
// Because k_elasticForceDirect only ever adds contributions for (component p,
// direction) pairs where interiorInDirection is true, and this kernel only
// ever contributes at (component c, direction i) pairs where it is false, the
// two remain disjoint per (component, direction) and sum with no double count
// -- the corner-averaging above is ACROSS the (possibly multiple) directions
// that are simultaneously non-interior at one cell, which is a separate axis
// from the k_elasticForceDirect/k_elasticBoundaryTraction split.
//
// This kernel is the ONLY source of the elastic outer-boundary contribution;
// k_viscousBodyForce's OWN traction application must stay disabled whenever
// this kernel also runs (see evalInternalBodyForce's includeTraction logic),
// to avoid applying the same boundary traction twice.
__global__ void k_elasticBoundaryTraction(CuField fField,
                                          const CuField stressTensor,  // 6-comp Voigt, ELASTIC-ONLY
                                          const CuBoundaryTraction traction,
                                          const real3 w,
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

  const real hs[3] = {1 / w.x, 1 / w.y, 1 / w.z};
  const int3 im1_arr[3] = {int3{-1, 0, 0}, int3{0,-1, 0}, int3{0, 0,-1}};
  const int3 ip1_arr[3] = {int3{ 1, 0, 0}, int3{0, 1, 0}, int3{0, 0, 1}};
  const int3 coo = grid.index2coord(idx);

  real fSum[3] = {0, 0, 0};
  int fCount[3] = {0, 0, 0};

  for (int i = 0; i < 3; i++) {
    int3 stressRow = tensorRowComps(i);  // [sigma_i0, sigma_i1, sigma_i2]
    int3 coo_im1 = mastergrid.wrap(coo + im1_arr[i]);
    int3 coo_ip1 = mastergrid.wrap(coo + ip1_arr[i]);
    bool im1_inGeo = system.inGeometry(coo_im1);
    bool ip1_inGeo = system.inGeometry(coo_ip1);

    const real3 sBdr3 = stressTensor.vectorAt(idx, stressRow);
    const real sBdr[3] = {sBdr3.x, sBdr3.y, sBdr3.z};

    if (!ip1_inGeo) {
      // +i face is a Neumann plane: sign_i = +1
      const real3 tVal3 = traction.getSide(i, 1).vectorAt(idx);
      const real tVal[3] = {tVal3.x, tVal3.y, tVal3.z};
#pragma unroll
      for (int c = 0; c < 3; c++) {
        fSum[c] += (tVal[c] - sBdr[c]) / hs[i];
        fCount[c] += 1;
      }
    }
    if (!im1_inGeo) {
      // -i face is a Neumann plane: sign_i = -1
      const real3 tVal3 = traction.getSide(i, -1).vectorAt(idx);
      const real tVal[3] = {tVal3.x, tVal3.y, tVal3.z};
#pragma unroll
      for (int c = 0; c < 3; c++) {
        fSum[c] += -(tVal[c] - sBdr[c]) / hs[i];
        fCount[c] += 1;
      }
    }
    // else (both interior in direction i): no boundary contribution from
    // this direction; k_elasticForceDirect handles it entirely.
  }

  real3 f;
  f.x = (fCount[0] > 0) ? fSum[0] / fCount[0] : 0;
  f.y = (fCount[1] > 0) ? fSum[1] / fCount[1] : 0;
  f.z = (fCount[2] > 0) ? fSum[2] / fCount[2] : 0;

  fField.setVectorInCell(idx, f);
}

Field evalInternalBodyForce(const Magnet* magnet) {
  Field fField(magnet->system(), 3);
  fField.makeZero();
  if (stressTensorAssuredZero(magnet)) {
    return fField;
  }

  int ncells = fField.grid().ncells();
  real3 w = 1. / magnet->cellsize();
  Grid mastergrid = magnet->world()->mastergrid();
  CuBoundaryTraction traction = magnet->boundaryTraction.cu();

  // Whether the elastic (C11/C12/C44) contribution -- and with it,
  // k_elasticBoundaryTraction's own traction application -- is present this
  // evaluation. If it's assured zero, the viscous kernel below is the only
  // source of the outer-boundary force and must apply the traction itself.
  const bool elasticPresent = !elasticityAssuredZero(magnet);

  // ---- elastic part: direct jump-aware derivatives of u (interior) ----
  //      + traction-BC boundary contribution (outer edge)
  if (elasticPresent) {
    CuField u = magnet->elasticDisplacement()->field().cu();
    CuParameter C11 = magnet->C11.cu();
    CuParameter C12 = magnet->C12.cu();
    CuParameter C44 = magnet->C44.cu();

    debugFieldStats(u, ncells, "u (elasticDisplacement, input)");

    Field gradU = evalDisplacementGradientTensor(magnet);
    debugFieldStats(gradU.cu(), ncells, "gradU (evalDisplacementGradientTensor)");
    FaceEigenstrainTraction fe = evalFaceEigenstrainTraction(magnet);
    debugFieldStats(fe.faceL.cu(), ncells, "sigmaFaceL");
    debugFieldStats(fe.faceR.cu(), ncells, "sigmaFaceR");

    Field elasticForce(magnet->system(), 3);
    cudaLaunch(ncells, k_elasticForceDirect, elasticForce.cu(), u, gradU.cu(),
               fe.faceL.cu(), fe.faceR.cu(), C11, C12, C44, w, mastergrid);
    debugFieldStats(elasticForce.cu(), ncells, "elasticForce (k_elasticForceDirect)");
    debugFieldSum(elasticForce, "elasticForce (k_elasticForceDirect)");

    // The free-surface traction condition must use the ELASTIC-ONLY stress
    // (C:eps_total - sigma_mel), not the total stress: the magnetoelastic
    // eigenstrain is an internal material effect, not an externally applied
    // load, so it must be subtracted before imposing a zero-traction outer
    // boundary condition. Mirrors magnum.np's dpd():
    //   sig_ii -= sig_m[...,:3]; sig_ij -= sig_m[...,3:]
    // Deliberately done here (not inside evalElasticStress itself), since
    // evalElasticStress also backs the user-facing stress_tensor quantity,
    // which should keep reporting the total (elastic + eigenstrain) stress.
    Field elasticStress = evalElasticStress(magnet);
    debugFieldStats(elasticStress.cu(), ncells, "elasticStress (before sigMel subtraction)");
    Field sigMel = evalMagnetoelasticStressVoigt(magnet);
    debugFieldStats(sigMel.cu(), ncells, "sigMel (evalMagnetoelasticStressVoigt)");
    elasticStress -= sigMel;
    debugFieldStats(elasticStress.cu(), ncells, "elasticStress (after sigMel subtraction)");

    Field boundaryForce(magnet->system(), 3);
    cudaLaunch(ncells, k_elasticBoundaryTraction, boundaryForce.cu(),
               elasticStress.cu(), traction, w, mastergrid);
    debugFieldStats(boundaryForce.cu(), ncells, "boundaryForce (k_elasticBoundaryTraction)");
    debugFieldSum(boundaryForce, "boundaryForce (k_elasticBoundaryTraction)");

    fField += elasticForce;
    fField += boundaryForce;
  }

  // ---- viscous part: unchanged, plain divergence of the viscous stress ----
  //      (no magnum.np analogue; interior AND boundary both use the original
  //      combined formula since there is no jump-aware alternative to port).
  //      Only apply this kernel's own traction term when the elastic part
  //      above did NOT already supply the outer-boundary traction.
  if (!viscousDampingAssuredZero(magnet)) {
    Field viscousStress = evalViscousStress(magnet);
    debugFieldStats(viscousStress.cu(), ncells, "viscousStress (evalViscousStress)");

    Field viscousForce(magnet->system(), 3);
    const bool viscousIncludesTraction = !elasticPresent;
    cudaLaunch(ncells, k_viscousBodyForce, viscousForce.cu(), viscousStress.cu(),
               traction, w, mastergrid, viscousIncludesTraction);
    debugFieldStats(viscousForce.cu(), ncells, "viscousForce (k_viscousBodyForce)");
    debugFieldSum(viscousForce, "viscousForce (k_viscousBodyForce)");

    fField += viscousForce;
  }

  debugFieldStats(fField.cu(), ncells, "fField (evalInternalBodyForce, final)");
  debugFieldSum(fField, "fField (evalInternalBodyForce, final)");

  return fField;
}

M_FieldQuantity internalBodyForceQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalInternalBodyForce, 3, "internal_body_force", "N/m3");
}
