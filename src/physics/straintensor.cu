#include "cudalaunch.hpp"
#include "derivativestencil.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "magnet.hpp"
#include "magnetoelasticfield.hpp"  // magnetoelasticAssuredZero
#include "parameter.hpp"
#include "straintensor.hpp"
#include "stresstensor.hpp"
#include "traction.hpp"

bool strainTensorAssuredZero(const Magnet* magnet) {
  return !magnet->enableElastodynamics();
}

// ============================================================
// Face-extrapolated magnetoelastic eigenstrain traction
// ============================================================
//
// For a single Ferromagnet sublattice, computes the row-i magnetoelastic stress
// sigma_mel_ic = (c==i ? B1 : B2) * m_i * m_c  at each cell's -i face ("L") and
// +i face ("R"), using m linearly extrapolated from the cell center to that
// face via the cell's own bulk gradient of m (matching magnum.np's
// mxl/mxr/myl/... construction in strain.py's epsilon()).
//
// Sign: B = +sigma_mel, with f_i = d_j sigma_mel_ij, sigma_mel_ii = B1 m_i^2,
// sigma_mel_ij = B2 m_i m_j (as in magnetoelasticforce.cu). This is the same
// number as magnum.np's offset B = -sigma_m, because sigma_mel = -sigma_m (= -C
// eps_m): there is no sign flip. The sign matters wherever sigma_mel enters a
// face condition -- at free surfaces as well as across magnetic/non-magnetic or
// B-contrast interfaces.
//
// Output layout (9 components each): faceL/faceR[3*i+c] = +sigma_mel_ic
// evaluated at this cell's -i / +i face respectively.
__global__ void k_faceEigenstrainTraction(CuField faceL,
                                          CuField faceR,
                                          const CuField m,
                                          const CuParameter B1,
                                          const CuParameter B2,
                                          const real3 w,
                                          const Grid mastergrid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = faceL.system;
  const Grid gridLocal = system.grid;

  if (!system.inGeometry(idx)) {
    if (gridLocal.cellInGrid(idx)) {
      for (int c = 0; c < faceL.ncomp; c++) {
        faceL.setValueInCell(idx, c, 0);
        faceR.setValueInCell(idx, c, 0);
      }
    }
    return;
  }

  const int3 coo = gridLocal.index2coord(idx);
  real gradM[3][3];
  deviceGradientTensor(gradM, m, system, mastergrid, coo, w);

  const real3 m0 = m.vectorAt(idx);
  const real m0arr[3] = {m0.x, m0.y, m0.z};
  const real b1 = B1.valueAt(idx);
  const real b2 = B2.valueAt(idx);
  const real hs[3] = {1 / w.x, 1 / w.y, 1 / w.z};  // cellsize components

#pragma unroll
  for (int i = 0; i < 3; i++) {
    // linear extrapolation of m from cell center to the -i / +i half-face,
    // using this cell's own d(m)/dx_i row
    real mL[3], mR[3];
#pragma unroll
    for (int c = 0; c < 3; c++) {
      real half = 0.5 * hs[i] * gradM[i][c];
      mL[c] = m0arr[c] - half;
      mR[c] = m0arr[c] + half;
    }

#pragma unroll
    for (int c = 0; c < 3; c++) {
      real coef = (c == i) ? b1 : b2;
      real sigL = coef * mL[i] * mL[c];
      real sigR = coef * mR[i] * mR[c];
      // NOT negated -- see the sign note above the kernel.
      faceL.setValueInCell(idx, 3 * i + c, sigL);
      faceR.setValueInCell(idx, 3 * i + c, sigR);
    }
  }
}

// ============================================================
// symmetrize gradient tensor into the 6-component Voigt strain
// ============================================================

__global__ void k_symmetrizeStrain(CuField strain, const CuField grad) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = strain.system;

  if (!system.inGeometry(idx)) {
    if (system.grid.cellInGrid(idx)) {
      for (int c = 0; c < strain.ncomp; c++)
        strain.setValueInCell(idx, c, 0);
    }
    return;
  }

  real g[3][3];
#pragma unroll
  for (int i = 0; i < 3; i++)
#pragma unroll
    for (int c = 0; c < 3; c++)
      g[i][c] = grad.valueAt(idx, 3 * i + c);

  for (int i = 0; i < 3; i++) {
    for (int j = i; j < 3; j++) {
      if (i == j) {
        strain.setValueInCell(idx, i, g[i][j]);
      } else {
        strain.setValueInCell(idx, i + j + 2, 0.5 * (g[i][j] + g[j][i]));
      }
    }
  }
}

// ============================================================
// Host-side orchestration
// ============================================================

// Computes and accumulates (via Field::operator+=) the face-extrapolated
// magnetoelastic eigenstrain traction of one Ferromagnet sublattice into
// faceL/faceR, mirroring the sublattice-summation pattern used for
// evalMagnetoelasticForce in elastodynamics.cu's evalEffectiveBodyForce.
static void accumulateFaceEigenstrain(const Ferromagnet* fm,
                                      int ncells,
                                      real3 w,
                                      const Grid& mastergrid,
                                      Field& faceL,
                                      Field& faceR) {
  if (magnetoelasticAssuredZero(fm))
    return;

  Field thisFaceL(fm->system(), 9);
  Field thisFaceR(fm->system(), 9);
  CuField m = fm->magnetization()->field().cu();
  CuParameter B1 = fm->B1.cu();
  CuParameter B2 = fm->B2.cu();

  cudaLaunch(ncells, k_faceEigenstrainTraction, thisFaceL.cu(), thisFaceR.cu(),
             m, B1, B2, w, mastergrid);

  faceL += thisFaceL;
  faceR += thisFaceR;
}

FaceEigenstrainTraction evalFaceEigenstrainTraction(const Magnet* magnet) {
  Field faceL(magnet->system(), 9);
  Field faceR(magnet->system(), 9);
  faceL.makeZero();
  faceR.makeZero();

  int ncells = magnet->grid().ncells();
  real3 w = 1 / magnet->cellsize();
  Grid mastergrid = magnet->world()->mastergrid();

  if (auto host = magnet->asHost()) {
    for (const Ferromagnet* sub : host->sublattices())
      accumulateFaceEigenstrain(sub, ncells, w, mastergrid, faceL, faceR);
  } else if (const Ferromagnet* fm = magnet->asFM()) {
    accumulateFaceEigenstrain(fm, ncells, w, mastergrid, faceL, faceR);
  }

  return {faceL, faceR};
}

// ============================================================
// magnum.np-style jump conditions with voids as C = 0
// ============================================================

// Face offsets B = sigma_mel (face-extrapolated) + B_eps, B_eps = Cc *
// (C-weighted mean of the tangential gradient of the two cells sharing the
// face). A non-solid neighbour has Cc = 0.
__global__ void k_buildJumpOffsets(CuField BL,
                                   CuField BR,
                                   const CuField sigL,
                                   const CuField sigR,
                                   const CuField gu,
                                   const bool useSigma,
                                   const bool useEps,
                                   const CuParameter C12,
                                   const CuParameter C44,
                                   const Grid mastergrid,
                                   const int3 thin) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = BL.system;
  const Grid grid = system.grid;

  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      for (int c = 0; c < BL.ncomp; c++) {
        BL.setValueInCell(idx, c, 0);
        BR.setValueInCell(idx, c, 0);
      }
    }
    return;
  }

  const int3 coo = grid.index2coord(idx);
  const int thinArr[3] = {thin.x, thin.y, thin.z};
  const int3 step_arr[3] = {int3{1, 0, 0}, int3{0, 1, 0}, int3{0, 0, 1}};

  for (int d = 0; d < 3; d++) {
    const int3 coo_m = mastergrid.wrap(coo - step_arr[d]);
    const int3 coo_p = mastergrid.wrap(coo + step_arr[d]);
    const bool solidM = system.inGeometry(coo_m);
    const bool solidP = system.inGeometry(coo_p);
    const int idx_m = solidM ? grid.coord2index(coo_m) : idx;
    const int idx_p = solidP ? grid.coord2index(coo_p) : idx;

    for (int k = 0; k < 3; k++) {
      real bl = useSigma ? sigL.valueAt(idx, 3 * d + k) : real(0);
      real br = useSigma ? sigR.valueAt(idx, 3 * d + k) : real(0);

      if (useEps && !thinArr[d]) {
        const CuParameter& Cc = (k == d) ? C12 : C44;
        const real Cs = Cc.valueAt(idx);
        const real gs = tangentialGrad(gu, idx, d, k);

        real ghat = gs;  // +d face
        if (solidP) {
          const real Cn = Cc.valueAt(idx_p);
          const real den = Cs + Cn;
          if (den > 0)
            ghat = (Cs * gs + Cn * tangentialGrad(gu, idx_p, d, k)) / den;
        }
        br += Cs * ghat;

        ghat = gs;  // -d face
        if (solidM) {
          const real Cn = Cc.valueAt(idx_m);
          const real den = Cs + Cn;
          if (den > 0)
            ghat = (Cs * gs + Cn * tangentialGrad(gu, idx_m, d, k)) / den;
        }
        bl += Cs * ghat;
      }

      BL.setValueInCell(idx, 3 * d + k, bl);
      BR.setValueInCell(idx, 3 * d + k, br);
    }
  }
}

// gu[3*d+k] = 1/2 (fwd + bwd). Solid neighbour: magnum's two-point jump
// formula. Non-solid neighbour (void or grid edge): C*grad + B = S, with face
// stress S = sense * traction (S = 0 for a free surface).
__global__ void k_gradientJump(CuField gu,
                               const CuField u,
                               const CuField BL,
                               const CuField BR,
                               const CuParameter C11,
                               const CuParameter C44,
                               const CuBoundaryTraction traction,
                               const bool useTraction,
                               const real3 w,
                               const Grid mastergrid,
                               const int3 thin) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = gu.system;
  const Grid grid = system.grid;

  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      for (int c = 0; c < gu.ncomp; c++)
        gu.setValueInCell(idx, c, 0);
    }
    return;
  }

  const int3 coo = grid.index2coord(idx);
  const real hs[3] = {1 / w.x, 1 / w.y, 1 / w.z};
  const int thinArr[3] = {thin.x, thin.y, thin.z};
  const int3 step_arr[3] = {int3{1, 0, 0}, int3{0, 1, 0}, int3{0, 0, 1}};

  for (int d = 0; d < 3; d++) {
    if (thinArr[d]) {
      for (int k = 0; k < 3; k++)
        gu.setValueInCell(idx, 3 * d + k, 0);
      continue;
    }

    const int3 coo_m = mastergrid.wrap(coo - step_arr[d]);
    const int3 coo_p = mastergrid.wrap(coo + step_arr[d]);
    const bool solidM = system.inGeometry(coo_m);
    const bool solidP = system.inGeometry(coo_p);
    const int idx_m = solidM ? grid.coord2index(coo_m) : idx;
    const int idx_p = solidP ? grid.coord2index(coo_p) : idx;
    const real h = hs[d];
    const real3 tP =
        useTraction ? traction.getSide(d, 1).vectorAt(idx) : real3{0, 0, 0};
    const real3 tM =
        useTraction ? traction.getSide(d, -1).vectorAt(idx) : real3{0, 0, 0};

    for (int k = 0; k < 3; k++) {
      const CuParameter& Cpar = (k == d) ? C11 : C44;
      const real C0 = Cpar.valueAt(idx);
      const real us = u.valueAt(idx, k);
      const real BRs = BR.valueAt(idx, 3 * d + k);
      const real BLs = BL.valueAt(idx, 3 * d + k);

      real fwd = 0, bwd = 0;
      if (solidP) {
        const real Cn = Cpar.valueAt(idx_p);
        const real D = C0 + Cn;
        if (D > 0)
          fwd = (2 * Cn * (u.valueAt(idx_p, k) - us) +
                 h * (BL.valueAt(idx_p, 3 * d + k) - BRs)) /
                (h * D);
      } else if (C0 > 0) {
        fwd = (comp3(tP, k) - BRs) / C0;
      }
      if (solidM) {
        const real Cn = Cpar.valueAt(idx_m);
        const real D = Cn + C0;
        if (D > 0)
          bwd = (2 * Cn * (us - u.valueAt(idx_m, k)) -
                 h * (BLs - BR.valueAt(idx_m, 3 * d + k))) /
                (h * D);
      } else if (C0 > 0) {
        bwd = (-comp3(tM, k) - BLs) / C0;
      }

      gu.setValueInCell(idx, 3 * d + k, 0.5 * (fwd + bwd));
    }
  }
}

static ElasticJumpData evalJumpDataImpl(
    const Magnet* magnet,
    const CuField& field,
    const CuParameter& C11,
    const CuParameter& C12,
    const CuParameter& C44,
    FaceEigenstrainTraction* sig,  // nullptr: no sigma_mel
    bool useTraction,
    int depth) {
  const int ncells = magnet->grid().ncells();
  const real3 w = 1 / magnet->cellsize();
  const Grid mastergrid = magnet->world()->mastergrid();
  const int3 thin = thinAxes(magnet->grid(), mastergrid);
  CuBoundaryTraction traction = magnet->boundaryTraction.cu();

  Field gu(magnet->system(), 9);
  Field BL(magnet->system(), 9);
  Field BR(magnet->system(), 9);

  const bool useSigma = (sig != nullptr);
  CuField sigL = useSigma ? sig->faceL.cu() : gu.cu();  // dummy when unused
  CuField sigR = useSigma ? sig->faceR.cu() : gu.cu();
  const bool noEps = false, withEps = true;

  // pass 0: offsets = sigma_mel only
  cudaLaunch(ncells, k_buildJumpOffsets, BL.cu(), BR.cu(), sigL, sigR, gu.cu(),
             useSigma, noEps, C12, C44, mastergrid, thin);
  cudaLaunch(ncells, k_gradientJump, gu.cu(), field, BL.cu(), BR.cu(), C11, C44,
             traction, useTraction, w, mastergrid, thin);

  // iterations: B_eps from the previous gradient, then a new gradient
  for (int it = 0; it < depth; it++) {
    cudaLaunch(ncells, k_buildJumpOffsets, BL.cu(), BR.cu(), sigL, sigR,
               gu.cu(), useSigma, withEps, C12, C44, mastergrid, thin);
    cudaLaunch(ncells, k_gradientJump, gu.cu(), field, BL.cu(), BR.cu(), C11,
               C44, traction, useTraction, w, mastergrid, thin);
  }

  return {gu, BL, BR};
}

ElasticJumpData evalElasticJumpData(const Magnet* magnet) {
  CuField u = magnet->elasticDisplacement()->field().cu();
  FaceEigenstrainTraction sig = evalFaceEigenstrainTraction(magnet);
  return evalJumpDataImpl(magnet, u, magnet->C11.cu(), magnet->C12.cu(),
                          magnet->C44.cu(), &sig, /*useTraction=*/true,
                          ELASTIC_JUMP_DEPTH);
}

ElasticJumpData evalViscousJumpData(const Magnet* magnet,
                                    bool useTensor,
                                    bool useTraction) {
  CuField v = magnet->elasticVelocity()->field().cu();
  if (useTensor)
    return evalJumpDataImpl(magnet, v, magnet->eta11.cu(), magnet->eta12.cu(),
                            magnet->eta44.cu(), nullptr, useTraction,
                            VISCOUS_JUMP_DEPTH);
  return evalJumpDataImpl(magnet, v, magnet->C11.cu(), magnet->C12.cu(),
                          magnet->C44.cu(), nullptr, useTraction,
                          VISCOUS_JUMP_DEPTH);
}

Field evalDisplacementGradientTensor(const Magnet* magnet) {
  if (strainTensorAssuredZero(magnet)) {
    Field grad(magnet->system(), 9);
    grad.makeZero();
    return grad;
  }
  return evalElasticJumpData(magnet).gu;
}

Field evalStrainTensor(const Magnet* magnet) {
  Field strain(magnet->system(), 6);
  if (strainTensorAssuredZero(magnet)) {
    strain.makeZero();
    return strain;
  }

  Field grad = evalDisplacementGradientTensor(magnet);
  int ncells = strain.grid().ncells();
  cudaLaunch(ncells, k_symmetrizeStrain, strain.cu(), grad.cu());
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
  const bool useTensor = !viscousDampingTensorAssuredZero(magnet);
  Field grad = evalViscousJumpData(magnet, useTensor, /*useTraction=*/false).gu;
  cudaLaunch(strainRate.grid().ncells(), k_symmetrizeStrain, strainRate.cu(),
             grad.cu());
  return strainRate;
}

M_FieldQuantity strainRateQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalStrainRate, 6, "strain_rate", "1/s");
}
