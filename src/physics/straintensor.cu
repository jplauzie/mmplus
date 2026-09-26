#include "cudalaunch.hpp"
#include "derivativestencil.hpp"
#include "ferromagnet.hpp"
#include "magnet.hpp"
#include "field.hpp"
#include "magnetoelasticfield.hpp"  // magnetoelasticAssuredZero
#include "parameter.hpp"
#include "straintensor.hpp"


bool strainTensorAssuredZero(const Magnet* magnet) {
  return !magnet->enableElastodynamics();
}

// ============================================================
// Kernel 1: bulk gradient tensor (no interior-interface jump treatment)
// ============================================================
//
// Writes the full 9-component gradient tensor of a 3-component field `f`
// (grad[3*i+c] = d f_c / d x_i) using the adaptive one-sided/central stencil.
// This is the exact stencil previously embedded directly in k_strainTensor,
// factored out via derivativestencil.hpp so it can also drive the eigenstrain
// face-extrapolation kernel below.
__global__ void k_bulkGradientTensor(CuField grad,
                                     const CuField f,
                                     const real3 w,
                                     const Grid mastergrid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = grad.system;
  const Grid gridLocal = system.grid;

  if (!system.inGeometry(idx)) {
    if (gridLocal.cellInGrid(idx)) {
      for (int c = 0; c < grad.ncomp; c++)
        grad.setValueInCell(idx, c, 0);
    }
    return;
  }

  const int3 coo = gridLocal.index2coord(idx);
  real g[3][3];
  deviceGradientTensor(g, f, system, mastergrid, coo, w);

#pragma unroll
  for (int i = 0; i < 3; i++)
#pragma unroll
    for (int c = 0; c < 3; c++)
      grad.setValueInCell(idx, 3 * i + c, g[i][c]);
}

// ============================================================
// Kernel 2: face-extrapolated magnetoelastic eigenstrain traction
// ============================================================
//
// For a single Ferromagnet sublattice, computes the row-i magnetoelastic stress
// sig_m_ic = (c==i ? B1 : B2) * m_i * m_c  at each cell's -i face ("L") and +i
// face ("R"), using m linearly extrapolated from the cell center to that face
// via the cell's own bulk gradient of m (matching magnum.np's mxl/mxr/myl/...
// construction in strain.py's epsilon()).
//
// Sign convention: mumax+'s magnetoelastic body force (k_magnetoelasticForce)
// is built as f_i = d(sigma_mel)_ij / dx_j with sigma_mel_ii = B1*m_i^2 and
// sigma_mel_ij = B2*m_i*m_j (i != j) -- i.e. the *positive* stress sigma_mel is
// what already appears, additively, in the divergence that produces the body
// force. The interior-interface jump formula in Kernel 3 / directSecondDerivative
// assumes a flux of the form C*grad(u) + B that is continuous across a face, so
// the offset here must be B = +sigma_mel_ic (NOT negated). This differs from
// magnum.np's own internal convention, where the corresponding offset is defined
// as B = -sigma_m because magnum's flux is built as C*grad(u) - sigma_m; the sign
// flip here is required to match mumax+'s own (opposite) force-kernel convention,
// not a discrepancy with magnum.
//
// Output layout (9 components each): faceL/faceR[3*i+c] = +sig_m_ic evaluated
// at this cell's -i / +i face respectively.
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
      // NOT negated -- see sign-convention note above the kernel.
      faceL.setValueInCell(idx, 3 * i + c, sigL);
      faceR.setValueInCell(idx, 3 * i + c, sigR);
    }
  }
}

// ============================================================
// Kernel 2b: cell-centered magnetoelastic stress (Voigt notation)
// ============================================================
//
// Companion to Kernel 2, but evaluated at the cell center (no face
// extrapolation) and written directly in 6-component Voigt order, matching
// the layout k_elasticStress/evalElasticStress already use for the elastic
// stress tensor: [xx, yy, zz, yz, xz, xy]. Same sign convention as
// k_faceEigenstrainTraction (see its comment) -- i.e. this is +sigma_mel, not
// negated.
__global__ void k_magnetoelasticStressVoigt(CuField sigMel,
                                            const CuField m,
                                            const CuParameter B1,
                                            const CuParameter B2) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = sigMel.system;

  if (!system.inGeometry(idx)) {
    if (system.grid.cellInGrid(idx)) {
      for (int c = 0; c < sigMel.ncomp; c++)
        sigMel.setValueInCell(idx, c, 0);
    }
    return;
  }

  const real3 m0 = m.vectorAt(idx);
  const real marr[3] = {m0.x, m0.y, m0.z};
  const real b1 = B1.valueAt(idx);
  const real b2 = B2.valueAt(idx);

  sigMel.setValueInCell(idx, 0, b1 * marr[0] * marr[0]);         // xx
  sigMel.setValueInCell(idx, 1, b1 * marr[1] * marr[1]);         // yy
  sigMel.setValueInCell(idx, 2, b1 * marr[2] * marr[2]);         // zz
  sigMel.setValueInCell(idx, 3, b2 * marr[1] * marr[2]);         // yz
  sigMel.setValueInCell(idx, 4, b2 * marr[0] * marr[2]);         // xz
  sigMel.setValueInCell(idx, 5, b2 * marr[0] * marr[1]);         // xy
}

// ============================================================
// Kernel 3: interior material-interface jump correction
// ============================================================
//
// Overwrites, at cells adjacent to an interior interface (i.e. not adjacent to
// the outer geometry boundary), the gradient components d(f_c)/dx_i with a
// traction-consistent two-point formula ported from magnum.np's
// first_derivative_with_jump_conditions (linear_elasticity/utils.py):
//
//   at a face between a "near" cell N and "far" cell F (uniform cellsize h_i):
//     contribution(N->this cell) = [2*C_F*(u_far - u_near) + h*(B_far - B_near)]
//                                   / (h * (C_near + C_far))
//
// with C = Cii (e.g. C11/eta11) when c == i (diagonal / normal strain), or
// Cshear (e.g. C44/eta44) when c != i (off-diagonal / shear strain), and B the
// magnetoelastic eigenstrain traction offset from Kernel 2 (see its sign-
// convention note), evaluated on the correct side of each face. The final
// derivative is the average of the -i-face and +i-face contributions.
//
// NOTE: magnum's elastic tangential-strain contribution to B (the C12/C44
// "companion derivative" term) is provably always equal on both sides of a
// face by construction in magnum's own code (see straintensor.hpp docstring
// above), so it never affects the jump and is intentionally omitted here.
//
// faceL/faceR may be all-zero fields (as used for strain rate, where there is
// no magnetoelastic eigenstrain-rate analogue); the formula degrades correctly.
//
// If both neighboring cells' weight (Cii or Cshear) are exactly zero on a given
// face (e.g. a purely eta44-damped magnet with eta11==0), that face's
// contribution is skipped and the bulk (Kernel 1) value is kept for that
// component, to avoid a division by zero.
__global__ void k_jumpCorrectGradient(CuField gradOut,
                                      const CuField gradIn,
                                      const CuField faceL,
                                      const CuField faceR,
                                      const CuField u,
                                      const CuParameter Cii,
                                      const CuParameter Cshear,
                                      const real3 w,
                                      const Grid mastergrid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = gradOut.system;
  const Grid gridLocal = system.grid;

  if (!system.inGeometry(idx)) {
    if (gridLocal.cellInGrid(idx)) {
      for (int c = 0; c < gradOut.ncomp; c++)
        gradOut.setValueInCell(idx, c, 0);
    }
    return;
  }

  const int3 coo = gridLocal.index2coord(idx);
  const real hs[3] = {1 / w.x, 1 / w.y, 1 / w.z};
  const real ws[3] = {w.x, w.y, w.z};
  const int3 step_arr[3] = {int3{1, 0, 0}, int3{0, 1, 0}, int3{0, 0, 1}};

  // start from the bulk (or previous-pass) gradient; only interior-interface
  // entries get overwritten below
  real g[3][3];
#pragma unroll
  for (int i = 0; i < 3; i++)
#pragma unroll
    for (int c = 0; c < 3; c++)
      g[i][c] = gradIn.valueAt(idx, 3 * i + c);

#pragma unroll
  for (int i = 0; i < 3; i++) {
    if (!interiorInDirection(system, mastergrid, coo, i))
      continue;  // keep the outer-boundary stencil value from Kernel 1

    int3 coo_im1 = mastergrid.wrap(coo - step_arr[i]);
    int3 coo_ip1 = mastergrid.wrap(coo + step_arr[i]);
    int idx_im1 = gridLocal.coord2index(coo_im1);
    int idx_ip1 = gridLocal.coord2index(coo_ip1);

    const real Wself_ii = Cii.valueAt(idx);
    const real Wim1_ii = Cii.valueAt(idx_im1);
    const real Wip1_ii = Cii.valueAt(idx_ip1);
    const real Wself_sh = Cshear.valueAt(idx);
    const real Wim1_sh = Cshear.valueAt(idx_im1);
    const real Wip1_sh = Cshear.valueAt(idx_ip1);

    const real3 uSelf3 = u.vectorAt(idx);
    const real3 uIm13 = u.vectorAt(coo_im1);
    const real3 uIp13 = u.vectorAt(coo_ip1);
    const real uSelf[3] = {uSelf3.x, uSelf3.y, uSelf3.z};
    const real uIm1[3] = {uIm13.x, uIm13.y, uIm13.z};
    const real uIp1[3] = {uIp13.x, uIp13.y, uIp13.z};

    // row i of the magnetoelastic traction offset (see Kernel 2 sign note):
    //   Bself_L / Bself_R : this cell's own -i / +i face value
    //   Bim1_R            : the -i neighbor's own +i face value (shared face)
    //   Bip1_L            : the +i neighbor's own -i face value (shared face)
    const int3 row = gradRowComps(i);
    const real3 Bself_L3 = faceL.vectorAt(idx, row);
    const real3 Bself_R3 = faceR.vectorAt(idx, row);
    const real3 Bim1_R3 = faceR.vectorAt(coo_im1, row);
    const real3 Bip1_L3 = faceL.vectorAt(coo_ip1, row);
    const real Bself_L[3] = {Bself_L3.x, Bself_L3.y, Bself_L3.z};
    const real Bself_R[3] = {Bself_R3.x, Bself_R3.y, Bself_R3.z};
    const real Bim1_R[3] = {Bim1_R3.x, Bim1_R3.y, Bim1_R3.z};
    const real Bip1_L[3] = {Bip1_L3.x, Bip1_L3.y, Bip1_L3.z};

#pragma unroll
    for (int c = 0; c < 3; c++) {
      const real Wself = (c == i) ? Wself_ii : Wself_sh;
      const real Wim1 = (c == i) ? Wim1_ii : Wim1_sh;
      const real Wip1 = (c == i) ? Wip1_ii : Wip1_sh;

      const real denomM = Wim1 + Wself;  // -i face
      const real denomP = Wself + Wip1;  // +i face

      real contribFromIm1, contribFromIp1;
      bool haveM = denomM > 0;
      bool haveP = denomP > 0;

      if (haveM) {
        contribFromIm1 = (2. * Wim1 * (uSelf[c] - uIm1[c]) +
                          hs[i] * (Bim1_R[c] - Bself_L[c])) / denomM;
      }
      if (haveP) {
        contribFromIp1 = (2. * Wip1 * (uIp1[c] - uSelf[c]) +
                          hs[i] * (Bip1_L[c] - Bself_R[c])) / denomP;
      }

      // NOTE: contribFromIm1/contribFromIp1 above have units of displacement
      // (they are h * derivative); multiplying by ws[i] = 1/h converts back
      // to a proper derivative, matching magnum's division by h_sum = 2*h.
      if (haveM && haveP) {
        g[i][c] = 0.5 * ws[i] * (contribFromIm1 + contribFromIp1);
      } else if (haveM) {
        g[i][c] = ws[i] * contribFromIm1;
      } else if (haveP) {
        g[i][c] = ws[i] * contribFromIp1;
      }
      // else: both faces have zero weight (e.g. eta11==0 identically);
      // leave g[i][c] at its bulk (Kernel 1 / previous-pass) value.
    }
  }

#pragma unroll
  for (int i = 0; i < 3; i++)
#pragma unroll
    for (int c = 0; c < 3; c++)
      gradOut.setValueInCell(idx, 3 * i + c, g[i][c]);
}

// ============================================================
// Kernel 4: symmetrize gradient tensor into the 6-component Voigt strain
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
static void accumulateFaceEigenstrain(const Ferromagnet* fm, int ncells, real3 w,
                                      const Grid& mastergrid, Field& faceL, Field& faceR) {
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

// Computes and accumulates the cell-centered magnetoelastic Voigt stress of
// one Ferromagnet sublattice into sigMel, mirroring accumulateFaceEigenstrain
// above and the sublattice-summation pattern used throughout for magnetoelastic
// quantities.
static void accumulateMagnetoelasticStress(const Ferromagnet* fm, int ncells,
                                           Field& sigMel) {
  if (magnetoelasticAssuredZero(fm))
    return;

  Field thisSigMel(fm->system(), 6);
  CuField m = fm->magnetization()->field().cu();
  CuParameter B1 = fm->B1.cu();
  CuParameter B2 = fm->B2.cu();

  cudaLaunch(ncells, k_magnetoelasticStressVoigt, thisSigMel.cu(), m, B1, B2);

  sigMel += thisSigMel;
}

Field evalMagnetoelasticStressVoigt(const Magnet* magnet) {
  Field sigMel(magnet->system(), 6);
  sigMel.makeZero();

  int ncells = magnet->grid().ncells();

  if (auto host = magnet->asHost()) {
    for (const Ferromagnet* sub : host->sublattices())
      accumulateMagnetoelasticStress(sub, ncells, sigMel);
  } else if (const Ferromagnet* fm = magnet->asFM()) {
    accumulateMagnetoelasticStress(fm, ncells, sigMel);
  }

  return sigMel;
}

// Shared pipeline for both evalStrainTensor/evalDisplacementGradientTensor and
// evalStrainRate:
//   1. bulk gradient tensor of `f` (u or v)
//   2. (optional) magnetoelastic eigenstrain face traction, summed over
//      sublattices; pass includeEigenstrain=false for strain rate
//   3. STRAIN_JUMP_ITERATIONS passes of interior-interface jump correction,
//      weighted by (Cii, Cshear) = (C11,C44) or (eta11,eta44)
// Returns the corrected 9-component gradient tensor (pre-symmetrization).
static Field evalGradientLike(const Magnet* magnet, const CuField& f,
                              const CuParameter& Cii, const CuParameter& Cshear,
                              bool includeEigenstrain) {
  int ncells = magnet->grid().ncells();
  real3 w = 1 / magnet->cellsize();
  Grid mastergrid = magnet->world()->mastergrid();

  Field gradA(magnet->system(), 9);
  cudaLaunch(ncells, k_bulkGradientTensor, gradA.cu(), f, w, mastergrid);
  debugFieldStats(gradA.cu(), ncells, "gradA (k_bulkGradientTensor, bulk)");

  Field faceL(magnet->system(), 9);
  Field faceR(magnet->system(), 9);
  if (includeEigenstrain) {
    FaceEigenstrainTraction fe = evalFaceEigenstrainTraction(magnet);
    faceL = fe.faceL;
    faceR = fe.faceR;
  } else {
    faceL.makeZero();
    faceR.makeZero();
  }

  Field gradB(magnet->system(), 9);
  for (int iter = 0; iter < STRAIN_JUMP_ITERATIONS; iter++) {
    cudaLaunch(ncells, k_jumpCorrectGradient, gradB.cu(), gradA.cu(),
               faceL.cu(), faceR.cu(), f, Cii, Cshear, w, mastergrid);
    debugFieldStats(gradB.cu(), ncells, "gradB (k_jumpCorrectGradient, after pass)");
    std::swap(gradA, gradB);
  }

  return gradA;
}

Field evalDisplacementGradientTensor(const Magnet* magnet) {
  Field grad(magnet->system(), 9);
  if (strainTensorAssuredZero(magnet)) {
    grad.makeZero();
    return grad;
  }

  CuField u = magnet->elasticDisplacement()->field().cu();
  CuParameter C11 = magnet->C11.cu();
  CuParameter C44 = magnet->C44.cu();

  return evalGradientLike(magnet, u, C11, C44, /*includeEigenstrain=*/true);
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

  CuField v = magnet->elasticVelocity()->field().cu();
  CuParameter eta11 = magnet->eta11.cu();
  CuParameter eta44 = magnet->eta44.cu();

  int ncellsDbg = strainRate.grid().ncells();
  debugFieldStats(v, ncellsDbg, "v (elasticVelocity, input to strain rate)");

  // no magnetoelastic eigenstrain-rate analogue (magnum.np has none either)
  Field grad = evalGradientLike(magnet, v, eta11, eta44, /*includeEigenstrain=*/false);
  debugFieldStats(grad.cu(), ncellsDbg, "gradV (strain-rate gradient, pre-symmetrize)");
  int ncells = strainRate.grid().ncells();
  cudaLaunch(ncells, k_symmetrizeStrain, strainRate.cu(), grad.cu());
  return strainRate;
}

M_FieldQuantity strainRateQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalStrainRate, 6, "strain_rate", "1/s");
}
