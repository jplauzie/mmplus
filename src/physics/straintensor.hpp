#pragma once

#include "quantityevaluator.hpp"

class Magnet;
class Field;

// Fixed number of interior-interface jump-correction passes applied when computing the
// strain tensor / strain rate across a material discontinuity (differing C11/C12/C44 or
// eta11/eta12/eta44 between neighboring cells). Kept as a single named constant (rather
// than hard-coded inline) so it is easy to promote to a per-Magnet runtime setting later,
// mirroring magnum.np's LLGWithLESolver(iteration_depth=...).
//
// NOTE: because the elastic (C12/C44) tangential-strain contribution to the jump
// condition cancels exactly in magnum.np's own formulation (its Bl/Br arrays are built
// so that only the magnetoelastic eigenstrain term is ever asymmetric across a face),
// additional passes beyond the first are currently mathematically inert here: neither
// the conductance-weighted term nor the eigenstrain offset depends on the corrected
// gradient itself, so a single pass already reproduces magnum's converged result for
// this term. The loop is kept so this remains true only incidentally, not structurally,
// should a future refinement reintroduce genuine cross-pass coupling.
constexpr int STRAIN_JUMP_ITERATIONS = 1;

bool strainTensorAssuredZero(const Magnet*);

Field evalStrainTensor(const Magnet*);
Field evalStrainRate(const Magnet*);

// Full 9-component (unsymmetrized) displacement gradient tensor, jump-corrected at
// interior material interfaces exactly as used internally by evalStrainTensor before
// symmetrization (layout: grad[3*i+c] = d(u_c)/dx_i). Exposed so internalbodyforce.cu
// and magnetoelasticforce.cu can consume the same jump-corrected first derivatives that
// magnum.np's force terms (_mixed_derivative) read from diff_data.gradient_ud, rather
// than recomputing a second, inconsistent bulk-only gradient of their own.
Field evalDisplacementGradientTensor(const Magnet*);

// The magnetoelastic eigenstrain traction offset, face-extrapolated and summed over all
// Ferromagnet sublattices, as used both by evalStrainTensor's jump correction and by
// internalbodyforce.cu's direct-derivative eigenstrain force. faceL/faceR layout:
// [3*i+c] = +sigma_m_ic at this cell's -i / +i half-face respectively.
//
// Sign convention: this is the SAME sign as mumax+'s own magnetoelastic body force
// (magnetoelasticforce.cu's k_magnetoelasticForce computes f_i = d(sigma_mel)_ij/dx_j
// with sigma_mel_ii = B1*m_i^2, sigma_mel_ij = B2*m_i*m_j), i.e. it is NOT negated the
// way magnum.np's internal B offset is (magnum's flux is built as C*grad(u) - sigma_m,
// so its offset is B = -sigma_m; mumax+'s flux is C*grad(u) + sigma_mel, so here
// B = +sigma_mel). Getting this sign right only matters where sigma_mel itself is
// discontinuous across a face (e.g. differing B1/B2 or a magnetic/non-magnetic
// interface); it has no effect in a single, uniform magnetic material.
//
// Returns all-zero fields if magnetoelasticAssuredZero holds for every sublattice.
struct FaceEigenstrainTraction {
  Field faceL;
  Field faceR;
};
FaceEigenstrainTraction evalFaceEigenstrainTraction(const Magnet*);

// Cell-centered magnetoelastic (eigen)stress in Voigt notation (6 components:
// [sig_mel_xx, sig_mel_yy, sig_mel_zz, sig_mel_yz, sig_mel_xz, sig_mel_xy]),
// summed over all Ferromagnet sublattices, using the SAME sign convention as
// FaceEigenstrainTraction and mumax+'s own magnetoelastic body force (i.e. NOT
// negated the way magnum.np's internal sigma_m is).
//
// This is the cell-centered counterpart of evalFaceEigenstrainTraction's
// face-extrapolated values (no extrapolation here, just m at cell center), and
// exists specifically so the outer-boundary free-surface traction condition
// can subtract it from the total elastic stress before applying a zero-
// traction condition -- mirroring magnum.np's dpd(): `sig_ii -= sig_m[...,:3]`
// / `sig_ij -= sig_m[...,3:]`. The physical free surface condition is
// (C:eps_total - sigma_mel)*n = 0, not C:eps_total*n = 0, since the eigenstrain
// is an internal material effect, not an externally applied traction.
//
// Returns an all-zero field if magnetoelasticAssuredZero holds for every
// sublattice.
Field evalMagnetoelasticStressVoigt(const Magnet*);

// Strain tensor quantity with 6 symmetric strain components
// [εxx, εyy, εzz, εxy, εxz, εyz],
// calculated according to ε = 1/2 (∇u + (∇u)^T).
//
// At interior interfaces between elastic regions with differing C11/C12/C44, the normal
// and shear components of ∇u are corrected using a traction-consistent (conductance-
// weighted) two-point jump formula, ported from magnum.np's
// linear_elasticity/strain.py and linear_elasticity/utils.py (first_derivative_with_
// jump_conditions / harmonic_mean), additionally accounting for the discontinuity in
// magnetoelastic eigenstrain (from B1, B2 and the magnetization) across such interfaces.
// At the outer geometry boundary, the existing one-sided/central adaptive stencil is
// used unchanged.
M_FieldQuantity strainTensorQuantity(const Magnet*);
// Strain rate tensor quantity with 6 symmetric strain components
// calculated according to dε/dt = 1/2 (∇v + (∇v)^T).
//
// Uses the same interior-interface jump correction as evalStrainTensor, weighted by
// eta11/eta44 (mumax's viscous-damping stiffness analogue) instead of C11/C44. There is
// no magnetoelastic eigenstrain-rate analogue (magnum.np has none either), so the
// eigenstrain offset term is omitted here.
M_FieldQuantity strainRateQuantity(const Magnet*);
