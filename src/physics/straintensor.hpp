#pragma once

#include "quantityevaluator.hpp"

class Magnet;
class Field;

// Number of B_eps iterations of the elastic jump-condition scheme (magnum.np's
// iteration_depth). Each iteration roughly halves the remaining error of the
// free-surface condition and of momentum conservation (see
// evalElasticJumpData).
constexpr int ELASTIC_JUMP_DEPTH = 4;

// Number of iterations for the viscous force / strain rate (same scheme applied
// to v).
constexpr int VISCOUS_JUMP_DEPTH = ELASTIC_JUMP_DEPTH;

bool strainTensorAssuredZero(const Magnet*);

Field evalStrainTensor(const Magnet*);
Field evalStrainRate(const Magnet*);

// Full 9-component (unsymmetrized) displacement gradient from
// evalElasticJumpData (layout: grad[3*d+k] = d(u_k)/dx_d). evalStrainTensor
// symmetrizes exactly this field.
Field evalDisplacementGradientTensor(const Magnet*);

// The magnetoelastic eigenstrain traction offset, face-extrapolated and summed
// over all Ferromagnet sublattices; this is the sigma_mel part of the offsets
// built by evalElasticJumpData. faceL/faceR layout: [3*i+c] = +sigma_mel_ic at
// this cell's -i / +i half-face respectively.
//
// Sign: B = +sigma_mel, with f_i = d_j sigma_mel_ij, sigma_mel_ii = B1 m_i^2,
// sigma_mel_ij = B2 m_i m_j (as in magnetoelasticforce.cu). This is the same
// number as magnum.np's offset B = -sigma_m, because sigma_mel = -sigma_m (= -C
// eps_m): there is no sign flip. The sign matters wherever sigma_mel enters a
// face condition -- at free surfaces as well as across magnetic/non-magnetic or
// B-contrast interfaces.
//
// Returns all-zero fields if magnetoelasticAssuredZero holds for every
// sublattice.
struct FaceEigenstrainTraction {
  Field faceL;
  Field faceR;
};
FaceEigenstrainTraction evalFaceEigenstrainTraction(const Magnet*);

// Jump-corrected displacement gradient plus the final face offsets, shared by
// the strain and the force. Layout of all three: [3*d + k], d = direction, k =
// component.
struct ElasticJumpData {
  Field gu;  // d_d u_k
  Field BL;  // total offset (sigma_mel + B_eps) at this cell's -d face
  Field BR;  // ... at the +d face
};
ElasticJumpData evalElasticJumpData(const Magnet*);

// Jump data of the elastic VELOCITY, built like evalElasticJumpData but
// weighted by the viscosity tensor (eta11, eta12, eta44) if useTensor, else by
// the stiffness (C11, C12, C44) (Rayleigh damping; the caller multiplies by the
// damping coefficient). No magnetoelastic offsets; traction only if
// useTraction.
ElasticJumpData evalViscousJumpData(const Magnet*,
                                    bool useTensor,
                                    bool useTraction);

// Strain tensor quantity with 6 symmetric strain components
// [εxx, εyy, εzz, εxy, εxz, εyz],
// calculated according to ε = 1/2 (∇u + (∇u)^T).
//
// ∇u comes from evalElasticJumpData (magnum.np-style jump conditions; voids and
// grid edges are C = 0 neighbours): solid-solid faces use a traction-consistent
// two-point formula with C11/C44 and offsets B = sigma_mel + B_eps; non-solid
// faces enforce C*grad + B = S. B_eps is iterated ELASTIC_JUMP_DEPTH times.
// Thin axes (one cell, non-periodic) have zero derivative (plane strain).
M_FieldQuantity strainTensorQuantity(const Magnet*);
// Strain rate tensor quantity with 6 symmetric strain components
// calculated according to dε/dt = 1/2 (∇v + (∇v)^T).
//
// Uses evalViscousJumpData: the same pipeline as the strain tensor, applied to
// v, weighted by (eta11, eta12, eta44) if the viscosity tensor is set,
// otherwise by (C11, C12, C44) (Rayleigh damping). No magnetoelastic offsets
// and no traction (viscous stress is traction-free).
M_FieldQuantity strainRateQuantity(const Magnet*);
