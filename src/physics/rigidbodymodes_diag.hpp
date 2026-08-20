#pragma once

#include "field.hpp"

// Diagnostic/verification utilities, intentionally decoupled from
// rigidbodymodes.cu (no shared kernels, no shared reduction code).
// Use these to independently sanity-check the rho-weighted rigid-body
// removal against a plain, unweighted calculation.

// Net translation, computed as a plain unweighted average of u over all
// cells in the field's geometry: T = (1/N) * sum_i u(i). No rho weighting.
// Intended for use directly on displacement snapshots, e.g.
// computeNetTranslationUnweighted(u0[i]) / computeNetTranslationUnweighted(u1[i]).
real3 computeNetTranslationUnweighted(const Field& u);

// Adds a constant translation to every cell in the field's geometry,
// in place. No rho weighting -- u(i) += T for all i in geometry.
// Intended for artificially perturbing u0/u1 to verify that
// removeRigidBodyModes (and the diagnostics above) correctly detect
// and/or remove an injected rigid translation.
void addTranslationUnweighted(Field& u, real3 T);

// rigidbodymodes_diag.hpp
void addRotationUnweighted(Field& u, real3 com, real3 omega);

// Net rotation about `com`, computed as a plain unweighted least-squares
// fit of u(r) = omega x r over all cells in the field's geometry. No rho
// weighting, no shared inertia tensor with rigidbodymodes.cu -- uses its
// own unweighted (geometric) inertia tensor computed inline.
real3 computeNetRotationUnweighted(const Field& u, real3 com);