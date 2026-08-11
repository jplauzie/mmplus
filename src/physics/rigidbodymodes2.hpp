#pragma once

#include <array>

#include "field.hpp"

class Magnet;

/// Orthonormal basis of the 6 rigid-body modes (3 translations + 3
/// rotations about the mass-weighted center of mass), built once per
/// magnet and cached, same usage pattern as RigidBodyGeometry in
/// rigidbodymodes.hpp.
///
/// Unlike the inertia-tensor approach in rigidbodymodes.cu, this follows
/// PETSc's MatNullSpaceCreateRigidBody(): construct the 6 modes
/// explicitly from geometry, then remove them from a field by direct
/// orthogonal projection (mass-weighted inner product) rather than
/// inverting an inertia tensor. See PR discussion for motivation.
struct RigidBodyModes2 {
  // 0,1,2 = translation along x,y,z. 3,4,5 = rotation about x,y,z
  // through the mass-weighted center of mass.
  std::array<Field, 6> modes;
};

/// TEMP DIAGNOSTIC: independently recomputes the inertia tensor as the
/// mass-weighted Gram matrix of the raw (un-orthogonalized) rotation
/// mode vectors: I_ij = <axis_i x r, axis_j x r>_rho. Mathematically
/// identical to RigidBodyGeometry::I from rigidbodymodes.cu, but
/// computed via a completely independent route -- useful as a
/// correctness cross-check between the two methods.
std::array<std::array<double, 3>, 3> inertiaTensorViaProjection2(const Magnet* magnet);

/// Build the orthonormal rigid-body mode basis for `magnet`. Expensive
/// (up to 21 mass-weighted reductions for Gram-Schmidt); call once and
/// cache, analogous to computeRigidBodyGeometry().
RigidBodyModes2 computeRigidBodyModes2(const Magnet* magnet);

/// Project the 6 rigid-body modes out of `f` in place, using the
/// mass-weighted inner product against the precomputed orthonormal
/// basis `modes`. Signature mirrors removeRigidBodyModes() for
/// drop-in comparison.
void removeRigidBodyModes2(Field& f, const RigidBodyModes2& modes, const Magnet* magnet);

/// Returns the 6 mass-weighted projection coefficients of `f` onto the
/// rigid-body mode basis, without modifying `f`. Indices 0-2 correspond
/// to translation (x,y,z), 3-5 to rotation (x,y,z) -- the direct-
/// projection analog of (T, omega) in the inertia-tensor method. Exposed
/// separately from removeRigidBodyModes2() for diagnostics/comparison.
std::array<double, 6> rigidBodyModeCoefficients2(const Field& f,
                                                  const RigidBodyModes2& modes,
                                                  const Magnet* magnet);