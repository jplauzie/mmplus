#pragma once
#include "datatypes.hpp"

class Field;
class Magnet;

// ---------------------------------------------------------------------------
// Exact (non-linearized) rigid-body alignment via Kabsch/Procrustes SVD.
//
// Unlike rigidbodymodes.cu (closed-form T, omega via inertia-tensor inverse)
// and rigidbodymodes2.cu (Gram-Schmidt projection onto the same omega x r
// basis), this file does NOT linearize the rotation. It recovers a genuine
// rotation matrix R in SO(3), valid for arbitrarily large rotations, by
// aligning the reference (undeformed) cell positions to the current
// (reference + displacement) positions.
//
// Intended use: NOT a per-step minimizer cleanup routine (that's what
// rigidbodymodes.cu / rigidbodymodes2.cu are for, and are cheaper). This is
// meant for one-off checks/corrections where a large built-in rotation is
// possible and must be measured or removed exactly -- e.g. validating or
// cleaning up a displacement field handed off from a dynamic solver before
// minimization begins.
// ---------------------------------------------------------------------------

// One-time reference-geometry cache: mass-weighted center of mass and the
// mass-weighted covariance of reference positions about it. Both depend only
// on the static grid/geometry/rho, not on any displacement field, so this is
// computed once and reused across calls, analogous to RigidBodyGeometry in
// rigidbodymodes.cu.
struct RigidBodyGeometry3 {
  double3 com0;      // reference (undeformed) mass-weighted center of mass
  double totalRho;   // total mass (sum of rho) in geometry
  double S0[3][3];   // Sum w*(r0-com0)(r0-com0)^T -- reference shape covariance
};

// Result of aligning the current configuration (com0's geometry + a
// displacement field u) back onto the reference configuration.
struct KabschResult {
  double R[3][3];  // rotation matrix, reference -> current (best-fit, exact)
  double3 T;       // translation: x_current ~= R*r0 + T
  double3 com;     // current (deformed) mass-weighted center of mass
};

RigidBodyGeometry3 computeRigidBodyGeometry3(const Magnet* magnet);

// Computes the best-fit rigid alignment (rotation + translation) of the
// current configuration (geom's reference positions + field u, interpreted
// as a displacement) onto the reference configuration. Exact for any
// rotation magnitude -- no small-angle assumption.
KabschResult computeKabschAlignment(const Field& u, const RigidBodyGeometry3& geom,
                                    const Magnet* magnet);

// Subtracts the rigid (translation + finite rotation) component from a
// displacement field u, in place, using the exact Kabsch alignment.
void removeRigidBodyModesKabsch(Field& u, const RigidBodyGeometry3& geom,
                                const Magnet* magnet);

// Rotation angle (radians, in [0, pi]) corresponding to R, via the standard
// trace formula: angle = acos((trace(R) - 1) / 2). Useful for checking
// whether a linearized (omega x r / theta x r) treatment would have been
// valid -- e.g. flag if this exceeds ~0.2 rad.
double kabschRotationAngle(const KabschResult& result);
