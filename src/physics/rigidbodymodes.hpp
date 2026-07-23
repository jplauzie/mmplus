#pragma once
#include <memory>
#include "datatypes.hpp"
#include "field.hpp"

class System;

struct RigidBodyGeometry {
  Field relPos;     // r - center of mass (COM) per cell, world units, zero outside geometry
                     // (precomputed once; purely geometric, independent of u)
  real Iinv[3][3];  // regularized pseudo-inverse of the inertia tensor
};

/// One-time per-magnet setup: mass-weighted (geometry-aware) COM and
/// inertia tensor, cached for reuse every minimizer step.
RigidBodyGeometry computeRigidBodyGeometry(std::shared_ptr<const System> system);

/// Subtracts the best-fit rigid translation + rotation from `f` in place 
/// (f as in field, not force! (magneto)-elastic forces don't care about absolute values of u and don't change).
/// Called on `u1` right after each step to anchor the physical displacement
/// state itself (preventing unconstrained rigid-mode drift from
/// accumulating across steps, analogous to magnetization's normalize).
/// Also safe to call on the linear diff `du`, or any other 3-component
/// field defined on the same system as `geom`.
void removeRigidBodyModes(Field& f, const RigidBodyGeometry& geom);