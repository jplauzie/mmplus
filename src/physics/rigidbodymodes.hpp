#pragma once
#include <memory>
#include "datatypes.hpp"
#include "field.hpp"

class System;

struct RigidBodyGeometry {
  Field relPos;     // r - COM per cell, world units, zero outside geometry
                     // (precomputed once; purely geometric, independent of u)
  real Iinv[3][3];  // regularized pseudo-inverse of the inertia tensor
};

/// One-time per-magnet setup: mass-weighted (geometry-aware) COM and
/// inertia tensor, cached for reuse every minimizer step.
RigidBodyGeometry computeRigidBodyGeometry(std::shared_ptr<const System> system);

/// Subtracts the best-fit rigid translation + rotation from `f` in place.
/// Intended to be called on the linear diff `du`, but works on any
/// 3-component field defined on the same system as `geom`.
void removeRigidBodyModes(Field& f, const RigidBodyGeometry& geom);