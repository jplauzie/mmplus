#pragma once
#include <memory>
#include "datatypes.hpp"

class Magnet;
class Field;

struct RigidBodyGeometry {
  double3 com;       // mass-weighted center of mass, world units
  double Iinv[3][3]; // regularized pseudo-inverse of the mass-weighted inertia tensor about com
  double totalRho;   // sum of rho over the geometry (cell volume cancels out
                     // everywhere it would appear, so it's never needed).
                     // Precomputed once; assumes rho is static for the
                     // duration of the minimization.
};

struct RigidModeMoments {
  double3 T;      // rigid translation
  double3 omega;  // rigid rotation
};

// Computes the rigid translation/rotation content of f without modifying it.
RigidModeMoments computeRigidModeMoments(const Field& f,
                                         const RigidBodyGeometry& geom,
                                         const Magnet* magnet);

/// One-time per-magnet setup: mass-weighted (geometry-aware) center of mass
/// and inertia tensor, cached for reuse every minimizer step. No per-cell
/// data is stored -- cell positions are cheap to recompute on the fly from
/// the grid. Computed and stored in double precision throughout: float32
/// storage here was found to introduce measurable asymmetry (a nonzero
/// off-diagonal Iinv term that should be exactly zero for symmetric
/// geometries), which showed up as basin-selection sensitivity in
/// magnetization-symmetry-breaking test cases.
RigidBodyGeometry computeRigidBodyGeometry(const Magnet* magnet);

/// Subtracts the best-fit rigid translation + rotation from `f` in place
/// (f as in field, not force! (magneto)elastic forces don't care about
/// absolute values of u and don't change).
/// `magnet` supplies rho for the mass weighting and must be the same magnet
/// that `geom` was computed from.
void removeRigidBodyModes(Field& f, const RigidBodyGeometry& geom,
                          const Magnet* magnet);