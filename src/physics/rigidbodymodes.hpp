#pragma once

#include <array>

class Field;
class Magnet;

struct RigidBodyGeometry {
  double3 com;
  double totalRho;
  double I[3][3];
  double Iinv[3][3];
};

struct RigidModeMoments {
  double3 T;
  double3 omega;
};

RigidBodyGeometry computeRigidBodyGeometry(const Magnet* magnet);

// isForce = false: kinematic fields (displacement u, velocity v). Fit is
//   rho-weighted directly on the field; correction subtracted flat.
// isForce = true: force fields. Fit is on the *specific force* f/rho
//   (rho-weighted -> the rho's cancel out of the sums, leaving plain
//   Sum(f)/Sum(rho) and Sum(r x f)); correction is scaled by rho(r)
//   before subtraction, since f = rho * a physically.
RigidModeMoments computeRigidModeMoments(const Field& f, const RigidBodyGeometry& geom,
                                         const Magnet* magnet, bool isForce);

void removeRigidBodyModes(Field& f, const RigidBodyGeometry& geom,
                          const Magnet* magnet, bool isForce);