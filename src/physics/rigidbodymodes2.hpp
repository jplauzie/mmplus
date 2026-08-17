#pragma once

#include <array>

class Field;
class Magnet;

struct RigidBodyModes2 {
  Field modes[6];  // 0-2: translations, 3-5: rotations. rho-scaled if built with isForce=true.
};

std::array<std::array<double, 3>, 3> inertiaTensorViaProjection2(const Magnet* magnet);

// isForce = false: kinematic modes -- bare direction/rotation fields,
//   rho-weighted inner product.
// isForce = true: force modes -- rho(r)-scaled direction/rotation fields,
//   1/rho-weighted inner product (the rho's cancel in the projection
//   coefficient, leaving plain unweighted dot sums -- see derivation notes
//   in rigidbodymodes2.cu).
double weightedDot(const Field& a, const Field& b, const Magnet* magnet, bool isForce);

std::array<double, 6> rigidBodyModeCoefficients2(const Field& f, const RigidBodyModes2& modes,
                                                  const Magnet* magnet, bool isForce);

RigidBodyModes2 computeRigidBodyModes2(const Magnet* magnet, bool isForce);

void removeRigidBodyModes2(Field& f, const RigidBodyModes2& modes,
                           const Magnet* magnet, bool isForce);