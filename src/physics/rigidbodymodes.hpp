#pragma once

#include <array>
#include "rigidbodymodes_common.hpp"

class Field;
class Magnet;

struct RigidModeMoments {
  double3 T;
  double3 omega;
};

struct RigidBodyGeometry {
  double3 com;
  double3 comUnweighted;    // plain geometric centroid (no rho anywhere)
  double totalRho;
  double ncellsInGeometry;  
  double I[3][3];
  double Iinv[3][3];
  double IUnweighted[3][3];
  double IinvUnweighted[3][3];
};

RigidBodyGeometry computeRigidBodyGeometry(const Magnet* magnet);

RigidModeMoments computeRigidModeMoments(const Field& f, const RigidBodyGeometry& geom,
                                         const Magnet* magnet, bool unweighted = false);

void removeRigidBodyModes(Field& f, const RigidBodyGeometry& geom,
                          const Magnet* magnet, bool unweighted = false);

