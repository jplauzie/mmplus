// rigidbodymodes_diag.hpp
#pragma once
#include "datatypes.hpp"  // wherever double3 etc. live in your codebase

class Field;
class Magnet;

struct NetForceTorqueDumb {
  double3 com;
  double3 netForce;
  double3 netTorque;
};

NetForceTorqueDumb computeNetForceTorqueDumbLoop(const Field& f, const Magnet* magnet);