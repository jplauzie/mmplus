#pragma once

#include <deque>
#include <vector>

#include "field.hpp"
#include "quantityevaluator.hpp"
#include "rigidbodymodes.hpp"
#include "rigidbodymodes2.hpp"   

class Ferromagnet;
class HostMagnet;
class Magnet;
class MumaxWorld;

/// Minimizer implements a steepest-descent (Barzilai-Borwein) relaxation of the magnetization, following Exl et al., JAP 115,
/// 17D118 (2014). When elastodynamics is enabled on a magnet, a parallel steepest-descent
/// relaxation of the elastic displacement towards mechanical equilibrium
/// (effective body force -> 0).
///
/// Both systems are stepped every iteration inside step(), each with their
/// own independently-tracked Barzilai-Borwein step size and convergence
/// history. Overall convergence requires both systems (if present) to have
/// converged.
class Minimizer {
 public:
  Minimizer(const Ferromagnet* magnet,
            real stopMaxMagDiff,
            int nMagDiffSamples,
            real stopMaxElDiff,
            int nElDiffSamples,
            real stepsizeEl = 1e-30, real stepsizeElFallback = 1e-30,
            int maxSteps = 200000,
            int rigidBodyModesInterval = 1,
            int rigidBodyModesDelay = 0,
            int rigidBodyModesMethod = 0);

  Minimizer(const HostMagnet* magnet,
            real stopMaxMagDiff,
            int nMagDiffSamples,
            real stopMaxElDiff,
            int nElDiffSamples,
            real stepsizeEl = 1e-30, real stepsizeElFallback = 1e-30,
            int maxSteps = 200000,
            int rigidBodyModesInterval = 1,
            int rigidBodyModesDelay = 0,
            int rigidBodyModesMethod = 0);

  Minimizer(const MumaxWorld* world,
            real stopMaxMagDiff,
            int nMagDiffSamples,
            real stopMaxElDiff,
            int nElDiffSamples,
            real stepsizeEl = 1e-30, real stepsizeElFallback = 1e-30,
            int maxSteps = 200000,
            int rigidBodyModesInterval = 1,
            int rigidBodyModesDelay = 0,
            int rigidBodyModesMethod = 0);

  void exec();

 private:
  void step();
  void stepMagnetic();
  void stepElastic();
  bool converged() const;
  void addMagDiff(real dm);
  void addElDiff(real du);
 

  // --- magnetic part ---
  std::vector<const Ferromagnet*> magnets_;
  std::vector<FM_FieldQuantity> torques_;
  std::vector<Field> t0, t1, m0, m1;
  std::vector<real> stepsizes_;
  std::deque<real> lastMagDiffs_;
  size_t nMagDiffSamples_;
  real stopMaxMagDiff_;
   

  // --- elastic part ---
  // One entry per independent Ferromagnet or per HostMagnet (AFM/NcAfm), not sublattice
  std::vector<const Magnet*> elMagnets_;
  std::vector<M_FieldQuantity> forces_;
  std::vector<RigidBodyGeometry> rigidGeoms_;
  std::vector<RigidBodyModes2> rigidModes2_;
  std::vector<Field> f0, f1, u0, u1;
  std::vector<real> elStepsizes_;
  std::deque<real> lastElDiffs_;
  size_t nElDiffSamples_;
  real stopMaxElDiff_;
  real stepsizeElInit_;
  real stepsizeElFallback_; 

  bool shouldRemoveRigidBodyModes() const;
  // NEW: applies whichever method is toggled to f; for the first ~20
  // steps, also computes and prints BOTH methods' output (T/omega vs.
  // projection coefficients) to std::cerr regardless of toggle, tagged
  // with `label`, so the two can be compared side by side. Only the
  // toggled method ever mutates f. Temporary debug scaffolding -- strip
  // or gate before merge.
  void applyRigidBodyModeRemoval(Field& f, size_t i, const char* label);

  
  int nsteps_;
  int maxSteps_;                  
  int rigidBodyModesInterval_;    
  int rigidBodyModesDelay_;  
  int rigidBodyModesMethod_;       
};
