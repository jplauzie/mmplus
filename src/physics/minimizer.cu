#include "antiferromagnet.hpp"
#include "cudalaunch.hpp"
#include "elastodynamics.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "hostmagnet.hpp"
#include "magnet.hpp"
#include "minimizer.hpp"
#include "mumaxworld.hpp"
#include "ncafm.hpp"
#include "reduce.hpp"
#include "torque.hpp"
#include "internalbodyforce.hpp"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <array>
#include <iostream>
#include <cmath>



namespace {

// Adds `magnet` to the elastic bookkeeping lists if it actually has
// elastodynamics. Used identically by all 3 constructors so "assuredZero" filter only lives in one place.
void addElasticIfPresent(const Magnet* magnet,
                         std::vector<const Magnet*>& elMagnets,
                         std::vector<M_FieldQuantity>& forces,
                         std::vector<RigidBodyGeometry>& rigidGeoms,
                         std::vector<RigidBodyModes2>& rigidModes2,
                         std::vector<RigidBodyModes2>& rigidModes2Force,
                         std::vector<RigidBodyGeometry3>& rigidGeoms3,
                         std::vector<RigidBodyGeometry4>& rigidGeoms4) {
  if (!elasticityAssuredZero(magnet)) {
    elMagnets.push_back(magnet);
    forces.push_back(effectiveBodyForceQuantity(magnet));

    RigidBodyGeometry g0 = computeRigidBodyGeometry(magnet);
    RigidBodyGeometry3 g3 = computeRigidBodyGeometry3(magnet);
    RigidBodyGeometry4 g4 = computeRigidBodyGeometry4(magnet);

    // One-time cross-check: g0/g3/g4 independently reduce the same
    // mass-weighted com/totalRho. A mismatch here means a bug in one of
    // the duplicated reduction kernels, not a physics issue -- worth
    // catching immediately rather than inferring it from noisy per-step
    // diagnostics later.
    auto dist = [](double3 a, double3 b) {
      return std::sqrt(std::pow(a.x-b.x,2) + std::pow(a.y-b.y,2) + std::pow(a.z-b.z,2));
    };
    double comDiff03 = dist(g0.com, g3.com0);
    double comDiff04 = dist(g0.com, g4.com0);
    double rhoDiff03 = std::abs(g0.totalRho - g3.totalRho);
    double rhoDiff04 = std::abs(g0.totalRho - g4.totalRho);

    constexpr double relTol = 1e-9;
    if (comDiff03 > relTol || comDiff04 > relTol ||
        rhoDiff03 > relTol * g0.totalRho || rhoDiff04 > relTol * g0.totalRho) {
      std::cerr << "[rbm-init] WARNING: reference geometry mismatch across methods -- "
                << "comDiff03=" << comDiff03 << " comDiff04=" << comDiff04
                << " rhoDiff03=" << rhoDiff03 << " rhoDiff04=" << rhoDiff04
                << std::endl;
    }

    rigidGeoms.push_back(g0);
    rigidModes2.push_back(computeRigidBodyModes2(magnet, false));
    rigidModes2Force.push_back(computeRigidBodyModes2(magnet, true));
    rigidGeoms3.push_back(g3);
    rigidGeoms4.push_back(g4);
  }
}
}

Minimizer::Minimizer(const Ferromagnet* magnet,
                     real stopMaxMagDiff,
                     int nMagDiffSamples,
                     real stopMaxElDiff,
                     int nElDiffSamples,
                     real stepsizeEl,
                     real stepsizeElFallback,
                     int maxSteps,
                     int rigidBodyModesInterval,
                     int rigidBodyModesDelay,
                     int rigidBodyModesMethod)
    : magnets_({magnet}),
      torques_({relaxTorqueQuantity(magnet)}),
      nMagDiffSamples_(nMagDiffSamples),
      stopMaxMagDiff_(stopMaxMagDiff),
      nElDiffSamples_(nElDiffSamples),
      stopMaxElDiff_(stopMaxElDiff),
      t0(1), t1(1), m0(1), m1(1) {
  stepsizes_ = {1e-14};

  addElasticIfPresent(magnet, elMagnets_, forces_, rigidGeoms_, rigidModes2_, rigidModes2Force_, rigidGeoms3_, rigidGeoms4_);
  f0.resize(elMagnets_.size());
  f1.resize(elMagnets_.size());
  u0.resize(elMagnets_.size());
  u1.resize(elMagnets_.size());
  elStepsizes_.assign(elMagnets_.size(), stepsizeEl);
  stepsizeElInit_ = stepsizeEl;
  stepsizeElFallback_ = stepsizeElFallback;
  maxSteps_ = maxSteps;
  rigidBodyModesInterval_ = rigidBodyModesInterval;
  rigidBodyModesDelay_ = rigidBodyModesDelay;
  rigidBodyModesMethod_ = rigidBodyModesMethod;
}

Minimizer::Minimizer(const HostMagnet* magnet,
                     real stopMaxMagDiff,
                     int nMagDiffSamples,
                     real stopMaxElDiff,
                     int nElDiffSamples,
                     real stepsizeEl,
                     real stepsizeElFallback,
                     int maxSteps,
                     int rigidBodyModesInterval,
                     int rigidBodyModesDelay,
                     int rigidBodyModesMethod)
    : magnets_(magnet->sublattices()),
      nMagDiffSamples_(nMagDiffSamples),
      stopMaxMagDiff_(stopMaxMagDiff),
      nElDiffSamples_(nElDiffSamples),
      stopMaxElDiff_(stopMaxElDiff),
      t0(magnets_.size()),
      t1(magnets_.size()),
      m0(magnets_.size()),
      m1(magnets_.size()) {
  stepsizes_.assign(magnets_.size(), 1e-14);

  for (auto sub : magnets_)
    torques_.push_back(relaxTorqueQuantity(sub));


  addElasticIfPresent(magnet, elMagnets_, forces_, rigidGeoms_, rigidModes2_, rigidModes2Force_, rigidGeoms3_, rigidGeoms4_);
  f0.resize(elMagnets_.size());
  f1.resize(elMagnets_.size());
  u0.resize(elMagnets_.size());
  u1.resize(elMagnets_.size());
  elStepsizes_.assign(elMagnets_.size(), stepsizeEl);
  stepsizeElInit_ = stepsizeEl;
  stepsizeElFallback_ = stepsizeElFallback;
  maxSteps_ = maxSteps;
  rigidBodyModesInterval_ = rigidBodyModesInterval;
  rigidBodyModesDelay_ = rigidBodyModesDelay;
  rigidBodyModesMethod_ = rigidBodyModesMethod;
}

Minimizer::Minimizer(const MumaxWorld* world,
                     real stopMaxMagDiff,
                     int nMagDiffSamples,
                     real stopMaxElDiff,
                     int nElDiffSamples,
                     real stepsizeEl,
                     real stepsizeElFallback,
                     int maxSteps,
                     int rigidBodyModesInterval,
                     int rigidBodyModesDelay,
                     int rigidBodyModesMethod)
    : nMagDiffSamples_(0),  // set below, once N is known
      stopMaxMagDiff_(stopMaxMagDiff),
      nElDiffSamples_(0),  // set below, once elMagnets_ is known
      stopMaxElDiff_(stopMaxElDiff) {
  // Total number of ferromagnets (FM instances or sublattices)
  size_t N = world->ferromagnets().size() + 2 * world->antiferromagnets().size();
  nMagDiffSamples_ = nMagDiffSamples * N;
  t0.resize(N);
  t1.resize(N);
  m0.resize(N);
  m1.resize(N);

  for (const auto pair : world->magnets()) {
    const Magnet* mag = pair.second;
    if (auto host = mag->asHost()) {
      for (auto sub : host->sublattices())
        magnets_.push_back(sub);
    } else if (const Ferromagnet* fm = mag->asFM()) {
      magnets_.push_back(fm);
    }
    // Elastic displacement lives at host granularity: exactly one entry per
    // independent Ferromagnet or per HostMagnet (AFM/NcAfm), never per
    // sublattice, so this check happens once per world->magnets() entry
    // regardless of how many magnetic sublattices it owns.
        addElasticIfPresent(mag, elMagnets_, forces_, rigidGeoms_, rigidModes2_, rigidModes2Force_, rigidGeoms3_, rigidGeoms4_);
  }

  for (auto magnet : magnets_)
    torques_.push_back(relaxTorqueQuantity(magnet));

  stepsizes_.assign(N, 1e-14);

  nElDiffSamples_ = nElDiffSamples * elMagnets_.size();
  f0.resize(elMagnets_.size());
  f1.resize(elMagnets_.size());
  u0.resize(elMagnets_.size());
  u1.resize(elMagnets_.size());
  elStepsizes_.assign(elMagnets_.size(), stepsizeEl);
  stepsizeElInit_ = stepsizeEl;
  stepsizeElFallback_ = stepsizeElFallback;
  maxSteps_ = maxSteps;
  rigidBodyModesInterval_ = rigidBodyModesInterval;
  rigidBodyModesDelay_ = rigidBodyModesDelay;
  rigidBodyModesMethod_ = rigidBodyModesMethod;
}


void Minimizer::exec() {
  nsteps_ = 0;
  lastMagDiffs_.clear();
  lastElDiffs_.clear();

  // Since this should be equilibrium, gracefully zero out the velocity
  for (size_t i = 0; i < elMagnets_.size(); i++){
    elMagnets_[i]->elasticVelocity()->set(real3{0, 0, 0});
  }

  // Check the state as handed off from the dynamic solver, before any
  // minimizer steps or rigid-mode removal have touched it.
  for (size_t i = 0; i < elMagnets_.size(); i++) {
    Field u0init = elMagnets_[i]->elasticDisplacement()->eval();
    RigidModeMoments m = computeRigidModeMoments(u0init, rigidGeoms_[i], elMagnets_[i], /*isForce=*/false);
    double thetaNorm = std::sqrt(m.omega.x * m.omega.x +
                                 m.omega.y * m.omega.y +
                                 m.omega.z * m.omega.z);
    std::cerr << std::setprecision(6)
              << "[rbm-init] magnet=" << i
              << " |theta|=" << thetaNorm << " rad"
              << " T=(" << m.T.x << "," << m.T.y << "," << m.T.z << ")"
              << (thetaNorm > 0.2 ? "  <-- LARGE: small-angle assumption likely invalid" : "")
              << std::endl;
  }

  bool magnetoelasticsActive = !elMagnets_.empty();
  if (!magnetoelasticsActive) {
    while (!converged())
      step();
    return;
  }


  while (!converged() && nsteps_ < maxSteps_) {
    step();
  }
  std::cerr << "steps: " << nsteps_ << " ." << std::endl;
}

__global__ void k_step(CuField mField,
                       const CuField m0Field,
                       const CuField torqueField,
                       real dt) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (!mField.cellInGrid(idx))
    return;

  real3 m0 = m0Field.vectorAt(idx);
  real3 t = torqueField.vectorAt(idx);

  real t2 = dt * dt * dot(t, t);
  real3 m = ((4 - t2) * m0 + 4 * dt * t) / (4 + t2);

  mField.setVectorInCell(idx, m);
}


__global__ void k_stepElastic(CuField uField,
                              const CuField u0Field,
                              const CuField forceField,
                              real dt) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (!uField.cellInGrid(idx))
    return;

  real3 u0 = u0Field.vectorAt(idx);
  real3 f = forceField.vectorAt(idx);

  uField.setVectorInCell(idx, u0 + dt * f);
}

static inline real BarzilianBorweinStepSize(Field& dm, Field& dtorque, int n) {
  real nom, div;
  if (n % 2 == 0) {
    nom = dotSum(dm, dm);
    div = dotSum(dm, dtorque);
  } else {
    nom = dotSum(dm, dtorque);
    div = dotSum(dtorque, dtorque);
  }
  if (div == 0.0)
    return 1e-14;  // TODO: figure out safe stepsize

  return nom / div;
}

void Minimizer::stepMagnetic() {
  for (size_t i = 0; i < magnets_.size(); i++) {

    m0[i] = magnets_[i]->magnetization()->eval();

    if (nsteps_ == 0)
      t0[i] = torques_[i].eval();
    else
      t0[i] = t1[i];

    m1[i] = Field(magnets_[i]->system(), 3);
    int ncells = m1[i].grid().ncells();

    cudaLaunch(ncells, k_step, m1[i].cu(), m0[i].cu(), t0[i].cu(), stepsizes_[i]);
  }

  for (size_t i = 0; i < magnets_.size(); i++)
    magnets_[i]->magnetization()->set(m1[i]);  // normalizes

  for (size_t i = 0; i < magnets_.size(); i++)
    t1[i] = torques_[i].eval();

  for (size_t i = 0; i < magnets_.size(); i++) {
    Field dm = add(real(+1), m1[i], real(-1), m0[i]);
    Field dt = add(real(-1), t1[i], real(+1), t0[i]);  // TODO: check sign difference

    stepsizes_[i] = BarzilianBorweinStepSize(dm, dt, nsteps_);

    addMagDiff(maxVecNorm(dm));
  }
}


void Minimizer::stepElastic() {
  for (size_t i = 0; i < elMagnets_.size(); i++) {
    u0[i] = elMagnets_[i]->elasticDisplacement()->eval();
    //if (shouldRemoveRigidBodyModes()) applyRigidBodyModeRemoval(u0[i], i, "u0");


    if (nsteps_ == 0) {
      elMagnets_[i]->elasticDisplacement()->set(u0[i]);
      f0[i] = forces_[i].eval();
      //if (shouldRemoveRigidBodyModes()) applyRigidBodyModeRemoval(f0[i], i, "f0");
    } else {
      f0[i] = f1[i];
    }

    real h = elStepsizes_[i];
    u1[i] = Field(elMagnets_[i]->system(), 3);
    int ncells = u1[i].grid().ncells();
    cudaLaunch(ncells, k_stepElastic, u1[i].cu(), u0[i].cu(), f0[i].cu(), h);
  }

  //cleaning u1 loses information for delicate symmetric starting states. Also should be redundant since this is all linear, anyway.
  for (size_t i = 0; i < elMagnets_.size(); i++) {
   if (shouldRemoveRigidBodyModes()) applyRigidBodyModeRemoval(u1[i], i, "u1");
  }

  for (size_t i = 0; i < elMagnets_.size(); i++) {
    elMagnets_[i]->elasticDisplacement()->set(u1[i]);
  }

  for (size_t i = 0; i < elMagnets_.size(); i++) {
    f1[i] = forces_[i].eval();
    if (shouldRemoveRigidBodyModes()) applyRigidBodyModeRemoval(f1[i], i, "f1");
  }

  for (size_t i = 0; i < elMagnets_.size(); i++) {
    Field du = add(real(+1), u1[i], real(-1), u0[i]);
    if (shouldRemoveRigidBodyModes()) applyRigidBodyModeRemoval(du, i, "du");

    Field dg = add(real(-1), f1[i], real(+1), f0[i]);
    if (shouldRemoveRigidBodyModes()) applyRigidBodyModeRemoval(dg, i, "dg");

    real maxDu = maxVecNorm(du);
    real maxDg = maxVecNorm(dg);
    real duScale;
    if (maxDu > 0) {
      duScale = real(1.0) / maxDu;
    } else {
      duScale = real(1.0);
    }

    real dgScale;
    if (maxDg > 0) {
      dgScale = real(1.0) / maxDg;
    } else {
      dgScale = real(1.0);
    }

    du = duScale * du;   // uses fieldops.hpp's operator*(real, const Field&)
    dg = dgScale * dg;

    // scaling trick to avoid float32 underflow issues in dudu kernel
    // ultimately the duScale and dgScale factors cancel out in the BB step
    real dudu = dotSum(du, du) / (duScale * duScale);
    real dudg = dotSum(du, dg) / (duScale * dgScale);
    real dgdg = dotSum(dg, dg) / (dgScale * dgScale);

    real nom, div;
    if (nsteps_ % 2 == 0) {
      //BB1
      nom = dudu;
      div = dudg;
      // safety, if BB1 fails, use BB2. Shouldn't be necessary with scaling trick
      if (nom == 0.0) {
        nom = dudg;
        div = dgdg;
      }
    } else {
      //BB2
      nom = dudg;
      div = dgdg;
    }

    //safety fallbacks, that shouldn't be necessary anymore
    if (div != 0.0) {
      real newHstep = nom / div;
      if (!std::isnan(newHstep) && !std::isinf(newHstep)) {
        elStepsizes_[i] = newHstep;
      } else {
        elStepsizes_[i] = stepsizeElFallback_;
      }
    } else {
      elStepsizes_[i] = stepsizeElFallback_;
    }

    real maxU = maxVecNorm(u1[i]);
    real relDu = (maxU > 0) ? (maxDu / maxU) : maxDu;

    addElDiff(relDu);
  }
}


void Minimizer::step() {
  stepMagnetic();
  stepElastic();
  nsteps_ += 1;
}

//require convergence for both magnetic and elastic parts, if present
bool Minimizer::converged() const {
  bool magConverged = true;
  if (!magnets_.empty()) {
    if (lastMagDiffs_.size() < nMagDiffSamples_)
      magConverged = false;
    else
      for (auto dm : lastMagDiffs_)
        if (dm > stopMaxMagDiff_) {
          magConverged = false;
          break;
        }
  }

  bool elConverged = true;
  if (!elMagnets_.empty()) {
    if (lastElDiffs_.size() < nElDiffSamples_)
      elConverged = false;
    else
      for (auto du : lastElDiffs_)
        if (du > stopMaxElDiff_) {
          elConverged = false;
          break;
        }
  }

  return magConverged && elConverged;
}

bool Minimizer::shouldRemoveRigidBodyModes() const {
  if (rigidBodyModesInterval_ <= 0)
    return false;
  if (nsteps_ < rigidBodyModesDelay_)
    return false;
  return nsteps_ % rigidBodyModesInterval_ == 0;
}

void Minimizer::applyRigidBodyModeRemoval(Field& f, size_t i, const char* label) {
  bool isForce = (label == std::string("f0") ||
                  label == std::string("f1") ||
                  label == std::string("dg"));
  // (u0, u1, du stay on the kinematic path)

  constexpr int kDiagSteps = 20;
  if (nsteps_ < kDiagSteps) {
    real rawNorm = maxVecNorm(f);

    if (isForce) {
      // Correct (rho-aware) cleaning, both implementations.
      Field cleaned0 = f, cleaned1 = f;
      removeRigidBodyModes(cleaned0, rigidGeoms_[i], elMagnets_[i], /*isForce=*/true);
      removeRigidBodyModes2(cleaned1, rigidModes2Force_[i], elMagnets_[i], /*isForce=*/true);

      // Naive/old comparison baseline: apply the *kinematic* (rho-unaware)
      // treatment directly to a force field. Known-wrong for spatially
      // varying rho -- kept here purely as a comparison signal, not a
      // candidate for actual use. Reuses rigidGeoms_/rigidModes2_ (the
      // kinematic geometry/basis) with isForce=false, which reproduces the
      // pre-fix behavior exactly.
      Field cleanedOld0 = f, cleanedOld1 = f;
      removeRigidBodyModes(cleanedOld0, rigidGeoms_[i], elMagnets_[i], /*isForce=*/false);
      removeRigidBodyModes2(cleanedOld1, rigidModes2_[i], elMagnets_[i], /*isForce=*/false);

      real n0    = maxVecNorm(cleaned0);
      real n1    = maxVecNorm(cleaned1);
      real nOld0 = maxVecNorm(cleanedOld0);
      real nOld1 = maxVecNorm(cleanedOld1);
      real diff0vsOld0 = maxVecNorm(add(real(1), cleaned0, real(-1), cleanedOld0));
      real diff1vsOld1 = maxVecNorm(add(real(1), cleaned1, real(-1), cleanedOld1));

      std::cerr << std::setprecision(8)
                << "[rbm-diag-force] step=" << nsteps_ << " field=" << label << " i=" << i
                << " raw|f|=" << rawNorm
                << " |new0|=" << n0 << " |new1|=" << n1
                << " |old0|=" << nOld0 << " |old1|=" << nOld1
                << " |new0-old0|=" << diff0vsOld0
                << " |new1-old1|=" << diff1vsOld1
                << std::endl;
    } else {
      // Linearized (small-rotation) methods.
      RigidModeMoments m0 = computeRigidModeMoments(f, rigidGeoms_[i], elMagnets_[i], /*isForce=*/false);
      double omegaNorm = std::sqrt(m0.omega.x*m0.omega.x + m0.omega.y*m0.omega.y + m0.omega.z*m0.omega.z);
      std::array<double, 6> c1raw = rigidBodyModeCoefficients2(f, rigidModes2_[i], elMagnets_[i], /*isForce=*/false);
      double c1RotNorm = std::sqrt(c1raw[3]*c1raw[3] + c1raw[4]*c1raw[4] + c1raw[5]*c1raw[5]);

      // Exact (large-rotation) methods.
      KabschResult kb = computeKabschAlignment(f, rigidGeoms3_[i], elMagnets_[i]);
      QuatAlignResult qr = computeQuaternionAlignment(f, rigidGeoms4_[i], elMagnets_[i]);
      double thetaKabsch = kabschRotationAngle(kb);
      double thetaQuat = quaternionRotationAngle(qr);

      // Apply all four to independent copies of the same raw f, diff directly
      // -- isolates per-cleaning discrepancy from trajectory effects.
      Field cleaned0 = f, cleaned1 = f, cleaned2 = f, cleaned3 = f;
      removeRigidBodyModes(cleaned0, rigidGeoms_[i], elMagnets_[i], /*isForce=*/false);
      removeRigidBodyModes2(cleaned1, rigidModes2_[i], elMagnets_[i], /*isForce=*/false);
      removeRigidBodyModesKabsch(cleaned2, rigidGeoms3_[i], elMagnets_[i]);
      removeRigidBodyModesQuaternion(cleaned3, rigidGeoms4_[i], elMagnets_[i]);

      real diff01 = maxVecNorm(add(real(1), cleaned0, real(-1), cleaned1));
      real diff02 = maxVecNorm(add(real(1), cleaned0, real(-1), cleaned2));
      real diff23 = maxVecNorm(add(real(1), cleaned2, real(-1), cleaned3));  // kabsch vs quat: should be ~0

      std::cerr << std::setprecision(8)
                << "[rbm-diag] step=" << nsteps_ << " field=" << label << " i=" << i
                << " raw|f|=" << rawNorm
                << " |omega|(lin0)=" << omegaNorm
                << " |rotcoef|(lin1)=" << c1RotNorm
                << " theta(kabsch)=" << thetaKabsch
                << " theta(quat)=" << thetaQuat
                << " thetaGap(kabsch-quat)=" << (thetaKabsch - thetaQuat)
                << " quatEigenGap=" << qr.eigenGap
                << " |0-1|=" << diff01
                << " |0-2|=" << diff02
                << " |2-3|=" << diff23
                << (thetaKabsch > 0.2 ? "  <-- large rotation: methods 0/1 (linearized) may be inaccurate" : "")
                << (qr.eigenGap < 1e-9 ? "  <-- rotation axis poorly determined by this configuration" : "")
                << std::endl;
    }
  }

  // ---- the actual removal: exactly one call, dispatched on field type ----
  if (isForce) {
    switch (rigidBodyModesMethod_) {
      case 0:
        removeRigidBodyModes(f, rigidGeoms_[i], elMagnets_[i], /*isForce=*/true);
        break;
      case 1:
        removeRigidBodyModes2(f, rigidModes2Force_[i], elMagnets_[i], /*isForce=*/true);
        break;
      case 2:
      case 3:
        throw std::runtime_error(
            "Minimizer: rigidBodyModesMethod_ 2/3 have no force-aware "
            "implementation yet; only methods 0 and 1 support force fields.");
      default:
        throw std::runtime_error("Minimizer: unknown rigidBodyModesMethod_ (expected 0-3).");
    }
  } else {
    switch (rigidBodyModesMethod_) {
      case 0: removeRigidBodyModes(f, rigidGeoms_[i], elMagnets_[i], /*isForce=*/false); break;
      case 1: removeRigidBodyModes2(f, rigidModes2_[i], elMagnets_[i], /*isForce=*/false); break;
      case 2: removeRigidBodyModesKabsch(f, rigidGeoms3_[i], elMagnets_[i]); break;
      case 3: removeRigidBodyModesQuaternion(f, rigidGeoms4_[i], elMagnets_[i]); break;
      default:
        throw std::runtime_error("Minimizer: unknown rigidBodyModesMethod_ (expected 0-3).");
    }
  }
}

void Minimizer::addMagDiff(real dm) {
  lastMagDiffs_.push_back(dm);
  if (lastMagDiffs_.size() > nMagDiffSamples_)
    lastMagDiffs_.pop_front();
}

void Minimizer::addElDiff(real du) {
  lastElDiffs_.push_back(du);
  if (lastElDiffs_.size() > nElDiffSamples_)
    lastElDiffs_.pop_front();
}