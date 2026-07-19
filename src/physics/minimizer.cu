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

namespace {

// Adds `magnet` to the elastic bookkeeping lists if it actually has
// elastodynamics enabled with nonzero stiffness. Used identically by all
// three constructors so the "assuredZero" filter only lives in one place.
void addElasticIfPresent(const Magnet* magnet,
                         std::vector<const Magnet*>& elMagnets,
                         std::vector<M_FieldQuantity>& forces) {
  if (!elasticityAssuredZero(magnet)) {
    elMagnets.push_back(magnet);
    forces.push_back(effectiveBodyForceQuantity(magnet));
  }
}

} 

Minimizer::Minimizer(const Ferromagnet* magnet,
                     real stopMaxMagDiff,
                     int nMagDiffSamples,
                     real stopMaxElDiff,
                     int nElDiffSamples,
                     real stepsizeEl,
                     real stepsizeElFallback)
    : magnets_({magnet}),
      torques_({relaxTorqueQuantity(magnet)}),
      nMagDiffSamples_(nMagDiffSamples),
      stopMaxMagDiff_(stopMaxMagDiff),
      nElDiffSamples_(nElDiffSamples),
      stopMaxElDiff_(stopMaxElDiff),
      t0(1), t1(1), m0(1), m1(1) {
  stepsizes_ = {1e-14};  

  addElasticIfPresent(magnet, elMagnets_, forces_);
  f0.resize(elMagnets_.size());
  f1.resize(elMagnets_.size());
  u0.resize(elMagnets_.size());
  u1.resize(elMagnets_.size());
  elStepsizes_.assign(elMagnets_.size(), stepsizeEl);
  stepsizeElInit_ = stepsizeEl;
  stepsizeElFallback_ = stepsizeElFallback;
  
}

Minimizer::Minimizer(const HostMagnet* magnet,
                     real stopMaxMagDiff,
                     int nMagDiffSamples,
                     real stopMaxElDiff,
                     int nElDiffSamples,
                     real stepsizeEl,
                     real stepsizeElFallback)
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

  
  addElasticIfPresent(magnet, elMagnets_, forces_);
  f0.resize(elMagnets_.size());
  f1.resize(elMagnets_.size());
  u0.resize(elMagnets_.size());
  u1.resize(elMagnets_.size());
  elStepsizes_.assign(elMagnets_.size(), stepsizeEl);
  stepsizeElInit_ = stepsizeEl;
  stepsizeElFallback_ = stepsizeElFallback;
}

Minimizer::Minimizer(const MumaxWorld* world,
                     real stopMaxMagDiff,
                     int nMagDiffSamples,
                     real stopMaxElDiff,
                     int nElDiffSamples,
                     real stepsizeEl,
                     real stepsizeElFallback)
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
    addElasticIfPresent(mag, elMagnets_, forces_);
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
}


void Minimizer::exec() {
  nsteps_ = 0;
  lastMagDiffs_.clear();
  lastElDiffs_.clear();

  bool magnetoelasticsActive = !elMagnets_.empty();

  if (!magnetoelasticsActive) {
    while (!converged())
      step();
    return;
  }

  const int maxSteps = 2000000;  // TODO: make configurable if this isn't enough. maybe not needed anymore?
  

  while (!converged() && nsteps_ < maxSteps) {
    step();
  }

  if (nsteps_ >= maxSteps)
    std::cerr << "Warning: Minimizer did not converge after " << maxSteps << " steps." << std::endl;
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

    if (nsteps_ == 0)
      f0[i] = forces_[i].eval();
    else
      f0[i] = f1[i];

    real h = elStepsizes_[i];

    u1[i] = Field(elMagnets_[i]->system(), 3);
    int ncells = u1[i].grid().ncells();
    cudaLaunch(ncells, k_stepElastic, u1[i].cu(), u0[i].cu(), f0[i].cu(), h);
  }

  for (size_t i = 0; i < elMagnets_.size(); i++)
    elMagnets_[i]->elasticDisplacement()->set(u1[i]);
  for (size_t i = 0; i < elMagnets_.size(); i++)
    f1[i] = forces_[i].eval();

  for (size_t i = 0; i < elMagnets_.size(); i++) {
    Field du = add(real(+1), u1[i], real(-1), u0[i]);
    Field dg = add(real(-1), f1[i], real(+1), f0[i]);

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


    // du += (duScale-1)*du  ==>  du = duScale*du
    addTo(du, duScale - real(1.0), du);  
    addTo(dg, dgScale - real(1.0), dg); 
    

    // scaling trick to avoid float32 underflow issues in dudu kernel
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

    //shouldn't be necessary anymore
    if (div != 0.0) {
      real newHstep = nom / div;
      if (!std::isnan(newHstep) && !std::isinf(newHstep))
        elStepsizes_[i] = newHstep;
      else
      // newH somehow nan or inf? fallback
        elStepsizes_[i] = stepsizeElFallback_;
    } else {
      //div somehow zero? fallback
      elStepsizes_[i] = stepsizeElFallback_;
    }

    addElDiff(maxDu);
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