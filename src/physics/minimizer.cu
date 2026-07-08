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

}  // namespace

Minimizer::Minimizer(const Ferromagnet* magnet,
                     real stopMaxMagDiff,
                     int nMagDiffSamples,
                     real stopMaxElDiff,
                     int nElDiffSamples,
                     real stepsize,
                     real stepsizeEl)
    : magnets_({magnet}),
      torques_({relaxTorqueQuantity(magnet)}),
      nMagDiffSamples_(nMagDiffSamples),
      stopMaxMagDiff_(stopMaxMagDiff),
      nElDiffSamples_(nElDiffSamples),
      stopMaxElDiff_(stopMaxElDiff),
      t0(1),
      t1(1),
      m0(1),
      m1(1) {
  stepsizes_ = {stepsize};
  stepsizeInit_ = stepsize;

  addElasticIfPresent(magnet, elMagnets_, forces_);
  f0.resize(elMagnets_.size());
  f1.resize(elMagnets_.size());
  u0.resize(elMagnets_.size());
  u1.resize(elMagnets_.size());
  elStepsizes_.assign(elMagnets_.size(), stepsizeEl);
  stepsizeElInit_ = stepsizeEl;
  // TODO: figure out how to make descent guess
  // TODO: check if input arguments are sane
}

Minimizer::Minimizer(const HostMagnet* magnet,
                     real stopMaxMagDiff,
                     int nMagDiffSamples,
                     real stopMaxElDiff,
                     int nElDiffSamples,
                     real stepsize,
                     real stepsizeEl)
    : magnets_(magnet->sublattices()),
      nMagDiffSamples_(nMagDiffSamples),
      stopMaxMagDiff_(stopMaxMagDiff),
      nElDiffSamples_(nElDiffSamples),
      stopMaxElDiff_(stopMaxElDiff),
      t0(magnets_.size()),
      t1(magnets_.size()),
      m0(magnets_.size()),
      m1(magnets_.size()) {
  stepsizes_.assign(magnets_.size(), stepsize);
  stepsizeInit_ = stepsize;
  for (auto sub : magnets_)
    torques_.push_back(relaxTorqueQuantity(sub));

  // Elastic displacement lives on the host, not per sublattice.
  addElasticIfPresent(magnet, elMagnets_, forces_);
  f0.resize(elMagnets_.size());
  f1.resize(elMagnets_.size());
  u0.resize(elMagnets_.size());
  u1.resize(elMagnets_.size());
  elStepsizes_.assign(elMagnets_.size(), stepsizeEl);
  stepsizeElInit_ = stepsizeEl;
}

Minimizer::Minimizer(const MumaxWorld* world,
                     real stopMaxMagDiff,
                     int nMagDiffSamples,
                     real stopMaxElDiff,
                     int nElDiffSamples,
                     real stepsize,
                     real stepsizeEl)
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

  stepsizes_.assign(N, stepsize);
  stepsizeInit_ = stepsize;

  nElDiffSamples_ = nElDiffSamples * elMagnets_.size();
  f0.resize(elMagnets_.size());
  f1.resize(elMagnets_.size());
  u0.resize(elMagnets_.size());
  u1.resize(elMagnets_.size());
  elStepsizes_.assign(elMagnets_.size(), stepsizeEl);
  stepsizeElInit_ = stepsizeEl;
}


void Minimizer::exec() {
  nsteps_ = 0;
  lastMagDiffs_.clear();
  lastElDiffs_.clear();
  const int maxSteps = 1000;  // TODO: make configurable if this isn't enough
  const int printEvery = 50;    // tweak: smaller = more frequent updates

  while (!converged() && nsteps_ < maxSteps) {
    step();

    if (nsteps_ % printEvery == 0) {
      real lastMagDiff = lastMagDiffs_.empty() ? -1 : lastMagDiffs_.back();
      real lastElDiff = lastElDiffs_.empty() ? -1 : lastElDiffs_.back();
      std::cerr << "[minimize] step " << nsteps_
                << "  magDiff=" << lastMagDiff
                << "  elDiff=" << lastElDiff
                << std::endl;
    }
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

// Plain steepest-descent step for the elastic displacement. Unlike
// magnetization, displacement has no unit-length constraint, so there is no
// Cayley-type renormalizing update here -- just u1 = u0 + dt * f.
// NOTE: assumes no pinning / fixed-volume projection (not yet implemented
// upstream). If/when those land, and if they are enforced at the force level
// (e.g. by zeroing the effective body force at pinned cells, mirroring how
// frozen_spins zeroes torque), this kernel needs no changes at all.
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

static inline real BarzilianBorweinStepSize(Field& dm, Field& dtorque, int n, real fallback) {
  real nom, div;
  if (n % 2 == 0) {
    nom = dotSum(dm, dm);
    div = dotSum(dm, dtorque);
  } else {
    nom = dotSum(dm, dtorque);
    div = dotSum(dtorque, dtorque);
  }

  if (div == 0.0)
    return fallback;

  real h = nom / div;

  if (!(h > 0.0) || std::isnan(h) || std::isinf(h))
    return fallback;

  // Heuristic upper bound: don't let a single bad curvature estimate produce
  // a step size wildly larger than the known-stable fallback scale. The
  // multiplier is a tuning knob, not physically derived.
  const real maxGrowthFactor = 10.0;
  if (h > maxGrowthFactor * fallback)
    h = maxGrowthFactor * fallback;

  return h;
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

    stepsizes_[i] = BarzilianBorweinStepSize(dm, dt, nsteps_, stepsizeInit_);

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

    real qnorm = maxVecNorm(f0[i]);

    // Clamp so a single step never moves displacement more than 10% of the
    // smallest cellsize -- a physically meaningful bound, not a numeric guess.
    real3 cs = elMagnets_[i]->system()->cellsize();
    real minCellSize = std::min({cs.x, cs.y, cs.z});
    real maxDu = real(0.1) * minCellSize;

    real h = elStepsizes_[i];
    real hBeforeClamp = h;
    bool clamped = false;
    if (qnorm > 0) {
      real step = h * qnorm;
      if (step > maxDu) {
        h *= maxDu / step;
        clamped = true;
      }
    }
    elStepsizes_[i] = h;

    std::cerr << "[elastic] step=" << nsteps_
              << " i=" << i
              << " h_pre=" << hBeforeClamp
              << " h_post=" << h
              << " clamped=" << clamped
              << " |f|=" << qnorm
              << std::endl;

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

    // For BB purposes we need a gradient-like quantity, and force = -gradient.
    // Near a stable equilibrium (f ~ -K*u), f1-f0 is *always* anti-correlated
    // with du (dot(du, f1-f0) < 0 for any positive-definite K), which is why
    // every step here was landing on a negative/degenerate step size and
    // falling back to the floor. Using dg = f0-f1 = -(f1-f0) restores the
    // sign BB expects, matching the reversed convention already used for
    // torque (t0-t1) in stepMagnetic().
    Field dg = add(real(-1), f1[i], real(+1), f0[i]);

    real maxDu = maxVecNorm(du);
    real maxDg = maxVecNorm(dg);
    real duScale = (maxDu > 0) ? real(1.0) / maxDu : real(1.0);
    real dgScale = (maxDg > 0) ? real(1.0) / maxDg : real(1.0);

    Field duScaled = duScale * du;
    Field dgScaled = dgScale * dg;

    // Exact, since Dot(a*k, b*k) = k^2 * Dot(a,b) -- dividing back out
    // recovers the true unscaled dot product without ever squaring/
    // multiplying raw quantities that are many orders of magnitude apart.
    real dudu = dotSum(duScaled, duScaled) / (duScale * duScale);
    real dudg = dotSum(duScaled, dgScaled) / (duScale * dgScale);
    real dgdg = dotSum(dgScaled, dgScaled) / (dgScale * dgScale);

    std::cerr << "[elastic] step=" << nsteps_
              << " i=" << i
              << " max_du=" << maxDu
              << " duScale=" << duScale
              << " dgScale=" << dgScale
              << " dot(du,du)=" << dudu
              << " dot(du,dg)=" << dudg
              << " dot(dg,dg)=" << dgdg
              << std::endl;

    real nom, div;
    bool usedFallback = false;
    if (nsteps_ % 2 == 0) {
      nom = dudu;
      div = dudg;
      if (nom == 0.0) {
        std::cerr << "[elastic] WARNING: BB1 underflow at step=" << nsteps_
                   << " i=" << i << " (dot(du,du)=0), falling back to BB2"
                   << std::endl;
        nom = dudg;
        div = dgdg;
        usedFallback = true;
      }
    } else {
      nom = dudg;
      div = dgdg;
    }

    if (div == 0.0) {
      std::cerr << "[elastic] WARNING: step-size denominator underflow at step="
                 << nsteps_ << " i=" << i << " (div=0)"
                 << (usedFallback ? " [after BB1 fallback]" : "")
                 << std::endl;
    } else {
      real newH = nom / div;
      if (newH > 0.0 && !std::isnan(newH) && !std::isinf(newH)) {
        elStepsizes_[i] = newH;
      } else {
        std::cerr << "[elastic] WARNING: negative/degenerate step size at step="
                   << nsteps_ << " i=" << i << " (newH=" << newH
                   << "), keeping previous h=" << elStepsizes_[i]
                   << std::endl;
      }
    }

    addElDiff(maxDu);
  }
}


void Minimizer::step() {
  stepMagnetic();
  stepElastic();
  nsteps_ += 1;
}

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