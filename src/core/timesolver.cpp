#include "timesolver.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <string>

#include "field.hpp"
#include "fieldquantity.hpp"
#include "reduce.hpp"
#include "rungekutta.hpp"
#include "stepper.hpp"

namespace {
// Largest omega_max*dt for which the method does not amplify the highest-frequency
// elastic mode. Stability only (no accuracy criterion), evaluated with the default
// stiffness damping, times a safety factor of 0.8.
real stableOmegaDt(RKmethod method) {
  const real safety = 0.8;
  switch (method) {
    case RKmethod::HEUN:             return 0.5;  // placeholder: anti-damped at any step
    case RKmethod::BOGACKI_SHAMPINE: return safety * 2.0;
    case RKmethod::CASH_KARP:        return safety * 2.6;
    case RKmethod::FEHLBERG:         return safety * 3.5;
    case RKmethod::DORMAND_PRINCE:   return safety * 2.4;
    default:                         return safety * 2.4;  // get() falls back to Dormand-Prince
  }
}
}  // namespace

std::unique_ptr<TimeSolver> TimeSolver::Factory::create() {
  return std::unique_ptr<TimeSolver>(new TimeSolver());
}

TimeSolver::TimeSolver() {
  setRungeKuttaMethod(RKmethod::FEHLBERG);
}

TimeSolver::~TimeSolver() {}

void TimeSolver::setRungeKuttaMethod(RKmethod method) {
  stepper_ = std::make_unique<RungeKuttaStepper>(this, method);
  cflTriggerCount_ = 0;
  cflNextReport_ = 1;
  cflMaxOvershoot_ = 1.0;
  if (!fixedTimeStep_) timestep_ = sensibleTimeStep();
  method_ = method;
}

void TimeSolver::setRungeKuttaMethod(const std::string& method) {
  setRungeKuttaMethod(getRungeKuttaMethodFromName(method));
}

RKmethod TimeSolver::getRungeKuttaMethod() {
  return method_;
}

real TimeSolver::sensibleTimeStep() const {
  if (eqs_.empty())
    return 0.0;  // Timestep is irrelevant if there are no equations to solve

  // find global maxima
  real maxRhs = 0;
  real maxNoise = 0;
  for (auto eq : eqs_) {
    if (real maxNorm = maxVecNorm(eq.rhs->eval()); maxNorm > maxRhs)
      maxRhs = maxNorm;

    if (eq.noiseTerm && !eq.noiseTerm->assuredZero()) {
      // TODO: replace by expected maximum prefactor of noise, depending on the
      // number of cells, without curand noise generation?
      real eqMaxNoise = maxVecNorm(eq.noiseTerm->eval());
      if (eqMaxNoise > maxNoise) maxNoise = eqMaxNoise;
    }
  }

  if (maxNoise == 0) {
    // Sensible timestep cannot be calculated if torque is zero
    if (maxRhs == 0) return sensibleTimestepDefault();
    // with RHS but no noise
    return sensibleFactor() / maxRhs;
  }
  // negligible RHS compared to noise
  // valid for small values in the binomial approximation:
  // sqrt(1 + 4 fR/N²) ≈ 1 + 2 fR/N²
  // but values still larger than numerical noise (~1e-7)
  real smallNumber = 1e-3;
  if (2 * sensibleFactor() * maxRhs / (maxNoise * maxNoise) < smallNumber) {
    // return solution of dt = sensibleFactor / (maxNoise / sqrt(dt))
    return pow(sensibleFactor() / maxNoise , 2);
  }
  // RHS and noise
  // returns solution of dt = sensibleFactor / (maxRhs + maxNoise/sqrt(dt))
  real D = pow(maxNoise, 2) + 4 * maxRhs * sensibleFactor();  // discriminant
  return pow((sqrt(D) - maxNoise) / (2 * maxRhs), 2);
}

void TimeSolver::setEquations(std::vector<DynamicEquation> eqs) {
  eqs_ = eqs;
  cflMaxOmega_ = -1.0;  // force a new stability estimate at the next step
  if (!fixedTimeStep_) timestep_ = sensibleTimeStep();
}

real TimeSolver::maxStableTimestep() {
  if (cflMaxOmega_ < 0) {  // estimate once, after all setup is done
    cflMaxOmega_ = 0.0;
    cflTriggerCount_ = 0;
    cflNextReport_ = 1;
    cflMaxOvershoot_ = 1.0;
    for (const auto& eq : eqs_)
      if (eq.maxOmega)
        cflMaxOmega_ = std::max(cflMaxOmega_, eq.maxOmega());
    if (cflMaxOmega_ > 0)
      std::cerr << "[cfl] estimated omega_max = " << cflMaxOmega_
                << " rad/s, stable timestep limit = "
                << stableOmegaDt(method_) / cflMaxOmega_ << " s ("
                << getRungeKuttaNameFromMethod(method_) << ")" << std::endl;
  }
  if (cflMaxOmega_ <= 0)
    return 0.0;  // no elastodynamics: no limit
  return stableOmegaDt(method_) / cflMaxOmega_;
}

void TimeSolver::setSensibleTimestepDefault(real dt) {
  if (dt < 0)
    throw std::runtime_error("The sensible timestep should be larger than zero.");
  sensibleTimestepDefault_ = dt;
}

void TimeSolver::adaptTimeStep(real correctionFactor) {
  if (fixedTimeStep_)
    return;

  if (std::isnan(correctionFactor))
    correctionFactor = 1.;

  correctionFactor *= headroom_;
  if (lowerBound_ >= upperBound_) {
    throw std::runtime_error("The lower bound should be lower than the upper bound.");
  }
  correctionFactor = correctionFactor > upperBound_ ? upperBound_ : correctionFactor;
  correctionFactor = correctionFactor < lowerBound_ ? lowerBound_ : correctionFactor;

  timestep_ *= correctionFactor;
}

void TimeSolver::step() {
  if (timestep_ <= 0)
    throw std::runtime_error(
        "Timesolver can not make a step because the timestep is smaller than "
        "or equal to zero.");

  const real dtMax = maxStableTimestep();
  if (dtMax > 0 && timestep_ > dtMax) {
    cflTriggerCount_++;
    cflMaxOvershoot_ = std::max(cflMaxOvershoot_, timestep_ / dtMax);

    // print only at trigger #1, #10, #100, ...
    if (cflTriggerCount_ >= cflNextReport_) {
      std::cerr << "WARNING: timestep " << timestep_
                << " s exceeds the estimated elastodynamics stability limit "
                << dtMax << " s (omega_max = " << cflMaxOmega_ << " rad/s, "
                << getRungeKuttaNameFromMethod(method_) << "). "
                << (fixedTimeStep_
                        ? "Fixed timestep kept; the simulation may become unstable."
                        : "Timestep reduced to the limit.")
                << " [triggered " << cflTriggerCount_
                << " time(s), largest overshoot x" << cflMaxOvershoot_
                << ", next report at trigger " << 10 * cflNextReport_ << "]"
                << std::endl;
      cflNextReport_ *= 10;
    }

    if (!fixedTimeStep_)
      timestep_ = dtMax;  // clamp on every trigger, whether or not it was reported
  }
  stepper_->step();
}

void TimeSolver::steps(unsigned int nSteps) {
  for (int i = 0; i < nSteps; i++) {
    step();
  }
}

void TimeSolver::runwhile(std::function<bool(void)> runcondition) {
  while (runcondition()) {
    step();
  }
}

void TimeSolver::run(real duration) {
  if (duration <= 0)
    return;
  real stoptime = time_ + duration;
  auto runcondition = [this, stoptime]() {
    return this->time() < stoptime - this->timestep();
  };
  runwhile(runcondition);

  // make final time step to end exactly at stoptime
  real oldTimestep = timestep();
  setTimeStep(stoptime - time_);
  step();
  postStep();
  if (fixedTimeStep_) setTimeStep(oldTimestep);
}
