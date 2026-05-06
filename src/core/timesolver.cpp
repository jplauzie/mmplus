#include "timesolver.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>

#include "field.hpp"
#include "fieldquantity.hpp"
#include "reduce.hpp"
#include "rungekutta.hpp"
#include "stepper.hpp"

std::unique_ptr<TimeSolver> TimeSolver::Factory::create() {
  return std::unique_ptr<TimeSolver>(new TimeSolver());
}

TimeSolver::TimeSolver() {
  setRungeKuttaMethod(RKmethod::FEHLBERG);
}

TimeSolver::~TimeSolver() {}

void TimeSolver::setRungeKuttaMethod(RKmethod method) {
  stepper_ = std::make_unique<RungeKuttaStepper>(this, method);
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
      // TODO: could be replaced by maximum prefactor of noise, without curand
      // noise generation, but with some sensible expected maximum
      real localMaxNoise = maxVecNorm(eq.noiseTerm->eval());
      if (localMaxNoise > maxNoise) maxNoise = localMaxNoise;
    }
  }

  if (maxRhs == 0) {
    // Sensible timestep cannot be calculated if torque is zero
    if (maxNoise == 0) return sensibleTimestepDefault();
    // or only noise
    // return solution of dt = sensibleFactor / (maxNoise / sqrt(dt))
    // TODO: check that this makes sense
    return pow(sensibleFactor() / maxNoise , 2);
  }
  // with rhs but no noise
  if (maxNoise == 0) return sensibleFactor() / maxRhs;

  // rhs and noise
  // returns solution of dt = sensibleFactor / (maxRhs + maxNoise/sqrt(dt))
  real D = pow(maxNoise, 2) + 4 * maxRhs * sensibleFactor();  // discriminant
  return pow((sqrt(D) - maxNoise) / (2 * maxRhs), 2);
}

void TimeSolver::setEquations(std::vector<DynamicEquation> eqs) {
  eqs_ = eqs;
  if (!fixedTimeStep_) timestep_ = sensibleTimeStep();
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
  if (fixedTimeStep_) setTimeStep(oldTimestep);
}
