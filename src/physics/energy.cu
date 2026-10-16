#include "afmexchange.hpp"
#include "altermagnet.hpp"
#include "anisotropy.hpp"
#include "antiferromagnet.hpp"
#include "atmexchange.hpp"
#include "cudalaunch.hpp"
#include "demag.hpp"
#include "dmi.hpp"
#include "elasticenergies.hpp"
#include "elastodynamics.hpp"
#include "energy.hpp"
#include "exchange.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "local_dmi.hpp"
#include "magnetoelasticfield.hpp"
#include "ncafm.hpp"
#include "reduce.hpp"
#include "world.hpp"
#include "zeeman.hpp"

__global__ void k_energyDensity(CuField edens,
                                const CuField mag,
                                const CuField hfield,
                                const CuParameter msat,
                                const real prefactor) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!edens.cellInGeometry(idx)) {
    if (edens.cellInGrid(idx))
      edens.setValueInCell(idx, 0, 0.0);
    return;
  }
  real Ms = msat.valueAt(idx);
  real3 h = hfield.vectorAt(idx);
  real3 m = mag.vectorAt(idx);
  edens.setValueInCell(idx, 0, -prefactor * Ms * dot(m, h));
}

Field evalEnergyDensity(const Ferromagnet* magnet, const Field& h, real prefactor) {
  Field edens(magnet->system(), 1);
  if (magnet->msat.assuredZero()) {
    edens.makeZero();
    return edens;
  }

  cudaLaunch(edens.grid().ncells(), k_energyDensity, edens.cu(),
             magnet->magnetization()->field().cu(), h.cu(), magnet->msat.cu(),
             prefactor);
  return edens;
}

real energyFromEnergyDensity(const Magnet* magnet, real edens) {
  int ncells = magnet->system()->cellsInGeo();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * cellVolume * edens;
}

Field evalTotalEnergyDensity(const Ferromagnet* magnet) {
  Field edens(magnet->system(), 1, 0.0);
  if (!exchangeAssuredZero(magnet)) {
    edens += evalExchangeEnergyDensity(magnet);
  }
  if (!anisotropyAssuredZero(magnet)) {
    edens += evalAnisotropyEnergyDensity(magnet);
  }
  if (!externalFieldAssuredZero(magnet)) {
    edens += evalZeemanEnergyDensity(magnet);
  }
  if (!inhomoDmiAssuredZero(magnet)) {
    edens += evalDmiEnergyDensity(magnet);
  }
  if (!homoDmiAssuredZero(magnet)) {
    edens += evalHomoDmiEnergyDensity(magnet);
  }
  if (!demagFieldAssuredZero(magnet)) {
    edens += evalDemagEnergyDensity(magnet);
  }
  if (!homoAfmExchangeAssuredZero(magnet)) {
    edens += evalHomoAfmExchangeEnergyDensity(magnet);
  }
  if (!inHomoAfmExchangeAssuredZero(magnet)) {
    edens += evalInHomoAfmExchangeEnergyDensity(magnet);
  }
  if (!atmExchangeAssuredZero(magnet)) {
    edens += evalAtmExchangeEnergyDensity(magnet);
  }
  // magnetoelastics; works if host or if sublattice
  if (!magnetoelasticAssuredZero(magnet)) {
    edens += evalMagnetoelasticEnergyDensity(magnet);
  }
  // elastics; only works if independent host
  if (!kineticEnergyAssuredZero(magnet)) {
    edens += evalKineticEnergyDensity(magnet);
  }
  if (!elasticityAssuredZero(magnet)) {
    edens += evalElasticEnergyDensity(magnet);
  }
  return edens;
}

// acc += |term| per cell
__global__ void k_addAbs(CuField acc, const CuField term) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!acc.cellInGrid(idx) || !acc.cellInGeometry(idx))
    return;
  acc.setValueInCell(idx, 0, acc.valueAt(idx, 0) + fabs(term.valueAt(idx, 0)));
}

// NOTE: keep the term list in sync with evalTotalEnergyDensity(const Ferromagnet*).
void evalTotalEnergyAndScale(const Ferromagnet* magnet, real& E, real& scale) {
  Field edens(magnet->system(), 1, 0.0);
  Field absAcc(magnet->system(), 1, 0.0);
  auto addTerm = [&](const Field& term) {
    edens += term;
    cudaLaunch(absAcc.grid().ncells(), k_addAbs, absAcc.cu(), term.cu());
  };

  if (!exchangeAssuredZero(magnet))
    addTerm(evalExchangeEnergyDensity(magnet));
  if (!anisotropyAssuredZero(magnet))
    addTerm(evalAnisotropyEnergyDensity(magnet));
  if (!externalFieldAssuredZero(magnet))
    addTerm(evalZeemanEnergyDensity(magnet));
  if (!inhomoDmiAssuredZero(magnet))
    addTerm(evalDmiEnergyDensity(magnet));
  if (!homoDmiAssuredZero(magnet))
    addTerm(evalHomoDmiEnergyDensity(magnet));
  if (!demagFieldAssuredZero(magnet))
    addTerm(evalDemagEnergyDensity(magnet));
  if (!homoAfmExchangeAssuredZero(magnet))
    addTerm(evalHomoAfmExchangeEnergyDensity(magnet));
  if (!inHomoAfmExchangeAssuredZero(magnet))
    addTerm(evalInHomoAfmExchangeEnergyDensity(magnet));
  if (!atmExchangeAssuredZero(magnet))
    addTerm(evalAtmExchangeEnergyDensity(magnet));
  if (!magnetoelasticAssuredZero(magnet))
    addTerm(evalMagnetoelasticEnergyDensity(magnet));
  if (!kineticEnergyAssuredZero(magnet))
    addTerm(evalKineticEnergyDensity(magnet));
  if (!elasticityAssuredZero(magnet))
    addTerm(evalElasticEnergyDensity(magnet));

  E = energyFromEnergyDensity(magnet, fieldComponentAverage(edens, 0));
  scale = energyFromEnergyDensity(magnet, fieldComponentAverage(absAcc, 0));
}

Field evalTotalEnergyDensity(const HostMagnet* magnet) {
  Field edens(magnet->system(), 1, 0.0);

  for (const Ferromagnet* sub : magnet->sublattices())
    edens += evalTotalEnergyDensity(sub);

  if (!kineticEnergyAssuredZero(magnet)) {
    edens += evalKineticEnergyDensity(magnet);
  }
  if (!elasticityAssuredZero(magnet)) {
    edens += evalElasticEnergyDensity(magnet);
  }
  return edens;
}

real evalTotalEnergy(const Magnet* magnet) {
  if (const Ferromagnet* mag = magnet->asFM())
    return energyFromEnergyDensity(mag, totalEnergyDensityQuantity(mag).average()[0]);
  if (const HostMagnet* mag = magnet->asHost())
    return energyFromEnergyDensity(mag, totalEnergyDensityQuantity(mag).average()[0]);
  throw std::invalid_argument("Cannot calculate energy of unknown magnet type.");
}
