#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Field;
class HostMagnet;
class Magnet;

// Returns the energy density of an effective field term.
//   edens = - prefactor * Msat * dot(m,h)
// The prefactor depends on the origin of the effective field term
Field evalEnergyDensity(const Ferromagnet*, const Field&, real prefactor);

real energyFromEnergyDensity(const Magnet*, real);

Field evalTotalEnergyDensity(const Ferromagnet*);
Field evalTotalEnergyDensity(const HostMagnet*);
real evalTotalEnergy(const Magnet*);

// Energy of a ferromagnet together with the noise scale of that energy:
//   scale = V * sum over cells and terms of |e_term|   (always >= |E|)
// Cancellation between terms and between cells makes |E| alone too small a scale
// for float32 rounding noise in E. E equals evalTotalEnergy(magnet).
// MUST list the same terms as evalTotalEnergyDensity(const Ferromagnet*).
void evalTotalEnergyAndScale(const Ferromagnet*, real& E, real& scale);

template <class T>
FieldQuantityEvaluator<T> totalEnergyDensityQuantity(const T* magnet) {
  return FieldQuantityEvaluator<T>(
      magnet, [](const T* m) { return evalTotalEnergyDensity(m); }, 1,
      "total_energy_density", "J/m3");
}

template <class T>
ScalarQuantityEvaluator<T> totalEnergyQuantity(const T* magnet) {
  return ScalarQuantityEvaluator<T>(
      magnet, [](const T* m) { return evalTotalEnergy(m); }, "total_energy", "J");
}
