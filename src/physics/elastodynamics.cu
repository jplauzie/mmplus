#include "antiferromagnet.hpp"
#include "cudalaunch.hpp"
#include "elasticdamping.hpp"
#include "internalbodyforce.hpp"
#include "elastodynamics.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "magnet.hpp"
#include "magnetoelasticfield.hpp"
#include "magnetoelasticforce.hpp"
#include "ncafm.hpp"
#include "parameter.hpp"
#include "reduce.hpp"
#include "straintensor.hpp"
#include "stresstensor.hpp"
#include <iostream>



bool elasticityAssuredZero(const Magnet* magnet) {
  return ((!magnet->enableElastodynamics()) ||
          (magnet->C11.assuredZero() && magnet->C12.assuredZero() &&
           magnet->C44.assuredZero()));
}

// ========== Effective Body Force ==========

Field evalEffectiveBodyForce(const Magnet* magnet) {
  Field fField = evalInternalBodyForce(magnet);  // safely 0 if assuredZero

  if (!magnet->externalBodyForce.assuredZero())
    fField += magnet->externalBodyForce;

  if (auto host = magnet->asHost()) {
    // add magnetoelastic force of all sublattices
    for (const Ferromagnet* sub : host->sublattices()) {
      if (!magnetoelasticAssuredZero(sub))
        fField += evalMagnetoelasticForce(sub);
    }
  }
  else {
    // add magnetoelastic force of independent ferromagnet
    const Ferromagnet* fm = magnet->asFM();
    if (!magnetoelasticAssuredZero(fm))
      fField += evalMagnetoelasticForce(fm);
  }

  return fField;
}

M_FieldQuantity effectiveBodyForceQuantity(const Magnet* magnet) {
    return M_FieldQuantity(magnet, evalEffectiveBodyForce, 3,
                           "effective_body_force", "N/m3");
}

// ========== Elastic Accelleration ==========

__global__ void k_divideByParam(CuField field, const CuParameter param) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = field.system;
  const Grid grid = system.grid;

  // When outside the geometry, set to zero and return early
  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      field.setVectorInCell(idx, real3{0, 0, 0});
    }
    return;
  }

  real p = param.valueAt(idx);
  if (p != 0) {
    field.setVectorInCell(idx, field.vectorAt(idx) / p);
  } else {
    field.setVectorInCell(idx, real3{0, 0, 0});  // substitue for infinity
  }
}

// Deterministic pseudo-random values in [-1, 1) inside the geometry, 0 outside.
__global__ void k_fillPseudoRandom(CuField f) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = f.system;
  if (!system.grid.cellInGrid(idx)) return;
  const bool inGeo = system.inGeometry(idx);
  for (int c = 0; c < f.ncomp; c++) {
    unsigned int s = 3u * (unsigned int)idx + (unsigned int)c;
    s = s * 747796405u + 2891336453u;  // PCG hash
    s = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    s = (s >> 22u) ^ s;
    f.setValueInCell(idx, c, inGeo ? (real)(s & 0xFFFFFFu) / (real)0x800000 - (real)1 : (real)0);
  }
}

// Largest angular frequency (rad/s) of the discrete elastic operator, estimated by
// power iteration on  A v = -(1/rho) div( C : strain(v) )  (no damping, no traction).
// The iteration converges from below, so 2% margin is added.
real estimateMaxOmega(const Magnet* magnet) {
  if (elasticityAssuredZero(magnet))
    return 0.0;

  Field v(magnet->system(), 3);
  const int ncells = v.grid().ncells();
  cudaLaunch(ncells, k_fillPseudoRandom, v.cu());
  v = (1 / std::sqrt(dotSum(v, v))) * v;  // entries ~ 1/sqrt(N): no overflow

  real lambda = 0;
  for (int it = 0; it < 50; it++) {
    Field strain = evalStrainTensorOf(magnet, v);
    Field stress = evalElasticStressFromStrain(magnet, strain);
    Field a = evalStressDivergence(magnet, stress, false);          // = -K v
    cudaLaunch(ncells, k_divideByParam, a.cu(), magnet->rho.cu());  // = -(1/rho) K v

    // ||a|| = amax * ||a/amax||; avoids squaring huge numbers (a ~ 1e22, a^2 ~ 1e45)
    const real amax = maxAbsValue(a);
    if (!(amax > 0) || !std::isfinite(amax)) {
      std::cerr << "[cfl] omega_max estimate failed at iteration " << it
                << ", max|a| = " << amax << std::endl;
      return 0.0;
    }
    Field aScaled = (1 / amax) * a;                     // entries in [-1, 1]
    const real scaledNorm = std::sqrt(dotSum(aScaled, aScaled));
    lambda = amax * scaledNorm;                          // ||A v|| with ||v|| = 1
    v = (1 / scaledNorm) * aScaled;                      // unit 2-norm again
  }
  return (real)1.02 * std::sqrt(lambda);
}

Field evalElasticAcceleration(const Magnet* magnet) {
  Field aField = evalEffectiveBodyForce(magnet) + evalElasticDamping(magnet);
  
  // divide by rho if possible
  if (!magnet->rho.assuredZero())
    cudaLaunch(aField.grid().ncells(), k_divideByParam,
               aField.cu(), magnet->rho.cu());

  return aField;
}

M_FieldQuantity elasticAccelerationQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalElasticAcceleration, 3,
                         "elastic_acceleration", "m/s2");
}

// ========== Elastic Velocity Quantity ==========

M_FieldQuantity elasticVelocityQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet,
       [](const Magnet* magnet){return magnet->elasticVelocity()->eval();},
                         3, "elastic_velocity", "m/s");
}
