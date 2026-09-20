#include "cudalaunch.hpp"
#include "elastodynamics.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "magnetoelasticfield.hpp"  // magnetoelasticAssuredZero
#include "magnetoelasticforce.hpp"
#include "parameter.hpp"


#include "internalbodyforce.hpp"  // evalStressDivergence

__global__ void k_magnetoelasticStress(CuField stress,
                                       const CuField m,
                                       const CuParameter B1,
                                       const CuParameter B2) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = stress.system;
  if (!system.inGeometry(idx)) {
    if (system.grid.cellInGrid(idx))
      for (int c = 0; c < stress.ncomp; c++) stress.setValueInCell(idx, c, 0);
    return;
  }
  const real3 m0 = m.vectorAt(idx);
  const real b1 = B1.valueAt(idx), b2 = B2.valueAt(idx);
  stress.setValueInCell(idx, 0, b1 * m0.x * m0.x);  // xx
  stress.setValueInCell(idx, 1, b1 * m0.y * m0.y);  // yy
  stress.setValueInCell(idx, 2, b1 * m0.z * m0.z);  // zz
  stress.setValueInCell(idx, 3, b2 * m0.x * m0.y);  // xy
  stress.setValueInCell(idx, 4, b2 * m0.x * m0.z);  // xz
  stress.setValueInCell(idx, 5, b2 * m0.y * m0.z);  // yz
}

Field evalMagnetoelasticForce(const Ferromagnet* magnet) {
  if (magnetoelasticAssuredZero(magnet)) {
    Field fField(magnet->system(), 3);
    fField.makeZero();
    return fField;
  }
  Field stress(magnet->system(), 6);
  cudaLaunch(stress.grid().ncells(), k_magnetoelasticStress, stress.cu(),
             magnet->magnetization()->field().cu(), magnet->B1.cu(), magnet->B2.cu());
  return evalStressDivergence(magnet, stress, false);  // traction only acts on the host
}


FM_FieldQuantity magnetoelasticForceQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalMagnetoelasticForce, 3,
                          "magnetoelastic_force", "N/m3");
}
