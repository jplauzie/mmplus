#include <stdexcept>
#include <vector>

#include "cudalaunch.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "reduce.hpp"
#include "rigidbodymodes.hpp"
#include "system.hpp"

namespace {

// result[i] = cross(a[i], b[i]), cell by cell
__global__ void k_crossProduct(CuField result, CuField a, CuField b) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!result.cellInGrid(idx))
    return;
  result.setVectorInCell(idx, cross(a.vectorAt(idx), b.vectorAt(idx)));
}

Field crossProduct(const Field& a, const Field& b) {
  Field result(a.system(), 3);
  int ncells = result.grid().ncells();
  cudaLaunch(ncells, k_crossProduct, result.cu(), a.cu(), b.cu());
  return result;
}

void invert3x3(const double I[3][3], real out[3][3]) {
  double det =
      I[0][0]*(I[1][1]*I[2][2] - I[1][2]*I[2][1]) -
      I[0][1]*(I[1][0]*I[2][2] - I[1][2]*I[2][0]) +
      I[0][2]*(I[1][0]*I[2][1] - I[1][1]*I[2][0]);

  if (det == 0.0)
    throw std::runtime_error("Rigid-body inertia tensor is singular even "
                             "after regularization.");

  double invDet = 1.0 / det;
  out[0][0] = real(( I[1][1]*I[2][2] - I[1][2]*I[2][1]) * invDet);
  out[0][1] = real((-I[0][1]*I[2][2] + I[0][2]*I[2][1]) * invDet);
  out[0][2] = real(( I[0][1]*I[1][2] - I[0][2]*I[1][1]) * invDet);
  out[1][0] = real((-I[1][0]*I[2][2] + I[1][2]*I[2][0]) * invDet);
  out[1][1] = real(( I[0][0]*I[2][2] - I[0][2]*I[2][0]) * invDet);
  out[1][2] = real((-I[0][0]*I[1][2] + I[0][2]*I[1][0]) * invDet);
  out[2][0] = real(( I[1][0]*I[2][1] - I[1][1]*I[2][0]) * invDet);
  out[2][1] = real((-I[0][0]*I[2][1] + I[0][1]*I[2][0]) * invDet);
  out[2][2] = real(( I[0][0]*I[1][1] - I[0][1]*I[1][0]) * invDet);
}

}  

RigidBodyGeometry computeRigidBodyGeometry(std::shared_ptr<const System> system) {
  Grid grid = system->grid();
  int ncells = grid.ncells();

  std::vector<bool> geomMask = system->geometry().getData();
  bool hasMask = !geomMask.empty();  // empty buffer == "everything included"

  // mass-weighted center of mass (COM) (host, one-time). Should probably be a cuda kernel
  double sx = 0, sy = 0, sz = 0;
  int count = 0;
  for (int i = 0; i < ncells; i++) {
    if (hasMask && !geomMask[i]) continue;
    real3 p = system->cellPosition(grid.index2coord(i));
    sx += p.x; sy += p.y; sz += p.z;
    count++;
  }
  if (count == 0)
    throw std::runtime_error("computeRigidBodyGeometry: empty geometry.");
  real3 com = {real(sx / count), real(sy / count), real(sz / count)};

  // inertia tensor about COM, and relPos data in the same pass 
  double I[3][3] = {{0}};
  std::vector<real> relPosData(3 * ncells, real(0));  // zero outside geometry
  for (int i = 0; i < ncells; i++) {
    if (hasMask && !geomMask[i]) continue;
    real3 p = system->cellPosition(grid.index2coord(i));
    real3 r = {p.x - com.x, p.y - com.y, p.z - com.z};

    relPosData[i]              = r.x;
    relPosData[ncells + i]     = r.y;
    relPosData[2 * ncells + i] = r.z;

    double x = r.x, y = r.y, z = r.z;
    I[0][0] += y*y + z*z;  I[0][1] += -x*y;      I[0][2] += -x*z;
    I[1][0] += -x*y;       I[1][1] += x*x + z*z; I[1][2] += -y*z;
    I[2][0] += -x*z;       I[2][1] += -y*z;      I[2][2] += x*x + y*y;
  }


  //Claude trick, this should probably be perpindicular axis theorem or something. temporary.
  // see Numerical Recipes 3rd Ed, 19.5, Linear Regularization Methods

  // Tikhonov regularization: thin/2D geometries make some rotation axes
  // genuinely undetectable (that block of I is singular, not just small).
  // Without this, the corresponding omega component may blow up on noise
  // instead of as zero.
  double trace = I[0][0] + I[1][1] + I[2][2];
  double lambda = 1e-10 * trace;
  for (int d = 0; d < 3; d++) I[d][d] += lambda;

  RigidBodyGeometry geom;
  geom.relPos = Field(system, 3);
  geom.relPos.setData(relPosData);
  invert3x3(I, geom.Iinv);
  return geom;
}

void removeRigidBodyModes(Field& f, const RigidBodyGeometry& geom) {
  // Translation: fieldAverage is already geometry-masked (divides by
  // cellsInGeo()), so this is exactly the mean rigid translation.
  std::vector<real> T = fieldAverage(f);

  // Rotation: L = sum_i cross(r_i, f_i); fieldAverage(L_field)*cellsInGeo()
  // recovers the sum from the mean, reusing the same masked reduction.
  Field Lfield = crossProduct(geom.relPos, f);
  std::vector<real> Lavg = fieldAverage(Lfield);
  int cellsInGeo = f.system()->cellsInGeo();
  real3 L = {Lavg[0] * cellsInGeo, Lavg[1] * cellsInGeo, Lavg[2] * cellsInGeo};

  real3 omega;
  omega.x = geom.Iinv[0][0]*L.x + geom.Iinv[0][1]*L.y + geom.Iinv[0][2]*L.z;
  omega.y = geom.Iinv[1][0]*L.x + geom.Iinv[1][1]*L.y + geom.Iinv[1][2]*L.z;
  omega.z = geom.Iinv[2][0]*L.x + geom.Iinv[2][1]*L.y + geom.Iinv[2][2]*L.z;

  // uRigid(r) = T + omega x r
  Field omegaCrossR = crossProduct(Field(f.system(), 3, omega), geom.relPos);
  real3 Treal3 = {T[0], T[1], T[2]};
  addTo(omegaCrossR, Treal3, Field(f.system(), 3, real3{1,1,1}));  // += T uniformly... 
  addTo(f, real(-1), omegaCrossR);
}