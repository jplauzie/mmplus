#include "cudalaunch.hpp"
#include "derivativestencil.hpp"
#include "elastodynamics.hpp"
#include "field.hpp"
#include "internalbodyforce.hpp"
#include "magnet.hpp"
#include "parameter.hpp"
#include "straintensor.hpp"
#include "stresstensor.hpp"
#include "traction.hpp"

__device__ inline real mixedTerm(const CuField& gu,
                                 int gcomp,
                                 const CuParameter& C,
                                 int idx,
                                 bool solidM,
                                 int idx_m,
                                 bool solidP,
                                 int idx_p,
                                 real h) {
  return directMixedDerivative(
      gu.valueAt(idx, gcomp), solidM ? gu.valueAt(idx_m, gcomp) : real(0),
      solidP ? gu.valueAt(idx_p, gcomp) : real(0), C.valueAt(idx),
      solidM ? C.valueAt(idx_m) : real(0), solidP ? C.valueAt(idx_p) : real(0),
      h);
}

// Elastic force, magnum.np structure.
// Conventions:
//   gu[3*d+k]     = d_d u_k   (d: derivative direction, k: displacement
//   component) BL/BR[3*d+k]  = flux offset for (direction d, component k) at
//   this cell's -d / +d face:
//                   sigma_mel (+ B_eps, the tangential companion term). The row
//                   is the differentiation direction d, not the force
//                   component.
//   Non-solid neighbour: enters the two-point formulas with C = 0, and its
//   "offset" is the face stress S = sense * t (t = outward traction, so -t on a
//   -d face; 0 = free surface).
// For force component a and direction d (skipped on thin axes: N_d = 1,
// non-periodic):
//   own-side flux difference of C d_d u_a  (C = C11 if a == d else C44), with
//   offset B[d][a], plus mixed term:  a == d: sum_{e != d} d_d( C12 * gu[e][e]
//   );   a != d: d_d( C44 * gu[a][d] ) (the mixed terms carry no offset).
// Not double counted: at solid-solid faces B_eps cancels between the two cells;
// at a non-solid face it is exactly balanced by the half-weight one-sided mixed
// term. Pointwise div(sigma_mel) is not included here: evalEffectiveBodyForce
// adds it separately.
__global__ void k_elasticForceMagnum(CuField fField,
                                     const CuField u,
                                     const CuField gu,
                                     const CuField BL,
                                     const CuField BR,
                                     const CuParameter C11,
                                     const CuParameter C12,
                                     const CuParameter C44,
                                     const CuBoundaryTraction traction,
                                     const bool useTraction,
                                     const real3 w,
                                     const Grid mastergrid,
                                     const int3 thin) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = fField.system;
  const Grid grid = system.grid;

  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx))
      fField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  const int3 coo = grid.index2coord(idx);
  const real hs[3] = {1 / w.x, 1 / w.y, 1 / w.z};
  const int thinArr[3] = {thin.x, thin.y, thin.z};
  const int3 step_arr[3] = {int3{1, 0, 0}, int3{0, 1, 0}, int3{0, 0, 1}};

  real f[3] = {0, 0, 0};

  for (int d = 0; d < 3; d++) {
    if (thinArr[d])
      continue;

    const int3 coo_m = mastergrid.wrap(coo - step_arr[d]);
    const int3 coo_p = mastergrid.wrap(coo + step_arr[d]);
    const bool solidM = system.inGeometry(coo_m);
    const bool solidP = system.inGeometry(coo_p);
    const int idx_m = solidM ? grid.coord2index(coo_m) : idx;
    const int idx_p = solidP ? grid.coord2index(coo_p) : idx;
    const real h = hs[d];
    const real3 tP =
        useTraction ? traction.getSide(d, 1).vectorAt(idx) : real3{0, 0, 0};
    const real3 tM =
        useTraction ? traction.getSide(d, -1).vectorAt(idx) : real3{0, 0, 0};

    for (int a = 0; a < 3; a++) {
      const int bc = 3 * d + a;  // (direction d, component a)

      // ---- own-side flux difference ----
      const CuParameter& Cn = (a == d) ? C11 : C44;
      const real Cs = Cn.valueAt(idx);
      const real Cm = solidM ? Cn.valueAt(idx_m) : real(0);
      const real Cp = solidP ? Cn.valueAt(idx_p) : real(0);
      const real us = u.valueAt(idx, a);
      const real um = solidM ? u.valueAt(idx_m, a) : real(0);
      const real up = solidP ? u.valueAt(idx_p, a) : real(0);
      const real BmR =
          solidM ? BR.valueAt(idx_m, bc) : -comp3(tM, a);  // S = -t on -d face
      const real BpL =
          solidP ? BL.valueAt(idx_p, bc) : comp3(tP, a);  // S = +t on +d face

      f[a] +=
          directSecondDerivative(us, um, up, Cs, Cm, Cp, BL.valueAt(idx, bc),
                                 BR.valueAt(idx, bc), BmR, BpL, h);

      // ---- mixed derivative(s) ----
      if (a == d) {
        for (int e = 0; e < 3; e++) {
          if (e == d)
            continue;
          f[a] += mixedTerm(gu, 3 * e + e, C12, idx, solidM, idx_m, solidP,
                            idx_p, h);
        }
      } else {
        f[a] +=
            mixedTerm(gu, 3 * a + d, C44, idx, solidM, idx_m, solidP, idx_p, h);
      }
    }
  }

  fField.setVectorInCell(idx, real3{f[0], f[1], f[2]});
}

__global__ void k_scaleByParameter(CuField f, const CuParameter p) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!f.cellInGeometry(idx))
    return;
  for (int c = 0; c < f.ncomp; c++)
    f.setValueInCell(idx, c, p.valueAt(idx) * f.valueAt(idx, c));
}

// Viscous force in flux form: the elastic operator acting on v, weighted by eta
// (tensor damping) or by C and then multiplied by the Rayleigh coefficient
// beta. Traction-free (symmetric, dissipative, momentum-conserving), as long as
// beta is uniform.
static Field evalViscousForceFlux(const Magnet* magnet, bool useTraction) {
  const int ncells = magnet->grid().ncells();
  const real3 w = 1. / magnet->cellsize();
  const Grid mastergrid = magnet->world()->mastergrid();
  const int3 thin = thinAxes(magnet->grid(), mastergrid);
  const bool useTensor = !viscousDampingTensorAssuredZero(magnet);

  CuField v = magnet->elasticVelocity()->field().cu();
  CuParameter A11 = useTensor ? magnet->eta11.cu() : magnet->C11.cu();
  CuParameter A12 = useTensor ? magnet->eta12.cu() : magnet->C12.cu();
  CuParameter A44 = useTensor ? magnet->eta44.cu() : magnet->C44.cu();
  CuBoundaryTraction traction = magnet->boundaryTraction.cu();

  ElasticJumpData jd = evalViscousJumpData(magnet, useTensor, useTraction);

  Field f(magnet->system(), 3);
  cudaLaunch(ncells, k_elasticForceMagnum, f.cu(), v, jd.gu.cu(), jd.BL.cu(),
             jd.BR.cu(), A11, A12, A44, traction, useTraction, w, mastergrid,
             thin);

  if (!useTensor) {
    CuParameter beta = magnet->stiffnessDamping.cu();
    CuField fcu = f.cu();
    cudaLaunch(ncells, k_scaleByParameter, fcu, beta);
  }
  return f;
}

Field evalInternalBodyForce(const Magnet* magnet) {
  Field fField(magnet->system(), 3);
  fField.makeZero();
  if (stressTensorAssuredZero(magnet)) {
    return fField;
  }

  int ncells = fField.grid().ncells();
  real3 w = 1. / magnet->cellsize();
  Grid mastergrid = magnet->world()->mastergrid();
  CuBoundaryTraction traction = magnet->boundaryTraction.cu();

  // Whether the elastic part is present. It applies the outer-boundary traction
  // itself (at void/grid-edge faces, in k_gradientJump and
  // k_elasticForceMagnum). If it is assured zero, the viscous part is the only
  // source of the boundary force and gets the traction instead.
  const bool elasticPresent = !elasticityAssuredZero(magnet);

  // ---- elastic part: k_elasticForceMagnum (jump conditions, voids as C = 0,
  // traction via the face stress)
  if (elasticPresent) {
    CuField u = magnet->elasticDisplacement()->field().cu();
    CuParameter C11 = magnet->C11.cu();
    CuParameter C12 = magnet->C12.cu();
    CuParameter C44 = magnet->C44.cu();
    const int3 thin = thinAxes(magnet->grid(), mastergrid);

    ElasticJumpData jd = evalElasticJumpData(magnet);

    Field elasticForce(magnet->system(), 3);
    cudaLaunch(ncells, k_elasticForceMagnum, elasticForce.cu(), u, jd.gu.cu(),
               jd.BL.cu(), jd.BR.cu(), C11, C12, C44, traction,
               /*useTraction=*/true, w, mastergrid, thin);
    debugFieldStats(elasticForce.cu(), ncells,
                    "elasticForce (k_elasticForceMagnum)");
    debugFieldSum(elasticForce, "elasticForce (k_elasticForceMagnum)");

    fField += elasticForce;
  }

  // ---- viscous part: same flux-form operator acting on v (traction-free) ----
  {
    // DEBUG: report the viscous state whenever it changes
    static int lastState = -1;
    const int state = (viscousDampingAssuredZero(magnet) ? 1 : 0) |
                      (viscousDampingTensorAssuredZero(magnet) ? 2 : 0) |
                      (viscousDampingRayleighAssuredZero(magnet) ? 4 : 0) |
                      (magnet->stiffnessDamping.assuredZero() ? 8 : 0) |
                      (magnet->eta.assuredZero() ? 16 : 0);
    if (state != lastState) {
      lastState = state;
      fprintf(stderr,
              "[viscous-debug] viscousAssuredZero=%d tensorAssuredZero=%d "
              "rayleighAssuredZero=%d stiffnessDamping.assuredZero=%d "
              "eta.assuredZero=%d\n",
              (state & 1) != 0, (state & 2) != 0, (state & 4) != 0,
              (state & 8) != 0, (state & 16) != 0);
    }
  }
  if (!viscousDampingAssuredZero(magnet)) {
    // the applied traction is only used here if there is no elastic part at all
    Field viscousForce =
        evalViscousForceFlux(magnet, /*useTraction=*/!elasticPresent);
    debugFieldStats(viscousForce.cu(), ncells, "viscousForce (flux form)");
    debugFieldSum(viscousForce, "viscousForce (flux form)");
    fField += viscousForce;
  }

  debugFieldStats(fField.cu(), ncells, "fField (evalInternalBodyForce, final)");
  debugFieldSum(fField, "fField (evalInternalBodyForce, final)");

  return fField;
}

M_FieldQuantity internalBodyForceQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalInternalBodyForce, 3,
                         "internal_body_force", "N/m3");
}
