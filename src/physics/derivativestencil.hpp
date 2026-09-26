#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "grid.hpp"
#include "field.hpp"

// ============================================================
// Debug diagnostics (always compiled in, gated at runtime)
// ============================================================
//
// A minimal, dependency-free way to answer three questions about any CuField
// without needing to know this codebase's own reduction/statistics API:
//   1. Is it all zero inside the geometry? (suggests the field was never
//      written, or a kernel returned early / launched with 0 cells)
//   2. Does it contain any NaN or Inf? (suggests a division by zero, an
//      uninitialized read, or an out-of-bounds neighbor index)
//   3. What is its max absolute value inside the geometry? (a quick magnitude
//      sanity check -- e.g. "is this exactly what I'd expect from a bulk
//      uniform-material central difference" vs "wildly larger than input")
//
// Usage: call debugFieldStats(field.cu(), ncells, "label") right after any
// cudaLaunch whose output you want to check. Prints to stderr and does a
// blocking cudaDeviceSynchronize + cudaMemcpy, so this is deliberately slow.
// It is a no-op unless the MUMAX_DEBUG_FIELDS environment variable is set, so
// it is safe to leave the call sites in place; set MUMAX_DEBUG_FIELDS=1 to
// enable the prints for a debugging run.

__global__ static void k_debugFieldStats(const CuField field, real* outMaxAbs,
                                  int* outNumNonfinite, int* outNumNonzero,
                                  int* outNumInGeometry) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = field.system;
  if (!system.grid.cellInGrid(idx))
    return;
  if (!system.inGeometry(idx))
    return;

  atomicAdd(outNumInGeometry, 1);

  for (int c = 0; c < field.ncomp; c++) {
    real v = field.valueAt(idx, c);
    if (isnan(v) || isinf(v)) {
      atomicAdd(outNumNonfinite, 1);
      continue;
    }
    if (v != 0)
      atomicAdd(outNumNonzero, 1);
    real av = v < 0 ? -v : v;
    // simple approximate max via atomicCAS on the bit pattern (fine for a
    // debug utility; not intended to be a general-purpose atomic max)
    unsigned int* addr = (unsigned int*)outMaxAbs;
    unsigned int old = *addr, assumed;
    float avf = (float)av;
    do {
      assumed = old;
      float oldf = __uint_as_float(assumed);
      if (avf <= oldf) break;
      old = atomicCAS(addr, assumed, __float_as_uint(avf));
    } while (assumed != old);
  }
}

inline void debugFieldStats(const CuField& field, int ncells, const char* label) {
  static const bool enabled = std::getenv("MUMAX_DEBUG_FIELDS") != nullptr;
  if (!enabled)
    return;

  real* d_maxAbs;
  int* d_numNonfinite;
  int* d_numNonzero;
  int* d_numInGeometry;
  cudaMalloc(&d_maxAbs, sizeof(real));
  cudaMalloc(&d_numNonfinite, sizeof(int));
  cudaMalloc(&d_numNonzero, sizeof(int));
  cudaMalloc(&d_numInGeometry, sizeof(int));
  real zero = 0;
  int zeroi = 0;
  cudaMemcpy(d_maxAbs, &zero, sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(d_numNonfinite, &zeroi, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_numNonzero, &zeroi, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_numInGeometry, &zeroi, sizeof(int), cudaMemcpyHostToDevice);

  int blockSize = 512;
  int gridSize = (ncells + blockSize - 1) / blockSize;
  k_debugFieldStats<<<gridSize, blockSize>>>(field, d_maxAbs, d_numNonfinite,
                                             d_numNonzero, d_numInGeometry);
  cudaDeviceSynchronize();

  real maxAbs;
  int numNonfinite, numNonzero, numInGeometry;
  cudaMemcpy(&maxAbs, d_maxAbs, sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(&numNonfinite, d_numNonfinite, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&numNonzero, d_numNonzero, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&numInGeometry, d_numInGeometry, sizeof(int), cudaMemcpyDeviceToHost);

  fprintf(stderr,
          "[debug] %-40s ncomp=%d inGeom=%d nonzero=%d nonfinite=%d maxAbs=%g\n",
          label, field.ncomp, numInGeometry, numNonzero, numNonfinite, (double)maxAbs);
  fflush(stderr);

  cudaFree(d_maxAbs);
  cudaFree(d_numNonfinite);
  cudaFree(d_numNonzero);
  cudaFree(d_numInGeometry);
}

// Host-side companion to debugFieldStats: prints the plain (non-absolute) sum
// of each component of `field` over the WHOLE grid (not just inGeometry --
// Field already zeroes outside geometry, so this is equivalent, but simpler
// and avoids a second kernel).
//
// This answers a different question than debugFieldStats' maxAbs: a term can
// have a large maxAbs from one noisy/localized cell while summing to ~0 (no
// net effect), or a modest maxAbs while summing to something large because
// it's biased in the same direction almost everywhere (a genuine net force,
// e.g. from an uncancelled boundary condition). Use this specifically to
// check whether a given contribution is responsible for a steady, non-
// physical net force on the whole magnet -- the kind of thing that shows up
// as rigid-body drift.
//
// Same gating as debugFieldStats: no-op unless MUMAX_DEBUG_FIELDS is set (to
// any non-empty value, including "0" -- see debugFieldStats' own caveat about
// getenv only checking presence, not value).
//
// Takes a Field (not CuField) since it needs Field::getData() to copy to the
// host; call it with the same Field you'd otherwise call field.cu() on.
inline void debugFieldSum(const Field& field, const char* label) {
  static const bool enabled = std::getenv("MUMAX_DEBUG_FIELDS") != nullptr;
  if (!enabled)
    return;

  std::vector<real> data = field.getData();
  int ncells = field.grid().ncells();
  int ncomp = static_cast<int>(data.size()) / ncells;

  fprintf(stderr, "[debug-sum] %-40s ", label);
  for (int c = 0; c < ncomp; c++) {
    double sum = 0;
    for (int i = 0; i < ncells; i++)
      sum += static_cast<double>(data[c * ncells + i]);
    fprintf(stderr, "comp%d_sum=%g  ", c, sum);
  }
  fprintf(stderr, "\n");
  fflush(stderr);
}

// Computes the full gradient tensor (∂f_c/∂x_i) of a 3-component field `f` at cell `idx`
// (grid coordinate `coo`), using the same adaptive one-sided/central finite-difference
// stencil previously duplicated between k_strainTensor (straintensor.cu) and
// k_magnetoelasticForce (magnetoelasticforce.cu).
//
// Result convention:
//   grad[i][c] = d f_c / d x_i
//     i = 0,1,2  -> differentiation direction (x,y,z)
//     c = 0,1,2  -> vector component of f being differentiated
//
// Caller must guarantee `system.inGeometry(idx)` is true; boundary/outside-geometry
// handling is left to the caller (as in the original kernels).
__device__ inline void deviceGradientTensor(real grad[3][3],
                                            const CuField& f,
                                            const CuSystem& system,
                                            const Grid& mastergrid,
                                            int3 coo,
                                            const real3& w) {
  const real ws[3] = {w.x, w.y, w.z};
  const int3 im2_arr[3] = {int3{-2, 0, 0}, int3{0, -2, 0}, int3{0, 0, -2}};
  const int3 im1_arr[3] = {int3{-1, 0, 0}, int3{0, -1, 0}, int3{0, 0, -1}};
  const int3 ip1_arr[3] = {int3{ 1, 0, 0}, int3{0,  1, 0}, int3{0, 0,  1}};
  const int3 ip2_arr[3] = {int3{ 2, 0, 0}, int3{0,  2, 0}, int3{0, 0,  2}};

  const real3 f_0 = f.vectorAt(coo);

#pragma unroll
  for (int i = 0; i < 3; i++) {
    real wi = ws[i];
    int3 coo_im2 = mastergrid.wrap(coo + im2_arr[i]);
    int3 coo_im1 = mastergrid.wrap(coo + im1_arr[i]);
    int3 coo_ip1 = mastergrid.wrap(coo + ip1_arr[i]);
    int3 coo_ip2 = mastergrid.wrap(coo + ip2_arr[i]);

    real3 dfdi;
    if (!system.inGeometry(coo_im1) && !system.inGeometry(coo_ip1)) {
      // --1-- zero
      dfdi = real3{0, 0, 0};
    } else if ((!system.inGeometry(coo_im2) || !system.inGeometry(coo_ip2)) &&
                system.inGeometry(coo_im1) && system.inGeometry(coo_ip1)) {
      // -111-, 1111-, -1111 central difference, ε ~ h^2
      dfdi = 0.5 * (f.vectorAt(coo_ip1) - f.vectorAt(coo_im1));
    } else if (!system.inGeometry(coo_im2) && !system.inGeometry(coo_ip1)) {
      // -11-- backward difference, ε ~ h^1
      dfdi = (f_0 - f.vectorAt(coo_im1));
    } else if (!system.inGeometry(coo_im1) && !system.inGeometry(coo_ip2)) {
      // --11- forward difference, ε ~ h^1
      dfdi = (-f_0 + f.vectorAt(coo_ip1));
    } else if (system.inGeometry(coo_im2) && !system.inGeometry(coo_ip1)) {
      // 111-- backward difference, ε ~ h^2
      dfdi = (0.5 * f.vectorAt(coo_im2) - 2.0 * f.vectorAt(coo_im1) + 1.5 * f_0);
    } else if (!system.inGeometry(coo_im1) && system.inGeometry(coo_ip1)) {
      // --111 forward difference, ε ~ h^2
      dfdi = (-0.5 * f.vectorAt(coo_ip2) + 2.0 * f.vectorAt(coo_ip1) - 1.5 * f_0);
    } else {
      // 11111 central difference, ε ~ h^4
      dfdi = ((2.0 / 3.0) * (f.vectorAt(coo_ip1) - f.vectorAt(coo_im1)) +
              (1.0 / 12.0) * (f.vectorAt(coo_im2) - f.vectorAt(coo_ip2)));
    }
    dfdi *= wi;

    grad[i][0] = dfdi.x;
    grad[i][1] = dfdi.y;
    grad[i][2] = dfdi.z;
  }
}

// True if both the -i and +i neighbors of `coo` (wrapped through mastergrid) are inside
// the geometry, i.e. this cell is not adjacent to the outer geometry/vacuum boundary in
// direction i. Interior material-interface (stiffness-jump) corrections are only ever
// applied where this holds; at the true outer boundary the one-sided stencils already
// computed by deviceGradientTensor are left untouched.
__device__ inline bool interiorInDirection(const CuSystem& system,
                                           const Grid& mastergrid,
                                           int3 coo, int dir) {
  const int3 step_arr[3] = {int3{1, 0, 0}, int3{0, 1, 0}, int3{0, 0, 1}};
  int3 coo_im1 = mastergrid.wrap(coo - step_arr[dir]);
  int3 coo_ip1 = mastergrid.wrap(coo + step_arr[dir]);
  return system.inGeometry(coo_im1) && system.inGeometry(coo_ip1);
}

__device__ inline int3 gradRowComps(int i) {
  return int3{3 * i, 3 * i + 1, 3 * i + 2};
}

// ============================================================
// Direct jump-aware derivative primitives
// ============================================================
//
// These two functions are the CUDA translation of magnum.np's
// linear_elasticity/utils.py `first_derivative_with_jump_conditions` (used, with
// zero B, inside `_mixed_derivative`) and llg_with_le_solver.py's
// `_2nd_derivative_homogeneous_Neumann` (used inside `_2nd_derivative`), both
// specialized to a uniform grid spacing h (mumax's cellsize is uniform per
// direction, unlike magnum's general non-uniform mesh).
//
// Both were verified symbolically against magnum's original (non-uniform-mesh)
// formulas substituted with dx_left = dx_right = h, confirming:
//   - the bulk (uniform-coefficient, zero-jump) limit reduces exactly to the
//     standard second-order finite-difference operator, and
//   - the coefficient-jump limit reduces to the conservative harmonic-mean
//     flux-difference scheme, a standard energy-consistent discretization for
//     discontinuous-coefficient elliptic operators.
//
// A single shared per-face helper underlies both: at a face between `self` and
// a `neighbor` cell (uniform spacing h), with B_left/B_right the two cells' own
// outward-facing eigenstrain offsets at that specific face (B_left belonging to
// whichever of the two cells has the lower grid coordinate on this axis, B_right
// to the higher one -- NOT "self" vs "neighbor"; this asymmetric-looking
// left/right labeling is what makes the im1-face and ip1-face contributions add
// up correctly, and was confirmed by direct symbolic verification against
// magnum's own construction rather than assumed by symmetry):
//   contrib(self,neighbor) = C_self*[2*C_neighbor*(u_neighbor-u_self)
//                                     + h*(B_right - B_left)]
//                             / (h^2 * (C_self+C_neighbor))
// The full second derivative at a cell is the SUM of its im1-face and ip1-face
// contributions (no further division by h -- the h^2 is already in the
// denominator above). This was verified symbolically to: (a) reduce to the
// standard 3-point second derivative C*(u_ip1-2*u_self+u_im1)/h^2 in the
// uniform-coefficient, zero-B limit, and (b) reduce to the classic
// conservative harmonic-mean flux-difference scheme in the coefficient-jump,
// zero-B limit.
__device__ inline real secondDerivFaceTerm(real C_self, real C_neighbor,
                                           real u_self, real u_neighbor,
                                           real B_left, real B_right,
                                           real h) {
  real denom = C_self + C_neighbor;
  if (denom <= 0) return 0;  // no coupling on this face; caller should fall back
  return C_self * (2 * C_neighbor * (u_neighbor - u_self) + h * (B_right - B_left)) /
         (h * h * denom);
}

// Direct second derivative d/dx_i( C * d(u_c)/dx_i ) at a cell, jump-aware
// across interior interfaces where C (or an eigenstrain-type offset B) is
// discontinuous. `u_self`/`u_im1`/`u_ip1` are the raw field values (not
// derivatives) of component c; `Bself_L`/`Bself_R` are this cell's own two
// half-face eigenstrain offsets (as produced by straintensor.cu's
// k_faceEigenstrainTraction), `Bim1_R`/`Bip1_L` the corresponding neighbor
// cells' own outward-facing values at the two shared faces. Matches magnum's
// _2nd_derivative_homogeneous_Neumann (ij_C = [i,i]).
__device__ inline real directSecondDerivative(real u_self, real u_im1, real u_ip1,
                                              real C_self, real C_im1, real C_ip1,
                                              real Bself_L, real Bself_R,
                                              real Bim1_R, real Bip1_L,
                                              real h) {
  // im1 face: self is the "right" cell of the pair -> B_left=Bim1_R, B_right=Bself_L
  real contribM = secondDerivFaceTerm(C_self, C_im1, u_self, u_im1,
                                      Bim1_R, Bself_L, h);
  // ip1 face: self is the "left" cell of the pair -> B_left=Bself_R, B_right=Bip1_L
  real contribP = secondDerivFaceTerm(C_self, C_ip1, u_self, u_ip1,
                                      Bself_R, Bip1_L, h);
  return contribM + contribP;
}

// Direct mixed derivative d/dx_j( C * g ) at a cell, where g is an already-
// computed (jump-corrected) first-derivative field (e.g. d(u_a)/dx_i for some
// other direction i != j), jump-aware across interfaces where C is
// discontinuous. Matches magnum's _mixed_derivative, which computes this as
// gradient_with_pbc(C*g, dim=j, C=[C]) with no B offset -- i.e. this is the
// SAME two-point jump formula as directSecondDerivative/Kernel 3, applied as a
// first derivative of the flux F = C*g, not a second derivative of g itself.
//
// contrib(self,neighbor) = 2*C_neighbor*(F_neighbor - F_self) / (h*(C_self+C_neighbor))
// with F = C*g, and the final derivative the average of the two face
// contributions (mirrors k_jumpCorrectGradient's weighting, but with B == 0).
__device__ inline real directMixedDerivative(real g_self, real g_im1, real g_ip1,
                                             real C_self, real C_im1, real C_ip1,
                                             real h) {
  const real F_self = C_self * g_self;
  const real F_im1  = C_im1  * g_im1;
  const real F_ip1  = C_ip1  * g_ip1;

  const real denM = C_self + C_im1;
  const real denP = C_self + C_ip1;
  const bool haveM = denM > 0;
  const bool haveP = denP > 0;

  const real fromM = haveM ? 2 * C_im1 * (F_self - F_im1) / (h * denM) : 0;
  const real fromP = haveP ? 2 * C_ip1 * (F_ip1 - F_self) / (h * denP) : 0;

  if (haveM && haveP) return 0.5 * (fromM + fromP);
  if (haveM) return fromM;
  if (haveP) return fromP;
  return 0;  // both faces have zero weight; no coupling to correct
}
