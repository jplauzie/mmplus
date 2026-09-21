#include "elastodynamics.hpp"
#include "internalbodyforce.hpp"
#include "cudalaunch.hpp"
#include "magnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "stresstensor.hpp"
#include "traction.hpp"
#include "voigtstiffness.hpp"
#include "straintensor.hpp"

__device__ inline real comp(const real3& v, int i) {
  if (i == 0) return v.x;
  if (i == 1) return v.y;
  return v.z;
}

__device__ inline int voigtIdx(int a, int b) {
  if (a == b) return a;
  if (a > b) { int t = a; a = b; b = t; }
  if (a == 0 && b == 1) return 3;  // (x,y) -> 3
  if (a == 0 && b == 2) return 4;  // (x,z) -> 4
  return 5;                        // (y,z) -> 5
}



__device__ inline real S_face(const CuVoigtStiffness& S,
                              int alpha, int beta,
                              int i0, int i1) {
  if (alpha > beta) { int t = alpha; alpha = beta; beta = t; }
  if (alpha == 0) {
    if (beta == 0) return S.C11.harmonicMean(i0, i1);
    if (beta == 1) return S.C12.harmonicMean(i0, i1);
    if (beta == 2) return S.C13.harmonicMean(i0, i1);
    if (beta == 3) return S.C14.harmonicMean(i0, i1);
    if (beta == 4) return S.C15.harmonicMean(i0, i1);
    return S.C16.harmonicMean(i0, i1);
  }
  if (alpha == 1) {
    if (beta == 1) return S.C22.harmonicMean(i0, i1);
    if (beta == 2) return S.C23.harmonicMean(i0, i1);
    if (beta == 3) return S.C24.harmonicMean(i0, i1);
    if (beta == 4) return S.C25.harmonicMean(i0, i1);
    return S.C26.harmonicMean(i0, i1);
  }
  if (alpha == 2) {
    if (beta == 2) return S.C33.harmonicMean(i0, i1);
    if (beta == 3) return S.C34.harmonicMean(i0, i1);
    if (beta == 4) return S.C35.harmonicMean(i0, i1);
    return S.C36.harmonicMean(i0, i1);
  }
  if (alpha == 3) {
    if (beta == 3) return 2.0 * S.C44.harmonicMean(i0, i1);
    if (beta == 4) return S.C45.harmonicMean(i0, i1);
    return S.C46.harmonicMean(i0, i1);
  }
  if (alpha == 4) {
    if (beta == 4) return 2.0 * S.C55.harmonicMean(i0, i1);
    return S.C56.harmonicMean(i0, i1);
  }
  return 2.0 * S.C66.harmonicMean(i0, i1);
}



__device__ int tensorComp(int row, int col) {
  return (row == col) ? row : row+col+2;
}

__device__ int3 tensorRowComps(int row) {
  return int3{tensorComp(row, 0), tensorComp(row, 1), tensorComp(row, 2)};
}

/**
 * Numerical divergence of stress with central five-point stencil in bulk material.
 * Lower order accuracy (three-point stencil) central difference is used in bulk
 * 1 cell away from boundary.
 * The traction is applied at the boundary, implemented using a custom
 * second-order-accurate three-point stencil.
*/
__global__ void k_internalBodyForce(CuField fField,
                                    const CuField stressTensor,
                                    const CuBoundaryTraction traction,
                                    const real3 w,  // 1 / cellsize
                                    const Grid mastergrid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = fField.system;
  const Grid grid = system.grid;

  // When outside the geometry, set to zero and return early
  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      fField.setVectorInCell(idx, real3{0, 0, 0});
    }
    return;
  }

  // array instead of real3 to get indexing [i]
  const real ws[3] = {w.x, w.y, w.z};
  const int3 im2_arr[3] = {int3{-2, 0, 0}, int3{0,-2, 0}, int3{0, 0,-2}};
  const int3 im1_arr[3] = {int3{-1, 0, 0}, int3{0,-1, 0}, int3{0, 0,-1}};
  const int3 ip1_arr[3] = {int3{ 1, 0, 0}, int3{0, 1, 0}, int3{0, 0, 1}};
  const int3 ip2_arr[3] = {int3{ 2, 0, 0}, int3{0, 2, 0}, int3{0, 0, 2}};
  const int3 coo = grid.index2coord(idx);
    
  real3 f = {0, 0, 0};  // elastic force vector
  for (int i = 0; i < 3; i++) {
    // i is {x, y, z} derivative direction and stress tensor row
    // f_j = ∂i σ_ij

    int3 stressRow = tensorRowComps(i);

    // translate in direction i
    int3 im2 = im2_arr[i], im1 = im1_arr[i];  // transl in direction -i
    int3 ip1 = ip1_arr[i], ip2 = ip2_arr[i];  // transl in direction +i

    int3 coo_im2 = mastergrid.wrap(coo + im2);
    int3 coo_im1 = mastergrid.wrap(coo + im1);
    int3 coo_ip1 = mastergrid.wrap(coo + ip1);
    int3 coo_ip2 = mastergrid.wrap(coo + ip2);

    bool im2_inGeo = system.inGeometry(coo_im2);
    bool im1_inGeo = system.inGeometry(coo_im1);
    bool ip1_inGeo = system.inGeometry(coo_ip1);
    bool ip2_inGeo = system.inGeometry(coo_ip2);

    if (!im1_inGeo && !ip1_inGeo) {
      // --1-- central difference of boundary stress, ε ~ h^2
      f += ws[i] * (traction.getSide(i, 1).vectorAt(idx)
                    // -1 from stencil * -1 from normal vector
                    + traction.getSide(i, -1).vectorAt(idx));
    } else if (!im1_inGeo) {
      // --11- left boundary, custom difference + traction BC,  ε ~ h^2
      f += ws[i] * (
        // stress row at coo_i-1/2 = boundary traction * negative sense of normal vector
        // -1 from stencil * -1 from normal vector
        4./3. * traction.getSide(i, -1).vectorAt(idx)
        + stressTensor.vectorAt(idx, stressRow)  // +3/3 weight
        + 1./3. * stressTensor.vectorAt(coo_ip1, stressRow)
      );
    } else if (!ip1_inGeo) {
      // -11-- right boundary, custom difference + traction BC,  ε ~ h^2
      f += ws[i] * (
        - 1./3. * stressTensor.vectorAt(coo_im1, stressRow)
        - stressTensor.vectorAt(idx, stressRow)  // -3/3 weight
        // stress row at coo_i+1/2 = boundary traction * positive sense of normal vector
        + 4./3. * traction.getSide(i, 1).vectorAt(idx)
      );
    } else if (!im2_inGeo || !ip2_inGeo) {
      // -111-, 1111-, -1111 central difference,  ε ~ h^2
      f += 0.5*ws[i] * (stressTensor.vectorAt(coo_ip1, stressRow) -
                        stressTensor.vectorAt(coo_im1, stressRow));
    } else {  // all 5 points are safe for sure
      // 11111 central difference,  ε ~ h^4
      f += ws[i] * ((4./6.) * (stressTensor.vectorAt(coo_ip1, stressRow) -
                               stressTensor.vectorAt(coo_im1, stressRow)) + 
                    (1./12.)* (stressTensor.vectorAt(coo_im2, stressRow) -
                               stressTensor.vectorAt(coo_ip2, stressRow)));
    }
  }

  fField.setVectorInCell(idx, f);
}

__global__ void k_internalBodyForceFlux(CuField fField,
                                        const CuField u,
                                        const CuField strain,
                                        const CuVoigtStiffness S,
                                        const CuBoundaryTraction traction,
                                        const real3 w,
                                        const Grid mastergrid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = fField.system;
  const Grid grid = system.grid;

  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) fField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  const real ws[3] = {w.x, w.y, w.z};
  const int3 e[3] = {int3{1,0,0}, int3{0,1,0}, int3{0,0,1}};
  const int3 coo = grid.index2coord(idx);

  // Cache current-cell strain (true strain, Voigt order xx,yy,zz,yz,xz,xy)
  real eps0[6];
  for (int k = 0; k < 6; k++) eps0[k] = strain.valueAt(idx, k);

  real f[3] = {0, 0, 0};

  for (int b = 0; b < 3; b++) {
    const int3 coo_p = mastergrid.wrap(coo + e[b]);
    const int3 coo_m = mastergrid.wrap(coo - e[b]);
    const bool p_in = system.inGeometry(coo_p);
    const bool m_in = system.inGeometry(coo_m);
    const int idx_p = grid.coord2index(coo_p);
    const int idx_m = grid.coord2index(coo_m);

    // ---------- +b face ----------
    real sig_p[3] = {0, 0, 0};
    if (p_in) {
      // Face strain: diagonal component (a=b) uses direct gradient,
      // off-diagonal uses average of cell strains.
      real eps_f[6];
      for (int k = 0; k < 6; k++) eps_f[k] = 0.5 * (eps0[k] + strain.valueAt(idx_p, k));
      eps_f[b] = (comp(u.vectorAt(coo_p), b) - comp(u.vectorAt(coo), b)) * ws[b];

      for (int a = 0; a < 3; a++) {
        const int alpha = voigtIdx(a, b);
        for (int beta = 0; beta < 6; beta++) {
          sig_p[a] += S_face(S, alpha, beta, idx, idx_p) * eps_f[beta];
        }
      }
    } else {
      // Traction at +b face: sigma_ab = t_a  (sign convention from existing kernel)
      const real3 t = traction.getSide(b, +1).vectorAt(idx);
      sig_p[0] = t.x; sig_p[1] = t.y; sig_p[2] = t.z;
    }

    // ---------- -b face ----------
    real sig_m[3] = {0, 0, 0};
    if (m_in) {
      real eps_f[6];
      for (int k = 0; k < 6; k++) eps_f[k] = 0.5 * (strain.valueAt(idx_m, k) + eps0[k]);
      eps_f[b] = (comp(u.vectorAt(coo), b) - comp(u.vectorAt(coo_m), b)) * ws[b];

      for (int a = 0; a < 3; a++) {
        const int alpha = voigtIdx(a, b);
        for (int beta = 0; beta < 6; beta++) {
          sig_m[a] += S_face(S, alpha, beta, idx_m, idx) * eps_f[beta];
        }
      }
    } else {
      // Traction at -b face: sigma_ab = -t_a
      const real3 t = traction.getSide(b, -1).vectorAt(idx);
      sig_m[0] = -t.x; sig_m[1] = -t.y; sig_m[2] = -t.z;
    }

    // ---------- accumulate ----------
    f[0] += (sig_p[0] - sig_m[0]) * ws[b];
    f[1] += (sig_p[1] - sig_m[1]) * ws[b];
    f[2] += (sig_p[2] - sig_m[2]) * ws[b];
  }

  fField.setVectorInCell(idx, real3{f[0], f[1], f[2]});
}

Field evalInternalBodyForce(const Magnet* magnet) {
  Field fField(magnet->system(), 3);
  if (stressTensorAssuredZero(magnet)) {
    fField.makeZero();
    return fField;
  }

  Field strain = evalStrainTensor(magnet);
  Field u = magnet->elasticDisplacement()->eval();

  CuVoigtStiffness S{
      magnet->C11.cu(), magnet->C12.cu(), magnet->C13.cu(),
      magnet->C14.cu(), magnet->C15.cu(), magnet->C16.cu(),
      magnet->C22.cu(), magnet->C23.cu(), magnet->C24.cu(),
      magnet->C25.cu(), magnet->C26.cu(),
      magnet->C33.cu(), magnet->C34.cu(), magnet->C35.cu(),
      magnet->C36.cu(),
      magnet->C44.cu(),magnet->C45.cu(), magnet->C46.cu(),
      magnet->C55.cu(), magnet->C56.cu(),
      magnet->C66.cu()
  };
  CuBoundaryTraction traction = magnet->boundaryTraction.cu();
  real3 w = 1.0 / magnet->cellsize();
  Grid mastergrid = magnet->world()->mastergrid();

  cudaLaunch(fField.grid().ncells(), k_internalBodyForceFlux,
             fField.cu(), u.cu(), strain.cu(), S, traction, w, mastergrid);

  return fField;
}

M_FieldQuantity internalBodyForceQuantity(const Magnet* magnet) {
  return M_FieldQuantity(magnet, evalInternalBodyForce, 3, "internal_body_force", "N/m3");
}


