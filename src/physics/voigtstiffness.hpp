#pragma once

#include "parameter.hpp"

/** 6x6 symmetric Voigt stiffness tensor passed by value to CUDA kernels.
 *
 *  Stored as the upper triangle of 21 CuParameters. The shear diagonal
 *  entries C44, C55, C66 are ENGINEERING stiffnesses (i.e. σ_yz = C44 * γ_yz
 *  with γ_yz = 2 ε_yz), matching the existing k_elasticStress convention.
 *  The at() accessor returns the TRUE-strain stiffness S[α][β].
 */
struct CuVoigtStiffness {
  CuParameter C11, C12, C13, C14, C15, C16;
  CuParameter C22, C23, C24, C25, C26;
  CuParameter C33, C34, C35, C36;
  CuParameter C44, C45, C46;
  CuParameter C55, C56;
  CuParameter C66;

  // S[alpha][beta], symmetric. alpha, beta in Voigt {0..5}.
  __device__ inline real at(int idx, int alpha, int beta) const;
};

__device__ inline real CuVoigtStiffness::at(int idx, int alpha, int beta) const {
  // Symmetric: canonicalize
  if (alpha > beta) { int t = alpha; alpha = beta; beta = t; }

  // Upper-triangle look-up
  if (alpha == 0) {
    if (beta == 0) return C11.valueAt(idx);
    if (beta == 1) return C12.valueAt(idx);
    if (beta == 2) return C13.valueAt(idx);
    if (beta == 3) return C14.valueAt(idx);
    if (beta == 4) return C15.valueAt(idx);
    return C16.valueAt(idx);
  }
  if (alpha == 1) {
    if (beta == 1) return C22.valueAt(idx);
    if (beta == 2) return C23.valueAt(idx);
    if (beta == 3) return C24.valueAt(idx);
    if (beta == 4) return C25.valueAt(idx);
    return C26.valueAt(idx);
  }
  if (alpha == 2) {
    if (beta == 2) return C33.valueAt(idx);
    if (beta == 3) return C34.valueAt(idx);
    if (beta == 4) return C35.valueAt(idx);
    return C36.valueAt(idx);
  }
  if (alpha == 3) {
    if (beta == 3) return 2.0 * C44.valueAt(idx);  // true-strain shear
    if (beta == 4) return C45.valueAt(idx);
    return C46.valueAt(idx);
  }
  if (alpha == 4) {
    if (beta == 4) return 2.0 * C55.valueAt(idx);  // true-strain shear
    return C56.valueAt(idx);
  }
  return 2.0 * C66.valueAt(idx);                    // true-strain shear
}
