#pragma once
#include "datatypes.hpp"  // real

// First-derivative stencil selection, shared by the strain kernel and the force
// kernel so that the force is the exact transpose of the strain operator.
// Input: whether the cells at offsets -2,-1,+1,+2 lie in the geometry.
// Output: w[k+2] is the weight of u(cell + k), k = -2..2. derivative = sum_k w[k+2]*u / h.
__device__ inline void derivativeStencil(bool im2, bool im1, bool ip1, bool ip2,
                                         real w[5]) {
  for (int k = 0; k < 5; k++) w[k] = 0;
  if (!im1 && !ip1) {
    // --1-- zero
  } else if ((!im2 || !ip2) && im1 && ip1) {
    // -111-, 1111-, -1111 central difference, h^2
    w[3] = 0.5; w[1] = -0.5;
  } else if (!im2 && !ip1) {
    // -11-- backward difference, h^1
    w[2] = 1.; w[1] = -1.;
  } else if (!im1 && !ip2) {
    // --11- forward difference, h^1
    w[2] = -1.; w[3] = 1.;
  } else if (im2 && !ip1) {
    // 111-- backward difference, h^2
    w[0] = 0.5; w[1] = -2.; w[2] = 1.5;
  } else if (!im1 && ip1) {
    // --111 forward difference, h^2
    w[4] = -0.5; w[3] = 2.; w[2] = -1.5;
  } else {
    // 11111 central difference, h^4
    w[3] = 2./3.; w[1] = -2./3.; w[0] = 1./12.; w[4] = -1./12.;
  }
}
