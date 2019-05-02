/*
 * Eigenvalues.cu
 *
 * David Eberly. (2014) A Robust Eigensolver for 3 × 3 Symmetric Matrices.
 * https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
 *
 */

#include <assert.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_fp16.h>
#include "Eigenvalues.cuh"

#define EIG_EPSILON 1e-30

template<typename T>
__global__ void EigenvaluesKernel
(
 T *d_hessian_reg, float *d_eigen_reg,
 int num_regions, int pitch
 )
{
  const int idx0 = blockIdx.x;
  const int idx1 = idx0 + pitch;
  const int idx2 = idx1 + pitch;
  const int idx3 = idx2 + pitch;
  const int idx4 = idx3 + pitch;
  const int idx5 = idx4 + pitch;

  double xx = (double)d_hessian_reg[idx0];
  double xy = (double)d_hessian_reg[idx1];
  double xz = (double)d_hessian_reg[idx2];
  double yy = (double)d_hessian_reg[idx3];
  double yz = (double)d_hessian_reg[idx4];
  double zz = (double)d_hessian_reg[idx5];

  double max0 = fmax(fabs(xx), fabs(xy));
  double max1 = fmax(fabs(xz), fabs(yy));
  double max2 = fmax(fabs(yz), fabs(zz));
  double maxAbsElement = fmax( fmax(max0, max1), max2);
  if(maxAbsElement < EIG_EPSILON) {
    // zero matrix
    d_eigen_reg[idx0] = 0;
    d_eigen_reg[idx1] = 0;
    d_eigen_reg[idx2] = 0;
    return;
  }

  double norm = xy*xy + xz*xz + yz*yz;
  double eig0, eig1, eig2;
  if (norm > EIG_EPSILON) {
    // acos(z) fails silently and return NaN if |z| > 1.
    // To avoid this condition due to rounding errors,
    // halfDet value is clamped to [-1,1]
    double traceDiv3 = (xx + yy + zz) / 3.0;
    double b00 = xx - traceDiv3;
    double b11 = yy - traceDiv3;
    double b22 = zz - traceDiv3;
    double denom = sqrt((b00*b00 + b11*b11 + b22*b22 + norm*2.0) / 6.0);
    double c00 = b11*b22 - yz * yz;
    double c01 = xy *b22 - yz * xz;
    double c02 = xy * yz - b11* xz;
    double det = (b00*c00 - xy*c01 + xz*c02) / (denom*denom*denom);
    double halfDet = det * 0.5;
    halfDet = fmin( fmax(halfDet, -1.0), 1.0);

    // The eigenvalues of B are ordered as beta0 <= beta1 <= beta2.
    // The number of digits in twoThirdsPi is chosen so that, whether float or double, the floating-point number is the closest to theoretical 2*pi/3.
    double angle = acos(halfDet) / 3.0;
    double const twoThirdsPi = (double)2.094395102393195;
    double beta2 = cos(angle) * 2.0;
    double beta0 = cos(angle + twoThirdsPi) * 2.0;
    double beta1 = -(beta0 + beta2);

    // the eigenvalues of A are ordered as alpha0 <= alpha1 <= alpha2.
    eig0 = traceDiv3 + denom * beta0;
    eig1 = traceDiv3 + denom * beta1;
    eig2 = traceDiv3 + denom * beta2;
  } else {
    eig0 = xx;
    eig1 = yy;
    eig2 = zz;
  }
  //d_eigen_reg[idx0] = (float)eig0;
  //d_eigen_reg[idx1] = (float)eig1;
  //d_eigen_reg[idx2] = (float)eig2;

  //structureness
  d_eigen_reg[idx0] = (float)(eig0*eig0 + eig1*eig1 + eig2*eig2);
  //blobness
  double B_denom = eig0*eig1;
  d_eigen_reg[idx1] = (float)(eig2*eig2 / B_denom);

}

template<typename T>
void _Eigenvalues
(
 T *d_hessian_reg, float *d_eigen_reg,
 int num_regions, int pitch
 ) {
  dim3 blocks(num_regions, 1, 1);
  dim3 threads(1,1,1);

  EigenvaluesKernel <<< blocks, threads>>>
    (
     d_hessian_reg, d_eigen_reg, num_regions, pitch
     );
  getLastCudaError("EigenvaluesKernel() execution failed\n");
}

// explicit instantiation
template void _Eigenvalues<float>
(
 float *d_hessian_reg, float *d_eigen_reg, int num_regions, int pitch
 );

/*template void _Eigenvalues<half>
(
 half *d_hessian_reg, float *d_eigen_reg, int num_regions, int pitch
 );
*/
