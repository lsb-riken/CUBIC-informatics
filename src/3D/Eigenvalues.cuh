#ifndef EIGENVALUES_CUH
#define EIGENVALUES_CUH

//#include <cuda_fp16.h>


// It seems redundant but we intentionally wrap
// CUDA-enabled function by pure C++ function
// in order to satisfy the two requirement at the same time:
//   1. use template in CUDA
//   2. separate declaration and implementation
//
// .cu is compiled by NVCC, .cuh included by .cpp is compiled by G++

template<typename T> void _Eigenvalues
(
 T *d_hessian_reg, float *d_eigen_reg,
 int num_regions, int pitch
 );

template<typename T> void Eigenvalues
(
 T *d_hessian_reg, float *d_eigen_reg,
 int num_regions, int pitch
 ) {
  _Eigenvalues<T>(d_hessian_reg, d_eigen_reg, num_regions, pitch);
}

template void Eigenvalues<float>
(
 float *d_hessian_reg, float *d_eigen_reg, int num_regions, int pitch
 );

/*template void Eigenvalues<half>
(
 half *d_hessian_reg, float *d_eigen_reg, int num_regions, int pitch
 );*/


#endif
