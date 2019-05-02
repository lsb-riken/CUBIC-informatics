#ifndef REGIONAL_FEATURES_CUH
#define REGIONAL_FEATURES_CUH

//#include <cuda_fp16.h>


void MaxIntensity
(
 int *d_labels, unsigned short *d_img,
 int *d_labels_tmp, unsigned short *d_img_tmp,
 int *d_labels_reg, unsigned short *d_maxint_reg,
 void *d_cub_tmp, size_t cub_tmp_bytes,
 int *d_num_regions, int w, int h, int d
 );

void MaxNormalized
(
 int *d_labels, float *d_norm,
 int *d_labels_tmp, float *d_norm_tmp,
 int *d_labels_reg, float *d_maxnorm_reg,
 void *d_cub_tmp, size_t cub_tmp_bytes,
 int *d_num_regions, int w, int h, int d
 );
void MinNormalized
(
 int *d_labels, float *d_norm,
 int *d_labels_tmp, float *d_norm_tmp,
 int *d_labels_reg, float *d_minnorm_reg,
 void *d_cub_tmp, size_t cub_tmp_bytes,
 int *d_num_regions, int w, int h, int d
 );

void SumNormalized
(
 int *d_labels, float *d_norm,
 int *d_labels_tmp, float *d_norm_tmp,
 int *d_labels_reg, float *d_sumnorm_reg,
 void *d_cub_tmp, size_t cub_tmp_bytes,
 int *d_num_regions, int w, int h, int d
 );

void RegionalSizeAndCentroid
(
 int *d_labels, float *d_grid,
 int *d_labels_tmp, float *d_grid_tmp,
 int *d_labels_reg, unsigned short *d_size_reg, float *d_grid_reg,
 void *d_cub_tmp, size_t cub_tmp_bytes,
 int *d_num_regions, int num_regions, int w, int h, int d
 );

void AverageNormalized
(
 int *d_labels, float *d_norm,
 int *d_labels_tmp, float *d_norm_tmp,
 int *d_labels_reg, float *d_avenorm_reg,
 unsigned short *d_size_reg,
 void *d_cub_tmp, size_t cub_tmp_bytes,
 int *d_num_regions, int &h_num_regions, int w, int h, int d
 );

// It seems redundant but we intentionally wrap
// CUDA-enabled function by pure C++ function
// in order to satisfy the two requirement at the same time:
//   1. use template in CUDA
//   2. separate declaration and implementation
//
// .cu is compiled by NVCC, .cuh included by .cpp is compiled by G++

template<typename T>
void _HessianFeatures
(
 int *d_labels, T *d_hessian,
 int *d_labels_tmp, T *d_hessian_tmp,
 int *d_labels_reg, T *d_hessian_reg,
 void *d_cub_tmp, size_t cub_tmp_bytes,
 int *d_num_regions, int w, int h, int d
 );

template<typename T>
void HessianFeatures
(
 int *d_labels, T *d_hessian,
 int *d_labels_tmp, T *d_hessian_tmp,
 int *d_labels_reg, T *d_hessian_reg,
 void *d_cub_tmp, size_t cub_tmp_bytes,
 int *d_num_regions, int w, int h, int d
 ) {
  _HessianFeatures<T>(d_labels, d_hessian,
                      d_labels_tmp, d_hessian_tmp,
                      d_labels_reg, d_hessian_reg,
                      d_cub_tmp, cub_tmp_bytes,
                      d_num_regions, w, h, d);
}

template void HessianFeatures<float>
(
 int *d_labels, float *d_hessian,
 int *d_labels_tmp, float *d_hessian_tmp,
 int *d_labels_reg, float *d_hessian_reg,
 void *d_cub_tmp, size_t cub_tmp_bytes,
 int *d_num_regions, int w, int h, int d
 );
/*template void HessianFeatures<half>
(
 int *d_labels, half *d_hessian,
 int *d_labels_tmp, half *d_hessian_tmp,
 int *d_labels_reg, half *d_hessian_reg,
 void *d_cub_tmp, size_t cub_tmp_bytes,
 int *d_num_regions, int w, int h, int d
 );*/


#endif