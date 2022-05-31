/*
 * RegionalFeatures.cu
 */

#include <assert.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include "RegionalFeatures.cuh"

struct MaxOp
{
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return (b > a) ? b : a;
    }
};
struct MinOp
{
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return (b < a) ? b : a;
    }
};

struct SumOp
{
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return a+b;
    }
};

void MaxIntensity
(
 int *d_labels, unsigned short *d_img,
 int *d_labels_tmp, unsigned short *d_img_tmp,
 int *d_labels_reg, unsigned short *d_maxint_reg,
 void *d_cub_tmp, size_t cub_tmp_bytes,
 int *d_num_regions, int w, int h, int d
 ) {
  int image_size = w*h*d;
  size_t   temp_storage_bytes = 0;
  MaxOp max_op;

  // Max Intensity for Original Image
  // Radix Sort
  cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, d_labels, d_labels_tmp, d_img, d_img_tmp, image_size);
  assert(temp_storage_bytes < cub_tmp_bytes);
  cub::DeviceRadixSort::SortPairs(d_cub_tmp, temp_storage_bytes, d_labels, d_labels_tmp, d_img, d_img_tmp, image_size);

  // Reduce
  temp_storage_bytes = 0;
  cub::DeviceReduce::ReduceByKey(NULL, temp_storage_bytes, d_labels_tmp, d_labels_reg, d_img_tmp, d_maxint_reg, d_num_regions, max_op, image_size);
  assert(temp_storage_bytes < cub_tmp_bytes);
  cub::DeviceReduce::ReduceByKey(d_cub_tmp, temp_storage_bytes, d_labels_tmp, d_labels_reg, d_img_tmp, d_maxint_reg, d_num_regions, max_op, image_size);

}

void MaxNormalized
(
 int *d_labels, float *d_norm,
 int *d_labels_tmp, float *d_norm_tmp,
 int *d_labels_reg, float *d_maxnorm_reg,
 void *d_cub_tmp, size_t cub_tmp_bytes,
 int *d_num_regions, int w, int h, int d
 ) {
  int image_size = w*h*d;
  size_t   temp_storage_bytes = 0;
  MaxOp max_op;

  // Max Intensity for Normalized Image
  temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, d_labels, d_labels_tmp, d_norm, d_norm_tmp, image_size);
  assert(temp_storage_bytes < cub_tmp_bytes);
  cub::DeviceRadixSort::SortPairs(d_cub_tmp, temp_storage_bytes, d_labels, d_labels_tmp, d_norm, d_norm_tmp, image_size);

  temp_storage_bytes = 0;
  cub::DeviceReduce::ReduceByKey(NULL, temp_storage_bytes, d_labels_tmp, d_labels_reg, d_norm_tmp, d_maxnorm_reg, d_num_regions, max_op, image_size);
  assert(temp_storage_bytes < cub_tmp_bytes);
  cub::DeviceReduce::ReduceByKey(d_cub_tmp, temp_storage_bytes, d_labels_tmp, d_labels_reg, d_norm_tmp, d_maxnorm_reg, d_num_regions, max_op, image_size);

}
void MinNormalized
(
 int *d_labels, float *d_norm,
 int *d_labels_tmp, float *d_norm_tmp,
 int *d_labels_reg, float *d_minnorm_reg,
 void *d_cub_tmp, size_t cub_tmp_bytes,
 int *d_num_regions, int w, int h, int d
 ) {
  int image_size = w*h*d;
  size_t   temp_storage_bytes = 0;
  MinOp min_op;

  // Min Intensity for Normalized Image
  temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, d_labels, d_labels_tmp, d_norm, d_norm_tmp, image_size);
  assert(temp_storage_bytes < cub_tmp_bytes);
  cub::DeviceRadixSort::SortPairs(d_cub_tmp, temp_storage_bytes, d_labels, d_labels_tmp, d_norm, d_norm_tmp, image_size);

  temp_storage_bytes = 0;
  cub::DeviceReduce::ReduceByKey(NULL, temp_storage_bytes, d_labels_tmp, d_labels_reg, d_norm_tmp, d_minnorm_reg, d_num_regions, min_op, image_size);
  assert(temp_storage_bytes < cub_tmp_bytes);
  cub::DeviceReduce::ReduceByKey(d_cub_tmp, temp_storage_bytes, d_labels_tmp, d_labels_reg, d_norm_tmp, d_minnorm_reg, d_num_regions, min_op, image_size);

}

void SumNormalized
(
 int *d_labels, float *d_norm,
 int *d_labels_tmp, float *d_norm_tmp,
 int *d_labels_reg, float *d_sumnorm_reg,
 void *d_cub_tmp, size_t cub_tmp_bytes,
 int *d_num_regions, int w, int h, int d
 ) {
  int image_size = w*h*d;
  size_t   temp_storage_bytes = 0;
  SumOp sum_op;

  // Sum Intensity for Normalized Image
  temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, d_labels, d_labels_tmp, d_norm, d_norm_tmp, image_size);
  assert(temp_storage_bytes < cub_tmp_bytes);
  cub::DeviceRadixSort::SortPairs(d_cub_tmp, temp_storage_bytes, d_labels, d_labels_tmp, d_norm, d_norm_tmp, image_size);

  temp_storage_bytes = 0;
  cub::DeviceReduce::ReduceByKey(NULL, temp_storage_bytes, d_labels_tmp, d_labels_reg, d_norm_tmp, d_sumnorm_reg, d_num_regions, sum_op, image_size);
  assert(temp_storage_bytes < cub_tmp_bytes);
  cub::DeviceReduce::ReduceByKey(d_cub_tmp, temp_storage_bytes, d_labels_tmp, d_labels_reg, d_norm_tmp, d_sumnorm_reg, d_num_regions, sum_op, image_size);

}

void RegionalSize
(
 int *d_labels_tmp, int *d_labels_reg, unsigned short *d_size_reg,
 void *d_cub_tmp, size_t cub_tmp_bytes,
 int *d_num_regions, int image_size
 ) {
  size_t   temp_storage_bytes = 0;

  // RunLengthEncode
  cub::DeviceRunLengthEncode::Encode(NULL, temp_storage_bytes, d_labels_tmp, d_labels_reg, d_size_reg, d_num_regions, image_size);
  assert(temp_storage_bytes < cub_tmp_bytes);
  cub::DeviceRunLengthEncode::Encode(d_cub_tmp, temp_storage_bytes, d_labels_tmp, d_labels_reg, d_size_reg, d_num_regions, image_size);

}

#define IG_BLOCKDIM_X 8
#define IG_BLOCKDIM_Y 8
#define IG_BLOCKDIM_Z 8
__global__ void initGridKernel
(
 float *d_grid, int axis, int w, int h, int d
) {
  const int baseX = blockIdx.x * IG_BLOCKDIM_X + threadIdx.x;
  const int baseY = blockIdx.y * IG_BLOCKDIM_Y + threadIdx.y;
  const int baseZ = blockIdx.z * IG_BLOCKDIM_Z + threadIdx.z;

  const int idx = (baseZ * h + baseY) * w + baseX;

  if(axis == 0) {
    d_grid[idx] = (float)baseX;
  } else if(axis == 1) {
    d_grid[idx] = (float)baseY;
  } else {
    d_grid[idx] = (float)baseZ;
  }

}

extern "C" void initGrid
(
 float *d_grid, int axis, int w, int h, int d
) {
  assert(w % (IG_BLOCKDIM_X) == 0);
  assert(h % (IG_BLOCKDIM_Y) == 0);
  assert(d % (IG_BLOCKDIM_Z) == 0);
  dim3 blocks(w / (IG_BLOCKDIM_X), h/(IG_BLOCKDIM_Y), d / (IG_BLOCKDIM_Z));
  dim3 threads(IG_BLOCKDIM_X, IG_BLOCKDIM_Y, IG_BLOCKDIM_Z);

  initGridKernel<<<blocks, threads>>>(d_grid, axis, w,h, d);
  getLastCudaError("Error: initGridKernel() kernel execution FAILED!");
}


__global__ void DivideKernel
(
 float *d_dst,
 unsigned short *d_denom
 ) {
  const int idx = blockIdx.x;
  d_dst[idx] /= d_denom[idx];
}

extern "C" void Divide
(
 float *d_dst,
 unsigned short *d_denom,
 int num_regions
 ) {
  dim3 blocks(num_regions, 1, 1);
  dim3 threads(1,1,1);

  DivideKernel<<<blocks, threads>>>(d_dst, d_denom);
  getLastCudaError("Error: DivideKernel() kernel execution FAILED!");
}

void RegionalSizeAndCentroid
(
 int *d_labels, float *d_grid,
 int *d_labels_tmp, float *d_grid_tmp,
 int *d_labels_reg, unsigned short *d_size_reg, float *d_grid_reg,
 void *d_cub_tmp, size_t cub_tmp_bytes,
 int *d_num_regions, int num_regions, int w, int h, int d
 ) {
  int image_size = w*h*d;
  size_t   temp_storage_bytes = 0;
  float *d_grid_reg_el;

  for(int i = 0; i < 3; i++) {
    d_grid_reg_el = d_grid_reg + i*image_size;

    initGrid(d_grid, i, w,h,d);

    // Radix Sort
    temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, d_labels, d_labels_tmp, d_grid, d_grid_tmp, image_size);
    assert(temp_storage_bytes < cub_tmp_bytes);
    cub::DeviceRadixSort::SortPairs(d_cub_tmp, temp_storage_bytes, d_labels, d_labels_tmp, d_grid, d_grid_tmp, image_size);

    // Reduce
    SumOp sum_op;
    temp_storage_bytes = 0;
    cub::DeviceReduce::ReduceByKey(NULL, temp_storage_bytes, d_labels_tmp, d_labels_reg, d_grid_tmp, d_grid_reg_el, d_num_regions, sum_op, image_size);
    assert(temp_storage_bytes < cub_tmp_bytes);
    cub::DeviceReduce::ReduceByKey(d_cub_tmp, temp_storage_bytes, d_labels_tmp, d_labels_reg, d_grid_tmp, d_grid_reg_el, d_num_regions, sum_op, image_size);
  }

  RegionalSize(d_labels_tmp, d_labels_reg, d_size_reg,
               d_cub_tmp, cub_tmp_bytes,
               d_num_regions, image_size);

  // Divide the sum by region size to get average
  for(int i = 0; i < 3; i++) {
    d_grid_reg_el = d_grid_reg + i*image_size;

    Divide(d_grid_reg_el, d_size_reg, num_regions);
  }
}

void AverageNormalized
(
 int *d_labels, float *d_norm,
 int *d_labels_tmp, float *d_norm_tmp,
 int *d_labels_reg, float *d_avenorm_reg,
 unsigned short *d_size_reg,
 void *d_cub_tmp, size_t cub_tmp_bytes,
 int *d_num_regions, int &h_num_regions, int w, int h, int d
 ) {
  int image_size = w*h*d;
  size_t   temp_storage_bytes = 0;
  SumOp sum_op;

  // Sum Intensity for Normalized Image
  temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, d_labels, d_labels_tmp, d_norm, d_norm_tmp, image_size);
  assert(temp_storage_bytes < cub_tmp_bytes);
  cub::DeviceRadixSort::SortPairs(d_cub_tmp, temp_storage_bytes, d_labels, d_labels_tmp, d_norm, d_norm_tmp, image_size);

  temp_storage_bytes = 0;
  cub::DeviceReduce::ReduceByKey(NULL, temp_storage_bytes, d_labels_tmp, d_labels_reg, d_norm_tmp, d_avenorm_reg, d_num_regions, sum_op, image_size);
  assert(temp_storage_bytes < cub_tmp_bytes);
  cub::DeviceReduce::ReduceByKey(d_cub_tmp, temp_storage_bytes, d_labels_tmp, d_labels_reg, d_norm_tmp, d_avenorm_reg, d_num_regions, sum_op, image_size);

  RegionalSize(d_labels_tmp, d_labels_reg, d_size_reg,
               d_cub_tmp, cub_tmp_bytes,
               d_num_regions, image_size);

  // get num_regions in host
  checkCudaErrors(cudaMemcpy(&h_num_regions, d_num_regions, sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "num_regions: " << h_num_regions << std::endl;

  // Divide sum by size to get average
  Divide(d_avenorm_reg, d_size_reg, h_num_regions);

}

template<typename T>
void _HessianFeatures
(
 int *d_labels, T *d_hessian,
 int *d_labels_tmp, T *d_hessian_tmp,
 int *d_labels_reg, T *d_hessian_reg,
 void *d_cub_tmp, size_t cub_tmp_bytes,
 int *d_num_regions, int w, int h, int d
 ) {
  int image_size = w*h*d;
  size_t   temp_storage_bytes = 0;

  T *d_hessian_el;
  T *d_hessian_el_reg;
  // Hessian Element
  for(int i = 0 ; i < 6; i++) {
    // next element
    d_hessian_el = d_hessian + image_size*i;
    d_hessian_el_reg = d_hessian_reg + image_size*i;

    // Radix Sort
    temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, d_labels, d_labels_tmp, d_hessian_el, d_hessian_tmp, image_size);
    // std::cout << "SortPairs@hessian " << temp_storage_bytes << " , " << cub_tmp_bytes << std::endl;
    assert(temp_storage_bytes < cub_tmp_bytes);
    cub::DeviceRadixSort::SortPairs(d_cub_tmp, temp_storage_bytes, d_labels, d_labels_tmp, d_hessian_el, d_hessian_tmp, image_size);

    // Regional Reduce
    SumOp sum_op;
    temp_storage_bytes = 0;
    cub::DeviceReduce::ReduceByKey(NULL, temp_storage_bytes, d_labels_tmp, d_labels_reg, d_hessian_tmp, d_hessian_el_reg, d_num_regions, sum_op, image_size);
    // std::cout << "ReduceByKey@hessian " << temp_storage_bytes << " , " << cub_tmp_bytes << std::endl;
    assert(temp_storage_bytes < cub_tmp_bytes);
    cub::DeviceReduce::ReduceByKey(d_cub_tmp, temp_storage_bytes, d_labels_tmp, d_labels_reg, d_hessian_tmp, d_hessian_el_reg, d_num_regions, sum_op, image_size);
  }
}


// explicit instantiation
template void _HessianFeatures<float>
(
 int *d_labels, float *d_hessian,
 int *d_labels_tmp, float *d_hessian_tmp,
 int *d_labels_reg, float *d_hessian_reg,
 void *d_cub_tmp, size_t cub_tmp_bytes,
 int *d_num_regions, int w, int h, int d
 );
/*template void _HessianFeatures<half>
(
 int *d_labels, half *d_hessian,
 int *d_labels_tmp, half *d_hessian_tmp,
 int *d_labels_reg, half *d_hessian_reg,
 void *d_cub_tmp, size_t cub_tmp_bytes,
 int *d_num_regions, int w, int h, int d
 );
*/