/*
 * RegionalFeatures.cu
 */

#include <assert.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cub/cub.cuh>
#include "MaxIntensity.cuh"

void MaxIntensity
(
 unsigned short *d_img,
 unsigned short *d_maxintensity,
 void *d_cub_tmp, size_t cub_tmp_bytes,
 int width, int height, int depth
 ) {

  size_t temp_storage_bytes = 0;

  cub::DeviceReduce::Max(NULL, temp_storage_bytes,
                         d_img, d_maxintensity, width*height);
  assert(temp_storage_bytes < cub_tmp_bytes);

  for(int i=0; i < depth; i++) {
    cub::DeviceReduce::Max(d_cub_tmp, temp_storage_bytes,
                           &d_img[i*width*height],
                           &d_maxintensity[i],
                           width*height);
  }
}
