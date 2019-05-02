/*
 * DoG3DFilter.cu
 * based on NVIDIA Toolkit Samples: convolutionSeparable
 *
 */

#include <assert.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "DoG3DFilter.cuh"

// diameter should be odd number
#define MAX_KERNEL_LENGTH 80
#define TRUNCATE 2.0
// max sigma = MAX_KERNEL_LENGTH/TRUNCATE = 20.0

/*
 * Convolution kernel storage
 * c_Kernel[0:MAX_KERNEL_LENGTH] : 1st XY
 * c_Kernel[MAX_KERNEL_LENGTH:MAX_KERNEL_LENGTH*2] : 1st Z
 * c_Kernel[MAX_KERNEL_LENGTH*2:MAX_KERNEL_LENGTH*3] : 2nd XY
 * c_Kernel[MAX_KERNEL_LENGTH*3:MAX_KERNEL_LENGTH*4] : 2nd Z
 */
__constant__ float c_Kernel[MAX_KERNEL_LENGTH * 4];
int radius_kernel[4];
float sigma_kernel[4];

// Row convolution
// width = 2560 = 5 * 2^9  = divisible by 8*16
// radius should be <= 16*2
// sigma should be < 16*2 / 2
#define   ROWS_BLOCKDIM_X 16
#define   ROWS_BLOCKDIM_Y 4
#define   ROWS_BLOCKDIM_Z 4
#define   ROWS_RESULT_STEPS 8
#define   ROWS_HALO_STEPS 2

// Column convolution
// height = 2160 = 3^3 * 5 * 2^4  = divisible by 9*16
// radius should be <= 16*2
// sigma should be < 16*2/2
#define   COLUMNS_BLOCKDIM_X 4
#define   COLUMNS_BLOCKDIM_Y 16
#define   COLUMNS_BLOCKDIM_Z 4
#define   COLUMNS_RESULT_STEPS 9
#define   COLUMNS_HALO_STEPS 2


// Layer convolution
// depth = 32 = 2^5  = divisible by 8*4
// radius should be <= 8*2
// sigma should be < 8*2/2
#define   LAYERS_BLOCKDIM_X 8
#define   LAYERS_BLOCKDIM_Y 8
#define   LAYERS_BLOCKDIM_Z 8
#define   LAYERS_RESULT_STEPS 4
#define   LAYERS_HALO_STEPS 2

// Subtract filter
#define SUB_BLOCKDIM_X 8
#define SUB_BLOCKDIM_Y 8
#define SUB_BLOCKDIM_Z 8



/*
 * Row convolution filter
 */
__global__ void convolutionRows3DKernel(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD,
    int kernel_index,
    int kernel_radius
)
{
    __shared__ float s_Data[ROWS_BLOCKDIM_Z][ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    //Offset to the left halo edge
    const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;
    const int baseZ = blockIdx.z * ROWS_BLOCKDIM_Z + threadIdx.z;

    d_Src += (baseZ * imageH + baseY) * imageW + baseX;
    d_Dst += (baseZ * imageH + baseY) * imageW + baseX;

    const float* kernel = &c_Kernel[kernel_index*MAX_KERNEL_LENGTH];

    //Load main data
    #pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
      s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
    }

    //Load left halo
    #pragma unroll

    for (int i = 0; i < ROWS_HALO_STEPS; i++) {
      s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X >= 0) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Load right halo
    #pragma unroll

    for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++) {
      s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX + i * ROWS_BLOCKDIM_X < imageW) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Compute and store results
    __syncthreads();
    #pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
        float sum = 0;

        //#pragma unroll

        for (int j = -kernel_radius; j <= kernel_radius; j++)
        {
          sum += kernel[kernel_radius - j] * s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
        }

        d_Dst[i * ROWS_BLOCKDIM_X] = sum;
    }
}

extern "C" void convolutionRows3D(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD,
    int kernel_index,
    int kernel_radius
)
{
    assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= kernel_radius);
    assert(imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
    assert(imageH % ROWS_BLOCKDIM_Y == 0);
    assert(imageD % ROWS_BLOCKDIM_Z == 0);

    dim3 blocks(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y, imageD / ROWS_BLOCKDIM_Z);
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y, ROWS_BLOCKDIM_Z);

    convolutionRows3DKernel<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageD,
        kernel_index,
        kernel_radius
    );
    getLastCudaError("convolutionRows3DKernel() execution failed\n");
}

/*
 * Column convolution filter
*/
__global__ void convolutionColumns3DKernel(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD,
    int kernel_index,
    int kernel_radius
)
{
    __shared__ float s_Data[COLUMNS_BLOCKDIM_Z][COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    const int baseZ = blockIdx.z * COLUMNS_BLOCKDIM_Z + threadIdx.z;
    d_Src += (baseZ * imageH + baseY) * imageW + baseX;
    d_Dst += (baseZ * imageH + baseY) * imageW + baseX;

    const float* kernel = &c_Kernel[kernel_index*MAX_KERNEL_LENGTH];

    //Main data
    #pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
      s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * imageW];
    }

    //Upper halo
    #pragma unroll

    for (int i = 0; i < COLUMNS_HALO_STEPS; i++) {
      s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y >= 0) ? d_Src[i * COLUMNS_BLOCKDIM_Y * imageW] : 0;
    }

    //Lower halo
    #pragma unroll

    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++) {
      s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (baseY + i * COLUMNS_BLOCKDIM_Y < imageH) ? d_Src[i * COLUMNS_BLOCKDIM_Y * imageW] : 0;
    }

    //Compute and store results
    __syncthreads();
    #pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
        float sum = 0;
        //#pragma unroll

        for (int j = -kernel_radius; j <= kernel_radius; j++) {
          sum += kernel[kernel_radius - j] * s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
        }

        d_Dst[i * COLUMNS_BLOCKDIM_Y * imageW] = sum;
    }
}

extern "C" void convolutionColumns3D(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD,
    int kernel_index,
    int kernel_radius
)
{
    assert(COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= kernel_radius);
    assert(imageW % COLUMNS_BLOCKDIM_X == 0);
    assert(imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);
    assert(imageD % COLUMNS_BLOCKDIM_Z == 0);

    dim3 blocks(imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y), imageD / COLUMNS_BLOCKDIM_Z);
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y, COLUMNS_BLOCKDIM_Z);

    convolutionColumns3DKernel<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageD,
        kernel_index,
        kernel_radius
    );
    getLastCudaError("convolutionColumns3DKernel() execution failed\n");
}

/*
 * Layer convolution filter
*/
__global__ void convolutionLayers3DKernel(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD,
    int kernel_index,
    int kernel_radius
)
{
    __shared__ float s_Data[LAYERS_BLOCKDIM_X][LAYERS_BLOCKDIM_Y][(LAYERS_RESULT_STEPS + 2 * LAYERS_HALO_STEPS) * LAYERS_BLOCKDIM_Z + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * LAYERS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * LAYERS_BLOCKDIM_Y + threadIdx.y;
    const int baseZ = (blockIdx.z * LAYERS_RESULT_STEPS - LAYERS_HALO_STEPS) * LAYERS_BLOCKDIM_Z + threadIdx.z;
    d_Src += (baseZ * imageH + baseY) * imageW + baseX;
    d_Dst += (baseZ * imageH + baseY) * imageW + baseX;

    const int pitch = imageW*imageH;
    const float* kernel = &c_Kernel[kernel_index*MAX_KERNEL_LENGTH];

    //Main data
    #pragma unroll

    for (int i = LAYERS_HALO_STEPS; i < LAYERS_HALO_STEPS + LAYERS_RESULT_STEPS; i++) {
      s_Data[threadIdx.x][threadIdx.y][threadIdx.z + i * LAYERS_BLOCKDIM_Z] = d_Src[i * LAYERS_BLOCKDIM_Z * pitch];
    }

    //Upper halo
    #pragma unroll

    for (int i = 0; i < LAYERS_HALO_STEPS; i++) {
      s_Data[threadIdx.x][threadIdx.y][threadIdx.z + i * LAYERS_BLOCKDIM_Z] = (baseZ + i * LAYERS_BLOCKDIM_Z >= 0) ? d_Src[i * LAYERS_BLOCKDIM_Z * pitch] : 0;
    }

    //Lower halo
    #pragma unroll

    for (int i = LAYERS_HALO_STEPS + LAYERS_RESULT_STEPS; i < LAYERS_HALO_STEPS + LAYERS_RESULT_STEPS + LAYERS_HALO_STEPS; i++) {
      s_Data[threadIdx.x][threadIdx.y][threadIdx.z + i * LAYERS_BLOCKDIM_Z]= (baseZ + i * LAYERS_BLOCKDIM_Z < imageD) ? d_Src[i * LAYERS_BLOCKDIM_Z * pitch] : 0;
    }

    //Compute and store results
    __syncthreads();
    #pragma unroll

    for (int i = LAYERS_HALO_STEPS; i < LAYERS_HALO_STEPS + LAYERS_RESULT_STEPS; i++) {
        float sum = 0;
        //#pragma unroll

        for (int j = -kernel_radius; j <= kernel_radius; j++) {
          sum += kernel[kernel_radius - j] * s_Data[threadIdx.x][threadIdx.y][threadIdx.z + i * LAYERS_BLOCKDIM_Z + j];
        }

        d_Dst[i * LAYERS_BLOCKDIM_Z * pitch] = sum;
    }
}

extern "C" void convolutionLayers3D(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD,
    int kernel_index,
    int kernel_radius
)
{
    assert(LAYERS_BLOCKDIM_Z * LAYERS_HALO_STEPS >= kernel_radius);
    assert(imageW % LAYERS_BLOCKDIM_X == 0);
    assert(imageH % LAYERS_BLOCKDIM_Y == 0);
    assert(imageD % (LAYERS_RESULT_STEPS * LAYERS_BLOCKDIM_Z) == 0);

    dim3 blocks(imageW / LAYERS_BLOCKDIM_X, imageH / LAYERS_BLOCKDIM_Y, imageD / (LAYERS_RESULT_STEPS * LAYERS_BLOCKDIM_Z));
    dim3 threads(LAYERS_BLOCKDIM_X, LAYERS_BLOCKDIM_Y, LAYERS_BLOCKDIM_Z);

    convolutionLayers3DKernel<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageD,
        kernel_index,
        kernel_radius
    );
    getLastCudaError("convolutionLayers3DKernel() execution failed\n");
}


__global__ void NormalizeSubtract3DKernel(float * img_src, const float * img_sub, const int width, const int height, const int depth, float normalizer) {
  const int baseX = blockIdx.x * SUB_BLOCKDIM_X + threadIdx.x;
  const int baseY = blockIdx.y * SUB_BLOCKDIM_Y + threadIdx.y;
  const int baseZ = blockIdx.z * SUB_BLOCKDIM_Z + threadIdx.z;

  const int idx = (baseZ * height + baseY) * width + baseX;
  img_src[idx] = (img_src[idx] - img_sub[idx]) * normalizer;

}

extern "C" void NormalizeSubtract3DFilter(float *d_src, const float *d_sub, const int width, const int height, const int depth, float normalizer) {
  assert(width % (SUB_BLOCKDIM_X) == 0);
  assert(height % (SUB_BLOCKDIM_Y) == 0);
  assert(depth % (SUB_BLOCKDIM_Z) == 0);
  dim3 blocks(width / (SUB_BLOCKDIM_X), height/(SUB_BLOCKDIM_Y), depth / (SUB_BLOCKDIM_Z));
  dim3 threads(SUB_BLOCKDIM_X, SUB_BLOCKDIM_Y, SUB_BLOCKDIM_Z);

  NormalizeSubtract3DKernel<<<blocks, threads>>>(d_src, d_sub, width, height, depth, normalizer);
  getLastCudaError("Error: Subtract3DKernel() kernel execution FAILED!");

}

/*
 * Gaussian 3D filter
 */

extern "C" int calcGaussianWeight(float *h_kernel, float sigma) {
  int lw = (int)(TRUNCATE * sigma + 0.5);
  int length = lw * 2 + 1;
  float p,sum;
  sum = 1.;
  h_kernel[lw] = 1.;
  sigma *= sigma;
  for(int i = 1; i <= lw; i++) {
    p = exp(-0.5 * i*i / sigma);
    h_kernel[lw-i] = p;
    h_kernel[lw+i] = p;
    sum += p * 2;
  }
  for(int i = 0; i < length; i++)
    h_kernel[i] /= sum;

  for(int i=length; i < MAX_KERNEL_LENGTH; i++)
    h_kernel[i] = 0;

  return lw;
}

extern "C" int initGaussian3DKernel(const float sigma_xy1, const float sigma_z1, const float sigma_xy2, const float sigma_z2){
  // save as global
  sigma_kernel[0] = sigma_xy1;
  sigma_kernel[1] = sigma_z1;
  sigma_kernel[2] = sigma_xy2;
  sigma_kernel[3] = sigma_z2;

  float *h_kernel = (float *)malloc(4*MAX_KERNEL_LENGTH*sizeof(float));

  radius_kernel[0] = calcGaussianWeight(&h_kernel[0], sigma_xy1);
  radius_kernel[1] = calcGaussianWeight(&h_kernel[MAX_KERNEL_LENGTH], sigma_z1);
  radius_kernel[2] = calcGaussianWeight(&h_kernel[MAX_KERNEL_LENGTH*2], sigma_xy2);
  radius_kernel[3] = calcGaussianWeight(&h_kernel[MAX_KERNEL_LENGTH*3], sigma_z2);

  cudaMemcpyToSymbol(c_Kernel, h_kernel, 4*MAX_KERNEL_LENGTH * sizeof(float));

  getLastCudaError("Error: MemcpyToSymbol failed!");

  free(h_kernel);

  return radius_kernel[3];
}

extern "C" void Gaussian3DFilter(float *d_img, float *d_temp, float *d_result, const int sigma_num, const int width, const int height, const int depth) {
  if(sigma_num == 1) {
    convolutionRows3D(d_result, d_img, width, height, depth, 0, radius_kernel[0]);
    convolutionColumns3D(d_temp, d_result, width, height, depth, 0, radius_kernel[0]);
    convolutionLayers3D(d_result, d_temp, width, height, depth, 1, radius_kernel[1]);
  } else if(sigma_num == 2){
    convolutionRows3D(d_result, d_img, width, height, depth, 2, radius_kernel[2]);
    convolutionColumns3D(d_temp, d_result, width, height, depth, 2, radius_kernel[2]);
    convolutionLayers3D(d_result, d_temp, width, height, depth, 3, radius_kernel[3]);
  }
  getLastCudaError("Error: convolutionRows3D() or convolutionColumns3D() or convolutionLayers3D() kernel execution FAILED!");
  //checkCudaErrors(cudaDeviceSynchronize());
}

extern "C" void DoG3DFilter(float *d_img, float *d_temp1, float *d_temp2, float *d_result, const int width, const int height, const int depth, float gamma) {
  float normalizer = powf(sigma_kernel[0], gamma*2)*powf(sigma_kernel[1], gamma);
  Gaussian3DFilter(d_img, d_temp1, d_temp2, 2, width, height, depth);
  Gaussian3DFilter(d_img, d_temp1, d_result, 1, width, height, depth);
  NormalizeSubtract3DFilter(d_result, d_temp2, width, height, depth, normalizer);
  //checkCudaErrors(cudaDeviceSynchronize());
}
