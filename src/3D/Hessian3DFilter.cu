/*
 * Hessian3DFilter.cu
 *
 */

#include <assert.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "Hessian3DFilter.cuh"


#define   HES_BLOCKDIM_X 16
#define   HES_BLOCKDIM_Y 4
#define   HES_BLOCKDIM_Z 4
#define   HES_RESULT_STEPS 8
#define   HES_HALO_STEPS 2

__global__ void HessianPositiveDefiniteKernel(
    char *d_hessian_pd,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD
)
{
  __shared__ float s_Data[HES_BLOCKDIM_Z+2][HES_BLOCKDIM_Y+2][(HES_RESULT_STEPS + 2 * HES_HALO_STEPS) * HES_BLOCKDIM_X];

  //Offset to the left halo edge
  const int baseX = (blockIdx.x * HES_RESULT_STEPS - HES_HALO_STEPS) * HES_BLOCKDIM_X + threadIdx.x;
  const int baseY = blockIdx.y * HES_BLOCKDIM_Y + threadIdx.y-1;
  const int baseZ = blockIdx.z * HES_BLOCKDIM_Z + threadIdx.z-1;
  const int idx = (baseZ * imageH + baseY) * imageW + baseX;

  d_Src += idx;  d_hessian_pd += idx;

  if(baseZ < 0 || baseZ >= imageD || baseY < 0 || baseY >= imageH) {
    for (int i = 0; i < HES_HALO_STEPS + HES_RESULT_STEPS + HES_HALO_STEPS; i++) {
      s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * HES_BLOCKDIM_X] = 0;
    }
    return;
  }

  //Load main data
#pragma unroll

  for (int i = HES_HALO_STEPS; i < HES_HALO_STEPS + HES_RESULT_STEPS; i++) {
    s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * HES_BLOCKDIM_X] = (float)d_Src[i * HES_BLOCKDIM_X];
  }

  //Load left halo
#pragma unroll

  for (int i = 0; i < HES_HALO_STEPS; i++) {
    s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * HES_BLOCKDIM_X] = (baseX + i * HES_BLOCKDIM_X >= 0) ? (float)d_Src[i * HES_BLOCKDIM_X] : 0;
  }

  //Load right halo
#pragma unroll

  for (int i = HES_HALO_STEPS + HES_RESULT_STEPS; i < HES_HALO_STEPS + HES_RESULT_STEPS + HES_HALO_STEPS; i++) {
    s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * HES_BLOCKDIM_X] = (baseX + i * HES_BLOCKDIM_X < imageW) ? (float)d_Src[i * HES_BLOCKDIM_X] : 0;
  }

  // yz edge is no need to compute
  if (threadIdx.z == 0 || threadIdx.z == HES_BLOCKDIM_Z+1 || threadIdx.y == 0 || threadIdx.y == HES_BLOCKDIM_Y+1)
    return;

  //Compute and store results
  __syncthreads();
#pragma unroll

  for (int i = HES_HALO_STEPS; i < HES_HALO_STEPS + HES_RESULT_STEPS; i++)
    {
      float xx,xy,xz,yy,yz,zz;
      xx = s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * HES_BLOCKDIM_X - 1]
        + s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * HES_BLOCKDIM_X + 1]
        - s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * HES_BLOCKDIM_X] * 2;
      xy = s_Data[threadIdx.z][threadIdx.y + 1][threadIdx.x + i * HES_BLOCKDIM_X + 1]
        + s_Data[threadIdx.z][threadIdx.y - 1][threadIdx.x + i * HES_BLOCKDIM_X - 1]
        - s_Data[threadIdx.z][threadIdx.y + 1][threadIdx.x + i * HES_BLOCKDIM_X - 1]
        - s_Data[threadIdx.z][threadIdx.y - 1][threadIdx.x + i * HES_BLOCKDIM_X + 1];
      xz = s_Data[threadIdx.z + 1][threadIdx.y][threadIdx.x + i * HES_BLOCKDIM_X + 1]
        + s_Data[threadIdx.z - 1][threadIdx.y][threadIdx.x + i * HES_BLOCKDIM_X - 1]
        - s_Data[threadIdx.z + 1][threadIdx.y][threadIdx.x + i * HES_BLOCKDIM_X - 1]
        - s_Data[threadIdx.z - 1][threadIdx.y][threadIdx.x + i * HES_BLOCKDIM_X + 1];
      yy = s_Data[threadIdx.z][threadIdx.y + 1][threadIdx.x + i * HES_BLOCKDIM_X]
        + s_Data[threadIdx.z][threadIdx.y - 1][threadIdx.x + i * HES_BLOCKDIM_X]
        - s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * HES_BLOCKDIM_X] * 2;
      yz = s_Data[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + i * HES_BLOCKDIM_X]
        + s_Data[threadIdx.z - 1][threadIdx.y - 1][threadIdx.x + i * HES_BLOCKDIM_X]
        - s_Data[threadIdx.z + 1][threadIdx.y - 1][threadIdx.x + i * HES_BLOCKDIM_X]
        - s_Data[threadIdx.z - 1][threadIdx.y + 1][threadIdx.x + i * HES_BLOCKDIM_X];
      zz = s_Data[threadIdx.z + 1][threadIdx.y][threadIdx.x + i * HES_BLOCKDIM_X]
        + s_Data[threadIdx.z - 1][threadIdx.y][threadIdx.x + i * HES_BLOCKDIM_X]
        - s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * HES_BLOCKDIM_X] * 2;

      xy *= 0.25; xz *= 0.25; yz *= 0.25;

      d_hessian_pd[i * HES_BLOCKDIM_X] = (xx < 0 && xx*yy-xy*xy < 0 && xx*yy*zz + 2*xy*yz*xz - xx*yz*yz - yy*xz*xz - zz*xy*xy < 0) ? 1 : 0;
    }
}

extern "C" void HessianPositiveDefinite(
    char *d_hessian_pd,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD
)
{
    assert(imageW % (HES_RESULT_STEPS * HES_BLOCKDIM_X) == 0);
    assert(imageH % HES_BLOCKDIM_Y == 0);
    assert(imageD % HES_BLOCKDIM_Z == 0);

    dim3 blocks(imageW / (HES_RESULT_STEPS * HES_BLOCKDIM_X), imageH / HES_BLOCKDIM_Y, imageD / HES_BLOCKDIM_Z);
    dim3 threads(HES_BLOCKDIM_X, HES_BLOCKDIM_Y+2, HES_BLOCKDIM_Z+2);

    HessianPositiveDefiniteKernel<<<blocks, threads>>>(
        d_hessian_pd,
        d_Src,
        imageW,
        imageH,
        imageD
    );
    getLastCudaError("HessianPositiveDefiniteKernel() execution failed\n");

}
