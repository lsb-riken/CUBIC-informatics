/*
 * Erosion3DFilter.cu
 *
 */

#include <assert.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "Erosion3DFilter.cuh"

/*
 * Row erosion filter
 */
// 2560 = 5 * 2^9  = divisible by 8*16
#define   ER_ROWS_BLOCKDIM_X 16
#define   ER_ROWS_BLOCKDIM_Y 4
#define   ER_ROWS_BLOCKDIM_Z 4
#define   ER_ROWS_RESULT_STEPS 8
#define   ER_ROWS_HALO_STEPS 1

__global__ void erosionRows3DKernel
(
 unsigned short *d_dst, unsigned short *d_src,
 int w, int h, int d,
 int kernel_radius
)
{
    __shared__ unsigned short smem[ER_ROWS_BLOCKDIM_Z][ER_ROWS_BLOCKDIM_Y][(ER_ROWS_RESULT_STEPS + 2 * ER_ROWS_HALO_STEPS) * ER_ROWS_BLOCKDIM_X];
    unsigned short *smem_thread = smem[threadIdx.z][threadIdx.y];

    //Offset to the left halo edge
    const int baseX = (blockIdx.x * ER_ROWS_RESULT_STEPS - ER_ROWS_HALO_STEPS) * ER_ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ER_ROWS_BLOCKDIM_Y + threadIdx.y;
    const int baseZ = blockIdx.z * ER_ROWS_BLOCKDIM_Z + threadIdx.z;

    d_src += (baseZ * h + baseY) * w + baseX;
    d_dst += (baseZ * h + baseY) * w + baseX;

    //Load main data
    #pragma unroll
    for (int i = ER_ROWS_HALO_STEPS; i < ER_ROWS_HALO_STEPS + ER_ROWS_RESULT_STEPS; i++) {
      smem_thread[threadIdx.x + i * ER_ROWS_BLOCKDIM_X] = d_src[i * ER_ROWS_BLOCKDIM_X];
    }

    //Load left halo
    #pragma unroll
    for (int i = 0; i < ER_ROWS_HALO_STEPS; i++) {
      smem_thread[threadIdx.x + i * ER_ROWS_BLOCKDIM_X] = (baseX + i * ER_ROWS_BLOCKDIM_X >= 0) ? d_src[i * ER_ROWS_BLOCKDIM_X] : 0;
    }

    //Load right halo
    #pragma unroll
    for (int i = ER_ROWS_HALO_STEPS + ER_ROWS_RESULT_STEPS; i < ER_ROWS_HALO_STEPS + ER_ROWS_RESULT_STEPS + ER_ROWS_HALO_STEPS; i++) {
      smem_thread[threadIdx.x + i * ER_ROWS_BLOCKDIM_X] = (baseX + i * ER_ROWS_BLOCKDIM_X < w) ? d_src[i * ER_ROWS_BLOCKDIM_X] : 0;
    }

    //Compute and store results
    __syncthreads();
    #pragma unroll
    for (int i = ER_ROWS_HALO_STEPS; i < ER_ROWS_HALO_STEPS + ER_ROWS_RESULT_STEPS; i++) {
      unsigned short *smem_kern = &smem_thread[threadIdx.x + i * ER_ROWS_BLOCKDIM_X - kernel_radius];
      unsigned short val = smem_kern[0];

      //#pragma unroll
      for (int j = 1; j <= 2*kernel_radius; j++) {
        val = min(val, smem_kern[j]);
      }
      d_dst[i * ER_ROWS_BLOCKDIM_X] = val;
    }
}


void erosionRows3D
(
 unsigned short *d_dst, unsigned short *d_src,
 int w, int h, int d,
 int kernel_radius
 )
{
  assert(ER_ROWS_BLOCKDIM_X * ER_ROWS_HALO_STEPS >= kernel_radius);
  assert(w % (ER_ROWS_RESULT_STEPS * ER_ROWS_BLOCKDIM_X) == 0);
  assert(h % ER_ROWS_BLOCKDIM_Y == 0);
  assert(d % ER_ROWS_BLOCKDIM_Z == 0);

  dim3 blocks(w / (ER_ROWS_RESULT_STEPS * ER_ROWS_BLOCKDIM_X), h / ER_ROWS_BLOCKDIM_Y, d / ER_ROWS_BLOCKDIM_Z);
  dim3 threads(ER_ROWS_BLOCKDIM_X, ER_ROWS_BLOCKDIM_Y, ER_ROWS_BLOCKDIM_Z);

  erosionRows3DKernel<<<blocks, threads>>>
    (
     d_dst, d_src, w,h,d, kernel_radius
     );
  getLastCudaError("erosionRows3DKernel() execution failed\n");
}

/*
 * Column erosion filter
*/
// 2160 = 3^3 * 5 * 2^4  = divisible by 9*16
#define   ER_COLUMNS_BLOCKDIM_X 4
#define   ER_COLUMNS_BLOCKDIM_Y 16
#define   ER_COLUMNS_BLOCKDIM_Z 4
#define   ER_COLUMNS_RESULT_STEPS 9
#define   ER_COLUMNS_HALO_STEPS 1

__global__ void erosionColumns3DKernel(
    unsigned short *d_dst, unsigned short *d_src,
    int w,int h,int d,
    int kernel_radius
)
{
    __shared__ unsigned short smem[ER_COLUMNS_BLOCKDIM_Z][ER_COLUMNS_BLOCKDIM_X][(ER_COLUMNS_RESULT_STEPS + 2 * ER_COLUMNS_HALO_STEPS) * ER_COLUMNS_BLOCKDIM_Y + 1];
    unsigned short *smem_thread = smem[threadIdx.z][threadIdx.x];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * ER_COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * ER_COLUMNS_RESULT_STEPS - ER_COLUMNS_HALO_STEPS) * ER_COLUMNS_BLOCKDIM_Y + threadIdx.y;
    const int baseZ = blockIdx.z * ER_COLUMNS_BLOCKDIM_Z + threadIdx.z;
    d_src += (baseZ * h + baseY) * w + baseX;
    d_dst += (baseZ * h + baseY) * w + baseX;

    //Main data
    #pragma unroll
    for (int i = ER_COLUMNS_HALO_STEPS; i < ER_COLUMNS_HALO_STEPS + ER_COLUMNS_RESULT_STEPS; i++) {
      smem_thread[threadIdx.y + i * ER_COLUMNS_BLOCKDIM_Y] = d_src[i * ER_COLUMNS_BLOCKDIM_Y * w];
    }

    //Upper halo
    #pragma unroll
    for (int i = 0; i < ER_COLUMNS_HALO_STEPS; i++) {
      smem_thread[threadIdx.y + i * ER_COLUMNS_BLOCKDIM_Y] = (baseY + i * ER_COLUMNS_BLOCKDIM_Y >= 0) ? d_src[i * ER_COLUMNS_BLOCKDIM_Y * w] : 0;
    }

    //Lower halo
    #pragma unroll
    for (int i = ER_COLUMNS_HALO_STEPS + ER_COLUMNS_RESULT_STEPS; i < ER_COLUMNS_HALO_STEPS + ER_COLUMNS_RESULT_STEPS + ER_COLUMNS_HALO_STEPS; i++) {
      smem_thread[threadIdx.y + i * ER_COLUMNS_BLOCKDIM_Y]= (baseY + i * ER_COLUMNS_BLOCKDIM_Y < h) ? d_src[i * ER_COLUMNS_BLOCKDIM_Y * w] : 0;
    }

    //Compute and store results
    __syncthreads();
    #pragma unroll
    for (int i = ER_COLUMNS_HALO_STEPS; i < ER_COLUMNS_HALO_STEPS + ER_COLUMNS_RESULT_STEPS; i++) {
      unsigned short *smem_kern = &smem_thread[threadIdx.y + i * ER_COLUMNS_BLOCKDIM_Y - kernel_radius];
      unsigned short val = smem_kern[0];

      //#pragma unroll
      for (int j = 1; j <= 2 * kernel_radius; j++) {
        val = min(val, smem_kern[j]);
      }
      d_dst[i * ER_COLUMNS_BLOCKDIM_Y * w] = val;
    }
}

void erosionColumns3D
(
 unsigned short *d_dst, unsigned short *d_src,
 int w,int h,int d, int kernel_radius
)
{
  assert(ER_COLUMNS_BLOCKDIM_Y * ER_COLUMNS_HALO_STEPS >= kernel_radius);
  assert(w % ER_COLUMNS_BLOCKDIM_X == 0);
  assert(h % (ER_COLUMNS_RESULT_STEPS * ER_COLUMNS_BLOCKDIM_Y) == 0);
  assert(d % ER_COLUMNS_BLOCKDIM_Z == 0);

  dim3 blocks(w / ER_COLUMNS_BLOCKDIM_X, h / (ER_COLUMNS_RESULT_STEPS * ER_COLUMNS_BLOCKDIM_Y), d / ER_COLUMNS_BLOCKDIM_Z);
  dim3 threads(ER_COLUMNS_BLOCKDIM_X, ER_COLUMNS_BLOCKDIM_Y, ER_COLUMNS_BLOCKDIM_Z);

  erosionColumns3DKernel<<<blocks, threads>>>
    (
     d_dst,d_src,w,h,d,kernel_radius
    );
  getLastCudaError("erosionColumns3DKernel() execution failed\n");
}

/*
 * Layer erosion filter
*/
// 32 = 2^5  = divisible by 8*4
#define   ER_LAYERS_BLOCKDIM_X 8
#define   ER_LAYERS_BLOCKDIM_Y 8
#define ER_LAYERS_BLOCKDIM_Z 4
#define ER_LAYERS_RESULT_STEPS 2
#define   ER_LAYERS_HALO_STEPS 2

__global__ void erosionLayers3DKernel(
    unsigned short *d_dst, unsigned short *d_src,
    int w, int h, int d,
    int kernel_radius
)
{
    __shared__ unsigned short smem[ER_LAYERS_BLOCKDIM_X][ER_LAYERS_BLOCKDIM_Y][(ER_LAYERS_RESULT_STEPS + 2 * ER_LAYERS_HALO_STEPS) * ER_LAYERS_BLOCKDIM_Z + 1];
    unsigned short *smem_thread = smem[threadIdx.x][threadIdx.y];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * ER_LAYERS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ER_LAYERS_BLOCKDIM_Y + threadIdx.y;
    const int baseZ = (blockIdx.z * ER_LAYERS_RESULT_STEPS - ER_LAYERS_HALO_STEPS) * ER_LAYERS_BLOCKDIM_Z + threadIdx.z;
    d_src += (baseZ * h + baseY) * w + baseX;
    d_dst += (baseZ * h + baseY) * w + baseX;

    const int pitch = w*h;

    //Main data
    #pragma unroll
    for (int i = ER_LAYERS_HALO_STEPS; i < ER_LAYERS_HALO_STEPS + ER_LAYERS_RESULT_STEPS; i++) {
      smem_thread[threadIdx.z + i * ER_LAYERS_BLOCKDIM_Z] = d_src[i * ER_LAYERS_BLOCKDIM_Z * pitch];
    }

    //Upper halo
    #pragma unroll
    for (int i = 0; i < ER_LAYERS_HALO_STEPS; i++) {
      smem_thread[threadIdx.z + i * ER_LAYERS_BLOCKDIM_Z] = (baseZ + i * ER_LAYERS_BLOCKDIM_Z >= 0) ? d_src[i * ER_LAYERS_BLOCKDIM_Z * pitch] : 0;
    }

    //Lower halo
    #pragma unroll
    for (int i = ER_LAYERS_HALO_STEPS + ER_LAYERS_RESULT_STEPS; i < ER_LAYERS_HALO_STEPS + ER_LAYERS_RESULT_STEPS + ER_LAYERS_HALO_STEPS; i++) {
      smem_thread[threadIdx.z + i * ER_LAYERS_BLOCKDIM_Z]= (baseZ + i * ER_LAYERS_BLOCKDIM_Z < d) ? d_src[i * ER_LAYERS_BLOCKDIM_Z * pitch] : 0;
    }

    //Compute and store results
    __syncthreads();
    #pragma unroll
    for (int i = ER_LAYERS_HALO_STEPS; i < ER_LAYERS_HALO_STEPS + ER_LAYERS_RESULT_STEPS; i++) {
      unsigned short *smem_kern = &smem_thread[threadIdx.z + i * ER_LAYERS_BLOCKDIM_Z - kernel_radius];
      unsigned short val = smem_kern[0];

      //#pragma unroll
      for (int j = 1; j <= 2*kernel_radius; j++) {
        val = min(val, smem_kern[j]);
      }
      d_dst[i * ER_LAYERS_BLOCKDIM_Z * pitch] = val;
    }
}

 void erosionLayers3D(
    unsigned short *d_dst, unsigned short *d_src,
    int w, int h, int d,
    int kernel_radius
)
{
    assert(ER_LAYERS_BLOCKDIM_Z * ER_LAYERS_HALO_STEPS >= kernel_radius);
    assert(w % ER_LAYERS_BLOCKDIM_X == 0);
    assert(h % ER_LAYERS_BLOCKDIM_Y == 0);
    assert(d % (ER_LAYERS_RESULT_STEPS * ER_LAYERS_BLOCKDIM_Z) == 0);

    dim3 blocks(w / ER_LAYERS_BLOCKDIM_X, h / ER_LAYERS_BLOCKDIM_Y, d / (ER_LAYERS_RESULT_STEPS * ER_LAYERS_BLOCKDIM_Z));
    dim3 threads(ER_LAYERS_BLOCKDIM_X, ER_LAYERS_BLOCKDIM_Y, ER_LAYERS_BLOCKDIM_Z);

    erosionLayers3DKernel<<<blocks, threads>>>
      (
       d_dst, d_src, w,h,d,kernel_radius
       );
    getLastCudaError("erosionLayers3DKernel() execution failed\n");
}



void Erosion3DFilter(unsigned short *d_img, unsigned short *d_temp, unsigned short *d_result, const int width, const int height, const int depth, int radius_xy, int radius_z) {

  erosionRows3D(d_result, d_img, width, height, depth, radius_xy);
  erosionColumns3D(d_temp, d_result, width, height, depth, radius_xy);
  erosionLayers3D(d_result, d_temp, width, height, depth, radius_z);

  getLastCudaError("Error: erosionRows3D() or erosionColumns3D() or erosionLayers3D() kernel execution FAILED!");
  //checkCudaErrors(cudaDeviceSynchronize());
}

