/*
 * Hessian3DFilter_element.cu
 *   save Hessian matrix element
 */

#include <assert.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "Hessian3DFilter_element.cuh"

/*
 * Row gradient filter
 */
// 2560 = 5 * 2^9  = divisible by 8*16
#define   ROWS_GRAD_BLOCKDIM_X 16
#define   ROWS_GRAD_BLOCKDIM_Y 4
#define ROWS_GRAD_BLOCKDIM_Z 4
#define ROWS_GRAD_RESULT_STEPS 8
#define   ROWS_GRAD_HALO_STEPS 1

__global__ void gradientRowsKernel(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD
)
{
    __shared__ float s_Data[ROWS_GRAD_BLOCKDIM_Z][ROWS_GRAD_BLOCKDIM_Y][(ROWS_GRAD_RESULT_STEPS + 2 * ROWS_GRAD_HALO_STEPS) * ROWS_GRAD_BLOCKDIM_X];

    //Offset to the left halo edge
    const int baseX = (blockIdx.x * ROWS_GRAD_RESULT_STEPS - ROWS_GRAD_HALO_STEPS) * ROWS_GRAD_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_GRAD_BLOCKDIM_Y + threadIdx.y;
    const int baseZ = blockIdx.z * ROWS_GRAD_BLOCKDIM_Z + threadIdx.z;

    d_Src += (baseZ * imageH + baseY) * imageW + baseX;
    d_Dst += (baseZ * imageH + baseY) * imageW + baseX;

    //Load main data
    #pragma unroll

    for (int i = ROWS_GRAD_HALO_STEPS; i < ROWS_GRAD_HALO_STEPS + ROWS_GRAD_RESULT_STEPS; i++) {
      s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_GRAD_BLOCKDIM_X] = d_Src[i * ROWS_GRAD_BLOCKDIM_X];
    }

    //Load left halo
    #pragma unroll

    for (int i = 0; i < ROWS_GRAD_HALO_STEPS; i++) {
      s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_GRAD_BLOCKDIM_X] = (baseX + i * ROWS_GRAD_BLOCKDIM_X >= 0) ? d_Src[i * ROWS_GRAD_BLOCKDIM_X] : 0;
    }

    //Load right halo
    #pragma unroll

    for (int i = ROWS_GRAD_HALO_STEPS + ROWS_GRAD_RESULT_STEPS; i < ROWS_GRAD_HALO_STEPS + ROWS_GRAD_RESULT_STEPS + ROWS_GRAD_HALO_STEPS; i++) {
      s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_GRAD_BLOCKDIM_X] = (baseX + i * ROWS_GRAD_BLOCKDIM_X < imageW) ? d_Src[i * ROWS_GRAD_BLOCKDIM_X] : 0;
    }

    //Compute and store results
    __syncthreads();
    #pragma unroll

    for (int i = ROWS_GRAD_HALO_STEPS; i < ROWS_GRAD_HALO_STEPS + ROWS_GRAD_RESULT_STEPS; i++)
    {
        float sum = 0;
        sum += s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_GRAD_BLOCKDIM_X + 1];
        sum -= s_Data[threadIdx.z][threadIdx.y][threadIdx.x + i * ROWS_GRAD_BLOCKDIM_X - 1];
        sum *= 0.5f;

        d_Dst[i * ROWS_GRAD_BLOCKDIM_X] = sum;
    }
}

extern "C" void gradientRows(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD
)
{
    assert(imageW % (ROWS_GRAD_RESULT_STEPS * ROWS_GRAD_BLOCKDIM_X) == 0);
    assert(imageH % ROWS_GRAD_BLOCKDIM_Y == 0);
    assert(imageD % ROWS_GRAD_BLOCKDIM_Z == 0);

    dim3 blocks(imageW / (ROWS_GRAD_RESULT_STEPS * ROWS_GRAD_BLOCKDIM_X),
                imageH / ROWS_GRAD_BLOCKDIM_Y,
                imageD / ROWS_GRAD_BLOCKDIM_Z);
    dim3 threads(ROWS_GRAD_BLOCKDIM_X, ROWS_GRAD_BLOCKDIM_Y, ROWS_GRAD_BLOCKDIM_Z);

    gradientRowsKernel<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageD
    );
    //checkCudaErrors(cudaPeekAtLastError());
    //checkCudaErrors(cudaDeviceSynchronize());
    getLastCudaError("gradientRowsKernel() execution failed\n");
}

/*
 * Column gradient filter
*/
// 2160 = 3^3 * 5 * 2^4  = divisible by 9*16
#define   COLUMNS_GRAD_BLOCKDIM_X 4
#define   COLUMNS_GRAD_BLOCKDIM_Y 16
#define COLUMNS_GRAD_BLOCKDIM_Z 4
#define COLUMNS_GRAD_RESULT_STEPS 9
#define   COLUMNS_GRAD_HALO_STEPS 1

__global__ void gradientColumnsKernel(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD
)
{
    __shared__ float s_Data[COLUMNS_GRAD_BLOCKDIM_Z][COLUMNS_GRAD_BLOCKDIM_X][(COLUMNS_GRAD_RESULT_STEPS + 2 * COLUMNS_GRAD_HALO_STEPS) * COLUMNS_GRAD_BLOCKDIM_Y + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_GRAD_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_GRAD_RESULT_STEPS - COLUMNS_GRAD_HALO_STEPS) * COLUMNS_GRAD_BLOCKDIM_Y + threadIdx.y;
    const int baseZ = blockIdx.z * COLUMNS_GRAD_BLOCKDIM_Z + threadIdx.z;
    d_Src += (baseZ * imageH + baseY) * imageW + baseX;
    d_Dst += (baseZ * imageH + baseY) * imageW + baseX;

    //Main data
    #pragma unroll

    for (int i = COLUMNS_GRAD_HALO_STEPS; i < COLUMNS_GRAD_HALO_STEPS + COLUMNS_GRAD_RESULT_STEPS; i++) {
      s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_GRAD_BLOCKDIM_Y] = d_Src[i * COLUMNS_GRAD_BLOCKDIM_Y * imageW];
    }

    //Upper halo
    #pragma unroll

    for (int i = 0; i < COLUMNS_GRAD_HALO_STEPS; i++) {
      s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_GRAD_BLOCKDIM_Y] = (baseY + i * COLUMNS_GRAD_BLOCKDIM_Y >= 0) ? d_Src[i * COLUMNS_GRAD_BLOCKDIM_Y * imageW] : 0;
    }

    //Lower halo
    #pragma unroll

    for (int i = COLUMNS_GRAD_HALO_STEPS + COLUMNS_GRAD_RESULT_STEPS; i < COLUMNS_GRAD_HALO_STEPS + COLUMNS_GRAD_RESULT_STEPS + COLUMNS_GRAD_HALO_STEPS; i++) {
      s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_GRAD_BLOCKDIM_Y]= (baseY + i * COLUMNS_GRAD_BLOCKDIM_Y < imageH) ? d_Src[i * COLUMNS_GRAD_BLOCKDIM_Y * imageW] : 0;
    }

    //Compute and store results
    __syncthreads();
    #pragma unroll

    for (int i = COLUMNS_GRAD_HALO_STEPS; i < COLUMNS_GRAD_HALO_STEPS + COLUMNS_GRAD_RESULT_STEPS; i++) {
        float sum = 0;
        sum += s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_GRAD_BLOCKDIM_Y + 1];
        sum -= s_Data[threadIdx.z][threadIdx.x][threadIdx.y + i * COLUMNS_GRAD_BLOCKDIM_Y - 1];
        sum *= 0.5f;

        d_Dst[i * COLUMNS_GRAD_BLOCKDIM_Y * imageW] = sum;
    }
}

extern "C" void gradientColumns(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD
)
{
    assert(imageW % COLUMNS_GRAD_BLOCKDIM_X == 0);
    assert(imageH % (COLUMNS_GRAD_RESULT_STEPS * COLUMNS_GRAD_BLOCKDIM_Y) == 0);
    assert(imageD % COLUMNS_GRAD_BLOCKDIM_Z == 0);

    dim3 blocks(imageW / COLUMNS_GRAD_BLOCKDIM_X,
                imageH / (COLUMNS_GRAD_RESULT_STEPS * COLUMNS_GRAD_BLOCKDIM_Y),
                imageD / COLUMNS_GRAD_BLOCKDIM_Z);
    dim3 threads(COLUMNS_GRAD_BLOCKDIM_X, COLUMNS_GRAD_BLOCKDIM_Y, COLUMNS_GRAD_BLOCKDIM_Z);

    gradientColumnsKernel<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageD
    );
    getLastCudaError("gradientColumnsKernel() execution failed\n");
    //checkCudaErrors(cudaPeekAtLastError());
    //checkCudaErrors(cudaDeviceSynchronize());
}


/*
 * Layer gradient filter
*/
// 32 = 2^5  = divisible by 8*4
#define   LAYERS_GRAD_BLOCKDIM_X 8
#define   LAYERS_GRAD_BLOCKDIM_Y 8
#define LAYERS_GRAD_BLOCKDIM_Z 4
#define LAYERS_GRAD_RESULT_STEPS 2
#define   LAYERS_GRAD_HALO_STEPS 2

__global__ void gradientLayersKernel(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD
)
{
    __shared__ float s_Data[LAYERS_GRAD_BLOCKDIM_X][LAYERS_GRAD_BLOCKDIM_Y][(LAYERS_GRAD_RESULT_STEPS + 2 * LAYERS_GRAD_HALO_STEPS) * LAYERS_GRAD_BLOCKDIM_Z + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * LAYERS_GRAD_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * LAYERS_GRAD_BLOCKDIM_Y + threadIdx.y;
    const int baseZ = (blockIdx.z * LAYERS_GRAD_RESULT_STEPS - LAYERS_GRAD_HALO_STEPS) * LAYERS_GRAD_BLOCKDIM_Z + threadIdx.z;
    d_Src += (baseZ * imageH + baseY) * imageW + baseX;
    d_Dst += (baseZ * imageH + baseY) * imageW + baseX;

    const int pitch = imageW*imageH;

    //Main data
    #pragma unroll

    for (int i = LAYERS_GRAD_HALO_STEPS; i < LAYERS_GRAD_HALO_STEPS + LAYERS_GRAD_RESULT_STEPS; i++) {
      s_Data[threadIdx.x][threadIdx.y][threadIdx.z + i * LAYERS_GRAD_BLOCKDIM_Z] = d_Src[i * LAYERS_GRAD_BLOCKDIM_Z * pitch];
    }

    //Upper halo
    #pragma unroll

    for (int i = 0; i < LAYERS_GRAD_HALO_STEPS; i++) {
      s_Data[threadIdx.x][threadIdx.y][threadIdx.z + i * LAYERS_GRAD_BLOCKDIM_Z] = (baseZ + i * LAYERS_GRAD_BLOCKDIM_Z >= 0) ? d_Src[i * LAYERS_GRAD_BLOCKDIM_Z * pitch] : 0;
    }

    //Lower halo
    #pragma unroll

    for (int i = LAYERS_GRAD_HALO_STEPS + LAYERS_GRAD_RESULT_STEPS; i < LAYERS_GRAD_HALO_STEPS + LAYERS_GRAD_RESULT_STEPS + LAYERS_GRAD_HALO_STEPS; i++) {
      s_Data[threadIdx.x][threadIdx.y][threadIdx.z + i * LAYERS_GRAD_BLOCKDIM_Z]= (baseZ + i * LAYERS_GRAD_BLOCKDIM_Z < imageD) ? d_Src[i * LAYERS_GRAD_BLOCKDIM_Z * pitch] : 0;
    }

    //Compute and store results
    __syncthreads();
    #pragma unroll

    for (int i = LAYERS_GRAD_HALO_STEPS; i < LAYERS_GRAD_HALO_STEPS + LAYERS_GRAD_RESULT_STEPS; i++) {
        float sum = 0;
        sum += s_Data[threadIdx.x][threadIdx.y][threadIdx.z + i * LAYERS_GRAD_BLOCKDIM_Z + 1];
        sum -= s_Data[threadIdx.x][threadIdx.y][threadIdx.z + i * LAYERS_GRAD_BLOCKDIM_Z - 1];
        sum *= 0.5f;

        d_Dst[i * LAYERS_GRAD_BLOCKDIM_Z * pitch] = sum;
    }
}

extern "C" void gradientLayers(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD
)
{
    assert(imageW % LAYERS_GRAD_BLOCKDIM_X == 0);
    assert(imageH % LAYERS_GRAD_BLOCKDIM_Y == 0);
    assert(imageD % (LAYERS_GRAD_RESULT_STEPS * LAYERS_GRAD_BLOCKDIM_Z) == 0);

    dim3 blocks(imageW / LAYERS_GRAD_BLOCKDIM_X,
                imageH / LAYERS_GRAD_BLOCKDIM_Y,
                imageD / (LAYERS_GRAD_RESULT_STEPS * LAYERS_GRAD_BLOCKDIM_Z));
    dim3 threads(LAYERS_GRAD_BLOCKDIM_X, LAYERS_GRAD_BLOCKDIM_Y, LAYERS_GRAD_BLOCKDIM_Z);

    gradientLayersKernel<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageD
    );
    getLastCudaError("gradientLayersKernel() execution failed\n");
    //checkCudaErrors(cudaPeekAtLastError());
    //checkCudaErrors(cudaDeviceSynchronize());
}


#define   PD_BLOCKDIM_X 8
#define   PD_BLOCKDIM_Y 8
#define   PD_BLOCKDIM_Z 8

__global__ void PositiveDefiniteKernel(
    char *hessian_pd,
    float *hessian,
    int imageW,
    int imageH,
    int imageD
)
{
  const int baseX = blockIdx.x * PD_BLOCKDIM_X + threadIdx.x;
  const int baseY = blockIdx.y * PD_BLOCKDIM_Y + threadIdx.y;
  const int baseZ = blockIdx.z * PD_BLOCKDIM_Z + threadIdx.z;
  const int size = imageW * imageH * imageD;
  const int idx = (baseZ * imageH + baseY) * imageW + baseX;

  float xx = hessian[idx];
  float xy = hessian[idx + size];
  float xz = hessian[idx + size*2];
  float yy = hessian[idx + size*3];
  float yz = hessian[idx + size*4];
  float zz = hessian[idx + size*5];

  // Sylvester's criterion
  hessian_pd[idx] = (
    xx < 0 &&
    xx*yy-xy*xy > 0 &&
    xx*yy*zz + 2*xy*yz*xz - xx*yz*yz - yy*xz*xz - zz*xy*xy < 0
  ) ? 1 : 0;

}

void PositiveDefinite(
    float *d_hessian,
    char *d_hessian_pd,
    int imageW,
    int imageH,
    int imageD
)
{
    assert(imageW % PD_BLOCKDIM_X == 0);
    assert(imageH % PD_BLOCKDIM_Y == 0);
    assert(imageD % PD_BLOCKDIM_Z == 0);

    dim3 blocks(imageW / PD_BLOCKDIM_X,
                imageH / PD_BLOCKDIM_Y,
                imageD / PD_BLOCKDIM_Z);
    dim3 threads(PD_BLOCKDIM_X, PD_BLOCKDIM_Y, PD_BLOCKDIM_Z);

    PositiveDefiniteKernel <<<blocks, threads>>>(
        d_hessian_pd,
        d_hessian,
        imageW,
        imageH,
        imageD
    );
    getLastCudaError("HessianPositiveDefiniteKernel() execution failed\n");
    //checkCudaErrors(cudaPeekAtLastError());
    //checkCudaErrors(cudaDeviceSynchronize());

}

void HessianPositiveDefiniteWithElement
(
 float *d_hessian, char *d_hessian_pd, float *d_src, float *d_temp,
 int w, int h, int d
 ) {
  const int size = w*h*d;

  float *Hxx = d_hessian;
  float *Hxy = &Hxx[size];
  float *Hxz = &Hxy[size];
  float *Hyy = &Hxz[size];
  float *Hyz = &Hyy[size];
  float *Hzz = &Hyz[size];
  // recycle as temporary storage
  float *Hx = d_temp;
  float *Hy = d_temp;
  float *Hz = d_temp;

  gradientRows(Hx, d_src, w,h,d);
  gradientRows(Hxx, Hx, w,h,d);
  gradientColumns(Hxy, Hx, w,h,d);
  gradientLayers(Hxz, Hx, w,h,d);

  gradientColumns(Hy, d_src, w,h,d);
  gradientColumns(Hyy, Hy, w,h,d);
  gradientLayers(Hyz, Hy, w,h,d);

  gradientLayers(Hz, d_src, w,h,d);
  gradientLayers(Hzz, Hz, w,h,d);

  PositiveDefinite(d_hessian, d_hessian_pd, w,h,d);

  checkCudaErrors(cudaDeviceSynchronize());
}
