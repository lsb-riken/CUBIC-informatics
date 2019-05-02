/*
 * NormalizeFilter.cu
 *
 */

#include <assert.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "NormalizeFilter.cuh"


/* --- for large size --- */
// width = 2560 = 5 * 2^9  = divisible by 5*128 = 5 * 2^7
// radius = 200 <= 128 * 2 = 256
// smem_size(minmax) = 2*128*(5+2*2)*2*2 = 9216 < 49152
// smem_size(uniform) = 4*128*(5+2*2)*2*2 = 18432 < 49152
constexpr int L_ROW_BLOCKDIM_X = 64;
constexpr int L_ROW_BLOCKDIM_Y = 2;
constexpr int L_ROW_BLOCKDIM_Z = 2;
constexpr int L_ROW_RESULT_STEPS = 5;
constexpr int L_ROW_HALO_STEPS = 4;

// height = 2160 = 3^3 * 5 * 2^4  = divisible by 144*5 = 3^2 * 5 * 2^4
// radius = 200 <= 144 * 2 = 288
// smem_size(minmax) = 2*144*(5+2*2)*2*2 = 10368 < 49152
// smem_size(uniform) = 4*144*(5+2*2)*2*2 = 20736 < 49152
constexpr int L_COL_BLOCKDIM_X = 2;
constexpr int L_COL_BLOCKDIM_Y = 72;
constexpr int L_COL_BLOCKDIM_Z = 2;
constexpr int L_COL_RESULT_STEPS = 5;
constexpr int L_COL_HALO_STEPS = 4;

// depth = 32 = 2^5  = divisible by 16*2 = 2^5
// radius = 30 < 16 * 2 = 32
// smem_size(minmax) = 2*16*(2+2*4)*8*8 = 20480 < 49152
// smem_size(uniform) = 4*16*(2+2*4)*8*8 = 40960 < 49152
constexpr int L_LAY_BLOCKDIM_X = 8;
constexpr int L_LAY_BLOCKDIM_Y = 8;
constexpr int L_LAY_BLOCKDIM_Z = 16;
constexpr int L_LAY_RESULT_STEPS = 2;
constexpr int L_LAY_HALO_STEPS = 2;


constexpr int NORM_BLOCKDIM_X = 8;
constexpr int NORM_BLOCKDIM_Y = 8;
constexpr int NORM_BLOCKDIM_Z = 8;


/*
 * Uniform Filter
*/
template<int BLOCKDIM_X, int BLOCKDIM_Y, int BLOCKDIM_Z,
  int RESULT_STEPS, int HALO_STEPS>
__global__ void uniformRows3DKernel
(
 float *d_dst, unsigned short *d_src,
 int w, int h, int d,
 int kernel_radius
)
{
    __shared__ float smem[BLOCKDIM_Z][BLOCKDIM_Y][(RESULT_STEPS + 2 * HALO_STEPS) * BLOCKDIM_X];
    float *smem_thread = smem[threadIdx.z][threadIdx.y];

    //Offset to the left halo edge
    const int baseX = (blockIdx.x * RESULT_STEPS - HALO_STEPS) * BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * BLOCKDIM_Y + threadIdx.y;
    const int baseZ = blockIdx.z * BLOCKDIM_Z + threadIdx.z;
    const float uniform_kernel = 1.0f / (2 * kernel_radius + 1);

    d_src += (baseZ * h + baseY) * w + baseX;
    d_dst += (baseZ * h + baseY) * w + baseX;

    //Load main data
    #pragma unroll
    for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++) {
      smem_thread[threadIdx.x + i * BLOCKDIM_X] = (float)d_src[i * BLOCKDIM_X];
    }

    //Load left halo (nearest repeat)
    #pragma unroll
    for (int i = 0; i < HALO_STEPS; i++) {
      smem_thread[threadIdx.x + i * BLOCKDIM_X] = (baseX + i * BLOCKDIM_X >= 0) ? (float)d_src[i * BLOCKDIM_X] : (float)d_src[-baseX];
    }

    //Load right halo (nearest repeat)
    #pragma unroll
    for (int i = HALO_STEPS + RESULT_STEPS; i < HALO_STEPS + RESULT_STEPS + HALO_STEPS; i++) {
      smem_thread[threadIdx.x + i * BLOCKDIM_X] = (baseX + i * BLOCKDIM_X < w) ? (float)d_src[i * BLOCKDIM_X] : (float)d_src[w-1 - baseX];
    }

    //Compute and store results
    __syncthreads();
    #pragma unroll
    for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++) {
      float *smem_kern = &smem_thread[threadIdx.x + i * BLOCKDIM_X - kernel_radius];
      float val = 0;

      //#pragma unroll
      for (int j = 0; j <= 2*kernel_radius; j++) {
        val += smem_kern[j];
      }
      d_dst[i * BLOCKDIM_X] = val * uniform_kernel;
    }
}

template<int BLOCKDIM_X, int BLOCKDIM_Y, int BLOCKDIM_Z,
  int RESULT_STEPS, int HALO_STEPS>
void uniformRows3D
(
 float *d_dst, unsigned short *d_src,
 int w, int h, int d,
 int kernel_radius
 )
{
  assert(BLOCKDIM_X * HALO_STEPS >= kernel_radius);
  assert(w % (RESULT_STEPS * BLOCKDIM_X) == 0);
  assert(h % BLOCKDIM_Y == 0);
  assert(d % BLOCKDIM_Z == 0);

  dim3 blocks(w / (RESULT_STEPS * BLOCKDIM_X), h / BLOCKDIM_Y, d / BLOCKDIM_Z);
  dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);

  uniformRows3DKernel<BLOCKDIM_X,BLOCKDIM_Y,BLOCKDIM_Z,RESULT_STEPS,HALO_STEPS><<<blocks, threads>>>
    (
     d_dst, d_src, w,h,d, kernel_radius
     );
  getLastCudaError("uniformRows3DKernel() execution failed\n");
  //checkCudaErrors(cudaDeviceSynchronize());
}

template<int BLOCKDIM_X, int BLOCKDIM_Y, int BLOCKDIM_Z,
  int RESULT_STEPS, int HALO_STEPS>
__global__ void uniformColumns3DKernel(
    float *d_dst, float *d_src,
    int w,int h,int d,
    int kernel_radius
)
{
    __shared__ float smem[BLOCKDIM_Z][BLOCKDIM_X][(RESULT_STEPS + 2 * HALO_STEPS) * BLOCKDIM_Y + 1];
    float *smem_thread = smem[threadIdx.z][threadIdx.x];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * RESULT_STEPS - HALO_STEPS) * BLOCKDIM_Y + threadIdx.y;
    const int baseZ = blockIdx.z * BLOCKDIM_Z + threadIdx.z;
    const float uniform_kernel = 1.0f / (2 * kernel_radius + 1);

    d_src += (baseZ * h + baseY) * w + baseX;
    d_dst += (baseZ * h + baseY) * w + baseX;

    //Main data
    #pragma unroll
    for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++) {
      smem_thread[threadIdx.y + i * BLOCKDIM_Y] = d_src[i * BLOCKDIM_Y * w];
    }

    //Upper halo (nearest repeat)
    #pragma unroll
    for (int i = 0; i < HALO_STEPS; i++) {
      smem_thread[threadIdx.y + i * BLOCKDIM_Y] = (baseY + i * BLOCKDIM_Y >= 0) ? d_src[i * BLOCKDIM_Y * w] : d_src[-baseY*w];
    }

    //Lower halo (nearest repeat)
    #pragma unroll
    for (int i = HALO_STEPS + RESULT_STEPS; i < HALO_STEPS + RESULT_STEPS + HALO_STEPS; i++) {
      smem_thread[threadIdx.y + i * BLOCKDIM_Y]= (baseY + i * BLOCKDIM_Y < h) ? d_src[i * BLOCKDIM_Y * w] : d_src[(h-1 - baseY)*w];
    }

    //Compute and store results
    __syncthreads();
    #pragma unroll
    for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++) {
      float *smem_kern = &smem_thread[threadIdx.y + i * BLOCKDIM_Y - kernel_radius];
      float val = 0;

      //#pragma unroll
      for (int j = 0; j <= 2 * kernel_radius; j++) {
        val += smem_kern[j];
      }
      d_dst[i * BLOCKDIM_Y * w] = val * uniform_kernel;
    }
}

template<int BLOCKDIM_X, int BLOCKDIM_Y, int BLOCKDIM_Z,
  int RESULT_STEPS, int HALO_STEPS>
void uniformColumns3D
(
 float *d_dst, float *d_src,
 int w,int h,int d, int kernel_radius
)
{
  assert(BLOCKDIM_Y * HALO_STEPS >= kernel_radius);
  assert(w % BLOCKDIM_X == 0);
  assert(h % (RESULT_STEPS * BLOCKDIM_Y) == 0);
  assert(d % BLOCKDIM_Z == 0);

  dim3 blocks(w / BLOCKDIM_X, h / (RESULT_STEPS * BLOCKDIM_Y), d / BLOCKDIM_Z);
  dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);

  uniformColumns3DKernel<BLOCKDIM_X,BLOCKDIM_Y,BLOCKDIM_Z,RESULT_STEPS,HALO_STEPS><<<blocks, threads>>>
    (
     d_dst,d_src,w,h,d,kernel_radius
    );
  getLastCudaError("uniformColumns3DKernel() execution failed\n");
  //checkCudaErrors(cudaDeviceSynchronize());
}

template<int BLOCKDIM_X, int BLOCKDIM_Y, int BLOCKDIM_Z,
  int RESULT_STEPS, int HALO_STEPS>
__global__ void uniformLayers3DKernel(
    float *d_dst, float *d_src,
    int w, int h, int d,
    int kernel_radius
)
{
    __shared__ float smem[BLOCKDIM_X][BLOCKDIM_Y][(RESULT_STEPS + 2 * HALO_STEPS) * BLOCKDIM_Z + 1];
    float *smem_thread = smem[threadIdx.x][threadIdx.y];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * BLOCKDIM_Y + threadIdx.y;
    const int baseZ = (blockIdx.z * RESULT_STEPS - HALO_STEPS) * BLOCKDIM_Z + threadIdx.z;
    const float uniform_kernel = 1.0f / (2 * kernel_radius + 1);

    d_src += (baseZ * h + baseY) * w + baseX;
    d_dst += (baseZ * h + baseY) * w + baseX;

    const int pitch = w*h;

    //Main data
    #pragma unroll
    for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++) {
      smem_thread[threadIdx.z + i * BLOCKDIM_Z] = d_src[i * BLOCKDIM_Z * pitch];
    }

    //Upper halo (nearest repeat)
    #pragma unroll
    for (int i = 0; i < HALO_STEPS; i++) {
      smem_thread[threadIdx.z + i * BLOCKDIM_Z] = (baseZ + i * BLOCKDIM_Z >= 0) ? d_src[i * BLOCKDIM_Z * pitch] : d_src[-baseZ*pitch];
    }

    //Lower halo (nearest repeat)
    #pragma unroll
    for (int i = HALO_STEPS + RESULT_STEPS; i < HALO_STEPS + RESULT_STEPS + HALO_STEPS; i++) {
      smem_thread[threadIdx.z + i * BLOCKDIM_Z]= (baseZ + i * BLOCKDIM_Z < d) ? d_src[i * BLOCKDIM_Z * pitch] : d_src[(d-1 - baseZ)*pitch];
    }

    //Compute and store results
    __syncthreads();
    #pragma unroll
    for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++) {
      float *smem_kern = &smem_thread[threadIdx.z + i * BLOCKDIM_Z - kernel_radius];
      float val = 0;

      //#pragma unroll
      for (int j = 0; j <= 2*kernel_radius; j++) {
        val += smem_kern[j];
      }
      d_dst[i * BLOCKDIM_Z * pitch] = val * uniform_kernel;
    }
}

template<int BLOCKDIM_X, int BLOCKDIM_Y, int BLOCKDIM_Z,
  int RESULT_STEPS, int HALO_STEPS>
void uniformLayers3D(
    float *d_dst, float *d_src,
    int w, int h, int d,
    int kernel_radius
)
{
    assert(BLOCKDIM_Z * HALO_STEPS >= kernel_radius);
    assert(w % BLOCKDIM_X == 0);
    assert(h % BLOCKDIM_Y == 0);
    assert(d % (RESULT_STEPS * BLOCKDIM_Z) == 0);

    dim3 blocks(w / BLOCKDIM_X, h / BLOCKDIM_Y, d / (RESULT_STEPS * BLOCKDIM_Z));
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);

    uniformLayers3DKernel<BLOCKDIM_X,BLOCKDIM_Y,BLOCKDIM_Z,RESULT_STEPS,HALO_STEPS><<<blocks, threads>>>
      (
       d_dst, d_src, w,h,d,kernel_radius
       );
    getLastCudaError("uniformLayers3DKernel() execution failed\n");
    //checkCudaErrors(cudaDeviceSynchronize());
}

/*
* MinMax(Erosion or Dilation) Filter
*/
template<int BLOCKDIM_X, int BLOCKDIM_Y, int BLOCKDIM_Z,
           int RESULT_STEPS, int HALO_STEPS, bool is_min>
__global__ void minmaxRows3DKernel
(
 unsigned short *d_dst, unsigned short *d_src,
 int w, int h, int d,
 int kernel_radius
)
{
    __shared__ unsigned short smem[BLOCKDIM_Z][BLOCKDIM_Y][(RESULT_STEPS + 2 * HALO_STEPS) * BLOCKDIM_X];
    unsigned short *smem_thread = smem[threadIdx.z][threadIdx.y];

    //Offset to the left halo edge
    const int baseX = (blockIdx.x * RESULT_STEPS - HALO_STEPS) * BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * BLOCKDIM_Y + threadIdx.y;
    const int baseZ = blockIdx.z * BLOCKDIM_Z + threadIdx.z;

    d_src += (baseZ * h + baseY) * w + baseX;
    d_dst += (baseZ * h + baseY) * w + baseX;

    //Load main data
    #pragma unroll
    for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++) {
      smem_thread[threadIdx.x + i * BLOCKDIM_X] = d_src[i * BLOCKDIM_X];
    }

    //Load left halo (nearest constant border)
    #pragma unroll
    for (int i = 0; i < HALO_STEPS; i++) {
      smem_thread[threadIdx.x + i * BLOCKDIM_X] = (baseX + i * BLOCKDIM_X >= 0) ? d_src[i * BLOCKDIM_X] : d_src[-baseX];
    }

    //Load right halo (nearest constant border)
    #pragma unroll
    for (int i = HALO_STEPS + RESULT_STEPS; i < HALO_STEPS + RESULT_STEPS + HALO_STEPS; i++) {
      smem_thread[threadIdx.x + i * BLOCKDIM_X] = (baseX + i * BLOCKDIM_X < w) ? d_src[i * BLOCKDIM_X] : d_src[w-1-baseX];
    }

    //Compute and store results
    __syncthreads();
    #pragma unroll
    for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++) {
      unsigned short *smem_kern = &smem_thread[threadIdx.x + i * BLOCKDIM_X - kernel_radius];
      unsigned short val = smem_kern[0];

      //#pragma unroll
      for (int j = 1; j <= 2*kernel_radius; j++) {
        if(is_min)
          val = min(val, smem_kern[j]);
        else
          val = max(val, smem_kern[j]);
      }
      d_dst[i * BLOCKDIM_X] = val;
    }
}

template<int BLOCKDIM_X, int BLOCKDIM_Y, int BLOCKDIM_Z,
           int RESULT_STEPS, int HALO_STEPS, bool is_min>
void minmaxRows3D
(
 unsigned short *d_dst, unsigned short *d_src,
 int w, int h, int d,
 int kernel_radius
 )
{
  assert(BLOCKDIM_X * HALO_STEPS >= kernel_radius);
  assert(w % (RESULT_STEPS * BLOCKDIM_X) == 0);
  assert(h % BLOCKDIM_Y == 0);
  assert(d % BLOCKDIM_Z == 0);

  dim3 blocks(w / (RESULT_STEPS * BLOCKDIM_X), h / BLOCKDIM_Y, d / BLOCKDIM_Z);
  dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);

  minmaxRows3DKernel<BLOCKDIM_X,BLOCKDIM_Y,BLOCKDIM_Z,RESULT_STEPS,HALO_STEPS, is_min><<<blocks, threads>>>
    (
     d_dst, d_src, w,h,d, kernel_radius
     );
  getLastCudaError("minmaxRows3DKernel() execution failed\n");
  //checkCudaErrors(cudaDeviceSynchronize());
}

template<int BLOCKDIM_X, int BLOCKDIM_Y, int BLOCKDIM_Z,
           int RESULT_STEPS, int HALO_STEPS, bool is_min>
__global__ void minmaxColumns3DKernel(
    unsigned short *d_dst, unsigned short *d_src,
    int w,int h,int d,
    int kernel_radius
)
{
    __shared__ unsigned short smem[BLOCKDIM_Z][BLOCKDIM_X][(RESULT_STEPS + 2 * HALO_STEPS) * BLOCKDIM_Y + 1];
    unsigned short *smem_thread = smem[threadIdx.z][threadIdx.x];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * RESULT_STEPS - HALO_STEPS) * BLOCKDIM_Y + threadIdx.y;
    const int baseZ = blockIdx.z * BLOCKDIM_Z + threadIdx.z;

    d_src += (baseZ * h + baseY) * w + baseX;
    d_dst += (baseZ * h + baseY) * w + baseX;

    //Main data
    #pragma unroll
    for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++) {
      smem_thread[threadIdx.y + i * BLOCKDIM_Y] = d_src[i * BLOCKDIM_Y * w];
    }

    //Upper halo (nearest constant border)
    #pragma unroll
    for (int i = 0; i < HALO_STEPS; i++) {
      smem_thread[threadIdx.y + i * BLOCKDIM_Y] = (baseY + i * BLOCKDIM_Y >= 0) ? d_src[i * BLOCKDIM_Y * w] : d_src[-baseY*w];
    }

    //Lower halo (nearest constant border)
    #pragma unroll
    for (int i = HALO_STEPS + RESULT_STEPS; i < HALO_STEPS + RESULT_STEPS + HALO_STEPS; i++) {
      smem_thread[threadIdx.y + i * BLOCKDIM_Y]= (baseY + i * BLOCKDIM_Y < h) ? d_src[i * BLOCKDIM_Y * w] : d_src[(h-1-baseY)*w];
    }

    //Compute and store results
    __syncthreads();
    #pragma unroll
    for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++) {
      unsigned short *smem_kern = &smem_thread[threadIdx.y + i * BLOCKDIM_Y - kernel_radius];
      unsigned short val = smem_kern[0];

      //#pragma unroll
      for (int j = 1; j <= 2 * kernel_radius; j++) {
        if(is_min)
          val = min(val, smem_kern[j]);
        else
          val = max(val, smem_kern[j]);
      }
      d_dst[i * BLOCKDIM_Y * w] = val;
    }
}

template<int BLOCKDIM_X, int BLOCKDIM_Y, int BLOCKDIM_Z,
           int RESULT_STEPS, int HALO_STEPS, bool is_min>
void minmaxColumns3D
(
 unsigned short *d_dst, unsigned short *d_src,
 int w,int h,int d, int kernel_radius
)
{
  assert(BLOCKDIM_Y * HALO_STEPS >= kernel_radius);
  assert(w % BLOCKDIM_X == 0);
  assert(h % (RESULT_STEPS * BLOCKDIM_Y) == 0);
  assert(d % BLOCKDIM_Z == 0);

  dim3 blocks(w / BLOCKDIM_X, h / (RESULT_STEPS * BLOCKDIM_Y), d / BLOCKDIM_Z);
  dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);

  minmaxColumns3DKernel<BLOCKDIM_X,BLOCKDIM_Y,BLOCKDIM_Z,RESULT_STEPS,HALO_STEPS, is_min><<<blocks, threads>>>
    (
     d_dst,d_src,w,h,d,kernel_radius
    );
  getLastCudaError("minmaxColumns3DKernel() execution failed\n");
  //checkCudaErrors(cudaDeviceSynchronize());
}

template<int BLOCKDIM_X, int BLOCKDIM_Y, int BLOCKDIM_Z,
           int RESULT_STEPS, int HALO_STEPS, bool is_min>
__global__ void minmaxLayers3DKernel(
    unsigned short *d_dst, unsigned short *d_src,
    int w, int h, int d,
    int kernel_radius
)
{
    __shared__ unsigned short smem[BLOCKDIM_X][BLOCKDIM_Y][(RESULT_STEPS + 2 * HALO_STEPS) * BLOCKDIM_Z + 1];
    unsigned short *smem_thread = smem[threadIdx.x][threadIdx.y];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * BLOCKDIM_Y + threadIdx.y;
    const int baseZ = (blockIdx.z * RESULT_STEPS - HALO_STEPS) * BLOCKDIM_Z + threadIdx.z;

    d_src += (baseZ * h + baseY) * w + baseX;
    d_dst += (baseZ * h + baseY) * w + baseX;

    const int pitch = w*h;

    //Main data
    #pragma unroll
    for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++) {
      smem_thread[threadIdx.z + i * BLOCKDIM_Z] = d_src[i * BLOCKDIM_Z * pitch];
    }

    //Upper halo (nearest constant border)
    #pragma unroll
    for (int i = 0; i < HALO_STEPS; i++) {
      smem_thread[threadIdx.z + i * BLOCKDIM_Z] = (baseZ + i * BLOCKDIM_Z >= 0) ? d_src[i * BLOCKDIM_Z * pitch] : d_src[-baseZ*w*h];
    }

    //Lower halo (nearest constant border)
    #pragma unroll
    for (int i = HALO_STEPS + RESULT_STEPS; i < HALO_STEPS + RESULT_STEPS + HALO_STEPS; i++) {
      smem_thread[threadIdx.z + i * BLOCKDIM_Z]= (baseZ + i * BLOCKDIM_Z < d) ? d_src[i * BLOCKDIM_Z * pitch] : d_src[(d-1-baseZ)*w*h];
    }

    //Compute and store results
    __syncthreads();
    #pragma unroll
    for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++) {
      unsigned short *smem_kern = &smem_thread[threadIdx.z + i * BLOCKDIM_Z - kernel_radius];
      unsigned short val = smem_kern[0];

      //#pragma unroll
      for (int j = 1; j <= 2*kernel_radius; j++) {
        if(is_min)
          val = min(val, smem_kern[j]);
        else
          val = max(val, smem_kern[j]);
      }
      d_dst[i * BLOCKDIM_Z * pitch] = val;
    }
}

template<int BLOCKDIM_X, int BLOCKDIM_Y, int BLOCKDIM_Z,
           int RESULT_STEPS, int HALO_STEPS, bool is_min>
void minmaxLayers3D(
    unsigned short *d_dst, unsigned short *d_src,
    int w, int h, int d,
    int kernel_radius
)
{
    assert(BLOCKDIM_Z * HALO_STEPS >= kernel_radius);
    assert(w % BLOCKDIM_X == 0);
    assert(h % BLOCKDIM_Y == 0);
    assert(d % (RESULT_STEPS * BLOCKDIM_Z) == 0);

    dim3 blocks(w / BLOCKDIM_X, h / BLOCKDIM_Y, d / (RESULT_STEPS * BLOCKDIM_Z));
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);

    minmaxLayers3DKernel<BLOCKDIM_X,BLOCKDIM_Y,BLOCKDIM_Z,RESULT_STEPS,HALO_STEPS,is_min><<<blocks, threads>>>
      (
       d_dst, d_src, w,h,d,kernel_radius
       );
    getLastCudaError("minmaxLayers3DKernel() execution failed\n");
    //checkCudaErrors(cudaDeviceSynchronize());
}


/*
 * Define Functions
 */

void UniformLarge3DFilter
(
 unsigned short *d_img, float *d_temp,
 float *d_result,
 int w, int h, int d, int radius_xy, int radius_z
) {
  uniformRows3D<L_ROW_BLOCKDIM_X,L_ROW_BLOCKDIM_Y,L_ROW_BLOCKDIM_Z,L_ROW_RESULT_STEPS,L_ROW_HALO_STEPS>(d_result, d_img, w,h,d,radius_xy);
  uniformColumns3D<L_COL_BLOCKDIM_X,L_COL_BLOCKDIM_Y,L_COL_BLOCKDIM_Z,L_COL_RESULT_STEPS,L_COL_HALO_STEPS>(d_temp, d_result, w,h,d,radius_xy);
  uniformLayers3D<L_LAY_BLOCKDIM_X,L_LAY_BLOCKDIM_Y,L_LAY_BLOCKDIM_Z,L_LAY_RESULT_STEPS,L_LAY_HALO_STEPS>(d_result, d_temp, w,h,d,radius_z);
}


void ErosionLarge3DFilter
(
 unsigned short *d_img, unsigned short *d_temp,
 unsigned short *d_result,
 int w, int h, int d, int radius_xy, int radius_z
 ) {
  minmaxRows3D<L_ROW_BLOCKDIM_X,L_ROW_BLOCKDIM_Y,L_ROW_BLOCKDIM_Z,L_ROW_RESULT_STEPS,L_ROW_HALO_STEPS,true>(d_result, d_img, w,h,d,radius_xy);
  minmaxColumns3D<L_COL_BLOCKDIM_X,L_COL_BLOCKDIM_Y,L_COL_BLOCKDIM_Z,L_COL_RESULT_STEPS,L_COL_HALO_STEPS,true>(d_temp, d_result, w,h,d,radius_xy);
  minmaxLayers3D<L_LAY_BLOCKDIM_X,L_LAY_BLOCKDIM_Y,L_LAY_BLOCKDIM_Z,L_LAY_RESULT_STEPS,L_LAY_HALO_STEPS,true>(d_result, d_temp, w,h,d,radius_z);
}

void DilationLarge3DFilter
(
 unsigned short *d_img, unsigned short *d_temp,
 unsigned short *d_result,
 int w, int h, int d, int radius_xy, int radius_z
 ) {
  minmaxRows3D<L_ROW_BLOCKDIM_X,L_ROW_BLOCKDIM_Y,L_ROW_BLOCKDIM_Z,L_ROW_RESULT_STEPS,L_ROW_HALO_STEPS,false>(d_result, d_img, w,h,d,radius_xy);
  minmaxColumns3D<L_COL_BLOCKDIM_X,L_COL_BLOCKDIM_Y,L_COL_BLOCKDIM_Z,L_COL_RESULT_STEPS,L_COL_HALO_STEPS,false>(d_temp, d_result, w,h,d,radius_xy);
  minmaxLayers3D<L_LAY_BLOCKDIM_X,L_LAY_BLOCKDIM_Y,L_LAY_BLOCKDIM_Z,L_LAY_RESULT_STEPS,L_LAY_HALO_STEPS,false>(d_result, d_temp, w,h,d,radius_z);
}

__global__ void Normalize3DKernel
(
 const unsigned short *d_src,
 const float *d_erosion,
 const float *d_dilation,
 float *d_dst, float min_intensity,
 const int width, const int height, const int depth
 ) {
  const int baseX = blockIdx.x * blockDim.x + threadIdx.x;
  const int baseY = blockIdx.y * blockDim.y + threadIdx.y;
  const int baseZ = blockIdx.z * blockDim.z + threadIdx.z;

  const int idx = (baseZ * height + baseY) * width + baseX;
  const float intensity = (float)d_src[idx];
  d_dst[idx] = (intensity >= min_intensity) ? (intensity-d_erosion[idx]) / (d_dilation[idx] - d_erosion[idx]) : 0;
}

__global__ void Copy3DKernel
(
 const unsigned short *d_src,
 float *d_dst, float min_intensity,
 const int width, const int height, const int depth
 ) {
  const int baseX = blockIdx.x * blockDim.x + threadIdx.x;
  const int baseY = blockIdx.y * blockDim.y + threadIdx.y;
  const int baseZ = blockIdx.z * blockDim.z + threadIdx.z;

  const int idx = (baseZ * height + baseY) * width + baseX;
  const float intensity = (float)d_src[idx];
  d_dst[idx] = (intensity >= min_intensity) ? intensity : 0;
}

void Normalize3DFilter
(
 unsigned short *d_img, float *d_norm,
 unsigned short *d_erosion_temp1, unsigned short *d_erosion_temp2,
 float *d_erosion_l, float *d_dilation_l,
 float min_intensity,
 const int width, const int height, const int depth,
 const int radius_large_xy, const int radius_large_z
 ) {
  if (radius_large_xy == 0 || radius_large_z == 0) {
    // skip normalize, just copy
    assert(width % (NORM_BLOCKDIM_X) == 0);
    assert(height % (NORM_BLOCKDIM_Y) == 0);
    assert(depth % (NORM_BLOCKDIM_Z) == 0);
    dim3 blocks(width / (NORM_BLOCKDIM_X), height/(NORM_BLOCKDIM_Y), depth / (NORM_BLOCKDIM_Z));
    dim3 threads(NORM_BLOCKDIM_X, NORM_BLOCKDIM_Y, NORM_BLOCKDIM_Z);
    Copy3DKernel<<<blocks, threads>>>(d_img,
                                      d_norm, min_intensity,
                                      width, height, depth);
    getLastCudaError("Error: Copy3DKernel() kernel execution FAILED!");
    //checkCudaErrors(cudaDeviceSynchronize());
  } else {
    float *d_uniform_temp = d_norm;

    ErosionLarge3DFilter(d_img, d_erosion_temp1, d_erosion_temp2,
                         width,height,depth,
                         radius_large_xy,radius_large_z);
    UniformLarge3DFilter(d_erosion_temp2, d_uniform_temp, d_erosion_l,
                         width,height,depth,
                         radius_large_xy,radius_large_z);

    DilationLarge3DFilter(d_img, d_erosion_temp1, d_erosion_temp2,
                          width,height,depth,
                          radius_large_xy,radius_large_z);
    UniformLarge3DFilter(d_erosion_temp2, d_uniform_temp, d_dilation_l,
                         width,height,depth,
                         radius_large_xy, radius_large_z);

    assert(width % (NORM_BLOCKDIM_X) == 0);
    assert(height % (NORM_BLOCKDIM_Y) == 0);
    assert(depth % (NORM_BLOCKDIM_Z) == 0);
    dim3 blocks(width / (NORM_BLOCKDIM_X), height/(NORM_BLOCKDIM_Y), depth / (NORM_BLOCKDIM_Z));
    dim3 threads(NORM_BLOCKDIM_X, NORM_BLOCKDIM_Y, NORM_BLOCKDIM_Z);

    Normalize3DKernel<<<blocks, threads>>>(d_img,
                                           d_erosion_l, d_dilation_l,
                                           d_norm, min_intensity,
                                           width, height, depth);
    getLastCudaError("Error: Normalize3DKernel() kernel execution FAILED!");
    //checkCudaErrors(cudaDeviceSynchronize());
  }
}
