/*
 * CCL3D.cu
 */

#include <assert.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "CCL3D.cuh"

#define CCL_BLOCK_SIZE_X 8
#define CCL_BLOCK_SIZE_Y 8
#define CCL_BLOCK_SIZE_Z 8

__device__ int d_isNotDone;

__global__ void initLabels(char* img, int* label, int w, int h, int d) {
    const int x = blockIdx.x * CCL_BLOCK_SIZE_X + threadIdx.x;
    const int y = blockIdx.y * CCL_BLOCK_SIZE_Y + threadIdx.y;
    const int z = blockIdx.z * CCL_BLOCK_SIZE_Z + threadIdx.z;
    const int index = (z*h + y)*w + x;

    if ( x >= w || y >= h || z >= d) return;

    label[index] = index * img[index];
}

__global__ void scanLabels(int* labels, int w, int h, int d) {
    const int x = blockIdx.x * CCL_BLOCK_SIZE_X + threadIdx.x;
    const int y = blockIdx.y * CCL_BLOCK_SIZE_Y + threadIdx.y;
    const int z = blockIdx.z * CCL_BLOCK_SIZE_Z + threadIdx.z;
    const int index = (z*h + y)*w + x;

    if (x >= w || y >= h || z >= d) return;

    const int Z1 = w*h; const int Y1 = w;

    int lcur = labels[index];
    if (lcur) {
      int lmin = index; // MAX
      // 26-neighbors
      int lne, pos;
      for (int Zdif = -Z1; Zdif <= Z1; Zdif += Z1) {
        for (int Ydif = -Y1; Ydif <= Y1; Ydif += Y1) {
          for (int Xdif = -1; Xdif <= 1; Xdif += 1) {
            pos = index + Zdif + Ydif + Xdif;
            lne = (pos >= 0 && pos < w*h*d) ? labels[pos] : 0; // circular boundary
            if (lne && lne < lmin) lmin = lne;
          }
        }
      }
      // need not (Xdif,Ydif,Zdif)=(0,0,0) but no problem

      if (lmin < lcur) {
        int lpa = labels[lcur];
        labels[lpa] = min(lpa, lmin);
        d_isNotDone = 1;
      }
    }
}

__global__ void analyseLabels(int* labels, int w, int h, int d) {
    const int x = blockIdx.x * CCL_BLOCK_SIZE_X + threadIdx.x;
    const int y = blockIdx.y * CCL_BLOCK_SIZE_Y + threadIdx.y;
    const int z = blockIdx.z * CCL_BLOCK_SIZE_Z + threadIdx.z;
    const int index = (z*h + y)*w + x;

    if (x >= w || y >= h || z >= d) return;

    int lcur = labels[index];
    if (lcur) {
      int r = labels[lcur];
      while(r != lcur) {
        lcur = labels[r];
        r = labels[lcur];
      }
      labels[index] = lcur;
    }
}

extern "C" void CCL(char *d_img, int *d_labels, int w, int h, int d) {
  //checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared)); //?

  assert(w % CCL_BLOCK_SIZE_X == 0);
  assert(h % CCL_BLOCK_SIZE_Y == 0);
  assert(d % CCL_BLOCK_SIZE_Z == 0);


  dim3 blocks (ceil(float(w)/CCL_BLOCK_SIZE_X),
               ceil(float(h)/CCL_BLOCK_SIZE_Y),
               ceil(float(d)/CCL_BLOCK_SIZE_Z));
  dim3 threads (CCL_BLOCK_SIZE_X, CCL_BLOCK_SIZE_Y, CCL_BLOCK_SIZE_Z);
  initLabels<<<blocks, threads>>>(d_img, d_labels, w, h, d);
  checkCudaErrors(cudaPeekAtLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  int isNotDone;
  while(1) {
    isNotDone = 0;
    checkCudaErrors(cudaMemcpyToSymbol(d_isNotDone, &isNotDone, sizeof(int), 0, cudaMemcpyHostToDevice));

    //std::cout << "scan " << std::endl;
    scanLabels<<<blocks, threads>>>(d_labels, w, h, d);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpyFromSymbol(&isNotDone, d_isNotDone, sizeof(int), 0, cudaMemcpyDeviceToHost));
    if(isNotDone) {
      //std::cout << "analysis " << std::endl;
      analyseLabels<<<blocks, threads>>>(d_labels, w, h, d);
      checkCudaErrors(cudaPeekAtLastError());
      checkCudaErrors(cudaDeviceSynchronize());
    }else {
      break;
    }
  }
}
