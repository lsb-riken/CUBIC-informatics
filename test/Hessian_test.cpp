#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <iostream>
#include "../src/3D/NormalizeFilter.cuh"
#include "../src/3D/DoG3DFilter.cuh"
#include "../src/3D/Hessian3DFilter_element.cuh"
#include "../src/3D/utils.h"

Parameters p;

unsigned short *h_img = NULL;
float *h_dog = NULL;
char *h_hessian_pd = NULL;
float *h_hessian = NULL;

unsigned short *d_img = NULL;
unsigned short *d_erosion_tmp1 = NULL;
unsigned short *d_erosion_tmp2 = NULL;
float *d_erosion_l = NULL;
float *d_dilation_l = NULL;
float *d_norm = NULL;
float *d_temp1 = NULL;
float *d_temp2 = NULL;
float *d_dog = NULL;
float *d_hessian = NULL;
float *d_hessian_tmp = NULL;
char *d_hessian_pd = NULL;

void initialize(const int devID) {
  // cuda device init
  cudaDeviceProp deviceProp;
  deviceProp.major = 0;
  deviceProp.minor = 0;
  cudaSetDevice(devID);
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
  //checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));

  printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
         deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
  cudaFree(0); // to measure cudaMalloc correctly


  // allocating page-locked host memory
  // h_img: WriteCombined is fast for host->device only
  checkCudaErrors(cudaHostAlloc(&h_img, p.image_size*sizeof(unsigned short), cudaHostAllocWriteCombined));
  checkCudaErrors(cudaHostAlloc(&h_hessian_pd, p.image_size*sizeof(char), 0));
  checkCudaErrors(cudaHostAlloc(&h_hessian, p.image_size*sizeof(float)*6, 0));
  checkCudaErrors(cudaHostAlloc(&h_dog, p.image_size*sizeof(float), 0));

  // prepare device memory
  checkCudaErrors(cudaMalloc((void **)&d_img, (p.image_size*sizeof(unsigned short))));
  checkCudaErrors(cudaMalloc((void **)&d_norm, (p.image_size*sizeof(float))));
  checkCudaErrors(cudaMalloc((void **)&d_dog, (p.image_size*sizeof(float))));
  checkCudaErrors(cudaMalloc((void **)&d_hessian_tmp, (p.image_size*sizeof(float))));
  checkCudaErrors(cudaMalloc((void **)&d_hessian, (p.image_size*sizeof(float) * 6)));
  checkCudaErrors(cudaMalloc((void **)&d_hessian_pd, (p.image_size*sizeof(char))));
  d_erosion_tmp1 = reinterpret_cast<unsigned short*>(d_hessian);
  d_erosion_tmp2 = d_erosion_tmp1 + p.image_size;
  d_erosion_l = d_hessian + p.image_size;
  d_dilation_l = d_erosion_l + p.image_size;
  d_temp1 = d_hessian;
  d_temp2 = d_temp1 + p.image_size;
}

void finalize() {
  checkCudaErrors(cudaFree(d_img));
  checkCudaErrors(cudaFree(d_norm));
  checkCudaErrors(cudaFree(d_dog));
  checkCudaErrors(cudaFree(d_hessian_tmp));
  checkCudaErrors(cudaFree(d_hessian));
  checkCudaErrors(cudaFree(d_hessian_pd));
  checkCudaErrors(cudaFreeHost(h_img));
  checkCudaErrors(cudaFreeHost(h_hessian_pd));
  checkCudaErrors(cudaFreeHost(h_hessian));
  checkCudaErrors(cudaFreeHost(h_dog));
}

int main(int argc, char** argv) {

  loadParamFile(argv[1], p);

  initialize(p.devID);

  if (p.n_stack != 1) {
    std::cout << "A single stack should be provided!" << std::endl;
    exit(2);
  }
  if(p.list_stack_length[0] != p.depth){
    std::cout << "depth(=" << p.list_stack_length[0] << ") should be " << p.depth << std::endl;
    exit(2);
  }

  initGaussian3DKernel(p.sigma_dog.small_xy,
                       p.sigma_dog.small_z,
                       p.sigma_dog.large_xy,
                       p.sigma_dog.large_z);

  std::cout << "sigma_xy1: " << p.sigma_dog.small_xy << "\t"
            << "sigma_z1: "  << p.sigma_dog.small_z << "\t"
            << "sigma_xy2: " << p.sigma_dog.large_xy << "\t"
            << "sigma_z2: "  << p.sigma_dog.large_z << std::endl;
  std::cout << "radius_large_xy: " << p.radius_norm.large_xy << "\t"
            << "radius_large_z: "  << p.radius_norm.large_z <<  std::endl;


  // load images to host
  for(int i=0; i < p.depth; i++) {
    loadImage(p.list_src_path[0][i], &h_img[i*p.image_size2D], p.image_size2D);
  }

  // copy images to device
  checkCudaErrors(cudaMemcpyAsync((unsigned short *)d_img, (unsigned short *)h_img, p.image_size*sizeof(unsigned short), cudaMemcpyHostToDevice));

  // launch kernels
  Normalize3DFilter(d_img, d_norm, d_erosion_tmp1, d_erosion_tmp2,
                    d_erosion_l, d_dilation_l,
                    p.min_intensity_truncate, p.width, p.height, p.depth,
                    p.radius_norm.large_xy, p.radius_norm.large_z);

  DoG3DFilter(d_norm, d_temp1, d_temp2, d_dog,
              p.width, p.height, p.depth, p.gamma_n);

  HessianPositiveDefiniteWithElement(d_hessian, d_hessian_pd,
                                     d_dog, d_hessian_tmp,
                                     p.width, p.height, p.depth);

  // download to check the result
  checkCudaErrors(cudaMemcpyAsync((char *)h_hessian_pd, (char *)d_hessian_pd, p.image_size*sizeof(char), cudaMemcpyDeviceToHost));
  if(p.verbose) {
    checkCudaErrors(cudaMemcpyAsync((char *)h_hessian, (char *)d_hessian, p.image_size*sizeof(float)*6, cudaMemcpyDeviceToHost));
  }
  checkCudaErrors(cudaMemcpyAsync((char *)h_dog, (char *)d_dog, p.image_size*sizeof(float), cudaMemcpyDeviceToHost));

  //barrier (without this, next depth kernel would overlap!!)
  checkCudaErrors(cudaDeviceSynchronize());

  // save images to disk
  if(p.verbose) {
    if(argc == 5) {
      saveImage(h_dog, argv[2], p.image_size);
      saveImage(h_hessian_pd, argv[3], p.image_size);
      saveImage(h_hessian, argv[4], p.image_size*6);
    } else if (argc == 4) {
      saveImage(h_dog, argv[2], p.image_size);
      saveImage(h_hessian_pd, argv[3], p.image_size);
      saveImage(h_hessian, "hessian.bin", p.image_size*6);
    }
  } else {
    if(argc == 4) {
      saveImage(h_dog, argv[2], p.image_size);
      saveImage(h_hessian_pd, argv[3], p.image_size);
    } else {
      saveImage(h_dog, "dog.bin", p.image_size);
      saveImage(h_hessian_pd, "hessian_pd.bin", p.image_size);
    }
  }

  finalize();
  return 0;
}
