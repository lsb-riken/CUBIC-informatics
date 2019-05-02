#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <iostream>
#include <chrono>
#include "../src/3D/NormalizeFilter.cuh"
#include "../src/3D/DoG3DFilter.cuh"
#include "../src/3D/Hessian3DFilter_element.cuh"
#include "../src/3D/CCL3D.cuh"
#include "../src/3D/RegionalFeatures.cuh"
#include "../src/3D/Eigenvalues.cuh"
#include "../src/3D/utils.h"

Parameters p;

// host memory
unsigned short *h_img = NULL;
float *h_norm = NULL;
int *h_labels = NULL;
int *h_labels_region = NULL;
unsigned short *h_size_region = NULL;
float *h_maxnorm_region = NULL;
float *h_eigen_region = NULL;
float *h_grid_region = NULL;
int h_num_regions;

// device memory
unsigned short *d_img = NULL;
float *d_hessian = NULL;
char *d_hessian_pd = NULL;
int *d_labels = NULL;
int *d_labels_tmp = NULL;
float *d_hessian_tmp = NULL;
int *d_labels_region = NULL;
unsigned short *d_size_region = NULL;
float *d_eigen_region = NULL;
int* d_num_regions = NULL;
void *d_cub_tmp = NULL;

// recycled device memory
float *d_temp1 = NULL;
float *d_temp2 = NULL;
float *d_norm = NULL;
float *d_dog = NULL;
float *d_hessian_region = NULL;
float *d_grid = NULL;
float *d_grid_region = NULL;
float *d_grid_tmp = NULL;
float *d_norm_tmp = NULL;
float *d_maxnorm_region = NULL;
unsigned short *d_erosion_tmp1 = NULL;
unsigned short *d_erosion_tmp2 = NULL;
float *d_erosion_l = NULL;
float *d_dilation_l = NULL;


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
  checkCudaErrors(cudaHostAlloc(&h_norm, p.image_size*sizeof(float), 0));
  checkCudaErrors(cudaHostAlloc(&h_labels, p.image_size*sizeof(int), 0));
  checkCudaErrors(cudaHostAlloc(&h_labels_region, p.image_size*sizeof(int), 0));
  checkCudaErrors(cudaHostAlloc(&h_maxnorm_region, p.image_size*sizeof(float), 0));
  checkCudaErrors(cudaHostAlloc(&h_size_region, p.image_size*sizeof(unsigned short), 0));
  checkCudaErrors(cudaHostAlloc(&h_eigen_region, 2*p.image_size*sizeof(float), 0));
  checkCudaErrors(cudaHostAlloc(&h_grid_region, 3*p.image_size*sizeof(float), 0));

  // prepare device memory
  checkCudaErrors(cudaMalloc((void **)&d_img, (p.image_size*sizeof(unsigned short))));
  checkCudaErrors(cudaMalloc((void **)&d_norm, (p.image_size*sizeof(float))));
  checkCudaErrors(cudaMalloc((void **)&d_labels, (p.image_size*sizeof(int))));
  checkCudaErrors(cudaMalloc((void **)&d_labels_tmp, (p.image_size*sizeof(int))));
  checkCudaErrors(cudaMalloc((void **)&d_labels_region, (p.image_size*sizeof(int))));
  checkCudaErrors(cudaMalloc((void **)&d_hessian, (6*p.image_size*sizeof(float))));
  checkCudaErrors(cudaMalloc((void **)&d_hessian_pd, (p.image_size*sizeof(char))));
  checkCudaErrors(cudaMalloc((void **)&d_hessian_tmp, (p.image_size*sizeof(float))));
  checkCudaErrors(cudaMalloc((void **)&d_size_region, (p.image_size*sizeof(unsigned short))));
  checkCudaErrors(cudaMalloc((void **)&d_eigen_region, (2*p.image_size*sizeof(float))));
  checkCudaErrors(cudaMalloc((void **)&d_num_regions, sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_cub_tmp, p.cub_tmp_bytes));

  // recycle device memory
  d_erosion_tmp1 = reinterpret_cast<unsigned short*>(d_hessian);
  d_erosion_tmp2 = d_erosion_tmp1 + p.image_size;
  d_erosion_l = d_hessian + p.image_size;
  d_dilation_l = d_erosion_l + p.image_size;
  d_temp1 = d_hessian;
  d_temp2 = d_temp1 + p.image_size;
  d_dog = reinterpret_cast<float*>(d_labels_tmp);
  d_hessian_region = d_hessian;
  d_grid = d_hessian;
  d_grid_region = d_hessian + p.image_size;
  d_maxnorm_region = d_hessian + 4*p.image_size;
  d_grid_tmp = d_hessian_tmp;
  d_norm_tmp = d_hessian_tmp;
}

void finalize() {
  checkCudaErrors(cudaFree(d_img));
  checkCudaErrors(cudaFree(d_norm));
  checkCudaErrors(cudaFree(d_labels));
  checkCudaErrors(cudaFree(d_labels_tmp));
  checkCudaErrors(cudaFree(d_labels_region));
  checkCudaErrors(cudaFree(d_hessian));
  checkCudaErrors(cudaFree(d_hessian_pd));
  checkCudaErrors(cudaFree(d_hessian_tmp));
  checkCudaErrors(cudaFree(d_size_region));
  checkCudaErrors(cudaFree(d_eigen_region));
  checkCudaErrors(cudaFree(d_num_regions));
  checkCudaErrors(cudaFree(d_cub_tmp));
  checkCudaErrors(cudaFreeHost(h_img));
  checkCudaErrors(cudaFreeHost(h_norm));
  checkCudaErrors(cudaFreeHost(h_labels));
  checkCudaErrors(cudaFreeHost(h_labels_region));
  checkCudaErrors(cudaFreeHost(h_size_region));
  checkCudaErrors(cudaFreeHost(h_eigen_region));
  checkCudaErrors(cudaFreeHost(h_grid_region));
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
  CCL(d_hessian_pd, d_labels, p.width, p.height, p.depth);

  HessianFeatures(d_labels, d_hessian,
                  d_labels_tmp, d_hessian_tmp,
                  d_labels_region, d_hessian_region,
                  d_cub_tmp, p.cub_tmp_bytes,
                  d_num_regions, p.width, p.height, p.depth);
  // get num_regions in host
  checkCudaErrors(cudaMemcpy(&h_num_regions, d_num_regions, sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "num_regions: " << h_num_regions << std::endl;

  Eigenvalues(d_hessian_region, d_eigen_region, h_num_regions, p.image_size);

  MaxNormalized(d_labels, d_norm, d_labels_tmp, d_norm_tmp,
                d_labels_region, d_maxnorm_region,
                d_cub_tmp, p.cub_tmp_bytes,
                d_num_regions, p.width, p.height, p.depth);

  RegionalSizeAndCentroid(d_labels, d_grid,
                          d_labels_tmp, d_grid_tmp,
                          d_labels_region, d_size_region, d_grid_region,
                          d_cub_tmp, p.cub_tmp_bytes,
                          d_num_regions, h_num_regions,
                          p.width, p.height, p.depth);

  // download to check the result
  checkCudaErrors(cudaMemcpyAsync((char *)h_norm, (char *)d_norm, p.image_size*sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpyAsync((char *)h_labels, (char *)d_labels, p.image_size*sizeof(int), cudaMemcpyDeviceToHost));

  // download only number of regions
  checkCudaErrors(cudaMemcpyAsync((char *)h_size_region, (char *)d_size_region, h_num_regions*sizeof(unsigned short), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpyAsync((char *)h_labels_region, (char *)d_labels_region, h_num_regions*sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpyAsync((char *)h_maxnorm_region, (char *)d_maxnorm_region, h_num_regions*sizeof(float), cudaMemcpyDeviceToHost));
  for(int i = 0; i < 3; i++) {
    checkCudaErrors(cudaMemcpyAsync((char *)(h_grid_region+i*p.image_size), (char *)(d_grid_region+i*p.image_size), h_num_regions*sizeof(float), cudaMemcpyDeviceToHost));
  }
  for(int i = 0; i < 2; i++) {
    checkCudaErrors(cudaMemcpyAsync((char *)(h_eigen_region+i*p.image_size), (char *)(d_eigen_region+i*p.image_size), h_num_regions*sizeof(float), cudaMemcpyDeviceToHost));
  }
  //barrier (without this, next depth kernel would overlap!!)
  checkCudaErrors(cudaDeviceSynchronize());

  std::cout << "Saving..." << std::endl;
  // save images to disk
  saveImage(h_labels, "labels.bin", p.image_size);
  saveImage(h_norm, "normalized.bin", p.image_size);
  saveImage(h_labels_region+1, "labels_region.bin", h_num_regions-1);
  saveImage(h_size_region+1, "size_region.bin", h_num_regions-1);
  saveImage(h_maxnorm_region+1, "max_normalized_region.bin", h_num_regions-1);
  saveImage(h_eigen_region+1, "structureness_region.bin", h_num_regions-1);
  saveImage(h_eigen_region+1+p.image_size, "blobness_region.bin", h_num_regions-1);
  saveImage(h_grid_region+1, "centroid_x.bin", h_num_regions-1);
  saveImage(h_grid_region+1+p.image_size, "centroid_y.bin", h_num_regions-1);
  saveImage(h_grid_region+1+p.image_size*2, "centroid_z.bin", h_num_regions-1);

  finalize();
  return 0;
}
