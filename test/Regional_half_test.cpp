#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_fp16.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <climits>
#include <vector>
#include "../src/3D/Hessian3DFilter_element.cuh"
#include "../src/3D/DoG3DFilter.h"
#include "../src/3D/CCL3D.h"
#include "../src/3D/RegionalFeatures.cuh"
#include "../src/3D/Eigenvalues.cuh"

const int width = 2560, height = 2160;
const int depth = 32;
const float gamma_n = 1.0;

unsigned short *h_img = NULL;
unsigned short *d_img = NULL;
float *d_temp1 = NULL;
float *d_temp2 = NULL;
float *d_dog = NULL;
half *d_hessian = NULL;
char *d_hessian_pd = NULL;
int *d_labels = NULL;

int *d_labels_tmp = NULL;
half *d_hessian_tmp = NULL;

int *d_labels_region = NULL;
unsigned short *d_size_region = NULL;
float *d_eigen_region = NULL;

int *h_labels = NULL;
int *h_labels_region = NULL;
unsigned short *h_size_region = NULL;
unsigned short *h_img_region = NULL;
half *h_hessian_region = NULL;
float *h_eigen_region = NULL;

void loadImage(const std::string fname, const unsigned short *h_img) {
  std::fstream f(fname, std::ios::in|std::ios::binary);

  if(!f.is_open()) {
    std::cout << "Unable to open " << fname << std::endl;
    exit(2);
  }
  f.read((char *)h_img, width*height*sizeof(unsigned short));
  f.close();
}

template<typename T>
void saveImage(const T* img_result, const std::string fname, const long long size=width*height*sizeof(T)) {
  //std::cout << "saveImage(" << fname << ")" << std::endl;
  std::fstream f(fname, std::ios::out|std::ios::binary);
  if(!f.is_open()) {
    std::cout << "Unable to open " << fname << std::endl;
    exit(2);
  }
  f.write((char *)img_result, size);
  f.close();

}

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
  checkCudaErrors(cudaHostAlloc(&h_img, width*height*depth*sizeof(unsigned short), cudaHostAllocWriteCombined));
  checkCudaErrors(cudaHostAlloc(&h_labels, width*height*depth*sizeof(int), 0));
  checkCudaErrors(cudaHostAlloc(&h_labels_region, width*height*depth*sizeof(int), 0));
  checkCudaErrors(cudaHostAlloc(&h_img_region, width*height*depth*sizeof(unsigned short), 0));
  checkCudaErrors(cudaHostAlloc(&h_hessian_region, 6*width*height*depth*sizeof(half), 0));
  checkCudaErrors(cudaHostAlloc(&h_size_region, width*height*depth*sizeof(unsigned short), 0));
  checkCudaErrors(cudaHostAlloc(&h_eigen_region, 2*width*height*depth*sizeof(float), 0));

  // prepare device memory
  checkCudaErrors(cudaMalloc((void **)&d_img, (width*height*depth*sizeof(unsigned short))));
  checkCudaErrors(cudaMalloc((void **)&d_labels, (width*height*depth*sizeof(int))));
  checkCudaErrors(cudaMalloc((void **)&d_labels_tmp, (width*height*depth*sizeof(int))));
  checkCudaErrors(cudaMalloc((void **)&d_labels_region, (width*height*depth*sizeof(int))));
  checkCudaErrors(cudaMalloc((void **)&d_hessian, (6*width*height*depth*sizeof(half))));
  checkCudaErrors(cudaMalloc((void **)&d_hessian_pd, (width*height*depth*sizeof(char))));
  checkCudaErrors(cudaMalloc((void **)&d_hessian_tmp, (width*height*depth*sizeof(half))));
  checkCudaErrors(cudaMalloc((void **)&d_size_region, (width*height*depth*sizeof(unsigned short))));
  checkCudaErrors(cudaMalloc((void **)&d_eigen_region, (2*width*height*depth*sizeof(float))));
  d_temp1 = reinterpret_cast<float*>(d_hessian);
  d_temp2 = d_temp1 + width*height*depth;
  d_dog = d_temp2 + width*height*depth;
}

void finalize() {
  checkCudaErrors(cudaFree(d_img));
  checkCudaErrors(cudaFree(d_labels));
  checkCudaErrors(cudaFree(d_labels_tmp));
  checkCudaErrors(cudaFree(d_labels_region));
  checkCudaErrors(cudaFree(d_hessian));
  checkCudaErrors(cudaFree(d_hessian_pd));
  checkCudaErrors(cudaFree(d_hessian_tmp));
  checkCudaErrors(cudaFree(d_size_region));
  checkCudaErrors(cudaFree(d_eigen_region));
  checkCudaErrors(cudaFreeHost(h_img));
  checkCudaErrors(cudaFreeHost(h_labels));
  checkCudaErrors(cudaFreeHost(h_labels_region));
  checkCudaErrors(cudaFreeHost(h_img_region));
  checkCudaErrors(cudaFreeHost(h_hessian_region));
  checkCudaErrors(cudaFreeHost(h_size_region));
  checkCudaErrors(cudaFreeHost(h_eigen_region));
}

int main(int argc, char** argv) {

  initialize(0);

  std::ifstream param_file(argv[1]);
  if(!param_file.is_open()) {
    std::cout << "Unable to open file." << std::endl;
    exit(2);
  }

  std::string line;
  std::string list_src_path[depth];
  std::string list_dst_path[depth];
  std::string sigma_str[4];
  float sigma_f[4];

  // load sigma params
  std::getline(param_file, line);
  std::istringstream stream(line);
  for(int j=0; j < 4; j++) {
    std::getline(stream, sigma_str[j], '\t');
    sigma_f[j] = std::stof(sigma_str[j]);
  }
  initGaussian3DKernel(sigma_f[0], sigma_f[1], sigma_f[2], sigma_f[3]);

  // load paths of images
  int count = 0;
  while(!param_file.eof()) {
    std::getline(param_file, line);
    if(line.length() == 0) {
      break;
    }
    std::istringstream stream(line);
    std::getline(stream, list_src_path[count], '\t');
    std::getline(stream, list_dst_path[count], '\t');
    count += 1;
  }
  param_file.close();

  if(count != depth){
    std::cout << "depth(=" << count << ") should be " << depth << std::endl;
    exit(2);
  }

  // load images to host
  for(int i=0; i < depth; i++) {
    loadImage(list_src_path[i], &h_img[i*width*height]);
  }

  // copy images to device
  checkCudaErrors(cudaMemcpyAsync((unsigned short *)d_img, (unsigned short *)h_img, width*height*count*sizeof(unsigned short), cudaMemcpyHostToDevice));
  // launch kernels
  DoG3DFilter(d_img, d_temp1, d_temp2, d_dog, width, height, depth, gamma_n);
  HessianPositiveDefiniteWithElement(d_hessian, d_hessian_pd, d_dog, width, height, depth);
  CCL(d_hessian_pd, d_labels, width, height, depth);

  int h_num_regions;
  int *d_num_regions;
  checkCudaErrors(cudaMalloc((void **)&d_num_regions, sizeof(int)));
  half *d_hessian_region = d_hessian;
  HessianFeatures(d_labels, d_hessian,
                  d_labels_tmp, d_hessian_tmp,
                  d_labels_region, d_hessian_region,
                  d_num_regions, width, height, depth);
  // get num_regions in host
  checkCudaErrors(cudaMemcpy(&h_num_regions, d_num_regions, sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "num_regions: " << h_num_regions << std::endl;

  Eigenvalues(d_hessian_region, d_eigen_region, h_num_regions, width*height*depth);

  unsigned short *d_img_tmp = reinterpret_cast<unsigned short*>(d_hessian_tmp);
  unsigned short *d_img_region = d_img;
  RegionalSizeAndMaxIntensity(d_labels, d_img, d_labels_tmp, d_img_tmp,
               d_labels_region, d_img_region, d_size_region,
               d_num_regions, width, height, depth);

  checkCudaErrors(cudaFree(d_num_regions));

  // download to check the result
  checkCudaErrors(cudaMemcpyAsync((char *)h_labels, (char *)d_labels, width*height*depth*sizeof(int), cudaMemcpyDeviceToHost));
  // download only number of regions
  checkCudaErrors(cudaMemcpyAsync((char *)h_size_region, (char *)d_size_region, h_num_regions*sizeof(unsigned short), cudaMemcpyDeviceToHost));
  int background_size = h_size_region[0];
  checkCudaErrors(cudaMemcpyAsync((char *)h_labels_region, (char *)d_labels_region, h_num_regions*sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpyAsync((char *)h_img_region, (char *)d_img_region, h_num_regions*sizeof(unsigned short), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpyAsync((char *)h_eigen_region, (char *)d_eigen_region, 2*width*height*depth*sizeof(float), cudaMemcpyDeviceToHost));
  //barrier (without this, next depth kernel would overlap!!)
  checkCudaErrors(cudaDeviceSynchronize());

  std::cout << "Saving..." << std::endl;
  // save images to disk
  saveImage<int>(h_labels, "labels.bin", width*height*depth*sizeof(int));
  saveImage<int>(h_labels_region, "labels_region.bin", h_num_regions*sizeof(int));
  saveImage<unsigned short>(h_size_region, "size_region.bin", h_num_regions*sizeof(unsigned short));
  saveImage<unsigned short>(h_img_region, "intensity_region.bin", h_num_regions*sizeof(unsigned short));

  saveImage<float>(h_eigen_region, "structureness_region.bin", h_num_regions*sizeof(float));
  saveImage<float>(h_eigen_region+width*height*depth, "blobness_region.bin", h_num_regions*sizeof(float));

  finalize();
  return 0;
}
